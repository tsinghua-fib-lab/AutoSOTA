import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import FOSTERNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy
from utils.hc_soinn_classifier import HCSOINNClassifier

# Please refer to https://github.com/G-U-N/ECCV22-FOSTER for the full source code to reproduce foster.

EPSILON = 1e-8


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = FOSTERNet(args, True)
        self._snet = None
        self.beta1 = args["beta1"]
        self.beta2 = args["beta2"]
        self.per_cls_weights = None
        self.is_teacher_wa = args["is_teacher_wa"]
        self.is_student_wa = args["is_student_wa"]
        self.lambda_okd = args["lambda_okd"]
        self.wa_value = args["wa_value"]
        self.oofc = args["oofc"].lower()

        # ---------------- HC-SOINN plugin (optional, eval-only head) ----------------
        self.use_hc_soinn = args.get("use_hc_soinn", False)
        self.hc_soinn = None
        if self.use_hc_soinn:
            logging.info("[FOSTER] HC-SOINN enabled (evaluation head)")
            self.hc_soinn = HCSOINNClassifier(
                max_prototypes_per_class=args.get("hcsoinn_max_proto_per_class", 20),
                alpha=args.get("hcsoinn_alpha", 0.5),
                tau_merge=args.get("hcsoinn_tau_merge", 0.2),
                tau_reject=args.get("hcsoinn_tau_reject", 2.0),
                linkage_method=args.get("hcsoinn_linkage", "average"),
                distance_metric=args.get("hcsoinn_distance", "cosine"),
                use_soinn_refinement=args.get("hcsoinn_use_soinn_refinement", True),
                soinn_ad=args.get("hcsoinn_soinn_ad", 20),
                soinn_lam=args.get("hcsoinn_soinn_lam", 20),
                soinn_threshold_scale=args.get("hcsoinn_soinn_threshold_scale", 0.5),
                soinn_max_iter=args.get("hcsoinn_soinn_max_iter", 3),
                soinn_max_degree_for_removal=args.get("hcsoinn_soinn_max_degree_for_removal", 1),
            )

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1
        if self._cur_task > 1:
            self._network = self._snet
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        self._network_module_ptr = self._network
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        if self._cur_task > 0:
            for p in self._network.backbones[0].parameters():
                p.requires_grad = False
            for p in self._network.oldfc.parameters():
                p.requires_grad = False

        logging.info("All params: {}".format(count_parameters(self._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(self._network, True))
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args["batch_size"],
            shuffle=True,
            num_workers=self.args["num_workers"],
            pin_memory=True,
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.args["batch_size"],
            shuffle=False,
            num_workers=self.args["num_workers"],
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)

        # Build/update HC-SOINN bank after finishing this task (class-incremental: only current task classes)
        if getattr(self, "use_hc_soinn", False) and self.hc_soinn is not None:
            try:
                self._build_hc_soinn_bank()
            except Exception as e:
                logging.error(f"[FOSTER] HC-SOINN build/compress error: {e}", exc_info=True)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    # ------------------------------------------------------------------ #
    # HC-SOINN helpers (feature extraction, bank build, eval)
    # ------------------------------------------------------------------ #
    def _get_eval_model(self):
        """
        Prefer the compressed student net (SNet) for evaluation when available.
        This matches FOSTER's intended workflow: boosting -> compression -> evaluate SNet.
        """
        return self._snet if self._snet is not None else self._network

    def _get_hc_soinn_feature_fn(self):
        """
        Return a feature extractor for HC-SOINN.
        We use the *last backbone* feature block only (fixed dim = out_dim) so the feature dimension
        stays constant across tasks even though FOSTER's full FC uses concatenated features.
        """
        model = self._get_eval_model()
        if isinstance(model, nn.DataParallel):
            model = model.module

        out_dim = getattr(model, "out_dim", None)

        def _fn(x):
            feats = model.extract_vector(x)
            # Keep only the newest backbone feature chunk to ensure consistent dimension across tasks
            if out_dim is not None and feats.shape[1] >= out_dim:
                feats = feats[:, -out_dim:]
            return feats

        return _fn

    def _build_hc_soinn_bank(self):
        """
        Build/update HC-SOINN prototypes using *current task* training data only (new classes),
        similar to coda_prompt._build_hc_soinn_bank().
        """
        feature_fn = self._get_hc_soinn_feature_fn()
        eval_model = self._get_eval_model()
        eval_model.eval()

        current_task_dataset = self.data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="test"
        )
        current_task_loader = DataLoader(
            current_task_dataset,
            batch_size=self.args["batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=self.args["num_workers"],
        )

        feats_list, lbs_list = [], []
        with torch.no_grad():
            for _, inputs, targets in current_task_loader:
                inputs = inputs.to(self._device)
                feats = feature_fn(inputs)
                feats = feats.detach().cpu()
                if len(feats.shape) == 1:
                    feats = feats.reshape(1, -1)
                feats_list.append(feats.numpy())
                lbs_list.append(targets.numpy())

        if len(feats_list) == 0:
            return

        feats_np = np.concatenate(feats_list, axis=0)
        lbs_np = np.concatenate(lbs_list, axis=0)
        self.hc_soinn.add_features(feats_np, lbs_np)
        self.hc_soinn.compress()

    def _eval_fc(self, loader):
        """Evaluate using the FOSTER FC logits (prefer SNet when available)."""
        model = self._get_eval_model()
        model.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)

    def _eval_hc_soinn(self, loader):
        """Evaluate using HC-SOINN prototypes (feature space: last backbone chunk)."""
        model = self._get_eval_model()
        model.eval()
        y_pred, y_true = [], []
        feature_fn = self._get_hc_soinn_feature_fn()
        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(loader):
                inputs = inputs.to(self._device)
                feats = feature_fn(inputs).detach().cpu()
                if len(feats.shape) == 1:
                    feats = feats.reshape(1, -1)
                feats_np = feats.numpy()
                topk_pred = self.hc_soinn.predict_topk(
                    feats_np, self.topk, self._total_classes, device=self._device
                )
                y_pred.append(topk_pred)
                y_true.append(targets.cpu().numpy())

        if len(y_pred) == 0:
            logging.warning("[FOSTER] No predictions generated from HC-SOINN evaluation")
            return np.array([]), np.array([])

        return np.concatenate(y_pred), np.concatenate(y_true)

    def eval_task(self):
        """
        Return a dict of evaluation results compatible with trainer.py (fc + optional hc_soinn).
        """
        results = {}
        y_pred_fc, y_true_fc = self._eval_fc(self.test_loader)
        results["fc"] = self._evaluate(y_pred_fc, y_true_fc)

        if getattr(self, "use_hc_soinn", False) and self.hc_soinn is not None:
            y_pred_hc, y_true_hc = self._eval_hc_soinn(self.test_loader)
            results["hc_soinn"] = self._evaluate(y_pred_hc, y_true_hc)

        return results

    def train(self):
        self._network_module_ptr.train()
        self._network_module_ptr.backbones[-1].train()
        if self._cur_task >= 1:
            self._network_module_ptr.backbones[0].eval()

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if hasattr(self._network, "module"):
            self._network_module_ptr = self._network.module
        if self._cur_task == 0:
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                momentum=0.9,
                lr=self.args["init_lr"],
                weight_decay=self.args["init_weight_decay"],
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.args["init_epochs"]
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:

            cls_num_list = [self.samples_old_class] * self._known_classes + [
                self.samples_new_class(i)
                for i in range(self._known_classes, self._total_classes)
            ]

            effective_num = 1.0 - np.power(self.beta1, cls_num_list)
            per_cls_weights = (1.0 - self.beta1) / np.array(effective_num)
            per_cls_weights = (
                per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            )

            logging.info("per cls weights : {}".format(per_cls_weights))
            self.per_cls_weights = torch.FloatTensor(per_cls_weights).to(self._device)

            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=self.args["lr"],
                momentum=0.9,
                weight_decay=self.args["weight_decay"],
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.args["boosting_epochs"]
            )
            if self.oofc == "az":
                for i, p in enumerate(self._network_module_ptr.fc.parameters()):
                    if i == 0:
                        p.data[
                            self._known_classes :, : self._network_module_ptr.out_dim
                        ] = torch.tensor(0.0)
            elif self.oofc != "ft":
                assert 0, "not implemented"
            self._feature_boosting(train_loader, test_loader, optimizer, scheduler)
            if self.is_teacher_wa:
                self._network_module_ptr.weight_align(
                    self._known_classes,
                    self._total_classes - self._known_classes,
                    self.wa_value,
                )
            else:
                logging.info("do not weight align teacher!")

            cls_num_list = [self.samples_old_class] * self._known_classes + [
                self.samples_new_class(i)
                for i in range(self._known_classes, self._total_classes)
            ]
            effective_num = 1.0 - np.power(self.beta2, cls_num_list)
            per_cls_weights = (1.0 - self.beta2) / np.array(effective_num)
            per_cls_weights = (
                per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            )
            logging.info("per cls weights : {}".format(per_cls_weights))
            self.per_cls_weights = torch.FloatTensor(per_cls_weights).to(self._device)
            self._feature_compression(train_loader, test_loader)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["init_epochs"]))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(
                    self._device, non_blocking=True
                ), targets.to(self._device, non_blocking=True)
                logits = self._network(inputs)["logits"]
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epochs"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epochs"],
                    losses / len(train_loader),
                    train_acc,
                )

            prog_bar.set_description(info)
        logging.info(info)

    def _feature_boosting(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["boosting_epochs"]))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.0
            losses_clf = 0.0
            losses_fe = 0.0
            losses_kd = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(
                    self._device, non_blocking=True
                ), targets.to(self._device, non_blocking=True)
                outputs = self._network(inputs)
                logits, fe_logits, old_logits = (
                    outputs["logits"],
                    outputs["fe_logits"],
                    outputs["old_logits"].detach(),
                )
                loss_clf = F.cross_entropy(logits / self.per_cls_weights, targets)
                loss_fe = F.cross_entropy(fe_logits, targets)
                loss_kd = self.lambda_okd * _KD_loss(
                    logits[:, : self._known_classes], old_logits, self.args["T"]
                )
                loss = loss_clf + loss_fe + loss_kd
                optimizer.zero_grad()
                loss.backward()
                if self.oofc == "az":
                    for i, p in enumerate(self._network_module_ptr.fc.parameters()):
                        if i == 0:
                            p.grad.data[
                                self._known_classes :,
                                : self._network_module_ptr.out_dim,
                            ] = torch.tensor(0.0)
                elif self.oofc != "ft":
                    assert 0, "not implemented"
                optimizer.step()
                losses += loss.item()
                losses_fe += loss_fe.item()
                losses_clf += loss_clf.item()
                losses_kd += (
                    self._known_classes / self._total_classes
                ) * loss_kd.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_fe {:.3f}, Loss_kd {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["boosting_epochs"],
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_fe / len(train_loader),
                    losses_kd / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_fe {:.3f}, Loss_kd {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["boosting_epochs"],
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_fe / len(train_loader),
                    losses_kd / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

    def _feature_compression(self, train_loader, test_loader):
        self._snet = FOSTERNet(self.args, True)
        self._snet.update_fc(self._total_classes)
        if len(self._multiple_gpus) > 1:
            self._snet = nn.DataParallel(self._snet, self._multiple_gpus)
        if hasattr(self._snet, "module"):
            self._snet_module_ptr = self._snet.module
        else:
            self._snet_module_ptr = self._snet
        self._snet.to(self._device)
        self._snet_module_ptr.backbones[0].load_state_dict(
            self._network_module_ptr.backbones[0].state_dict()
        )
        self._snet_module_ptr.copy_fc(self._network_module_ptr.oldfc)
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self._snet.parameters()),
            lr=self.args["lr"],
            momentum=0.9,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=self.args["compression_epochs"]
        )
        self._network.eval()
        prog_bar = tqdm(range(self.args["compression_epochs"]))
        for _, epoch in enumerate(prog_bar):
            self._snet.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(
                    self._device, non_blocking=True
                ), targets.to(self._device, non_blocking=True)
                dark_logits = self._snet(inputs)["logits"]
                with torch.no_grad():
                    outputs = self._network(inputs)
                    logits, old_logits, fe_logits = (
                        outputs["logits"],
                        outputs["old_logits"],
                        outputs["fe_logits"],
                    )
                loss_dark = self.BKD(dark_logits, logits, self.args["T"])
                loss = loss_dark
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(dark_logits[: targets.shape[0]], dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._snet, test_loader)
                info = "SNet: Task {}, Epoch {}/{} => Loss {:.3f},  Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["compression_epochs"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "SNet: Task {}, Epoch {}/{} => Loss {:.3f},  Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["compression_epochs"],
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)
        if len(self._multiple_gpus) > 1:
            self._snet = self._snet.module
        if self.is_student_wa:
            self._snet.weight_align(
                self._known_classes,
                self._total_classes - self._known_classes,
                self.wa_value,
            )
        else:
            logging.info("do not weight align student!")

        self._snet.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(self._device, non_blocking=True)
            with torch.no_grad():
                outputs = self._snet(inputs)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        cnn_accy = self._evaluate(y_pred, y_true)
        logging.info("darknet eval: ")
        logging.info("CNN top1 curve: {}".format(cnn_accy["top1"]))
        logging.info("CNN top5 curve: {}".format(cnn_accy["top5"]))

    @property
    def samples_old_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._known_classes

    def samples_new_class(self, index):
        if self.args["dataset"] == "cifar100":
            return 500
        else:
            return self.data_manager.getlen(index)

    def BKD(self, pred, soft, T):
        pred = torch.log_softmax(pred / T, dim=1)
        soft = torch.softmax(soft / T, dim=1)
        soft = soft * self.per_cls_weights
        soft = soft / soft.sum(1)[:, None]
        return -1 * torch.mul(soft, pred).sum() / pred.shape[0]


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
