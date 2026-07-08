import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.inc_net import CosineIncrementalNet
from utils.toolkit import target2onehot, tensor2numpy

EPSILON = 1e-8
num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)
        
        # HC-SOINN plugin
        self.use_hc_soinn = args.get("use_hc_soinn", False)
        if self.use_hc_soinn:
            from utils.hc_soinn_classifier import HCSOINNClassifier
            logging.info("Initializing HC-SOINNClassifier for iCaRL")
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
            )
        else:
            self.hc_soinn = None

        # STAR feature alignment (trajectory-only)
        self.use_feature_alignment = args.get("use_feature_alignment", False)
        self.use_full_task_rehearsal = args.get("use_full_task_rehearsal", False)
        self.star = None
        if self.use_feature_alignment and self.use_hc_soinn and self.hc_soinn is not None:
            from utils.STAR import STARAligner

            def feature_extractor(x):
                # Use the same representation as NCM / HC-SOINN: extract_vector
                if isinstance(self._network, nn.DataParallel):
                    feats = self._network.module.extract_vector(x)
                else:
                    feats = self._network.extract_vector(x)
                return feats

            self.star = STARAligner(
                hc_soinn=self.hc_soinn,
                feature_extractor=feature_extractor,
                device=self._device,
                use_full_task_rehearsal=self.use_full_task_rehearsal,
                star_lambda=args.get("star_lambda", 0.3),
            )
            if self.use_full_task_rehearsal:
                logging.info("[iCaRL] STAR alignment initialized (FULL TASK REHEARSAL mode)")
            else:
                logging.info("[iCaRL] STAR alignment initialized (trajectory)")

    def after_task(self):
        # STAR: select anchors for current task (for next task alignment)
        if self.star is not None and hasattr(self, "data_manager") and self.data_manager is not None:
            try:
                current_task_classes = set(range(self._known_classes, self._total_classes))
                if len(current_task_classes) > 0:
                    dataset = self.data_manager.get_dataset(
                        np.arange(self._known_classes, self._total_classes),
                        source="train",
                        mode="test",
                    )
                    self.star.select_anchors_for_current_task(
                        dataset=dataset,
                        batch_size=self.args["batch_size"],
                        num_workers=num_workers,
                        current_task_classes=current_task_classes,
                    )
            except Exception as e:
                logging.error(f"[iCaRL] STAR anchor selection error: {e}", exc_info=True)

        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        # keep a handle for after_task() / STAR anchor selection
        self.data_manager = data_manager
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        
        # Build HC-SOINN bank after training and building exemplar memory
        if getattr(self, "use_hc_soinn", False) and self.hc_soinn is not None:
            try:
                self._build_hc_soinn_bank(data_manager)
            except Exception as e:
                logging.error(f"HC-SOINN build error: {e}", exc_info=True)

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        if self._cur_task == 0:
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=self.args["init_lr"],
                weight_decay=self.args["init_weight_decay"],
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"]
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(
                self._network.parameters(),
                lr=self.args["lrate"],
                momentum=0.9,
                weight_decay=self.args["weight_decay"],
            )  # 1e-5
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.args["milestones"], gamma=self.args["lrate_decay"]
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["init_epoch"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
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
                    self.args["init_epoch"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    losses / len(train_loader),
                    train_acc,
                )

            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["epochs"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss_clf = F.cross_entropy(logits, targets)
                loss_kd = _KD_loss(
                    logits[:, : self._known_classes],
                    self._old_network(inputs)["logits"],
                    self.args["T"],
                )

                loss = loss_clf + loss_kd

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
                    self.args["epochs"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)


    def _build_hc_soinn_bank(self, data_manager):
        """Handle build hc soinn bank."""
        if not self.use_hc_soinn:
            return
        
        logging.info(f"Building HC-SOINN bank: adding new classes ({self._known_classes}-{self._total_classes-1})")
        
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="test"
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args["batch_size"],
            shuffle=False,
            num_workers=num_workers
        )
        
        vectors, targets_np = self._extract_vectors(train_loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
        feats_np = vectors
        lbs_np = targets_np
        
        if len(feats_np) > 0:
            self.hc_soinn.add_features(feats_np, lbs_np)
            
            try:
                self.hc_soinn.compress()
                logging.info("HC-SOINN bank built successfully")
            except Exception as e:
                logging.error(f"HC-SOINN compress error: {e}", exc_info=True)
    
    def _eval_hc_soinn(self, loader):
        """Handle eval hc soinn."""
        if not self.use_hc_soinn or self.hc_soinn is None:
            return None, None
        
        self._network.eval()
        y_pred, y_true = [], []
        
        vectors, targets_np = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
        
        topk_pred = self.hc_soinn.predict_topk(
            vectors, k=1, return_distances=False
        )
        
        y_pred = topk_pred
        y_true = targets_np
        
        if len(y_pred) > 0:
            return y_pred, y_true
        else:
            logging.warning("No predictions generated from HC-SOINN evaluation")
            return None, None
    
    def eval_task(self):
        """Handle eval task."""
        # STAR: pre-evaluation alignment (align old classes before evaluation)
        if self.star is not None and self._cur_task > 0 and getattr(self, "use_hc_soinn", False) and self.hc_soinn is not None:
            try:
                current_task_classes = set(range(self._known_classes, self._total_classes))
                logging.info(
                    f"[STAR] Pre-evaluation alignment: Aligning old classes before evaluation "
                    f"(task {self._cur_task}, current task classes: {current_task_classes})"
                )
                self.star.align_old_classes(cur_task=self._cur_task, current_task_classes=current_task_classes)
            except Exception as e:
                logging.error(f"[iCaRL] STAR align_old_classes error: {e}", exc_info=True)

        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        results = {"fc": cnn_accy}
        if nme_accy is not None:
            results["ncm"] = nme_accy
        
        if getattr(self, "use_hc_soinn", False) and self.hc_soinn is not None:
            y_pred_hc, y_true_hc = self._eval_hc_soinn(self.test_loader)
            if y_pred_hc is not None:
                results["hc_soinn"] = self._evaluate(y_pred_hc, y_true_hc)
        
        return results


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
