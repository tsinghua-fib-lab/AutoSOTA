import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import OurNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from utils.hc_soinn_classifier import HCSOINNClassifier
from utils.cluster_structure_analyzer import ClusterStructureAnalyzer
from utils.STAR import STARAligner
import random

num_workers = 8


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]


def compute_orthogonality_loss(previous_weights_list, current_weights, epsilon=1e-8):
    total_ortho_loss = 0.0
    current_norm = torch.norm(current_weights.flatten())
    current_normalized = current_weights.flatten() / (current_norm + epsilon)

    for prev_weights in previous_weights_list:
        # Normalize previous weights
        prev_norm = torch.norm(prev_weights.flatten())
        prev_normalized = prev_weights.flatten() / (prev_norm + epsilon)

        # Compute absolute dot product (should be close to 0 for orthogonal vectors)
        dot_product = torch.abs(torch.sum(prev_normalized * current_normalized))

        total_ortho_loss += dot_product

    # Average over all previous tasks
    if len(previous_weights_list) > 0:
        total_ortho_loss /= len(previous_weights_list)

    return total_ortho_loss

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = OurNet(args, True)

        self.args = args
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.init_cls = args["init_cls"]
        self.inc = args["increment"]

        self.use_exemplars = args["use_old_data"]
        self.use_init_ptm = args["use_init_ptm"]
        self.use_diagonal = args["use_diagonal"]

        self.recalc_sim = args["recalc_sim"]
        self.alpha = args["alpha"] # forward_reweight is divide by _cur_task
        self.beta = args["beta"]

        self.moni_adam = args["moni_adam"]
        self.adapter_num = args["adapter_num"]

        if self.moni_adam:
            self.use_init_ptm = True
            self.alpha = 1
            self.beta = 1

        # HC-SOINN plugin
        self.use_hc_soinn = args.get("use_hc_soinn", False)
        self.hcsoinn_feature_mode = str(args.get("hcsoinn_feature_mode", "diagonal")).lower().strip()
        if self.use_hc_soinn:
            logging.info("Initializing HC-SOINNClassifier")
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
            logging.info(f"[CL-LoRA] HC-SOINN feature mode: {self.hcsoinn_feature_mode}")
        
        self._class_means = None


        self.analyze_cluster_structure_drift = args.get("analyze_cluster_structure_drift", False)
        self.cluster_analyzer = None

        if self.analyze_cluster_structure_drift:
            # 
            def feature_extractor(x):
                if isinstance(self._network, nn.DataParallel):
                    backbone = self._network.module.backbone
                else:
                    backbone = self._network.backbone
                
                features_list = backbone.forward_test(x, use_init_ptm=self.use_init_ptm)
                
                if self.use_init_ptm:
                    task1_feat = features_list[1][:, 0, :]  # [B, 768]
                else:
                    task1_feat = features_list[0][:, 0, :]  # [B, 768]
                
                return task1_feat

            self.cluster_analyzer = ClusterStructureAnalyzer(
                feature_extractor=feature_extractor,
                device=self._device,
                args=args
            )

        self.use_feature_alignment = args.get("use_feature_alignment", False)
        self.use_full_task_rehearsal = args.get("use_full_task_rehearsal", False)
        self.star = None
        if self.use_feature_alignment and self.use_hc_soinn:
            logging.info("Initializing STARAligner for CL-LoRA")

            def feature_extractor(x, class_id=None):
                """Extract diagonal-segment features for each sample/class adapter."""
                if isinstance(self._network, nn.DataParallel):
                    backbone = self._network.module.backbone
                else:
                    backbone = self._network.backbone

                feats_full = backbone(x, test=True, use_init_ptm=self.use_init_ptm)  # [B, concat_dim]
                out_dim = self._network.out_dim
                offset = self._get_segment_offset()

                # If no class_id is provided, fall back to full features (not recommended for diagonal mode)
                if class_id is None:
                    return feats_full

                # scalar class_id: slice the same segment for the whole batch
                if isinstance(class_id, (int, np.integer)):
                    adapter_idx = self._get_class_adapter_idx(int(class_id))
                    seg_idx = adapter_idx + offset
                    return feats_full[:, seg_idx * out_dim:(seg_idx + 1) * out_dim]

                # batch class_id: per-sample slicing
                if torch.is_tensor(class_id):
                    class_ids = class_id.detach().cpu().numpy().astype(np.int64)
                else:
                    class_ids = np.asarray(class_id, dtype=np.int64)

                B = feats_full.shape[0]
                out = torch.empty((B, out_dim), device=feats_full.device, dtype=feats_full.dtype)
                for i in range(B):
                    cls = int(class_ids[i])
                    adapter_idx = self._get_class_adapter_idx(cls)
                    seg_idx = adapter_idx + offset
                    s = seg_idx * out_dim
                    e = s + out_dim
                    out[i] = feats_full[i, s:e]
                return out

            self.star = STARAligner(
                hc_soinn=self.hc_soinn,
                feature_extractor=feature_extractor,
                device=self._device,
                use_full_task_rehearsal=self.use_full_task_rehearsal,
                star_lambda=args.get("star_lambda", 0.3),
            )
            if self.use_full_task_rehearsal:
                logging.info("STAR alignment initialized (FULL TASK REHEARSAL mode - for performance upper bound)")
            else:
                logging.info("STAR alignment initialized (anchor mode: all SOINN nodes + NCM points)")

    def after_task(self):
        """Handle after task."""

        if self.cluster_analyzer is not None:
            if self._cur_task == 0:
                init_cls = self.args.get("init_cls", 10)
                dataset_loader = lambda: self.data_manager.get_dataset(
                    np.arange(0, init_cls), source="train", mode="test"
                )
                self.cluster_analyzer.save_task1_samples(
                    dataset_loader=dataset_loader,
                    batch_size=self.batch_size,
                    num_workers=num_workers
                )
            else:
                self.cluster_analyzer.compute_procrustes_distances(self._cur_task)

        if self.star is not None:
            current_task_classes = set(range(self._known_classes, self._total_classes))
            dataset = self.data_manager.get_dataset(
                np.arange(self._known_classes, self._total_classes),
                source="train",
                mode="test",
            )
            self.star.select_anchors_for_current_task(
                dataset=dataset,
                batch_size=self.batch_size,
                num_workers=num_workers,
                current_task_classes=current_task_classes,
            )

        self._known_classes = self._total_classes
        self._network.freeze()
        self._network.backbone.add_adapter_to_list()

    def get_cls_range(self, task_id):
        if task_id == 0:
            start_cls = 0
            end_cls = self.init_cls
        else:
            start_cls = self.init_cls + (task_id - 1) * self.inc
            end_cls = start_cls + self.inc

        return start_cls, end_cls

    def replace_fc_proxy(self):
        model = self._network
        model = model.eval()
        model.fc.weight.data[self._known_classes:self._total_classes, :] = model.proxy_fc.weight.data
        model.fc.bias.data[self._known_classes:self._total_classes] = model.proxy_fc.bias.data

    def replace_fc(self, train_loader):
        model = self._network
        model = model.eval()

        with torch.no_grad():
            # replace proto for each adapter in the current task
            if self.use_init_ptm:
                start_idx = -1
            else:
                start_idx = 0

            for index in range(start_idx, self._cur_task + 1):
                if self.moni_adam:
                    if index > self.adapter_num - 1:
                        break
                # only use the diagonal feature, index = -1 denotes using init PTM, index = self._cur_task denotes the last adapter's feature
                elif self.use_diagonal and index != -1 and index != self._cur_task:
                    continue

                embedding_list, label_list = [], []
                for i, batch in enumerate(train_loader):
                    (_, data, label) = batch
                    data = data.to(self._device)
                    label = label.to(self._device)
                    embedding = model.backbone.forward_proto(data, adapt_index=index)
                    embedding_list.append(embedding.cpu())
                    label_list.append(label.cpu())

                embedding_list = torch.cat(embedding_list, dim=0)
                label_list = torch.cat(label_list, dim=0)

                class_list = np.unique(self.train_dataset_for_protonet.labels)
                for class_index in class_list:
                    data_index = (label_list == class_index).nonzero().squeeze(-1)
                    embedding = embedding_list[data_index]
                    proto = embedding.mean(0)
                    if self.use_init_ptm:
                        model.fc.weight.data[class_index, (index+1)*self._network.out_dim:(index+2)*self._network.out_dim] = proto
                    else:
                        model.fc.weight.data[class_index, index*self._network.out_dim:(index+1)*self._network.out_dim] = proto
        return

    def get_A_B_Ahat(self, task_id):
        if self.use_init_ptm:
            start_dim = (task_id + 1) * self._network.out_dim
            end_dim = start_dim + self._network.out_dim
        else:
            start_dim = task_id * self._network.out_dim
            end_dim = start_dim + self._network.out_dim

        start_cls, end_cls = self.get_cls_range(task_id)

        # W(Ti)  i is the i-th task index, T is the cur task index, W is a T*T matrix
        A = self._network.fc.weight.data[self._known_classes:, start_dim : end_dim]
        #A = self._network.fc.weight.data[0:, start_dim : end_dim]
        # W(TT)
        B = self._network.fc.weight.data[self._known_classes:, -self._network.out_dim:]
        #B = self._network.fc.weight.data[0:, -self._network.out_dim:]
        # W(ii)
        A_hat = self._network.fc.weight.data[start_cls : end_cls, start_dim : end_dim]

        return A.cpu(), B.cpu(), A_hat.cpu()

    def solve_similarity(self):
        for task_id in range(self._cur_task):
            # print('Solve_similarity adapter:{}'.format(task_id))
            start_cls, end_cls = self.get_cls_range(task_id=task_id)

            A, B, A_hat = self.get_A_B_Ahat(task_id=task_id)

            # calculate similarity matrix between A_hat(old_cls1) and A(new_cls1).
            similarity = torch.zeros(len(A_hat), len(A))
            for i in range(len(A_hat)):
                for j in range(len(A)):
                    similarity[i][j] = torch.cosine_similarity(A_hat[i], A[j], dim=0)

            # softmax the similarity, it will be failed if not use it
            similarity = F.softmax(similarity, dim=1)

            # weight the combination of B(new_cls2)
            B_hat = torch.zeros(A_hat.shape[0], B.shape[1])
            for i in range(len(A_hat)):
                for j in range(len(A)):
                    B_hat[i] += similarity[i][j] * B[j]

            # B_hat(old_cls2)
            self._network.fc.weight.data[start_cls : end_cls, -self._network.out_dim:] = B_hat.to(self._device)

    def solve_sim_reset(self):
        for task_id in range(self._cur_task):
            if self.moni_adam and task_id > self.adapter_num - 2:
                break

            if self.use_init_ptm:
                range_dim = range(task_id + 2, self._cur_task + 2)
            else:
                range_dim = range(task_id + 1, self._cur_task + 1)
            for dim_id in range_dim:
                if self.moni_adam and dim_id > self.adapter_num:
                    break
                # print('Solve_similarity adapter:{}, {}'.format(task_id, dim_id))
                start_cls, end_cls = self.get_cls_range(task_id=task_id)

                start_dim = dim_id * self._network.out_dim
                end_dim = (dim_id + 1) * self._network.out_dim

                # Use the element above the diagonal to calculate
                if self.use_init_ptm:
                    start_cls_old = self.init_cls + (dim_id - 2) * self.inc
                    end_cls_old = self._total_classes
                    start_dim_old = (task_id + 1) * self._network.out_dim
                    end_dim_old = (task_id + 2) * self._network.out_dim
                else:
                    start_cls_old = self.init_cls + (dim_id - 1) * self.inc
                    end_cls_old = self._total_classes
                    start_dim_old = task_id * self._network.out_dim
                    end_dim_old = (task_id + 1) * self._network.out_dim

                A = self._network.fc.weight.data[start_cls_old:end_cls_old, start_dim_old:end_dim_old].cpu()
                B = self._network.fc.weight.data[start_cls_old:end_cls_old, start_dim:end_dim].cpu()
                A_hat = self._network.fc.weight.data[start_cls:end_cls, start_dim_old:end_dim_old].cpu()

                # calculate similarity matrix between A_hat(old_cls1) and A(new_cls1).
                similarity = torch.zeros(len(A_hat), len(A))
                for i in range(len(A_hat)):
                    for j in range(len(A)):
                        similarity[i][j] = torch.cosine_similarity(A_hat[i], A[j], dim=0)

                # softmax the similarity, it will be failed if not use it
                similarity = F.softmax(similarity, dim=1) # dim=1, not dim=0

                # weight the combination of B(new_cls2)
                B_hat = torch.zeros(A_hat.shape[0], B.shape[1])
                for i in range(len(A_hat)):
                    for j in range(len(A)):
                        B_hat[i] += similarity[i][j] * B[j]

                # B_hat(old_cls2)
                self._network.fc.weight.data[start_cls : end_cls, start_dim : end_dim] = B_hat.to(self._device)

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)

        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        self.data_manager = data_manager
        self.train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="train", )
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        self.test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        self.train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="test", )
        self.train_loader_for_protonet = DataLoader(self.train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        self._network.add_fc()
        self.replace_fc(self.train_loader_for_protonet)

        if self.star is not None and self._cur_task > 0:
            current_task_classes = set(range(self._known_classes, self._total_classes))
            self.star.align_old_classes(
                cur_task=self._cur_task,
                current_task_classes=current_task_classes,
            )

        self._build_ncm_classifier()
        
        if self.use_hc_soinn:
            self._build_hc_soinn_bank()
            try:
                self.hc_soinn.compress()
            except Exception as e:
                logging.error(f"HC-SOINN compress error: {e}", exc_info=True)

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        if self._cur_task == 0 or self.init_cls == self.inc:
            optimizer = self.get_optimizer(lr=self.args["init_lr"])
            scheduler = self.get_scheduler(optimizer, self.args["init_epochs"])
        else:
            # for base 0 setting, the later_lr and later_epochs are not used
            # for base N setting, the later_lr and later_epochs are used
            if "later_lr" not in self.args or self.args["later_lr"] == 0:
                self.args["later_lr"] = self.args["init_lr"]
            if "later_epochs" not in self.args or self.args["later_epochs"] == 0:
                self.args["later_epochs"] = self.args["init_epochs"]

            optimizer = self.get_optimizer(lr=self.args["later_lr"])
            scheduler = self.get_scheduler(optimizer, self.args["later_epochs"])

        self._init_train(train_loader, test_loader, optimizer, scheduler)


    def get_optimizer(self, lr):
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                momentum=0.9,
                lr=lr,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lr,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lr,
                weight_decay=self.weight_decay
            )

        return optimizer

    def get_scheduler(self, optimizer, epoch):
        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epoch, eta_min=self.min_lr)
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        # torch.autograd.set_detect_anomaly(True)
        if self.moni_adam:
            if self._cur_task > self.adapter_num - 1:
                return

        if self._cur_task == 0 or self.init_cls == self.inc:
            epochs = self.args['init_epochs']
        else:
            epochs = self.args['later_epochs']

        prog_bar = tqdm(range(epochs))

        for _, epoch in enumerate(prog_bar):
            self._network.train()

            losses = 0.0
            correct, total = 0, 0

            if not self._network.backbone.msa_adapt:

                for name, param in self._network.backbone.cur_adapter[0].named_parameters():
                    print(f"Parameter: {name}, Requires Gradient: {param.requires_grad}")
            else:
                for name, param in self._network.backbone.cur_adapter[0][1].named_parameters():
                    print(f"Parameter: {name}, Requires Gradient: {param.requires_grad}")
                for name, param in self._network.backbone.cur_adapter[-1][1].named_parameters():
                    print(f"Parameter: {name}, Requires Gradient: {param.requires_grad}")


            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                aux_targets = targets.clone()

                aux_targets = torch.where(
                    aux_targets - self._known_classes >= 0,
                    aux_targets - self._known_classes,
                    -1,
                )
                output = self._network(inputs, test=False)

                logits = output["logits"]

                loss = F.cross_entropy(logits, aux_targets.long())

                if self._cur_task > 0:
                    kd_ratio = 5.
                    Temperature = 2

                    out_new, out_teacher = self._network.forward_kd(inputs, self._cur_task)
                    out_new_logits = out_new["logits"]
                    out_teacher_logits = out_teacher["logits"]
                    loss_kd = kd_ratio * _KD_loss(out_new_logits, out_teacher_logits, T=Temperature)

                    optimizer.zero_grad()

                    loss_kd.backward()

                    for j in range(len(self._network.backbone.general_pos)):
                        pos = self._network.backbone.adapt_pos.index(self._network.backbone.general_pos[j])
                        for jj in range(len(self._network.backbone.msa)):
                            if self._network.backbone.msa[jj] == 1:
                                temp_weights = 1. * torch.norm(self._network.backbone.old_adapter_list[self._cur_task-1][pos][jj].lora_A.weight,dim=1)
                                temp_weights = 1. * len(temp_weights) * temp_weights / torch.sum(temp_weights)
                                self._network.backbone.cur_adapter[pos][jj].lora_A.weight.grad = temp_weights.unsqueeze(1) * self._network.backbone.cur_adapter[pos][jj].lora_A.weight.grad
                    optimizer.step()
                if self._cur_task > 0:
                    orth_loss_specific = compute_orthogonality_loss(self._network.backbone.block_weight_list, self._network.backbone.block_weight)
                    loss += 0.0001 * orth_loss_specific


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)

                correct += preds.eq(aux_targets.expand_as(preds)).cpu().sum()
                total += len(aux_targets)

            if scheduler:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)


    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model.forward(inputs, test=True)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        calc_task_acc = True

        if calc_task_acc:
            task_correct, task_acc, total = 0, 0, 0

        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)

            with torch.no_grad():
                outputs = self._network.forward(inputs, test=True)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

            # calculate the accuracy by using task_id
            if calc_task_acc:
                task_ids = (targets - self.init_cls) // self.inc + 1
                task_logits = torch.zeros(outputs.shape).to(self._device)
                for i, task_id in enumerate(task_ids):
                    if task_id == 0:
                        start_cls = 0
                        end_cls = self.init_cls
                    else:
                        start_cls = self.init_cls + (task_id-1)*self.inc
                        end_cls = self.init_cls + task_id*self.inc
                    task_logits[i, start_cls:end_cls] += outputs[i, start_cls:end_cls]
                # calculate the accuracy of task_id
                pred_task_ids = (torch.max(outputs, dim=1)[1] - self.init_cls) // self.inc + 1
                task_correct += (pred_task_ids.cpu() == task_ids).sum()

                pred_task_y = torch.max(task_logits, dim=1)[1]
                task_acc += (pred_task_y.cpu() == targets).sum()
                total += len(targets)

        if calc_task_acc:
            logging.info("Task correct: {}".format(tensor2numpy(task_correct) * 100 / total))
            logging.info("Task acc: {}".format(tensor2numpy(task_acc) * 100 / total))

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _get_hc_soinn_feature_fn(self):
        """Handle feature fn."""
        def feature_fn(x):
            if isinstance(self._network, nn.DataParallel):
                backbone = self._network.module.backbone
            else:
                backbone = self._network.backbone
            
            feats = backbone(x, test=True, use_init_ptm=self.use_init_ptm)
            
            if not isinstance(feats, torch.Tensor):
                raise TypeError(f"Expected tensor, got {type(feats)}")
            
            if len(feats.shape) == 1:
                feats = feats.reshape(1, -1)
            elif len(feats.shape) != 2:
                raise ValueError(f"Expected 2D features [B, D], got shape {feats.shape}")
            
            return feats
        
        return feature_fn

    def _get_class_adapter_idx(self, class_id: int) -> int:
        """Handle get class adapter idx."""
        if class_id < self.init_cls:
            return 0
        return ((class_id - self.init_cls) // self.inc) + 1

    def _get_segment_offset(self) -> int:
        """Handle get segment offset."""
        return 1 if self.use_init_ptm else 0

    def _build_hc_soinn_bank(self):
        """Handle build hc soinn bank."""
        if not self.use_hc_soinn:
            return

        self._network.eval()

        hc_bank_empty = (not hasattr(self, "hc_soinn")) or (len(getattr(self.hc_soinn, "class_clusters", {})) == 0)
        if hc_bank_empty:
            logging.info(f"HC-SOINN bank is empty!!!")
        
        feature_fn = self._get_hc_soinn_feature_fn()

        def add_from_loader(loader):
            feats, lbs = [], []
            with torch.no_grad():
                for _, inputs, targets in loader:
                    inputs = inputs.to(self._device)
                    batch_feats_full = feature_fn(inputs)  # [B, concat_dim]
                    if isinstance(batch_feats_full, torch.Tensor):
                        batch_feats_full = batch_feats_full.detach().cpu().numpy()
                    targets_np = targets.numpy()

                    # --- HC-SOINN diagonal mode: only store the class-owned segment ---
                    if self.hcsoinn_feature_mode == "diagonal":
                        out_dim = self._network.out_dim
                        offset = self._get_segment_offset()
                        batch_feats_diag = np.zeros((batch_feats_full.shape[0], out_dim), dtype=np.float32)

                        for i in range(batch_feats_full.shape[0]):
                            cls = int(targets_np[i])
                            adapter_idx = self._get_class_adapter_idx(cls)
                            seg_idx = adapter_idx + offset
                            start = seg_idx * out_dim
                            end = start + out_dim
                            if end > batch_feats_full.shape[1]:
                                raise ValueError(
                                    f"HC-SOINN diagonal feature slice out of range: "
                                    f"cls={cls}, adapter_idx={adapter_idx}, seg_idx={seg_idx}, "
                                    f"slice=[{start}:{end}], feat_dim={batch_feats_full.shape[1]}"
                                )
                            batch_feats_diag[i] = batch_feats_full[i, start:end]

                        feats.append(batch_feats_diag)
                        lbs.append(targets_np)
                    else:
                        # "full": store full concat features
                        feats.append(batch_feats_full)
                        lbs.append(targets_np)
            if len(feats) == 0:
                return
            feats_np = np.concatenate(feats, axis=0)
            lbs_np = np.concatenate(lbs, axis=0)
            self.hc_soinn.add_features(feats_np, lbs_np)

        logging.info(f"Building HC-SOINN bank: adding new classes ({self._known_classes}-{self._total_classes-1})")
        current_task_dataset = self.data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="test"
        )
        current_task_loader = DataLoader(
            current_task_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
        )
        add_from_loader(current_task_loader)


    def _eval_hc_soinn(self, loader):
        """Handle eval hc soinn."""
        self._network.eval()
        y_pred, y_true = [], []
        feature_fn = self._get_hc_soinn_feature_fn()
        out_dim = self._network.out_dim
        num_adapters = self._cur_task + 1
        offset = self._get_segment_offset()
        
        hc_soinn = self.hc_soinn
        all_classes = sorted(hc_soinn.class_mu.keys())
        
        if len(all_classes) == 0:
            logging.warning("HC-SOINN has no prototypes, returning zero predictions")
            for _, (_, inputs, targets) in enumerate(loader):
                batch_size = inputs.shape[0]
                topk_pred = np.zeros((batch_size, self.topk), dtype=np.int64)
                y_pred.append(topk_pred)
                y_true.append(targets.cpu().numpy())
            return np.concatenate(y_pred), np.concatenate(y_true)
        
        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(loader):
                inputs = inputs.to(self._device)
                feats = feature_fn(inputs)  # [B, num_adapters * out_dim]
                feats = feats.detach().cpu()
                if len(feats.shape) == 1:
                    feats = feats.reshape(1, -1)
                elif len(feats.shape) != 2:
                    raise ValueError(f"Expected 2D features [B, D], got shape {feats.shape}")
                
                batch_size = feats.shape[0]
                feats_t = torch.from_numpy(feats.numpy()).float().to(self._device)  # [B, num_adapters * out_dim]
                
                all_logits = []
                
                for adapter_idx in range(num_adapters):
                    seg_idx = adapter_idx + offset
                    adapter_feats = feats_t[:, seg_idx * out_dim:(seg_idx + 1) * out_dim]  # [B, out_dim]
                    
                    if adapter_idx == 0:
                        start_cls = 0
                        end_cls = self.init_cls
                    else:
                        start_cls = self.init_cls + (adapter_idx - 1) * self.inc
                        end_cls = start_cls + self.inc
                    
                    ncm_centers = []
                    valid_classes = []
                    for cls in all_classes:
                        if start_cls <= cls < end_cls:
                            cls_mu = hc_soinn.class_mu[cls]
                            if cls_mu.shape[0] == out_dim:
                                ncm_centers.append(cls_mu)
                                valid_classes.append(cls)
                    
                    if len(ncm_centers) == 0:
                        adapter_logits = torch.zeros((batch_size, end_cls - start_cls), device=self._device)
                    else:
                        ncm_centers_t = torch.from_numpy(np.stack(ncm_centers)).float().to(self._device)  # [C, out_dim]
                        
                        adapter_feats_norm = torch.nn.functional.normalize(adapter_feats, p=2, dim=1)
                        ncm_centers_norm = torch.nn.functional.normalize(ncm_centers_t, p=2, dim=1)
                        
                        sim_ncm = torch.mm(adapter_feats_norm, ncm_centers_norm.t())  # [B, C]
                        
                        dist_sub_list = []
                        for cls in valid_classes:
                            clusters = hc_soinn.class_clusters.get(cls, [])
                            if clusters:
                                cluster_centers = []
                                for c in clusters:
                                    if c.center.shape[0] == out_dim:
                                        cluster_centers.append(c.center)
                                
                                if cluster_centers:
                                    cluster_centers_t = torch.from_numpy(np.stack(cluster_centers)).float().to(self._device)
                                    cluster_centers_norm = torch.nn.functional.normalize(cluster_centers_t, p=2, dim=1)
                                    sim_cluster = torch.mm(adapter_feats_norm, cluster_centers_norm.t())  # [B, num_clusters]
                                    dist_cluster = 1.0 - sim_cluster
                                    min_dist, _ = dist_cluster.min(dim=1)  # [B]
                                    dist_sub_list.append(min_dist)
                                else:
                                    cls_idx = valid_classes.index(cls)
                                    dist_sub_list.append(1.0 - sim_ncm[:, cls_idx])
                            else:
                                cls_idx = valid_classes.index(cls)
                                dist_sub_list.append(1.0 - sim_ncm[:, cls_idx])
                        
                        if dist_sub_list:
                            dist_sub = torch.stack(dist_sub_list, dim=1)  # [B, C]
                        else:
                            dist_sub = 1.0 - sim_ncm
                        
                        dist_ncm = 1.0 - sim_ncm  # [B, C]
                        final_scores = hc_soinn.alpha * dist_ncm + (1.0 - hc_soinn.alpha) * dist_sub  # [B, C]
                        
                        adapter_logits = torch.full((batch_size, end_cls - start_cls), float('inf'), device=self._device)
                        for i, cls in enumerate(valid_classes):
                            if start_cls <= cls < end_cls:
                                adapter_logits[:, cls - start_cls] = final_scores[:, i]
                    
                    all_logits.append(adapter_logits)
                
                combined_logits = torch.cat(all_logits, dim=1)  # [B, total_classes]
                
                _, topk_indices = torch.topk(combined_logits, k=self.topk, dim=1, largest=False)
                topk_pred = topk_indices.cpu().numpy()
                
                y_pred.append(topk_pred)
                y_true.append(targets.cpu().numpy())

        if len(y_pred) == 0:
            logging.warning("No predictions generated from HC-SOINN evaluation")
            return np.array([]), np.array([])

        return np.concatenate(y_pred), np.concatenate(y_true)

    def _build_ncm_classifier(self):
        """Handle build ncm classifier."""
        model = self._network
        model.eval()
        
        need_rebuild_all = (self._class_means is None) or (self._class_means.shape[0] == 0)
        
        if need_rebuild_all:
            logging.info(f"Building NCM classifier: computing all seen classes (0-{self._total_classes-1}) [first task or after checkpoint load]")
            
            self._class_means = np.zeros((self._total_classes, self._network.out_dim))
            
            with torch.no_grad():
                for class_idx in range(self._total_classes):
                    if class_idx < self.init_cls:
                        adapter_idx = 0
                    else:
                        adapter_idx = ((class_idx - self.init_cls) // self.inc) + 1
                    
                    class_dataset = self.data_manager.get_dataset(
                        np.array([class_idx]), source="train", mode="test"
                    )
                    class_loader = DataLoader(
                        class_dataset,
                        batch_size=self.batch_size,
                        shuffle=False,
                        drop_last=False,
                        num_workers=num_workers
                    )
                    
                    embedding_list = []
                    for _, (_, inputs, targets) in enumerate(class_loader):
                        inputs = inputs.to(self._device)
                        if self.use_init_ptm:
                            if adapter_idx == 0:
                                adapt_index = -1
                            else:
                                adapt_index = adapter_idx - 1
                        else:
                            adapt_index = adapter_idx
                        
                        embeddings = model.backbone.forward_proto(inputs, adapt_index=adapt_index)
                        embedding_list.append(embeddings.cpu())
                    
                    if len(embedding_list) > 0:
                        embedding_list = torch.cat(embedding_list, dim=0)
                        proto = embedding_list.mean(0)
                        self._class_means[class_idx, :] = proto.numpy()
            
            logging.info(f"NCM classifier built: computed class means for {self._total_classes} classes")
        else:
            logging.info(f"Building NCM classifier: preserving previous class means, computing new classes ({self._known_classes}-{self._total_classes-1})")
            
            if self._class_means.shape[0] < self._total_classes:
                new_class_means = np.zeros((self._total_classes, self._network.out_dim))
                new_class_means[:self._class_means.shape[0]] = self._class_means
                self._class_means = new_class_means
                logging.info(f"Extended NCM classifier: preserved {self._class_means.shape[0] - (self._total_classes - self._known_classes)} previous class means")
            
            with torch.no_grad():
                for class_idx in range(self._known_classes, self._total_classes):
                    if class_idx < self.init_cls:
                        adapter_idx = 0
                    else:
                        adapter_idx = ((class_idx - self.init_cls) // self.inc) + 1
                    
                    class_dataset = self.data_manager.get_dataset(
                        np.array([class_idx]), source="train", mode="test"
                    )
                    class_loader = DataLoader(
                        class_dataset,
                        batch_size=self.batch_size,
                        shuffle=False,
                        drop_last=False,
                        num_workers=num_workers
                    )
                    
                    embedding_list = []
                    for _, (_, inputs, targets) in enumerate(class_loader):
                        inputs = inputs.to(self._device)
                        if self.use_init_ptm:
                            if adapter_idx == 0:
                                adapt_index = -1
                            else:
                                adapt_index = adapter_idx - 1
                        else:
                            adapt_index = adapter_idx
                        
                        embeddings = model.backbone.forward_proto(inputs, adapt_index=adapt_index)
                        embedding_list.append(embeddings.cpu())
                    
                    if len(embedding_list) > 0:
                        embedding_list = torch.cat(embedding_list, dim=0)
                        proto = embedding_list.mean(0)
                        self._class_means[int(class_idx), :] = proto.numpy()
            
            logging.info(f"NCM classifier updated: computed class means for classes {self._known_classes}-{self._total_classes-1} (total: {self._total_classes} classes)")

    def _eval_ncm(self, loader):
        """Handle eval ncm."""
        self._network.eval()
        y_pred, y_true = [], []
        out_dim = self._network.out_dim
        num_adapters = self._cur_task + 1
        
        if self._class_means is None or self._class_means.shape[0] == 0:
            logging.warning("NCM classifier not built, returning zero predictions")
            for _, (_, inputs, targets) in enumerate(loader):
                batch_size = inputs.shape[0]
                topk_pred = np.zeros((batch_size, self.topk), dtype=np.int64)
                y_pred.append(topk_pred)
                y_true.append(targets.cpu().numpy())
            return np.concatenate(y_pred), np.concatenate(y_true)
        
        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(loader):
                inputs = inputs.to(self._device)
                batch_size = inputs.shape[0]
                
                all_logits = []
                
                for adapter_idx in range(num_adapters):
                    if self.use_init_ptm:
                        if adapter_idx == 0:
                            adapt_index = -1
                        else:
                            adapt_index = adapter_idx - 1
                    else:
                        adapt_index = adapter_idx
                    
                    adapter_feats = self._network.backbone.forward_proto(inputs, adapt_index=adapt_index)  # [B, out_dim]
                    
                    if adapter_idx == 0:
                        start_cls = 0
                        end_cls = self.init_cls
                    else:
                        start_cls = self.init_cls + (adapter_idx - 1) * self.inc
                        end_cls = start_cls + self.inc
                    
                    ncm_centers = []
                    valid_classes = []
                    for cls in range(start_cls, min(end_cls, self._total_classes)):
                        if cls < self._class_means.shape[0] and np.any(self._class_means[cls] != 0):
                            ncm_centers.append(self._class_means[cls])
                            valid_classes.append(cls)
                    
                    if len(ncm_centers) == 0:
                        adapter_logits = torch.zeros((batch_size, end_cls - start_cls), device=self._device)
                    else:
                        ncm_centers_t = torch.from_numpy(np.stack(ncm_centers)).float().to(self._device)  # [C, out_dim]
                        
                        adapter_feats_norm = torch.nn.functional.normalize(adapter_feats, p=2, dim=1)
                        ncm_centers_norm = torch.nn.functional.normalize(ncm_centers_t, p=2, dim=1)
                        
                        sim_ncm = torch.mm(adapter_feats_norm, ncm_centers_norm.t())  # [B, C]
                        
                        dist_ncm = 1.0 - sim_ncm  # [B, C]
                        
                        adapter_logits = torch.full((batch_size, end_cls - start_cls), float('inf'), device=self._device)
                        for i, cls in enumerate(valid_classes):
                            if start_cls <= cls < end_cls:
                                adapter_logits[:, cls - start_cls] = dist_ncm[:, i]
                    
                    all_logits.append(adapter_logits)
                
                combined_logits = torch.cat(all_logits, dim=1)  # [B, total_classes]
                
                _, topk_indices = torch.topk(combined_logits, k=self.topk, dim=1, largest=False)
                topk_pred = topk_indices.cpu().numpy()
                
                y_pred.append(topk_pred)
                y_true.append(targets.cpu().numpy())
        
        if len(y_pred) == 0:
            logging.warning("No predictions generated from NCM evaluation")
            return np.array([]), np.array([])
        
        return np.concatenate(y_pred), np.concatenate(y_true)

    def eval_task(self):
        """Handle eval task."""
        results = {}
        
        y_pred, y_true = self._eval_cnn(self.test_loader)
        results["fc"] = self._evaluate(y_pred, y_true)
        
        if hasattr(self, "_class_means") and self._class_means is not None:
            y_pred_ncm, y_true_ncm = self._eval_ncm(self.test_loader)
            results["ncm"] = self._evaluate(y_pred_ncm, y_true_ncm)
        
        if getattr(self, "use_hc_soinn", False):
            y_pred_hc, y_true_hc = self._eval_hc_soinn(self.test_loader)
            results["hc_soinn"] = self._evaluate(y_pred_hc, y_true_hc)
        
        return results

