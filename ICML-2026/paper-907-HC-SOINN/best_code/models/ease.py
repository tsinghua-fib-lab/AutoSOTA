import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import EaseNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from utils.cluster_structure_analyzer import ClusterStructureAnalyzer
from utils.hc_soinn_classifier import HCSOINNClassifier

num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = EaseNet(args, True)
        
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

        self.analyze_cluster_structure_drift = args.get("analyze_cluster_structure_drift", False)
        self.cluster_analyzer = None

        if self.analyze_cluster_structure_drift:
            def feature_extractor(x):
                if isinstance(self._network, nn.DataParallel):
                    feats = self._network.module.extract_vector(x)
                else:
                    feats = self._network.extract_vector(x)
                return feats

            self.cluster_analyzer = ClusterStructureAnalyzer(
                feature_extractor=feature_extractor,
                device=self._device,
                args=args
            )

    def after_task(self):
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

        self._known_classes = self._total_classes
        self._network.freeze()
        self._network.backbone.add_adapter_to_list()
        
        if getattr(self, "use_hc_soinn", False):
            try:
                self.hc_soinn.compress()
            except Exception as e:
                logging.error(f"HC-SOINN compress error: {e}", exc_info=True)
    
    def get_cls_range(self, task_id):
        if task_id == 0:
            start_cls = 0
            end_cls = self.init_cls
        else:
            start_cls = self.init_cls + (task_id - 1) * self.inc
            end_cls = start_cls + self.inc
        
        return start_cls, end_cls
        
    # (proxy_fc = cls * dim)
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

            if self.use_exemplars and self._cur_task > 0:
                embedding_list = []
                label_list = []
                dataset = self.data_manager.get_dataset(np.arange(0, self._known_classes), source="train", mode="test", )
                loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
                for i, batch in enumerate(loader):
                    (_, data, label) = batch
                    data = data.to(self._device)
                    label = label.to(self._device)
                    embedding = model.backbone.forward_proto(data, adapt_index=self._cur_task)
                    embedding_list.append(embedding.cpu())
                    label_list.append(label.cpu())
                embedding_list = torch.cat(embedding_list, dim=0)
                label_list = torch.cat(label_list, dim=0)
                
                class_list = np.unique(dataset.labels)
                for class_index in class_list:
                    # print('adapter index:{}, Replacing...{}'.format(self._cur_task, class_index))
                    data_index = (label_list == class_index).nonzero().squeeze(-1)
                    embedding = embedding_list[data_index]
                    proto = embedding.mean(0)
                    model.fc.weight.data[class_index, -self._network.out_dim:] = proto                
        
        if self.use_diagonal or self.use_exemplars:
            return
        
        if self.recalc_sim:
            self.solve_sim_reset()
        else:
            self.solve_similarity()
    
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
        # W(TT)
        B = self._network.fc.weight.data[self._known_classes:, -self._network.out_dim:]
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
        # self._network.show_trainable_params()
        
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
        self.replace_fc(self.train_loader_for_protonet)
        
        if getattr(self, "use_hc_soinn", False):
            self._build_hc_soinn_bank()

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
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                aux_targets = targets.clone()
                aux_targets = torch.where(
                    aux_targets - self._known_classes >= 0,
                    aux_targets - self._known_classes,
                    -1,
                )
                aux_targets = aux_targets.long()
                
                output = self._network(inputs, test=False)
                logits = output["logits"]

                loss = F.cross_entropy(logits, aux_targets)

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

            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
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

    def _expand_hc_soinn_dimension(self):
        """Handle expand hc soinn dimension."""
        if not hasattr(self, "hc_soinn"):
            return
            
        target_dim = self._network.feature_dim
        logging.info(f"Expanding HC-SOINN dimension to {target_dim}")
        
        fc_weights = self._network.fc.weight.data.cpu().numpy()
        
        expanded_count = 0
        
        cls_scales = {} 
        cls_paddings = {}

        for cls, mu in self.hc_soinn.class_mu.items():
            if mu.shape[0] < target_dim:
                current_dim = mu.shape[0]
                pad_len = target_dim - current_dim
                
                if cls < len(fc_weights):
                    w_old = fc_weights[cls, :current_dim]
                    w_new = fc_weights[cls, current_dim:]
                    
                    
                    w_old_norm = np.linalg.norm(w_old)
                    mu_norm = np.linalg.norm(mu)
                    
                    if w_old_norm > 1e-6:
                        scale = mu_norm / w_old_norm
                    else:
                        scale = 1.0
                        logging.warning(f"Class {cls} old FC weight norm is too small ({w_old_norm}), using scale 1.0")

                    if w_new.shape[0] < pad_len:
                         logging.warning(f"Class {cls} FC weight dim {w_new.shape[0]} < pad_len {pad_len}, padding with zeros")
                         w_new = np.concatenate([w_new, np.zeros(pad_len - w_new.shape[0], dtype=w_new.dtype)])
                    elif w_new.shape[0] > pad_len:
                        w_new = w_new[:pad_len]

                    real_padding = w_new * scale
                    
                    
                    cls_scales[cls] = scale
                    cls_paddings[cls] = w_new
                else:
                    logging.warning(f"Class {cls} not found in FC weights, padding with zeros")
                    real_padding = np.zeros(pad_len, dtype=mu.dtype)
                    cls_scales[cls] = 1.0
                    cls_paddings[cls] = np.zeros(pad_len, dtype=mu.dtype)

                new_mu = np.concatenate([mu, real_padding])
                self.hc_soinn.class_mu[cls] = new_mu
                expanded_count += 1
                
                if hasattr(self.hc_soinn, "class_mu_raw") and cls in self.hc_soinn.class_mu_raw:
                    mu_raw = self.hc_soinn.class_mu_raw[cls]
                    mu_raw_norm = np.linalg.norm(mu_raw)
                    w_old_norm = np.linalg.norm(fc_weights[cls, :current_dim])
                    if w_old_norm > 1e-6:
                        scale_raw = mu_raw_norm / w_old_norm
                    else:
                        scale_raw = 1.0
                    
                    real_padding_raw = cls_paddings[cls] * scale_raw
                    new_mu_raw = np.concatenate([mu_raw, real_padding_raw])
                    self.hc_soinn.class_mu_raw[cls] = new_mu_raw
        
        if expanded_count > 0:
            logging.info(f"Expanded {expanded_count} classes in class_mu using Scaled EASE synthesis")

        expanded_cluster_count = 0
        for cls, clusters in self.hc_soinn.class_clusters.items():
            w_new = cls_paddings.get(cls)
            
            if cls < len(fc_weights) and clusters:
                 current_dim = clusters[0].center.shape[0]
                 w_old_norm = np.linalg.norm(fc_weights[cls, :current_dim])
            else:
                 w_old_norm = 1.0

            for cluster in clusters:
                if cluster.center.shape[0] < target_dim:
                    current_dim = cluster.center.shape[0]
                    pad_len = target_dim - current_dim
                    
                    if w_new is not None:
                        cluster_norm = np.linalg.norm(cluster.center)
                        if w_old_norm > 1e-6:
                            scale = cluster_norm / w_old_norm
                        else:
                            scale = 1.0
                        
                        if w_new.shape[0] < pad_len:
                            w_new_pad = np.concatenate([w_new, np.zeros(pad_len - w_new.shape[0], dtype=w_new.dtype)])
                        elif w_new.shape[0] > pad_len:
                            w_new_pad = w_new[:pad_len]
                        else:
                            w_new_pad = w_new

                        real_padding = w_new_pad * scale
                    else:
                        real_padding = np.zeros(pad_len, dtype=cluster.center.dtype)

                    new_center = np.concatenate([cluster.center, real_padding])
                    cluster.center = new_center
                    expanded_cluster_count += 1
                    
                    if cluster.center_raw is not None:
                        cluster_raw_norm = np.linalg.norm(cluster.center_raw)
                        if w_old_norm > 1e-6:
                            scale_raw = cluster_raw_norm / w_old_norm
                        else:
                            scale_raw = 1.0
                        
                        real_padding_raw = w_new_pad * scale_raw
                        new_center_raw = np.concatenate([cluster.center_raw, real_padding_raw])
                        cluster.center_raw = new_center_raw
        
        if expanded_cluster_count > 0:
            logging.info(f"Expanded {expanded_cluster_count} clusters in class_clusters using Scaled EASE synthesis")

    def _build_hc_soinn_bank(self):
        """Handle build hc soinn bank."""
        if not self.use_hc_soinn:
            return

        self._network.eval()
        
        self._expand_hc_soinn_dimension()

        hc_bank_empty = (not hasattr(self, "hc_soinn")) or (len(getattr(self.hc_soinn, "class_clusters", {})) == 0)
        if hc_bank_empty:
            logging.info(f"HC-SOINN bank is empty!!!")
        
        feature_fn = self._get_hc_soinn_feature_fn()

        def add_from_loader(loader):
            feats, lbs = [], []
            with torch.no_grad():
                for _, inputs, targets in loader:
                    inputs = inputs.to(self._device)
                    batch_feats = feature_fn(inputs)
                    if isinstance(batch_feats, torch.Tensor):
                        batch_feats = batch_feats.detach().cpu().numpy()
                    lbs.append(targets.numpy())
                    feats.append(batch_feats)
            if len(feats) == 0:
                return
            feats_np = np.concatenate(feats, axis=0)
            lbs_np = np.concatenate(lbs, axis=0)
            return feats_np, lbs_np

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
        feats_np, lbs_np = add_from_loader(current_task_loader)
        
        if feats_np is not None and len(feats_np) > 0:
            self.hc_soinn.add_features(feats_np, lbs_np)

    def _eval_hc_soinn(self, loader):
        """Handle eval hc soinn."""
        self._network.eval()
        y_pred, y_true = [], []
        feature_fn = self._get_hc_soinn_feature_fn()
        
        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(loader):
                inputs = inputs.to(self._device)
                feats = feature_fn(inputs)
                feats = feats.detach().cpu()
                if len(feats.shape) == 1:
                    feats = feats.reshape(1, -1)
                elif len(feats.shape) != 2:
                    raise ValueError(f"Expected 2D features [B, D], got shape {feats.shape}")
                feats_np = feats.numpy()
                
                topk_pred = self.hc_soinn.predict_topk(
                    feats_np, self.topk, self._total_classes, device=self._device,
                    use_ease_reweighting=True,
                    cur_task=self._cur_task,
                    task_increment=self.inc,
                    init_cls=self.init_cls,
                    ease_alpha=self.alpha
                )
                y_pred.append(topk_pred)
                y_true.append(targets.cpu().numpy())

        if len(y_pred) == 0:
            logging.warning("No predictions generated from HC-SOINN evaluation")
            return np.array([]), np.array([])

        return np.concatenate(y_pred), np.concatenate(y_true)

    def eval_task(self):
        """Handle eval task."""
        results = {}
        
        y_pred, y_true = self._eval_cnn(self.test_loader)
        results["fc"] = self._evaluate(y_pred, y_true)
        
        if getattr(self, "use_hc_soinn", False):
            y_pred_hc, y_true_hc = self._eval_hc_soinn(self.test_loader)
            results["hc_soinn"] = self._evaluate(y_pred_hc, y_true_hc)
        
        return results