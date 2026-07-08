import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import PromptVitNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from utils.hc_soinn_classifier import HCSOINNClassifier
from utils.STAR import STARAligner
from utils.hc_soinn_node_stats import log_hc_soinn_dataset_end_summary
from utils.cluster_structure_analyzer import ClusterStructureAnalyzer
import os

# tune the model at first session with vpt, and then conduct simple shot.
num_workers = 0

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        
        self.use_kac = args.get("use_kac", False)
        if self.use_kac:
            logging.info("KAC classifier enabled for DualPrompt")
            kac_config = args.get("kac_config", {})
            logging.info(f"KAC config: grid_min={kac_config.get('grid_min', -2.0)}, "
                        f"grid_max={kac_config.get('grid_max', 2.0)}, "
                        f"num_grids={kac_config.get('num_grids', 16)}")
    
        self._network = PromptVitNet(args, True)

        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.args = args

        # HC-SOINN plugin
        self.use_hc_soinn = args.get("use_hc_soinn", False)
        self.use_hc_soinn_fc_fusion = args.get("use_hc_soinn_fc_fusion", False)
        self.hc_soinn_fc_fusion_b = float(args.get("hc_soinn_fc_fusion_b", 0.3))
        self.hc_soinn_fc_fusion_a = float(args.get("hc_soinn_fc_fusion_a", 1.0 - self.hc_soinn_fc_fusion_b))
        self.hc_soinn_fc_calib_temp = float(args.get("hc_soinn_fc_calib_temp", 1.0))
        self.hc_soinn_hc_calib_temp = float(args.get("hc_soinn_hc_calib_temp", 1.0))
        if self.use_hc_soinn:
            logging.info("Initializing HC-SOINNClassifier for DualPrompt")
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
            # Adaptive prototype allocation
            if args.get("hcsoinn_adaptive_proto", False):
                self.hc_soinn.use_adaptive_proto = True
                self.hc_soinn.total_proto_budget = args.get("hcsoinn_total_proto_budget", None)
            if self.use_hc_soinn_fc_fusion:
                logging.info(
                    "HC-SOINN + Calibrated FC fusion enabled: "
                    f"a={self.hc_soinn_fc_fusion_a:.3f}, b={self.hc_soinn_fc_fusion_b:.3f}, "
                    f"fc_temp={self.hc_soinn_fc_calib_temp:.3f}, hc_temp={self.hc_soinn_hc_calib_temp:.3f}"
                )
        
        # STAR Feature Alignment
        self.use_feature_alignment = args.get("use_feature_alignment", False)
        self.use_full_task_rehearsal = args.get("use_full_task_rehearsal", False)
        self.star = None
        self.class_to_task_map = {} # Map class_id to task_id for E-Prompt selection

        if self.use_feature_alignment and self.use_hc_soinn:
            logging.info("Initializing STAR Aligner for DualPrompt")
            
            def feature_extractor(x, class_id=None):
                # DualPrompt requires task_id to select the correct E-Prompt
                task_id = self._cur_task
                if class_id is not None:
                    # If class_id is provided, try to find its original task_id
                    if class_id in self.class_to_task_map:
                        task_id = self.class_to_task_map[class_id]
                    else:
                        # Fallback: if not mapped, assume current task (or handle error)
                        pass
                
                return self._network.extract_vector(x, task_id=task_id)

            self.star = STARAligner(
                hc_soinn=self.hc_soinn,
                feature_extractor=feature_extractor,
                device=self._device,
                use_full_task_rehearsal=self.use_full_task_rehearsal,
                star_lambda=args.get("star_lambda", 0.3),
            )

        self.analyze_cluster_structure_drift = args.get("analyze_cluster_structure_drift", False)
        self.cluster_analyzer = None
        
        if self.analyze_cluster_structure_drift:
            def feature_extractor(x):
                return self._network.extract_vector(x, task_id=self._cur_task)
            
            self.cluster_analyzer = ClusterStructureAnalyzer(
                feature_extractor=feature_extractor,
                device=self._device,
                args=args
            )

        self.use_ncm = args.get("use_ncm", False)
        self._class_means = None

        # Freeze the parameters for ViT.
        if self.args["freeze"]:
            for p in self._network.original_backbone.parameters():
                p.requires_grad = False
        
            # freeze args.freeze[blocks, patch_embed, cls_token] parameters
            for n, p in self._network.backbone.named_parameters():
                if n.startswith(tuple(self.args["freeze"])):
                    p.requires_grad = False
        
        total_params = sum(p.numel() for p in self._network.backbone.parameters())
        logging.info(f'{total_params:,} model total parameters.')
        total_trainable_params = sum(p.numel() for p in self._network.backbone.parameters() if p.requires_grad)
        logging.info(f'{total_trainable_params:,} model training parameters.')
        
        if self.use_kac and hasattr(self._network.backbone, 'head'):
            head_params = sum(p.numel() for p in self._network.backbone.head.parameters() if p.requires_grad)
            logging.info(f'  - Head (KAC) parameters: {head_params:,}')

        # if some parameters are trainable, print the key name and corresponding parameter number
        if total_params != total_trainable_params:
            for name, param in self._network.backbone.named_parameters():
                if param.requires_grad:
                    logging.info("{}: {}".format(name, param.numel()))

    def after_task(self):
        if self.star is not None:
            dataset = self.data_manager.get_dataset(
                np.arange(self._known_classes, self._total_classes),
                source="train", mode="test"
            )
            current_task_classes = set(range(self._known_classes, self._total_classes))
            self.star.select_anchors_for_current_task(
                dataset=dataset,
                batch_size=self.batch_size,
                num_workers=num_workers,
                current_task_classes=current_task_classes
            )
            
        if self.cluster_analyzer is not None:
            if self._cur_task == 0:
                init_cls = self._total_classes
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
        log_hc_soinn_dataset_end_summary(self)

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        
        # Update class_to_task_map
        for cls in range(self._known_classes, self._total_classes):
            self.class_to_task_map[cls] = self._cur_task
            
        # self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train")
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        
        self._build_classifiers()

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)
            
        if self._cur_task > 0:
            self._init_prompt(optimizer)

        if self._cur_task > 0 and self.args["reinit_optimizer"]:
            optimizer = self.get_optimizer()
            
        self._init_train(train_loader, test_loader, optimizer, scheduler)

    def get_optimizer(self):
        trainable_params = filter(lambda p: p.requires_grad, self._network.parameters())
        
        if self.use_kac and hasattr(self._network.backbone, 'head'):
            head_param_count = sum(p.numel() for p in self._network.backbone.head.parameters() if p.requires_grad)
            logging.info(f"KAC classifier parameters included in optimizer: {head_param_count:,}")
        
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                trainable_params, 
                momentum=0.9, 
                lr=self.init_lr,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                trainable_params,
                lr=self.init_lr, 
                weight_decay=self.weight_decay
            )
            
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                trainable_params,
                lr=self.init_lr, 
                weight_decay=self.weight_decay
            )

        return optimizer
    
    def get_scheduler(self, optimizer):
        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler

    def _init_prompt(self, optimizer):
        args = self.args
        model = self._network.backbone
        task_id = self._cur_task

        # Transfer previous learned prompt params to the new prompt
        if args["prompt_pool"] and args["shared_prompt_pool"]:
            prev_start = (task_id - 1) * args["top_k"]
            prev_end = task_id * args["top_k"]

            cur_start = prev_end
            cur_end = (task_id + 1) * args["top_k"]

            if (prev_end > args["size"]) or (cur_end > args["size"]):
                pass
            else:
                cur_idx = (slice(None), slice(None), slice(cur_start, cur_end)) if args["use_prefix_tune_for_e_prompt"] else (slice(None), slice(cur_start, cur_end))
                prev_idx = (slice(None), slice(None), slice(prev_start, prev_end)) if args["use_prefix_tune_for_e_prompt"] else (slice(None), slice(prev_start, prev_end))

                with torch.no_grad():
                    model.e_prompt.prompt.grad.zero_()
                    model.e_prompt.prompt[cur_idx] = model.e_prompt.prompt[prev_idx]
                    optimizer.param_groups[0]['params'] = model.parameters()
                
        # Transfer previous learned prompt param keys to the new prompt
        if args["prompt_pool"] and args["shared_prompt_key"]:
            prev_start = (task_id - 1) * args["top_k"]
            prev_end = task_id * args["top_k"]

            cur_start = prev_end
            cur_end = (task_id + 1) * args["top_k"]

            if (prev_end > args["size"]) or (cur_end > args["size"]):
                pass
            else:
                cur_idx = (slice(cur_start, cur_end))
                prev_idx = (slice(prev_start, prev_end))

            with torch.no_grad():
                model.e_prompt.prompt_key.grad.zero_()
                model.e_prompt.prompt_key[cur_idx] = model.e_prompt.prompt_key[prev_idx]
                optimizer.param_groups[0]['params'] = model.parameters()

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.backbone.train()
            self._network.original_backbone.eval()

            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
            
                output = self._network(inputs, task_id=self._cur_task, train=True)
                logits = output["logits"][:, :self._total_classes]
                logits[:, :self._known_classes] = float('-inf')

                loss = F.cross_entropy(logits, targets.long())
                if self.args["pull_constraint"] and 'reduce_sim' in output:
                    loss = loss - self.args["pull_constraint_coeff"] * output['reduce_sim']

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if (epoch + 1) % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['tuned_epoch'],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['tuned_epoch'],
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)

    def _eval_cnn(self, loader):
        """Handle eval cnn."""
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs, task_id=self._cur_task)["logits"][:, :self._total_classes]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]
    
    def _eval_kac(self, loader):
        """Handle eval kac."""
        return self._eval_cnn(loader)

    def _build_classifiers(self):
        """Handle build classifiers."""
        logging.info(f"Building classifiers for task {self._cur_task} (total classes: {self._total_classes})")
        
        self._network.to(self._device)
        self._network.eval()
        
        if self.star is not None and self._cur_task > 0:
            current_task_classes = set(range(self._known_classes, self._total_classes))
            self.star.align_old_classes(
                cur_task=self._cur_task,
                current_task_classes=current_task_classes
            )
        
        if getattr(self, "use_ncm", False):
            self._build_ncm_classifier()
        
        if getattr(self, "use_hc_soinn", False):
            self._build_hc_soinn_bank()
        
        logging.info("All classifiers built successfully")

    def _build_ncm_classifier(self):
        """Handle build ncm classifier."""
        ncm_classes = self._total_classes
        need_rebuild_all = (self._class_means is None) or (self._class_means.shape[0] == 0)
        
        if need_rebuild_all:
            logging.info(f"Building NCM classifier: computing all seen classes (0-{ncm_classes-1})")
            all_train_dataset = self.data_manager.get_dataset(
                np.arange(0, ncm_classes), source="train", mode="test"
            )
            all_train_loader = DataLoader(
                all_train_dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                num_workers=num_workers
            )
            
            self._class_means = np.zeros((ncm_classes, self.feature_dim))
            embedding_list, label_list = [], []
            
            with torch.no_grad():
                for _, (_, inputs, targets) in enumerate(all_train_loader):
                    inputs = inputs.to(self._device)
                    embeddings = self._network.extract_vector(inputs, task_id=self._cur_task)
                    embedding_list.append(embeddings.cpu())
                    label_list.append(targets.cpu())
            
            if len(embedding_list) > 0:
                embedding_list = torch.cat(embedding_list, dim=0)
                label_list = torch.cat(label_list, dim=0)
                for class_index in np.unique(label_list.numpy()):
                    data_index = (label_list == class_index).nonzero().squeeze(-1)
                    self._class_means[int(class_index), :] = embedding_list[data_index].mean(0).numpy()
        else:
            logging.info(f"Building NCM classifier: computing new classes ({self._known_classes}-{self._total_classes-1})")
            if self._class_means.shape[0] < self._total_classes:
                new_class_means = np.zeros((self._total_classes, self.feature_dim))
                new_class_means[:self._class_means.shape[0]] = self._class_means
                self._class_means = new_class_means
            
            current_task_dataset = self.data_manager.get_dataset(
                np.arange(self._known_classes, self._total_classes), source="train", mode="test"
            )
            current_task_loader = DataLoader(
                current_task_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers
            )
            
            embedding_list, label_list = [], []
            with torch.no_grad():
                for _, (_, inputs, targets) in enumerate(current_task_loader):
                    inputs = inputs.to(self._device)
                    embeddings = self._network.extract_vector(inputs, task_id=self._cur_task)
                    embedding_list.append(embeddings.cpu())
                    label_list.append(targets.cpu())
            
            if len(embedding_list) > 0:
                embedding_list = torch.cat(embedding_list, dim=0)
                label_list = torch.cat(label_list, dim=0)
                for class_index in np.unique(label_list.numpy()):
                    data_index = (label_list == class_index).nonzero().squeeze(-1)
                    self._class_means[int(class_index), :] = embedding_list[data_index].mean(0).numpy()

        for class_index in range(ncm_classes):
            if np.any(self._class_means[class_index] != 0):
                self._network.ncm_fc.weight.data[int(class_index), :] = torch.from_numpy(self._class_means[int(class_index), :]).float().to(self._device)

    def _get_soinn_feature_fn(self):
        def feature_fn(x):
            feats = self._network.extract_vector(x, task_id=self._cur_task)
            return feats
        return feature_fn

    def _build_hc_soinn_bank(self):
        feature_fn = self._get_soinn_feature_fn()
        logging.info(f"Building HC-SOINN bank: adding new classes ({self._known_classes}-{self._total_classes-1})")
        
        current_task_dataset = self.data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="test"
        )
        current_task_loader = DataLoader(
            current_task_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers
        )
        
        feats, lbs = [], []
        with torch.no_grad():
            for _, inputs, targets in current_task_loader:
                inputs = inputs.to(self._device)
                batch_feats = feature_fn(inputs)
                lbs.append(targets.numpy())
                feats.append(batch_feats.cpu().numpy())
        
        if len(feats) > 0:
            all_feats = np.concatenate(feats, axis=0)
            all_lbs = np.concatenate(lbs, axis=0)
            
            
            self.hc_soinn.add_features(all_feats, all_lbs)
            self.hc_soinn.compress()

    def _eval_ncm_fc(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(loader):
                inputs = inputs.to(self._device)
                features = self._network.extract_vector(inputs, task_id=self._cur_task)
                ncm_output = self._network.ncm_fc(features)
                ncm_logits = ncm_output['logits'][:, :self._total_classes]
                predicts = torch.topk(ncm_logits, k=self.topk, dim=1, largest=True, sorted=True)[1]
                y_pred.append(predicts.cpu().numpy())
                y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)

    def _eval_hc_soinn(self, loader):
        self._network.eval()
        

        y_pred, y_true = [], []
        feature_fn = self._get_soinn_feature_fn()
        
        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(loader):
                inputs = inputs.to(self._device)
                feats = feature_fn(inputs)
                
                feats = F.normalize(feats, p=2, dim=1)
                feats_np = feats.cpu().numpy()

                topk_pred = self.hc_soinn.predict_topk(feats_np, self.topk, self._total_classes, device=self._device)
                y_pred.append(topk_pred)
                y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)

    def _eval_hc_soinn_fc_fusion(self, loader):
        """
        Fusion experiment:
        result = a * HC-SOINN + b * (Calibrated FC), where calibrated scores are probabilities.
        """
        self._network.eval()
        y_pred, y_true = [], []
        feature_fn = self._get_soinn_feature_fn()

        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(loader):
                inputs = inputs.to(self._device)
                fc_logits = self._network(inputs, task_id=self._cur_task)["logits"][:, :self._total_classes]
                feats = feature_fn(inputs)
                feats = F.normalize(feats, p=2, dim=1)
                hc_logits = self.hc_soinn.predict_class_logits(
                    feats, self._total_classes, device=self._device
                )

                fc_probs = torch.softmax(fc_logits / self.hc_soinn_fc_calib_temp, dim=1)
                hc_probs = torch.softmax(hc_logits / self.hc_soinn_hc_calib_temp, dim=1)
                fused_probs = self.hc_soinn_fc_fusion_a * hc_probs + self.hc_soinn_fc_fusion_b * fc_probs

                predicts = torch.topk(fused_probs, k=self.topk, dim=1, largest=True, sorted=True)[1]
                y_pred.append(predicts.cpu().numpy())
                y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)

    def eval_task(self):
        """Handle eval task."""
        results = {}
        
        if self.use_kac:
            y_pred_kac, y_true_kac = self._eval_kac(self.test_loader)
            results["kac"] = self._evaluate(y_pred_kac, y_true_kac)
            results["fc"] = results["kac"]
        else:
            y_pred_fc, y_true_fc = self._eval_cnn(self.test_loader)
            results["fc"] = self._evaluate(y_pred_fc, y_true_fc)
        
        if getattr(self, "use_hc_soinn", False):
            y_pred_hc, y_true_hc = self._eval_hc_soinn(self.test_loader)
            results["hc_soinn"] = self._evaluate(y_pred_hc, y_true_hc)
            if self.use_hc_soinn_fc_fusion:
                y_pred_mix, y_true_mix = self._eval_hc_soinn_fc_fusion(self.test_loader)
                results["hc_soinn_fc_fusion"] = self._evaluate(y_pred_mix, y_true_mix)
        
        if getattr(self, "use_ncm", False) and self._class_means is not None:
            y_pred_ncm, y_true_ncm = self._eval_ncm_fc(self.test_loader)
            results["ncm"] = self._evaluate(y_pred_ncm, y_true_ncm)
        
        return results

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs, task_id=self._cur_task)["logits"][:, :self._total_classes]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)