import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.optim import Optimizer
import math
import time
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import CodaPromptVitNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from utils.knn_classifier import KNNClassifier
from utils.hc_soinn_classifier import HCSOINNClassifier
from utils.cluster_structure_analyzer import ClusterStructureAnalyzer
from utils.STAR import STARAligner
from utils.hc_soinn_node_stats import log_hc_soinn_dataset_end_summary
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
try:
    # import umap
    # not use
    UMAP_AVAILABLE = False
except ImportError:
    UMAP_AVAILABLE = False
    logging.warning("UMAP not available, falling back to t-SNE")

# tune the model at first session with vpt, and then conduct simple shot.
num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
    
        self.use_kac = args.get("use_kac", False)
        if self.use_kac:
            logging.info("KAC classifier enabled")
            kac_config = args.get("kac_config", {})
            logging.info(f"KAC config: grid_min={kac_config.get('grid_min', -2.0)}, "
                        f"grid_max={kac_config.get('grid_max', 2.0)}, "
                        f"num_grids={kac_config.get('num_grids', 16)}")
        
        self._network = CodaPromptVitNet(args, True)

        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.args = args
        # KNN plugin
        self.use_knn = args.get("use_knn", False)
        self.knn_upperbound = args.get("knn_upperbound", False)
        self.knn_prune_zero_usage = args.get("knn_prune_zero_usage", False)
        if self.use_knn:
            knn_metric = args.get("knn_metric", "euclidean")
            logging.info(f"Initializing KNNClassifier with metric: {knn_metric}")
            if self.knn_upperbound:
                logging.info("KNN Upperbound mode enabled: will rebuild KNN bank with all seen tasks at eval time")
            self.knn = KNNClassifier(
                metric=knn_metric,
                use_all_samples=args.get("knn_use_all", True),
                k_neighbors=args.get("knn_k", None),
            )
            logging.info(f"KNNClassifier initialized with metric: {self.knn.metric}")
        
        # HC-SOINN plugin
        self.use_hc_soinn = args.get("use_hc_soinn", False)
        self.use_hc_soinn_fc_fusion = args.get("use_hc_soinn_fc_fusion", False)
        self.hc_soinn_fc_fusion_b = float(args.get("hc_soinn_fc_fusion_b", 0.3))
        self.hc_soinn_fc_fusion_a = float(args.get("hc_soinn_fc_fusion_a", 1.0 - self.hc_soinn_fc_fusion_b))
        self.hc_soinn_fc_calib_temp = float(args.get("hc_soinn_fc_calib_temp", 1.0))
        self.hc_soinn_hc_calib_temp = float(args.get("hc_soinn_hc_calib_temp", 1.0))
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
                coarse_topk=args.get("hcsoinn_coarse_topk", None),
                enable_inference_profiling=args.get("hcsoinn_profile_inference", False),
                profile_sync_cuda=args.get("hcsoinn_profile_sync_cuda", True),
            )
            if self.use_hc_soinn_fc_fusion:
                logging.info(
                    "HC-SOINN + Calibrated FC fusion enabled: "
                    f"a={self.hc_soinn_fc_fusion_a:.3f}, b={self.hc_soinn_fc_fusion_b:.3f}, "
                    f"fc_temp={self.hc_soinn_fc_calib_temp:.3f}, hc_temp={self.hc_soinn_hc_calib_temp:.3f}"
                )
        total_params = sum(p.numel() for p in self._network.parameters())
        logging.info(f'{total_params:,} total parameters.')
        
        fc_params = sum(p.numel() for p in self._network.fc.parameters() if p.requires_grad)
        prompt_params = sum(p.numel() for p in self._network.prompt.parameters() if p.requires_grad)
        total_trainable_params = fc_params + prompt_params
        logging.info(f'{total_trainable_params:,} fc and prompt training parameters.')
        if self.use_kac:
            logging.info(f'  - FC (KAC) parameters: {fc_params:,}')
            logging.info(f'  - Prompt parameters: {prompt_params:,}')
        
        self.test_only_first_task_classes = args.get("test_only_first_task_classes", False)
        if self.test_only_first_task_classes:
            self.init_cls = args.get("init_cls", 10)
        
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

        self.use_feature_alignment = args.get("use_feature_alignment", False)
        self.use_full_task_rehearsal = args.get("use_full_task_rehearsal", False)
        self.star = None
        
        if self.use_feature_alignment and self.use_hc_soinn:
            def feature_extractor(x):
                if isinstance(self._network, nn.DataParallel):
                    feats = self._network.module(x, pen=True, train=False)
                else:
                    feats = self._network(x, pen=True, train=False)
                
                if isinstance(feats, tuple):
                    feats = feats[0]
                return feats
            
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
        
        self.use_ncm = args.get("use_ncm", True)
        if not self.use_ncm:
            logging.info("NCM classifier disabled")
        self._class_means = None

    def after_task(self):
        """Handle after task."""
        # ========== Prepare current task class set ==========
        # Get the classes that belong to the current task (to exclude from alignment)
        current_task_classes = set(range(self._known_classes, self._total_classes))
        
        
        if self.star is not None:
            dataset = self.data_manager.get_dataset(
                np.arange(self._known_classes, self._total_classes),
                source="train", mode="test"
            )
            self.star.select_anchors_for_current_task(
                dataset=dataset,
                batch_size=self.batch_size,
                num_workers=num_workers,
                current_task_classes=current_task_classes
            )

        if getattr(self, "use_knn", False) and getattr(self, "knn_prune_zero_usage", False):
            try:
                if hasattr(self, "knn") and hasattr(self.knn, "prune_zero_usage"):
                    current_task_classes = list(range(self._known_classes, self._total_classes))
                    if len(current_task_classes) > 0:
                        logging.info(
                            f"[coda_prompt] Pruning KNN nodes with zero usage for current task classes: "
                            f"{current_task_classes} (task {self._cur_task})"
                        )
                        self.knn.prune_zero_usage(current_task_classes)
            except Exception as e:
                logging.error(f"Error during KNN prune_zero_usage in coda_prompt.after_task: {e}", exc_info=True)

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
        
        if getattr(self, "use_hc_soinn", False) and self.args.get("visualize_tsne", False):
            try:
                init_cls = self.args.get("init_cls", 10)
                self._visualize_hc_soinn_tsne(target_classes=list(range(min(10, init_cls))))
            except Exception as e:
                logging.error(f"Error during t-SNE visualization in coda_prompt.after_task: {e}", exc_info=True)
        
        self._known_classes = self._total_classes

        log_hc_soinn_dataset_end_summary(self)

        if self.args.get("save_checkpoint", False):
            self.save_checkpoint()
        

    def incremental_train(self, data_manager):
        self._cur_task += 1
        
        if self.args.get("load_all_checkpoints", False):
            self._load_checkpoint_for_task()
            expected_total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
            if self._total_classes != expected_total_classes:
                logging.warning(
                    f"Total classes mismatch: checkpoint has {self._total_classes}, "
                    f"expected {expected_total_classes}. Using expected value."
                )
                self._total_classes = expected_total_classes
            logging.info("Learning on {}-{} (checkpoint loaded, skipping training)".format(self._known_classes, self._total_classes))
            
            if self.test_only_first_task_classes:
                test_classes = np.arange(0, self.init_cls)
            else:
                test_classes = np.arange(0, self._total_classes)
            test_dataset = data_manager.get_dataset(test_classes, source="test", mode="test")
            self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
            
            if self._cur_task > 0:
                try:
                    if self._network.module.prompt is not None:
                        self._network.module.prompt.process_task_count()
                except:
                    if self._network.prompt is not None:
                        self._network.prompt.process_task_count()
            
            train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train")
            self.train_dataset = train_dataset
            self.data_manager = data_manager
            if self.test_only_first_task_classes:
                test_classes = np.arange(0, self.init_cls)
            else:
                test_classes = np.arange(0, self._total_classes)
            test_dataset = data_manager.get_dataset(test_classes, source="test", mode="test")
            self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
            
            self._build_classifiers()
            
            return
        
        if self._cur_task > 0:
            try:
                if self._network.module.prompt is not None:
                    self._network.module.prompt.process_task_count()
            except:
                if self._network.prompt is not None:
                    self._network.prompt.process_task_count()

        if not self.args.get("load_all_checkpoints", False):
            self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
            # self._network.update_fc(self._total_classes)
            logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train")
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
        if self.test_only_first_task_classes:
            test_classes = np.arange(0, self.init_cls)
            logging.info(f"实验模式：测试数据集只包含类别 {test_classes}")
        else:
            test_classes = np.arange(0, self._total_classes)
        test_dataset = data_manager.get_dataset(test_classes, source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        profile_training_time = self._should_profile_training_time()
        if profile_training_time:
            self._last_training_time_breakdown = {
                "pure_train_sec": 0.0,
                "periodic_eval_sec": 0.0,
                "classifier_build_sec": 0.0,
            }

        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        if profile_training_time:
            self._sync_training_time_profile()
            build_start = time.perf_counter()
        self._build_classifiers()
        if profile_training_time:
            self._sync_training_time_profile()
            self._last_training_time_breakdown["classifier_build_sec"] = time.perf_counter() - build_start

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)

        self.data_weighting()
        self._init_train(train_loader, test_loader, optimizer, scheduler)

    def _should_profile_training_time(self):
        return bool(self.args.get("training_time_profile_last_task", False)) and (
            self._cur_task == int(self.args.get("nb_tasks", 0)) - 1
        )

    def _sync_training_time_profile(self):
        if (
            bool(self.args.get("training_time_sync_cuda", True))
            and self._device.type == "cuda"
            and torch.cuda.is_available()
        ):
            torch.cuda.synchronize(device=self._device)

    def get_last_training_time_breakdown(self):
        return getattr(self, "_last_training_time_breakdown", {})


    def _build_classifiers(self):
        """Handle build classifiers."""
        
        self._network.to(self._device)
        self._network.eval()
        
        if self.star is not None and self._cur_task > 0:
            current_task_classes = set(range(self._known_classes, self._total_classes))
            logging.info(
                f"[STAR] Pre-evaluation alignment: Aligning old classes before evaluation "
                f"(task {self._cur_task}, current task classes: {current_task_classes})"
            )
            self.star.align_old_classes(
                cur_task=self._cur_task,
                current_task_classes=current_task_classes
            )
        
        if getattr(self, "use_ncm", True):
            self._build_ncm_classifier()
        
        if getattr(self, "use_knn", False):
            if getattr(self, "knn_upperbound", False):
                self._build_knn_bank_upperbound()
            else:
                self._build_knn_bank()
        
        if getattr(self, "use_hc_soinn", False):
            self._build_hc_soinn_bank()
        
    
    def _build_ncm_classifier(self):
        """Handle build ncm classifier."""
        if self.test_only_first_task_classes:
            ncm_classes = self.init_cls
        else:
            ncm_classes = self._total_classes
        
        need_rebuild_all = (self._class_means is None) or (self._class_means.shape[0] == 0)
        
        if need_rebuild_all:
            if self.test_only_first_task_classes:
                logging.info(f"Building NCM classifier (实验模式): computing first task classes (0-{ncm_classes-1}) [first task or after checkpoint load]")
            else:
                logging.info(f"Building NCM classifier: computing all seen classes (0-{ncm_classes-1}) [first task or after checkpoint load]")
            all_train_dataset = self.data_manager.get_dataset(
                np.arange(0, ncm_classes), source="train", mode="test"
            )
            all_train_loader = DataLoader(
                all_train_dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                drop_last=False, 
                num_workers=num_workers
            )
            
            self._class_means = np.zeros((ncm_classes, self.feature_dim))
            
            embedding_list = []
            label_list = []
            
            with torch.no_grad():
                for _, (_, inputs, targets) in enumerate(all_train_loader):
                    inputs = inputs.to(self._device)
                    if isinstance(self._network, nn.DataParallel):
                        embeddings = self._network.module.extract_vector(inputs)
                    else:
                        embeddings = self._network.extract_vector(inputs)
                    embedding_list.append(embeddings.cpu())
                    label_list.append(targets.cpu())
            
            if len(embedding_list) > 0:
                embedding_list = torch.cat(embedding_list, dim=0)
                label_list = torch.cat(label_list, dim=0)
                
                class_list = np.unique(label_list.numpy())
                for class_index in class_list:
                    data_index = (label_list == class_index).nonzero().squeeze(-1)
                    embedding = embedding_list[data_index]
                    proto = embedding.mean(0)
                    self._class_means[int(class_index), :] = proto.cpu().numpy()
                
                logging.info(f"NCM classifier built: computed class means for {len(class_list)} classes (total: {ncm_classes} classes)")
            else:
                logging.warning("No training data available for NCM classifier")
        else:
            if self.test_only_first_task_classes:
                logging.info(f"Building NCM classifier (实验模式): preserving first task class means (0-{ncm_classes-1}), skipping new classes")
                return
            else:
                logging.info(f"Building NCM classifier: preserving previous class means, computing new classes ({self._known_classes}-{self._total_classes-1})")
            
            if self._class_means.shape[0] < self._total_classes:
                new_class_means = np.zeros((self._total_classes, self.feature_dim))
                new_class_means[:self._class_means.shape[0]] = self._class_means
                self._class_means = new_class_means
                logging.info(f"Extended NCM classifier: preserved {self._class_means.shape[0] - (self._total_classes - self._known_classes)} previous class means")
            
            current_task_dataset = self.data_manager.get_dataset(
                np.arange(self._known_classes, self._total_classes), source="train", mode="test"
            )
            current_task_loader = DataLoader(
                current_task_dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                drop_last=False, 
                num_workers=num_workers
            )
            
            embedding_list = []
            label_list = []
            
            with torch.no_grad():
                for _, (_, inputs, targets) in enumerate(current_task_loader):
                    inputs = inputs.to(self._device)
                    if isinstance(self._network, nn.DataParallel):
                        embeddings = self._network.module.extract_vector(inputs)
                    else:
                        embeddings = self._network.extract_vector(inputs)
                    embedding_list.append(embeddings.cpu())
                    label_list.append(targets.cpu())
            
            if len(embedding_list) > 0:
                embedding_list = torch.cat(embedding_list, dim=0)
                label_list = torch.cat(label_list, dim=0)
                
                class_list = np.unique(label_list.numpy())
                for class_index in class_list:
                    data_index = (label_list == class_index).nonzero().squeeze(-1)
                    embedding = embedding_list[data_index]
                    proto = embedding.mean(0)
                    self._class_means[int(class_index), :] = proto.cpu().numpy()
                
                logging.info(f"NCM classifier updated: computed class means for {len(class_list)} new classes (total: {self._total_classes} classes)")
            else:
                logging.warning("No training data available for NCM classifier update")
        
        if self.test_only_first_task_classes:
            ncm_classes = self.init_cls
        else:
            ncm_classes = self._total_classes
        
        if isinstance(self._network, nn.DataParallel):
            for class_index in range(ncm_classes):
                if np.any(self._class_means[class_index] != 0):
                    self._network.module.ncm_fc.weight.data[int(class_index), :] = torch.from_numpy(self._class_means[int(class_index), :]).float().to(self._device)
        else:
            for class_index in range(ncm_classes):
                if np.any(self._class_means[class_index] != 0):
                    self._network.ncm_fc.weight.data[int(class_index), :] = torch.from_numpy(self._class_means[int(class_index), :]).float().to(self._device)
    
    def _build_knn_bank(self):
        """Handle build knn bank."""
        knn_bank_empty = (not hasattr(self.knn, 'class_to_features')) or (len(self.knn.class_to_features) == 0)
        
        if knn_bank_empty:
            all_train_dataset = self.data_manager.get_dataset(
                np.arange(0, self._total_classes), source="train", mode="test"
            )
            all_train_loader = DataLoader(
                all_train_dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                drop_last=False, 
                num_workers=num_workers
            )
            
            def feature_fn(x):
                if isinstance(self._network, nn.DataParallel):
                    feats = self._network.module(x, pen=True, train=False)
                else:
                    feats = self._network(x, pen=True, train=False)
                
                if isinstance(feats, tuple):
                    feats = feats[0]
                
                if not isinstance(feats, torch.Tensor):
                    raise TypeError(f"Expected tensor, got {type(feats)}")
                
                if len(feats.shape) == 1:
                    feats = feats.reshape(1, -1)
                elif len(feats.shape) != 2:
                    raise ValueError(f"Expected 2D features [B, D], got shape {feats.shape}")
                
                return feats
            
            self.knn.clear()
            with torch.no_grad():
                self.knn.add_from_loader(all_train_loader, feature_fn, self._device)
            
            logging.info(f"KNN bank built: computed features for {self._total_classes} classes")
        else:
            
            
            def feature_fn(x):
                if isinstance(self._network, nn.DataParallel):
                    feats = self._network.module(x, pen=True, train=False)
                else:
                    feats = self._network(x, pen=True, train=False)
                
                if isinstance(feats, tuple):
                    feats = feats[0]
                
                if not isinstance(feats, torch.Tensor):
                    raise TypeError(f"Expected tensor, got {type(feats)}")
                
                if len(feats.shape) == 1:
                    feats = feats.reshape(1, -1)
                elif len(feats.shape) != 2:
                    raise ValueError(f"Expected 2D features [B, D], got shape {feats.shape}")
                
                return feats
            
            current_task_dataset = self.data_manager.get_dataset(
                np.arange(self._known_classes, self._total_classes), source="train", mode="test"
            )
            current_task_loader = DataLoader(
                current_task_dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                drop_last=False, 
                num_workers=num_workers
            )
            
            with torch.no_grad():
                self.knn.add_from_loader(current_task_loader, feature_fn, self._device)
            
            logging.info(f"KNN bank updated: added features for classes {self._known_classes}-{self._total_classes-1} (previous features preserved)")
    
    def _build_knn_bank_upperbound(self):
        """Handle build knn bank upperbound."""
        logging.info("KNN upperbound mode: skipping build at training end (will be rebuilt at eval time with latest model)")
    
    def _build_knn_bank_upperbound_at_eval(self):
        """Handle build knn bank upperbound at eval."""
        
        self.knn.clear()
        
        self._network.eval()
        
        def feature_fn(x):
            if isinstance(self._network, nn.DataParallel):
                feats = self._network.module(x, pen=True, train=False)
            else:
                feats = self._network(x, pen=True, train=False)
            
            if isinstance(feats, tuple):
                feats = feats[0]
            
            if not isinstance(feats, torch.Tensor):
                raise TypeError(f"Expected tensor, got {type(feats)}")
            
            if len(feats.shape) == 1:
                feats = feats.reshape(1, -1)
            elif len(feats.shape) != 2:
                raise ValueError(f"Expected 2D features [B, D], got shape {feats.shape}")
            
            return feats
        
        all_train_dataset = self.data_manager.get_dataset(
            np.arange(0, self._total_classes), source="train", mode="test"
        )
        all_train_loader = DataLoader(
            all_train_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            drop_last=False, 
            num_workers=num_workers
        )
        
        with torch.no_grad():
            self.knn.add_from_loader(all_train_loader, feature_fn, self._device)
        
        logging.info(f"KNN bank rebuilt in upperbound mode with {self._total_classes} classes using latest model features")

    def _get_hc_soinn_feature_fn(self):
        """Handle feature fn."""
        def feature_fn(x):
            if isinstance(self._network, nn.DataParallel):
                feats = self._network.module(x, pen=True, train=False)
            else:
                feats = self._network(x, pen=True, train=False)
            
            if isinstance(feats, tuple):
                feats = feats[0]
            
            if not isinstance(feats, torch.Tensor):
                raise TypeError(f"Expected tensor, got {type(feats)}")
            
            if len(feats.shape) == 1:
                feats = feats.reshape(1, -1)
            elif len(feats.shape) != 2:
                raise ValueError(f"Expected 2D features [B, D], got shape {feats.shape}")
            
            return feats
        
        return feature_fn

    def _build_hc_soinn_bank(self):
        """Handle build hc soinn bank."""
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
            self.hc_soinn.add_features(feats_np, lbs_np)

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

        try:
            self.hc_soinn.compress()
        except Exception as e:
            logging.error(f"HC-SOINN compress error: {e}", exc_info=True)

    def data_weighting(self):
        self.dw_k = torch.tensor(np.ones(self._total_classes + 1, dtype=np.float32))
        self.dw_k = self.dw_k.to(self._device)

    def get_optimizer(self):
        if len(self._multiple_gpus) > 1:
            params = list(self._network.module.prompt.parameters()) + list(self._network.module.fc.parameters())
        else:
            params = list(self._network.prompt.parameters()) + list(self._network.fc.parameters())
        
        if self.use_kac:
            fc_param_count = sum(p.numel() for p in self._network.fc.parameters() if p.requires_grad)
            logging.info(f"KAC classifier parameters included in optimizer: {fc_param_count:,}")
        
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(params, momentum=0.9, lr=self.init_lr,weight_decay=self.weight_decay)
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(params, lr=self.init_lr, weight_decay=self.weight_decay)
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(params, lr=self.init_lr, weight_decay=self.weight_decay)

        return optimizer

    def get_scheduler(self, optimizer):
        if self.args["scheduler"] == 'cosine':
            scheduler = CosineSchedule(optimizer, K=self.args["tuned_epoch"])
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        profile_training_time = self._should_profile_training_time()
        if profile_training_time and not hasattr(self, "_last_training_time_breakdown"):
            self._last_training_time_breakdown = {
                "pure_train_sec": 0.0,
                "periodic_eval_sec": 0.0,
                "classifier_build_sec": 0.0,
            }

        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.train()

            losses = 0.0
            correct, total = 0, 0
            if profile_training_time:
                self._sync_training_time_profile()
                train_start = time.perf_counter()
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
            
                # logits
                logits, prompt_loss = self._network(inputs, train=True)
                logits = logits[:, :self._total_classes]

                logits[:, :self._known_classes] = float('-inf')
                dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
                loss_supervised = (F.cross_entropy(logits, targets.long()) * dw_cls).mean()

                # ce loss
                loss = loss_supervised + prompt_loss.sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            if profile_training_time:
                self._sync_training_time_profile()
                self._last_training_time_breakdown["pure_train_sec"] += time.perf_counter() - train_start

            if scheduler:
                scheduler.step()
            
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if (epoch + 1) % 5 == 0:
                if profile_training_time:
                    self._sync_training_time_profile()
                    eval_start = time.perf_counter()
                test_acc = self._compute_accuracy(self._network, test_loader)
                if profile_training_time:
                    self._sync_training_time_profile()
                    self._last_training_time_breakdown["periodic_eval_sec"] += time.perf_counter() - eval_start
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

    def _eval_fc(self, loader):
        """Handle eval fc."""
        self._network.eval()
        y_pred, y_true = [], []
        if self.test_only_first_task_classes:
            eval_classes = self.init_cls
        else:
            eval_classes = self._total_classes
        
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)[:, :eval_classes]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)
    
    def _eval_kac(self, loader):
        """Handle eval kac."""
        return self._eval_fc(loader)
    
    def _eval_knn(self, loader):
        """Handle eval knn."""
        self._network.eval()
        y_pred, y_true = [], []
        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(loader):
                inputs = inputs.to(self._device)
                if isinstance(self._network, nn.DataParallel):
                    feats = self._network.module(inputs, pen=True, train=False)
                else:
                    feats = self._network(inputs, pen=True, train=False)
                
                if isinstance(feats, tuple):
                    feats = feats[0]
                
                if not isinstance(feats, torch.Tensor):
                    raise TypeError(f"Expected tensor, got {type(feats)}")
                
                feats = feats.detach().cpu()
                
                if len(feats.shape) == 1:
                    feats = feats.reshape(1, -1)
                elif len(feats.shape) != 2:
                    raise ValueError(f"Expected 2D features [B, D], got shape {feats.shape}")
                
                feats_np = feats.numpy()
                topk_pred = self.knn.predict_topk(
                    feats_np, self.topk, self._total_classes, device=self._device, track_usage=True
                )
                y_pred.append(topk_pred)
                y_true.append(targets.cpu().numpy())
        
        if len(y_pred) == 0:
            logging.warning("No predictions generated from KNN evaluation")
            return np.array([]), np.array([])
        
        return np.concatenate(y_pred), np.concatenate(y_true)

    def _eval_ncm_fc(self, loader):
        """Handle eval ncm fc."""
        self._network.eval()
        y_pred, y_true = [], []
        if self.test_only_first_task_classes:
            ncm_classes = self.init_cls
        else:
            ncm_classes = self._total_classes
        
        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(loader):
                inputs = inputs.to(self._device)
                if isinstance(self._network, nn.DataParallel):
                    features = self._network.module.extract_vector(inputs)
                    ncm_output = self._network.module.ncm_fc(features)
                else:
                    features = self._network.extract_vector(inputs)
                    ncm_output = self._network.ncm_fc(features)
                ncm_logits = ncm_output['logits']
                ncm_logits = ncm_logits[:, :ncm_classes]
                predicts = torch.topk(
                    ncm_logits, k=self.topk, dim=1, largest=True, sorted=True
                )[1]
                y_pred.append(predicts.cpu().numpy())
                y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)

    def _eval_hc_soinn(self, loader):
        """Handle eval hc soinn."""
        self._network.eval()
        y_pred, y_true = [], []
        feature_fn = self._get_hc_soinn_feature_fn()
        profile_on = bool(self.args.get("hcsoinn_profile_inference", False))
        profile_sync_cuda = bool(self.args.get("hcsoinn_profile_sync_cuda", True))

        def _sync_if_needed():
            if profile_on and profile_sync_cuda and self._device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(device=self._device)

        batch_count = 0
        sample_count = 0
        feature_extract_sec = 0.0
        feature_to_numpy_sec = 0.0
        hc_predict_sec = 0.0

        if profile_on and hasattr(self.hc_soinn, "set_inference_profiling"):
            self.hc_soinn.set_inference_profiling(True, reset=True, sync_cuda=profile_sync_cuda)

        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(loader):
                inputs = inputs.to(self._device)

                _sync_if_needed()
                t0 = time.perf_counter()
                feats = feature_fn(inputs)
                _sync_if_needed()
                feature_extract_sec += time.perf_counter() - t0

                _sync_if_needed()
                t0 = time.perf_counter()
                feats = feats.detach().cpu()
                if len(feats.shape) == 1:
                    feats = feats.reshape(1, -1)
                elif len(feats.shape) != 2:
                    raise ValueError(f"Expected 2D features [B, D], got shape {feats.shape}")
                feats_np = feats.numpy()
                feature_to_numpy_sec += time.perf_counter() - t0

                _sync_if_needed()
                t0 = time.perf_counter()
                topk_pred = self.hc_soinn.predict_topk(
                    feats, self.topk, self._total_classes, device=self._device
                )
                _sync_if_needed()
                hc_predict_sec += time.perf_counter() - t0

                y_pred.append(topk_pred)
                y_true.append(targets.cpu().numpy())
                batch_count += 1
                sample_count += int(targets.shape[0])

        if len(y_pred) == 0:
            logging.warning("No predictions generated from HC-SOINN evaluation")
            return np.array([]), np.array([])

        if profile_on:
            total_eval_sec = feature_extract_sec + feature_to_numpy_sec + hc_predict_sec
            logging.info("=" * 72)
            logging.info(
                f"[HC-SOINN Inference Profile] task={self._cur_task}, "
                f"batches={batch_count}, samples={sample_count}, total={total_eval_sec:.6f}s"
            )
            if total_eval_sec > 0:
                logging.info(
                    f"  feature_extract: {feature_extract_sec:.6f}s "
                    f"({feature_extract_sec / total_eval_sec * 100:.2f}%)"
                )
                logging.info(
                    f"  feature_to_numpy: {feature_to_numpy_sec:.6f}s "
                    f"({feature_to_numpy_sec / total_eval_sec * 100:.2f}%)"
                )
                logging.info(
                    f"  hcsoinn_predict_topk: {hc_predict_sec:.6f}s "
                    f"({hc_predict_sec / total_eval_sec * 100:.2f}%)"
                )

            if hasattr(self.hc_soinn, "get_profile_stats"):
                profile = self.hc_soinn.get_profile_stats(reset=False)
                calls = int(profile.get("calls", 0))
                total_sec = float(profile.get("total_sec", 0.0))
                logging.info(
                    f"  predict_topk_internal: calls={calls}, total={total_sec:.6f}s, "
                    f"avg/call={profile.get('avg_ms_per_call', 0.0):.3f}ms, "
                    f"avg/sample={profile.get('avg_ms_per_sample', 0.0):.3f}ms"
                )
                step_sec = profile.get("steps_sec", {})
                step_ratio = profile.get("steps_ratio", {})
                for step_name, sec in step_sec.items():
                    ratio = float(step_ratio.get(step_name, 0.0)) * 100.0
                    logging.info(f"    - {step_name}: {sec:.6f}s ({ratio:.2f}%)")
            logging.info("=" * 72)

        return np.concatenate(y_pred), np.concatenate(y_true)

    def _eval_hc_soinn_fc_fusion(self, loader):
        """
        Fusion experiment:
        result = a * HC-SOINN + b * (Calibrated FC), where calibrated scores are probabilities.
        """
        self._network.eval()
        y_pred, y_true = [], []
        feature_fn = self._get_hc_soinn_feature_fn()
        eval_classes = self.init_cls if self.test_only_first_task_classes else self._total_classes

        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(loader):
                inputs = inputs.to(self._device)
                fc_logits = self._network(inputs)[:, :eval_classes]
                feats = feature_fn(inputs)
                feats = F.normalize(feats, p=2, dim=1)
                hc_logits = self.hc_soinn.predict_class_logits(
                    feats, eval_classes, device=self._device
                )

                fc_probs = torch.softmax(fc_logits / self.hc_soinn_fc_calib_temp, dim=1)
                hc_probs = torch.softmax(hc_logits / self.hc_soinn_hc_calib_temp, dim=1)
                fused_probs = self.hc_soinn_fc_fusion_a * hc_probs + self.hc_soinn_fc_fusion_b * fc_probs

                predicts = torch.topk(fused_probs, k=self.topk, dim=1, largest=True, sorted=True)[1]
                y_pred.append(predicts.cpu().numpy())
                y_true.append(targets.cpu().numpy())

        if len(y_pred) == 0:
            logging.warning("No predictions generated from HC-SOINN + FC fusion evaluation")
            return np.array([]), np.array([])
        return np.concatenate(y_pred), np.concatenate(y_true)

    def _profile_sync_if_needed(self):
        """Synchronize CUDA timers when speed profiling uses GPU."""
        if (
            bool(self.args.get("speed_profile_sync_cuda", True))
            and self._device.type == "cuda"
            and torch.cuda.is_available()
        ):
            torch.cuda.synchronize(device=self._device)

    def _profile_classifier_once(self, classifier_name):
        """Profile one classifier on the current test loader."""
        classifier_name = classifier_name.lower()
        if classifier_name == "hc_soinn_star":
            classifier_name = "hc_soinn"

        self._network.eval()
        y_pred, y_true = [], []
        backbone_sec = 0.0
        classifier_sec = 0.0
        batches = 0
        samples = 0
        eval_classes = self.init_cls if self.test_only_first_task_classes else self._total_classes

        if classifier_name in ("fc", "kac", "ncm"):
            network = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
        elif classifier_name == "hc_soinn":
            if not hasattr(self, "hc_soinn"):
                raise RuntimeError("HC-SOINN classifier is not initialized")
            feature_fn = self._get_hc_soinn_feature_fn()
        else:
            raise ValueError(f"Unsupported speed profile classifier: {classifier_name}")

        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(self.test_loader):
                self._profile_sync_if_needed()
                t0 = time.perf_counter()
                inputs = inputs.to(self._device)
                if classifier_name == "hc_soinn":
                    features = feature_fn(inputs)
                else:
                    features = network.extract_vector(inputs)
                self._profile_sync_if_needed()
                backbone_sec += time.perf_counter() - t0

                self._profile_sync_if_needed()
                t0 = time.perf_counter()
                if classifier_name in ("fc", "kac"):
                    outputs = network.fc(features)[:, :eval_classes]
                    predicts = torch.topk(
                        outputs, k=self.topk, dim=1, largest=True, sorted=True
                    )[1]
                    batch_pred = predicts.cpu().numpy()
                elif classifier_name == "ncm":
                    ncm_output = network.ncm_fc(features)
                    ncm_logits = ncm_output["logits"] if isinstance(ncm_output, dict) else ncm_output
                    predicts = torch.topk(
                        ncm_logits[:, :eval_classes], k=self.topk, dim=1, largest=True, sorted=True
                    )[1]
                    batch_pred = predicts.cpu().numpy()
                else:
                    features = features.detach().cpu()
                    if len(features.shape) == 1:
                        features = features.reshape(1, -1)
                    elif len(features.shape) != 2:
                        raise ValueError(f"Expected 2D features [B, D], got shape {features.shape}")
                    batch_pred = self.hc_soinn.predict_topk(
                        features, self.topk, self._total_classes, device=self._device
                    )
                self._profile_sync_if_needed()
                classifier_sec += time.perf_counter() - t0

                y_pred.append(batch_pred)
                y_true.append(targets.cpu().numpy())
                batches += 1
                samples += int(targets.shape[0])

        if len(y_pred) == 0:
            raise RuntimeError(f"No predictions generated for {classifier_name} speed profile")

        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        acc = self._evaluate(y_pred, y_true)
        total_sec = backbone_sec + classifier_sec
        backbone_pct = (backbone_sec / total_sec * 100.0) if total_sec > 0 else 0.0
        classifier_pct = (classifier_sec / total_sec * 100.0) if total_sec > 0 else 0.0

        return {
            "classifier": classifier_name,
            "task": int(self._cur_task),
            "samples": int(samples),
            "batches": int(batches),
            "total_sec": float(total_sec),
            "backbone_sec": float(backbone_sec),
            "backbone_pct": float(backbone_pct),
            "classifier_sec": float(classifier_sec),
            "classifier_pct": float(classifier_pct),
            "top1": float(acc["top1"]),
            "top5": float(acc[f"top{self.topk}"]),
        }

    def profile_last_task_classifiers(self):
        """Run configured single-classifier speed profiles for the last task."""
        classifiers = self.args.get("speed_profile_classifiers", [])
        if isinstance(classifiers, str):
            classifiers = [classifiers]
        repeats = int(self.args.get("speed_profile_repeats", 3))
        profile_results = []

        for classifier_name in classifiers:
            display_name = str(classifier_name)
            logging.info("=" * 72)
            logging.info(
                f"[Speed Profile] classifier={display_name}, task={self._cur_task}, repeats={repeats}"
            )
            for repeat_idx in range(repeats):
                result = self._profile_classifier_once(display_name)
                result["classifier"] = display_name
                result["repeat"] = repeat_idx + 1
                profile_results.append(result)
                logging.info(
                    "[Speed Profile] "
                    f"classifier={display_name}, repeat={repeat_idx + 1}/{repeats}, "
                    f"total={result['total_sec']:.6f}s, "
                    f"backbone={result['backbone_sec']:.6f}s ({result['backbone_pct']:.2f}%), "
                    f"classifier={result['classifier_sec']:.6f}s ({result['classifier_pct']:.2f}%), "
                    f"top1={result['top1']:.2f}, top5={result['top5']:.2f}"
                )
            logging.info("=" * 72)

        return profile_results

    def eval_task(self):
        """Handle eval task."""
        eval_fc = bool(self.args.get("eval_fc", True))

        if getattr(self, "use_knn", False) and getattr(self, "knn_upperbound", False):
            self._build_knn_bank_upperbound_at_eval()
        
        results = {}
        
        if eval_fc:
            if self.use_kac:
                y_pred_kac, y_true_kac = self._eval_kac(self.test_loader)
                results["kac"] = self._evaluate(y_pred_kac, y_true_kac)
                results["fc"] = results["kac"]
            else:
                y_pred_fc, y_true_fc = self._eval_fc(self.test_loader)
                results["fc"] = self._evaluate(y_pred_fc, y_true_fc)
        else:
            logging.info("Skipping FC evaluation (eval_fc=false)")
        
        if getattr(self, "use_knn", False):
            y_pred_knn, y_true_knn = self._eval_knn(self.test_loader)
            results["knn"] = self._evaluate(y_pred_knn, y_true_knn)
        
        if getattr(self, "use_hc_soinn", False):
            y_pred_hc, y_true_hc = self._eval_hc_soinn(self.test_loader)
            results["hc_soinn"] = self._evaluate(y_pred_hc, y_true_hc)
            if self.use_hc_soinn_fc_fusion:
                y_pred_mix, y_true_mix = self._eval_hc_soinn_fc_fusion(self.test_loader)
                results["hc_soinn_fc_fusion"] = self._evaluate(y_pred_mix, y_true_mix)
        
        if getattr(self, "use_ncm", True) and hasattr(self, "_class_means") and self._class_means is not None:
            y_pred_ncm, y_true_ncm = self._eval_ncm_fc(self.test_loader)
            results["ncm"] = self._evaluate(y_pred_ncm, y_true_ncm)
        
        return results

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        if self.test_only_first_task_classes:
            eval_classes = self.init_cls
        else:
            eval_classes = self._total_classes
        
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)[:, :eval_classes]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def save_checkpoint(self):
        """Handle save checkpoint."""
        checkpoint_dir = self.args.get("checkpoint_dir", "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_name = self.args.get("checkpoint_name", 
            f"{self.args['model_name']}_{self.args['dataset']}_{self.args['init_cls']}_{self.args['increment']}")
        checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_name}_task{self._cur_task}.pkl")
        
        model = self._network
        if isinstance(model, nn.DataParallel):
            model = model.module
        
        save_dict = {
            "task": self._cur_task,
            "known_classes": self._known_classes,
            "total_classes": self._total_classes,
            "backbone_state_dict": model.backbone.state_dict(),
            "fc_state_dict": model.fc.state_dict(),
            "prompt_state_dict": model.prompt.state_dict() if model.prompt is not None else None,
        }
        
        torch.save(save_dict, checkpoint_path)
        logging.info(f"Checkpoint saved to {checkpoint_path}")

    def _visualize_hc_soinn_tsne(self, target_classes=None):
        """Handle visualize hc soinn tsne."""
        if target_classes is None:
            init_cls = self.args.get("init_cls", 10)
            target_classes = list(range(min(10, init_cls)))
        
        if len(target_classes) == 0:
            logging.warning(f"No classes to visualize for task {self._cur_task}, skipping visualization")
            return
        
        logging.info(f"Visualizing HC-SOINN t-SNE for task {self._cur_task} classes: {target_classes}")
        
        if not hasattr(self, 'hc_soinn') or not hasattr(self.hc_soinn, 'class_mu') or len(self.hc_soinn.class_mu) == 0:
            logging.warning("HC-SOINN class centers are empty, skipping visualization")
            return
        
        available_classes = [cls for cls in target_classes if cls in self.hc_soinn.class_mu]
        if len(available_classes) == 0:
            logging.warning("No target classes found in HC-SOINN, skipping visualization")
            return
        
        if not hasattr(self, 'data_manager'):
            logging.warning("DataManager not found, skipping visualization")
            return
        
        try:
            train_dataset = self.data_manager.get_dataset(
                np.array(available_classes),
                source="train",
                mode="test"
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers
            )
            
            sample_features = []
            sample_labels = []
            
            self._network.eval()
            with torch.no_grad():
                for _, inputs, targets in train_loader:
                    inputs = inputs.to(self._device)
                    feats = self._network.extract_vector(inputs)
                    feats_np = feats.detach().cpu().numpy()
                    sample_features.append(feats_np)
                    sample_labels.append(targets.numpy())
            
            if len(sample_features) == 0:
                logging.warning("No sample features extracted, skipping visualization")
                return
            
            sample_features = np.concatenate(sample_features, axis=0)
            sample_labels = np.concatenate(sample_labels, axis=0)
            
            mask = np.isin(sample_labels, available_classes)
            sample_features = sample_features[mask]
            sample_labels = sample_labels[mask]
            
            if len(sample_features) == 0:
                logging.warning("No samples found for target classes, skipping visualization")
                return
            
            sample_features = sample_features / (np.linalg.norm(sample_features, axis=1, keepdims=True) + 1e-8)
            
            ncm_features = []
            ncm_labels = []
            
            for cls in available_classes:
                if cls in self.hc_soinn.class_mu:
                    ncm_features.append(self.hc_soinn.class_mu[cls])
                    ncm_labels.append(cls)
            
            if len(ncm_features) == 0:
                logging.warning("No NCM centers found, skipping visualization")
                return
            
            ncm_features = np.stack(ncm_features, axis=0)
            ncm_labels = np.array(ncm_labels)
            
            ncm_features = ncm_features / (np.linalg.norm(ncm_features, axis=1, keepdims=True) + 1e-8)
            
            cluster_features = []
            cluster_labels = []
            cluster_class_mapping = []
            class_cluster_edges = {}
            
            for cls in available_classes:
                if cls in self.hc_soinn.class_clusters:
                    clusters = self.hc_soinn.class_clusters[cls]
                    if len(clusters) > 0:
                        cls_cluster_centers = np.stack([c.center for c in clusters], axis=0)
                        cluster_features.append(cls_cluster_centers)
                        cluster_labels.extend([cls] * len(clusters))
                        start_idx = len(cluster_class_mapping)
                        for i in range(len(clusters)):
                            cluster_class_mapping.append((cls, i))
                        
                        if hasattr(self.hc_soinn, 'class_edges') and cls in self.hc_soinn.class_edges:
                            class_cluster_edges[cls] = self.hc_soinn.class_edges[cls]
            
            if len(cluster_features) > 0:
                cluster_features = np.concatenate(cluster_features, axis=0)
                cluster_labels = np.array(cluster_labels)
                cluster_features = cluster_features / (np.linalg.norm(cluster_features, axis=1, keepdims=True) + 1e-8)
            else:
                cluster_features = np.empty((0, ncm_features.shape[1]))
                cluster_labels = np.array([])
            
            all_features_list = [sample_features, ncm_features]
            all_labels_list = [sample_labels, ncm_labels]
            
            if len(cluster_features) > 0:
                all_features_list.append(cluster_features)
                all_labels_list.append(cluster_labels)
            
            all_features = np.vstack(all_features_list)
            
            if len(all_features) < 2:
                logging.warning("Not enough features for t-SNE, skipping visualization")
                return
            
            sample_end = len(sample_features)
            ncm_end = sample_end + len(ncm_features)
            
            model_name = self.args.get("model_name", "coda_prompt")
            dataset = self.args.get("dataset", "cifar224")
            init_cls = self.args.get("init_cls", 10)
            increment = self.args.get("increment", 10)
            use_star = self.args.get("use_feature_alignment", False)
            
            if use_star:
                vis_base_dir = f"logs/{model_name}/{dataset}/{init_cls}/{increment}/hcsoinn_star_visualizations"
            else:
                vis_base_dir = f"logs/{model_name}/{dataset}/{init_cls}/{increment}/hcsoinn_visualizations"
            os.makedirs(vis_base_dir, exist_ok=True)
            
            umap_model_path = os.path.join(vis_base_dir, "umap_model.pkl")
            
            reference_umap_path = self.args.get("reference_umap_model_path", None)
            if reference_umap_path:
                if not os.path.isabs(reference_umap_path):
                    reference_umap_path = os.path.join(os.getcwd(), reference_umap_path)
                if os.path.exists(reference_umap_path):
                    umap_model_path = reference_umap_path
                    logging.info(f"Using reference UMAP model from {reference_umap_path} for cross-experiment comparison")
                else:
                    logging.warning(f"Reference UMAP model not found at {reference_umap_path}, will use local model or fallback to t-SNE")
            
            if UMAP_AVAILABLE:
                if self._cur_task == 0 and reference_umap_path is None:
                    logging.info(f"Fitting UMAP model for fixed coordinate system ({len(all_features)} points)...")
                    reducer = umap.UMAP(
                        n_components=2,
                        n_neighbors=15,
                        min_dist=0.1,
                        metric='cosine',
                        random_state=42
                    )
                    embeddings = reducer.fit_transform(all_features)
                    import pickle
                    with open(umap_model_path, 'wb') as f:
                        pickle.dump(reducer, f)
                    logging.info(f"UMAP model saved to {umap_model_path}")
                else:
                    import pickle
                    if os.path.exists(umap_model_path):
                        logging.info(f"Loading UMAP model from {umap_model_path} for fixed coordinate system...")
                        with open(umap_model_path, 'rb') as f:
                            reducer = pickle.load(f)
                        embeddings = reducer.transform(all_features)
                        logging.info(f"Transformed {len(all_features)} points using fixed UMAP model")
                    else:
                        logging.warning(f"UMAP model not found at {umap_model_path}, falling back to t-SNE")
                        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_features) - 1), max_iter=1000)
                        embeddings = tsne.fit_transform(all_features)
            else:
                logging.info(f"Computing t-SNE for {len(all_features)} points ({len(sample_features)} samples + {len(ncm_features)} NCM + {len(cluster_features)} clusters)...")
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_features) - 1), max_iter=1000)
                embeddings = tsne.fit_transform(all_features)
            
            sample_embeddings = embeddings[:sample_end]
            ncm_embeddings = embeddings[sample_end:ncm_end]
            cluster_embeddings = embeddings[ncm_end:] if len(cluster_features) > 0 else None
            
            plt.figure(figsize=(16, 12))
            
            num_classes = len(available_classes)
            if num_classes <= 10:
                colors = plt.cm.tab10(np.linspace(0, 1, 10))[:num_classes]
            else:
                colors = plt.cm.tab20(np.linspace(0, 1, 20))[:num_classes]
            
            cls_to_color = {cls: colors[i] for i, cls in enumerate(available_classes)}
            
            for idx, cls in enumerate(available_classes):
                cls_mask = sample_labels == cls
                if np.any(cls_mask):
                    plt.scatter(
                        sample_embeddings[cls_mask, 0],
                        sample_embeddings[cls_mask, 1],
                        c=[cls_to_color[cls]],
                        marker='o',
                        s=30,
                        alpha=0.4,
                        label=f'Samples (Class {cls})' if idx == 0 else '',
                        edgecolors='none',
                        zorder=1
                    )
            
            if cluster_embeddings is not None and len(cluster_embeddings) > 0:
                for cls in available_classes:
                    if cls in class_cluster_edges and len(class_cluster_edges[cls]) > 0:
                        cls_mask = cluster_labels == cls
                        cls_indices = np.where(cls_mask)[0]
                        if len(cls_indices) == 0:
                            continue
                        
                        cls_cluster_start = cls_indices[0]
                        cls_edges = class_cluster_edges[cls]
                        
                        for node_idx, neighbors in cls_edges.items():
                            if node_idx >= len(cls_indices):
                                continue
                            global_idx_i = cls_cluster_start + node_idx
                            if global_idx_i >= len(cluster_embeddings):
                                continue
                            
                            try:
                                limited_neighbors = list(neighbors)[:2]
                            except (TypeError, ValueError):
                                limited_neighbors = []
                            
                            for neighbor_idx in limited_neighbors:
                                if neighbor_idx >= len(cls_indices):
                                    continue
                                global_idx_j = cls_cluster_start + neighbor_idx
                                if global_idx_j >= len(cluster_embeddings):
                                    continue
                                
                                if node_idx < neighbor_idx:
                                    x_coords = [cluster_embeddings[global_idx_i, 0], cluster_embeddings[global_idx_j, 0]]
                                    y_coords = [cluster_embeddings[global_idx_i, 1], cluster_embeddings[global_idx_j, 1]]
                                    plt.plot(
                                        x_coords, y_coords,
                                        color=cls_to_color[cls],
                                        alpha=0.4,
                                        linewidth=2.0,
                                        zorder=2
                                    )
                
                for idx, cls in enumerate(available_classes):
                    cls_mask = cluster_labels == cls
                    if np.any(cls_mask):
                        plt.scatter(
                            cluster_embeddings[cls_mask, 0],
                            cluster_embeddings[cls_mask, 1],
                            c=[cls_to_color[cls]],
                            marker='o',
                            s=120,
                            alpha=0.8,
                            label=f'HC-SOINN (Class {cls})' if idx == 0 else '',
                            edgecolors='black',
                            linewidths=1.5,
                            zorder=3
                        )
            
            for idx, cls in enumerate(available_classes):
                cls_mask = ncm_labels == cls
                if np.any(cls_mask):
                    plt.scatter(
                        ncm_embeddings[cls_mask, 0],
                        ncm_embeddings[cls_mask, 1],
                        c='black',
                        marker='x',
                        s=180,
                        alpha=0.9,
                        linewidths=3.5,
                        zorder=9
                    )
                    plt.scatter(
                        ncm_embeddings[cls_mask, 0],
                        ncm_embeddings[cls_mask, 1],
                        c=[cls_to_color[cls]],
                        marker='x',
                        s=150,
                        alpha=0.9,
                        label=f'NCM (Class {cls})',
                        linewidths=2.5,
                        zorder=10
                    )
            
            if UMAP_AVAILABLE and (self._cur_task == 0 or os.path.exists(umap_model_path)):
                plt.title(f'HC-SOINN UMAP Visualization (Task {self._cur_task}, Fixed Coordinate)', fontsize=14, fontweight='bold')
                plt.xlabel('UMAP Dimension 1', fontsize=12)
                plt.ylabel('UMAP Dimension 2', fontsize=12)
            else:
                plt.title(f'HC-SOINN t-SNE Visualization (Task {self._cur_task})', fontsize=14, fontweight='bold')
                plt.xlabel('t-SNE Dimension 1', fontsize=12)
                plt.ylabel('t-SNE Dimension 2', fontsize=12)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            vis_dir = vis_base_dir
            
            vis_path = os.path.join(vis_dir, f"tsne_task{self._cur_task}.png")
            plt.savefig(vis_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"HC-SOINN t-SNE visualization saved to {vis_path}")
            logging.info(f"  - Samples: {len(sample_features)}")
            logging.info(f"  - NCM centers: {len(ncm_features)}")
            logging.info(f"  - Cluster prototypes: {len(cluster_features)}")
            
        except Exception as e:
            logging.error(f"Error in HC-SOINN t-SNE visualization: {e}", exc_info=True)

    def load_checkpoint(self, checkpoint_path):
        """Handle load checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self._device)
        
        model = self._network
        if isinstance(model, nn.DataParallel):
            model = model.module
        
        if "backbone_state_dict" in checkpoint:
            model.backbone.load_state_dict(checkpoint["backbone_state_dict"])
            logging.info("Backbone loaded successfully")
        
        if "fc_state_dict" in checkpoint:
            fc_state = checkpoint["fc_state_dict"]
            current_fc_state = model.fc.state_dict()
            
            if fc_state["weight"].shape[0] != current_fc_state["weight"].shape[0]:
                logging.warning(
                    f"FC layer size mismatch: checkpoint has {fc_state['weight'].shape[0]} classes, "
                    f"current model has {current_fc_state['weight'].shape[0]} classes. "
                    f"Loading compatible weights only."
                )
                min_classes = min(fc_state["weight"].shape[0], current_fc_state["weight"].shape[0])
                current_fc_state["weight"][:min_classes] = fc_state["weight"][:min_classes]
                if "bias" in fc_state and "bias" in current_fc_state:
                    current_fc_state["bias"][:min_classes] = fc_state["bias"][:min_classes]
                model.fc.load_state_dict(current_fc_state)
            else:
                model.fc.load_state_dict(fc_state)
            logging.info("FC layer loaded successfully")
        
        if "prompt_state_dict" in checkpoint and checkpoint["prompt_state_dict"] is not None:
            if model.prompt is not None:
                model.prompt.load_state_dict(checkpoint["prompt_state_dict"])
                logging.info("Prompt loaded successfully")
            else:
                logging.warning("Checkpoint contains prompt state but current model has no prompt")
        
        if "task" in checkpoint:
            self._cur_task = checkpoint["task"]
            logging.info(f"Restored task: {self._cur_task}")
        
        if "known_classes" in checkpoint:
            self._known_classes = checkpoint["known_classes"]
            logging.info(f"Restored known_classes: {self._known_classes}")
        
        if "total_classes" in checkpoint:
            self._total_classes = checkpoint["total_classes"]
            logging.info(f"Restored total_classes: {self._total_classes}")
        
        self._network.to(self._device)
        logging.info("Checkpoint loaded successfully")
    
    def _load_checkpoint_for_task(self):
        """Handle load checkpoint for task."""
        checkpoint_dir = self.args.get("checkpoint_dir", "checkpoints")
        checkpoint_name = self.args.get("checkpoint_name", 
            f"{self.args['model_name']}_{self.args['dataset']}_{self.args['init_cls']}_{self.args['increment']}")
        checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_name}_task{self._cur_task}.pkl")
        
        if os.path.exists(checkpoint_path):
            logging.info(f"Loading checkpoint for task {self._cur_task}: {checkpoint_path}")
            self.load_checkpoint(checkpoint_path)
        else:
            raise FileNotFoundError(
                f"Checkpoint not found for task {self._cur_task}: {checkpoint_path}. "
                f"Please ensure all checkpoints are saved with save_checkpoint=True"
            )


class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class CosineSchedule(_LRScheduler):

    def __init__(self, optimizer, K):
        self.K = K
        super().__init__(optimizer, -1)

    def cosine(self, base_lr):
        if self.K <= 1:
            return base_lr
        return base_lr * math.cos((99 * math.pi * (self.last_epoch)) / (200 * (self.K-1)))

    def get_lr(self):
        return [self.cosine(base_lr) for base_lr in self.base_lrs]