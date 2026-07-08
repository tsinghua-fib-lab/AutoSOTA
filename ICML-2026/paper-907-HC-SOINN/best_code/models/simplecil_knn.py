import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet,SimpleCosineIncrementalNet,SimpleVitNet,SimpleVitNetKNN
from models.base import BaseLearner
from utils.knn_classifier import KNNClassifier
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


num_workers = 8
batch_size = 128

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SimpleVitNetKNN(args, True)
        self.args = args
        self.knn_prune_zero_usage = args.get("knn_prune_zero_usage", False)
        self.knn = KNNClassifier(
            metric=args.get("knn_metric", "euclidean"),
            use_all_samples=args.get("knn_use_all", True),
            k_neighbors=args.get("knn_k", None),
        )

    def after_task(self):
        """Handle after task."""
        if getattr(self, "knn_prune_zero_usage", False):
            try:
                if hasattr(self, "knn") and hasattr(self.knn, "prune_zero_usage"):
                    current_task_classes = list(range(self._known_classes, self._total_classes))
                    if len(current_task_classes) > 0:
                        logging.info(
                            f"Pruning KNN nodes with zero usage for current task classes: "
                            f"{current_task_classes} (task {self._cur_task})"
                        )
                        self.knn.prune_zero_usage(current_task_classes)
            except Exception as e:
                logging.error(f"Error during KNN prune_zero_usage in after_task: {e}", exc_info=True)

        self._known_classes = self._total_classes

        self._visualize_knn_tsne()

    def incremental_train(self, data_manager):
        logging.info(f"Starting incremental_train: cur_task={self._cur_task}, known_classes={self._known_classes}")
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        logging.info(f"After update: cur_task={self._cur_task}, total_classes={self._total_classes}, known_classes={self._known_classes}")
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train", )
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        train_dataset_for_knn = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train", 
            mode="test"
        )
        self.train_loader_for_knn = DataLoader(
            train_dataset_for_knn, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, self.train_loader_for_knn)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, train_loader_for_knn):
        """Handle train."""
        self._network.to(self._device)
        self._network.eval()

        def feature_fn(x):
            return self._network.extract_vector(x)

        self.knn.add_from_loader(train_loader_for_knn, feature_fn, self._device)

    def _eval_cnn(self, loader):
        """Handle eval cnn."""
        self._network.eval()
        y_pred, y_true = [], []

        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(loader):
                inputs = inputs.to(self._device)

                features = self._network.extract_vector(inputs).detach().cpu().numpy()

                batch_topk = self.knn.predict_topk(features, self.topk, self._total_classes, device=self._device, track_usage=True)

                y_pred.append(batch_topk)
                y_true.append(targets.numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)
    
    def eval_task(self):
        """Handle eval task."""
        y_pred, y_true = self._eval_cnn(self.test_loader)
        knn_accy = self._evaluate(y_pred, y_true)

        return {"knn": knn_accy}

    def _visualize_knn_tsne(self):
        """Handle visualize knn tsne."""
        init_cls = self.args.get("init_cls", 10)
        if init_cls < 5:
            logging.warning(f"Initial classes ({init_cls}) < 5, skipping KNN visualization")
            return
        
        target_classes = list(range(5))
        
        if not hasattr(self.knn, 'class_to_features') or len(self.knn.class_to_features) == 0:
            logging.warning("KNN feature bank is empty, skipping visualization")
            return
        
        available_classes = [cls for cls in target_classes if cls in self.knn.class_to_features]
        if len(available_classes) == 0:
            logging.warning("No target classes found in KNN bank, skipping visualization")
            return
        
        if not hasattr(self, 'data_manager'):
            logging.warning("DataManager not found, skipping visualization")
            return
        
        try:
            train_dataset = self.data_manager.get_dataset(
                np.arange(0, min(5, init_cls)), 
                source="train", 
                mode="test"
            )
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
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
            
            sample_features = np.concatenate(sample_features, axis=0)
            sample_labels = np.concatenate(sample_labels, axis=0)
            
            mask = np.isin(sample_labels, target_classes)
            sample_features = sample_features[mask]
            sample_labels = sample_labels[mask]
            
            if len(sample_features) == 0:
                logging.warning("No samples found for target classes, skipping visualization")
                return
            
            sample_features = sample_features / (np.linalg.norm(sample_features, axis=1, keepdims=True) + 1e-8)
            
            knn_features = []
            knn_labels = []
            knn_usage_counts = []
            
            for cls in available_classes:
                if cls not in self.knn.class_to_features:
                    continue
                
                cls_chunks = self.knn.class_to_features[cls]
                if len(cls_chunks) == 0:
                    continue
                
                cls_feats = np.concatenate(cls_chunks, axis=0)
                cls_labels = np.full((cls_feats.shape[0],), cls, dtype=np.int64)
                
                cls_feats = cls_feats / (np.linalg.norm(cls_feats, axis=1, keepdims=True) + 1e-8)
                
                knn_features.append(cls_feats)
                knn_labels.append(cls_labels)
                
                if cls in self.knn.usage_counts:
                    usage = self.knn.usage_counts[cls]
                    if len(usage) < cls_feats.shape[0]:
                        extended_usage = np.zeros(cls_feats.shape[0], dtype=np.int64)
                        extended_usage[:len(usage)] = usage
                        usage = extended_usage
                    elif len(usage) > cls_feats.shape[0]:
                        usage = usage[:cls_feats.shape[0]]
                    knn_usage_counts.append(usage)
                else:
                    knn_usage_counts.append(np.zeros(cls_feats.shape[0], dtype=np.int64))
            
            if len(knn_features) == 0:
                logging.warning("No KNN nodes found, skipping visualization")
                return
            
            knn_features = np.concatenate(knn_features, axis=0)
            knn_labels = np.concatenate(knn_labels, axis=0)
            knn_usage_counts = np.concatenate(knn_usage_counts, axis=0)
            
            logging.info(f"Computing t-SNE for {len(knn_features)} KNN nodes...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
            embeddings = tsne.fit_transform(knn_features)
            
            plt.figure(figsize=(12, 10))
            
            colors = plt.cm.tab10(np.linspace(0, 1, 10))[:5]
            
            max_usage = knn_usage_counts.max() if len(knn_usage_counts) > 0 else 0
            if max_usage > 0:
                normalized_usage = 0.2 + 0.8 * (knn_usage_counts / max_usage)
            else:
                normalized_usage = np.full(len(knn_usage_counts), 0.2)
            
            for cls in available_classes:
                cls_mask = knn_labels == cls
                if np.any(cls_mask):
                    cls_alpha = normalized_usage[cls_mask]
                    plt.scatter(
                        embeddings[cls_mask, 0],
                        embeddings[cls_mask, 1],
                        c=[colors[cls]],
                        marker='o',
                        s=30,
                        alpha=cls_alpha,
                        label=f'Class {cls}',
                        edgecolors='black',
                        linewidths=0.5
                    )
            
            plt.title(f'KNN t-SNE Visualization (Task {self._cur_task})', fontsize=14, fontweight='bold')
            plt.xlabel('t-SNE Dimension 1', fontsize=12)
            plt.ylabel('t-SNE Dimension 2', fontsize=12)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            model_name = self.args.get("model_name", "simplecil_knn")
            dataset = self.args.get("dataset", "cifar224")
            init_cls = self.args.get("init_cls", 10)
            increment = self.args.get("increment", 10)
            
            vis_dir = f"logs/{model_name}/{dataset}/{init_cls}/{increment}/knn_visualizations"
            os.makedirs(vis_dir, exist_ok=True)
            
            vis_path = os.path.join(vis_dir, f"tsne_task{self._cur_task}.png")
            plt.savefig(vis_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"KNN t-SNE visualization saved to {vis_path}")
            
            self._log_usage_statistics(knn_usage_counts, knn_labels, available_classes)
            
        except Exception as e:
            logging.error(f"Error in KNN t-SNE visualization: {e}", exc_info=True)

    def _log_usage_statistics(self, usage_counts: np.ndarray, labels: np.ndarray, available_classes: list):
        """Handle log usage statistics."""
        if len(usage_counts) == 0:
            logging.warning("No usage counts to statistics")
            return
        
        bins = [
            (0, 0, "0次"),
            (1, 1, "1次"),
            (2, 4, "2-4次"),
            (5, 10, "5-10次"),
            (11, 20, "11-20次"),
            (21, 50, "21-50次"),
            (51, 100, "51-100次"),
            (101, float('inf'), "100次以上")
        ]
        
        logging.info("=" * 80)
        logging.info(f"KNN节点使用频率统计 (Task {self._cur_task})")
        logging.info("=" * 80)
        
        total_nodes = len(usage_counts)
        logging.info(f"\n总体统计 (总节点数: {total_nodes}):")
        for min_val, max_val, label in bins:
            if max_val == float('inf'):
                count = np.sum(usage_counts >= min_val)
            else:
                count = np.sum((usage_counts >= min_val) & (usage_counts <= max_val))
            percentage = (count / total_nodes * 100) if total_nodes > 0 else 0.0
            logging.info(f"  {label:12s}: {count:6d} 个节点 ({percentage:5.2f}%)")
        
        for cls in available_classes:
            cls_mask = labels == cls
            cls_usage = usage_counts[cls_mask]
            cls_total = np.sum(cls_mask)
            
            if cls_total == 0:
                continue
            
            for min_val, max_val, label in bins:
                if max_val == float('inf'):
                    count = np.sum(cls_usage >= min_val)
                else:
                    count = np.sum((cls_usage >= min_val) & (cls_usage <= max_val))
                percentage = (count / cls_total * 100) if cls_total > 0 else 0.0
                logging.info(f"    {label:12s}: {count:6d} 个节点 ({percentage:5.2f}%)")
            
            if len(cls_usage) > 0:
                logging.info(f"    平均使用次数: {cls_usage.mean():.2f}")
                logging.info(f"    最大使用次数: {cls_usage.max()}")
                logging.info(f"    最小使用次数: {cls_usage.min()}")
                logging.info(f"    中位数使用次数: {np.median(cls_usage):.2f}")
        
        logging.info(f"\n总体详细统计:")
        logging.info(f"  平均使用次数: {usage_counts.mean():.2f}")
        logging.info(f"  最大使用次数: {usage_counts.max()}")
        logging.info(f"  最小使用次数: {usage_counts.min()}")
        logging.info(f"  中位数使用次数: {np.median(usage_counts):.2f}")
        logging.info(f"  标准差: {usage_counts.std():.2f}")
        
        logging.info("=" * 80)

