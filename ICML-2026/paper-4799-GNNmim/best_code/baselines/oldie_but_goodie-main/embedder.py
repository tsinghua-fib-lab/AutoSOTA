import numpy as np
import os.path as osp
import torch
from torch_geometric.data import NeighborSampler
from torch_geometric.utils.loop import remove_self_loops, add_self_loops, add_remaining_self_loops
from torch_geometric.utils import is_undirected, to_undirected
from torch_geometric.utils import to_dense_adj

from utils import get_missing_feature_mask
from data_utils import get_dataset, set_train_val_test_split
from filling_strategies import filling
from collections import Counter

class embedder:
    def __init__(self, args, seed):
        # args.device = torch.device(
        #     f"cuda:{args.gpu}"
        #     if torch.cuda.is_available() and not (args.dataset == "OGBN-Products" and args.model == "lp")
        #     else "cpu"
        # )
        args.device = torch.device("cpu")
        dataset, evaluator = get_dataset(name=args.dataset, homophily=args.homophily)

        split_idx = dataset.get_idx_split() if hasattr(dataset, "get_idx_split") else None
        n_nodes, n_features = dataset.data.x.shape        
        num_classes = dataset.num_classes

        data = set_train_val_test_split(
            seed=seed, data=dataset.data, split_idx=split_idx, dataset=args.dataset,
        )

        missing_feature_mask = get_missing_feature_mask(
            rate=args.missing_rate, n_nodes=n_nodes, n_features=n_features, type=args.mask_type,
        )

        x = data.x.clone()
        x[~missing_feature_mask] = float("nan")

        # Filling
        if args.embedder in ['MLP', 'GNN']:
            filled_features = filling(args.filling_method, data.edge_index, x, missing_feature_mask, args.hop, args.replace, args.num_iterations, args.normalize_feature)
        else:
            filled_features = torch.full_like(x, float("nan"))
        
        x = torch.where(missing_feature_mask, x, filled_features)
        
        self.x = x.to(args.device)
        # self.adj = adj
        self.edge_index = data.edge_index.to(args.device)
        self.evaluator = evaluator

        self.train_mask = data.train_mask.to(args.device)
        self.val_mask = data.val_mask.to(args.device)
        self.test_mask = data.test_mask.to(args.device)
        self.labels = data.y.to(args.device)
        self.missing_feature_mask = missing_feature_mask.to(args.device)

        args.n_nodes = n_nodes
        args.n_feat = n_features
        args.n_hid = args.hidden_dim
        args.n_class = num_classes
        args.n_layer = args.num_layers
        args.n_head = args.num_heads

        self.args = args