import numpy as np
import os.path as osp
import torch

from filling_strategies import filling
from collections import Counter

class embedder:
    def __init__(self, args):
        # args.device = torch.device(
        #     f"cuda:{args.gpu}"
        #     if torch.cuda.is_available() and not (args.dataset == "OGBN-Products" and args.model == "lp")
        #     else "cpu"
        # )
        # args.device = torch.device("cpu")
        # dataset, evaluator = get_dataset(name=args.dataset, homophily=args.homophily)

        # split_idx = dataset.get_idx_split() if hasattr(dataset, "get_idx_split") else None
        # n_nodes, n_features = dataset.data.x.shape        
        # num_classes = dataset.num_classes

        # data = set_train_val_test_split(
        #     seed=seed, data=dataset.data, split_idx=split_idx, dataset=args.dataset,
        # )

        # missing_feature_mask = get_missing_feature_mask(
        #     rate=args.missing_rate, n_nodes=n_nodes, n_features=n_features, type=args.mask_type,
        # )
        data = args.data
        x = data.masks[args.seed]['X_incomp']
        train_mis = 100 * torch.isnan(x[args.train_mask]).sum().item() / x[args.train_mask].numel()
        val_mis = 100 * torch.isnan(x[args.val_mask]).sum().item() / x[args.val_mask].numel()
        test_mis = 100 * torch.isnan(x[args.test_mask]).sum().item() / x[args.test_mask].numel()
        # print(f"Train missing: {train_mis:.2f}% val missing: {val_mis:.2f}, test_missing: {test_mis:.2f}")
        missing_feature_mask = ~data.masks[args.seed]['mask'].bool()
        
        x[~missing_feature_mask] = float("nan")
        # print(x)
        # x_mis = 100 * torch.isnan(x).sum().item() / x.numel()
        # print(f"Total: {x_mis:.2f}")
        # Filling
        if args.embedder in ['MLP', 'GNN']:
            filled_features = filling(args.filling_method, data.edge_index, x, missing_feature_mask, args.hop, args.replace, args.num_iterations, args.normalize_feature)
        else:
            filled_features = torch.full_like(x, float("nan"))
    
        x = torch.where(missing_feature_mask.to('cpu'), x.to('cpu'), filled_features.to('cpu'))

        self.x = x.to(args.device)
        # self.adj = adj
        self.edge_index = data.edge_index.to(args.device)

        self.train_mask = args.train_mask.to(args.device)
        self.val_mask = args.val_mask.to(args.device)
        self.test_mask = args.test_mask.to(args.device)
        self.labels = data.y.to(args.device)
        self.missing_feature_mask = missing_feature_mask.to(args.device)

        args.n_nodes = args.n_nodes
        args.n_feat = args.n_feat
        args.n_hid = args.n_hid
        args.n_class = args.n_class
        args.n_layer = args.num_layers
        args.n_head = args.num_heads
        self.x_original = args.x_original

        self.args = args