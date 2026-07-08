from utils import test_eval
from construct_P_features import construct_features
from model import PromptGADModel
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

class Detector:
    def __init__(self, args):
        self.args = args
        d_model = getattr(args, 'd_model', 64)
        nhead = getattr(args, 'nhead', 4)
        num_layers = getattr(args, 'num_layers', 2)
        dim_feedforward = getattr(args, 'dim_feedforward', 128)
        k_shot = getattr(args, 'k', 10)

        self.model = PromptGADModel(
            input_feature_dim=5,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            k_shot=k_shot
        ).to(args.device)

        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=args.lr,
                                    weight_decay=args.weight_decay)

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4.0]).to(args.device))

    def get_intra_dataset_batch(self, features, labels, k, batch_size):
        norm_indices = (labels == 0).nonzero().squeeze(1)
        ano_indices = (labels == 1).nonzero().squeeze(1)
        device = features.device

        perm_norm = torch.randperm(len(norm_indices)).to(device)
        perm_ano = torch.randperm(len(ano_indices)).to(device)

        sup_norm_idx = norm_indices[perm_norm[:k]]

        if len(ano_indices) < k:
            sup_ano_idx = ano_indices[torch.randint(0, len(ano_indices), (k,)).to(device)]
        else:
            sup_ano_idx = ano_indices[perm_ano[:k]]

        support_norm = features[sup_norm_idx]
        support_ano = features[sup_ano_idx]

        mask = torch.ones(len(features), dtype=torch.bool).to(device)
        mask[sup_norm_idx] = False
        mask[sup_ano_idx] = False

        remain_indices = torch.arange(len(features)).to(device)[mask]
        remain_labels = labels[remain_indices]

        remain_norm_idx = remain_indices[remain_labels == 0]
        remain_ano_idx = remain_indices[remain_labels == 1]

        n_query_ano = min(50, len(remain_ano_idx))
        n_query_norm = min(500, len(remain_norm_idx))

        if n_query_ano > 0:
            batch_ano = remain_ano_idx[torch.randperm(len(remain_ano_idx))[:n_query_ano]]
        else:
            batch_ano = torch.tensor([], dtype=torch.long).to(device)

        if n_query_norm > 0:
            batch_norm = remain_norm_idx[torch.randperm(len(remain_norm_idx))[:n_query_norm]]
        else:
            batch_norm = torch.tensor([], dtype=torch.long).to(device)

        query_indices = torch.cat([batch_norm, batch_ano])

        if len(query_indices) == 0:
            return support_norm, support_ano, None, None

        perm_query = torch.randperm(len(query_indices)).to(device)
        query_indices = query_indices[perm_query]

        query_features = features[query_indices]
        query_labels = labels[query_indices].float().unsqueeze(1)

        return support_norm, support_ano, query_features, query_labels

    def train_mixed(self, train_datasets):
        print(f"\n=== Start Source Domains Training (Datasets: {len(train_datasets)}) ===")

        train_data_cache = []
        for dataset in train_datasets:
            feats, lbls = construct_features(self.args, dataset, mode='all')
            train_data_cache.append((dataset.name, feats, lbls))
            print(f"  [Preload] {dataset.name}: {feats.shape[0]} nodes")

        self.model.train()
        batches_per_dataset = 20

        for epoch in range(self.args.epoch):
            epoch_loss = 0.0
            total_batches = 0

            for d_name, d_feats, d_lbls in train_data_cache:
                for _ in range(batches_per_dataset):
                    sup_norm, sup_ano, q_feat, q_lbl = self.get_intra_dataset_batch(
                        d_feats, d_lbls, self.args.k, self.args.batch_size
                    )

                    if q_feat is None or q_feat.size(0) == 0:
                        continue

                    self.optimizer.zero_grad()

                    output_logits_po = self.model(sup_norm, sup_ano, q_feat)
                    output_logits_ne = self.model(sup_ano, sup_norm, q_feat)

                    loss_po = self.criterion(output_logits_po, q_lbl)
                    loss_ne = self.criterion(output_logits_ne, 1 - q_lbl)
                    loss = loss_po

                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    total_batches += 1

            if (epoch + 1) % (self.args.epoch / 5) == 0:
                avg_loss = epoch_loss / total_batches if total_batches > 0 else 0.0
                print(f"  Epoch [{epoch + 1}/{self.args.epoch}], Avg Loss: {avg_loss:.4f}")

        print("--- Source Domains Training Completed ---")

    def test_one_dataset(self, dataset_obj):
        d_name = dataset_obj.name
        print(f"\n--- [Testing Target Domain] {d_name} ---")
        self.model.eval()
        features, labels = construct_features(self.args, dataset_obj, mode='all')

        num_test_runs = 5
        final_probs = torch.zeros_like(labels, dtype=torch.float)

        with torch.no_grad():
            for run in range(num_test_runs):
                sup_norm, sup_ano, _, _ = self.get_intra_dataset_batch(features, labels, self.args.k, 0)

                if sup_norm is None:
                    continue

                batch_size = self.args.batch_size
                run_probs = []

                for i in range(0, len(features), batch_size):
                    batch_feat = features[i: i + batch_size]
                    logits = self.model(sup_norm, sup_ano, batch_feat)
                    run_probs.append(logits.squeeze())

                if len(run_probs) > 0:
                    run_probs = torch.cat(run_probs)
                    if run_probs.shape[0] != final_probs.shape[0]:
                        run_probs = run_probs[:final_probs.shape[0]]
                    final_probs += run_probs

            final_probs /= num_test_runs

        test_score = test_eval(labels, final_probs)
        print(f"  Result {d_name} - AUROC: {test_score['AUROC']:.4f}, AUPRC: {test_score['AUPRC']:.4f}")
        return test_score