import time
from contextlib import redirect_stdout
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from info_nce import InfoNCE
from torch.utils.data import DataLoader, TensorDataset

from utils.client import Client
from utils.utils import cls_acc_2


def apply_local_differential_privacy_per_row(
    features: torch.Tensor, clip_C: float = 1.0, sigma: float = 0.1
) -> torch.Tensor:
    if features.numel() == 0:
        return features
    norms = torch.norm(features, p=2, dim=1, keepdim=True) + 1e-10
    clip_coef = torch.clamp(clip_C / norms, max=1.0)
    clipped = features * clip_coef
    noise_std = sigma * clip_C
    noise = torch.randn_like(clipped) * noise_std
    return clipped + noise


def calculate_intra_class_cross_client_distance(global_prototypes_dict):
    class_distances = {}
    for label, prototype_list in global_prototypes_dict.items():
        num_clients_for_class = len(prototype_list)
        if num_clients_for_class < 2:
            continue
        stacked_protos = torch.stack(prototype_list).view(num_clients_for_class, -1)
        normalized_protos = F.normalize(stacked_protos, p=2, dim=1)
        sim_matrix = torch.mm(normalized_protos, normalized_protos.t())
        row_idx, col_idx = torch.triu_indices(num_clients_for_class, num_clients_for_class, offset=1)
        pairwise_similarities = sim_matrix[row_idx, col_idx]
        pairwise_distances = 1.0 - pairwise_similarities
        class_distances[label] = pairwise_distances.mean().item()

    if len(class_distances) > 0:
        overall_avg_distance = sum(class_distances.values()) / len(class_distances)
    else:
        overall_avg_distance = float("nan")
    return overall_avg_distance, class_distances


class Server:
    """FedSPA federated server: aggregates client visual prototypes and updates global semantic prototypes."""

    def __init__(
        self,
        args,
        text_features: torch.Tensor,
        clients: List[Client],
        output_file: str,
    ):
        self.clients = clients
        self.output_file = output_file
        self.text_features = text_features
        self.global_semantic_prototypes = text_features
        self.args = args
        self.device = text_features.device
        self.num_classes = self.text_features.shape[0]
        self.total_training_time = 0.0

    def compute_prototypes_from_clients(self):
        all_few_shot_features = []
        all_few_shot_labels = []

        for client in self.clients:
            if client.few_shot_image_features is not None:
                all_few_shot_features.append(client.few_shot_image_features)
                all_few_shot_labels.append(client.few_shot_labels)
                if getattr(self.args, "use_dp", False):
                    all_few_shot_features[-1] = apply_local_differential_privacy_per_row(
                        features=all_few_shot_features[-1],
                        clip_C=getattr(self.args, "dp_clip_C", 1.0),
                        sigma=getattr(self.args, "dp_sigma", 0.1),
                    )

        if not all_few_shot_features:
            return torch.empty(0, device=self.device), torch.empty(0, device=self.device)

        all_features = torch.cat(all_few_shot_features, dim=0).to(self.device)
        all_labels = torch.cat(all_few_shot_labels, dim=0).to(self.device)
        unique_labels = torch.unique(all_labels, sorted=True)
        num_classes = len(unique_labels)
        feature_dim = all_features.shape[-1]

        prototypes = torch.zeros((num_classes, feature_dim), device=self.device)
        prototypes_labels = torch.eye(num_classes, device=self.device)

        for i, label in enumerate(unique_labels):
            mask = all_labels == label
            prototypes[i] = all_features[mask].mean(dim=0)

        return prototypes, prototypes_labels

    def update_global_text_features(self, prototypes, labels):
        epochs = self.args.local_epochs_server
        tau = 0.01
        lam_reg = self.args.lam_reg

        dataset = TensorDataset(prototypes, labels)
        loader = DataLoader(
            dataset, batch_size=self.args.global_batch_size, shuffle=True, drop_last=False
        )

        test_adapter = nn.Parameter(self.global_semantic_prototypes.clone().detach().to(self.device))
        optimizer = torch.optim.AdamW([test_adapter], lr=self.args.global_lr, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(loader))
        criterion = InfoNCE(temperature=tau, reduction="mean")

        for ep in range(epochs):
            running_loss = 0.0
            for q, lbl in loader:
                q = q.to(self.device)
                lbl = lbl.to(self.device)
                k_pos = test_adapter[lbl]
                loss_infonce = criterion(q, k_pos)
                reg = -lam_reg * torch.sum(test_adapter * self.text_features) / test_adapter.size(0)
                loss = loss_infonce + reg
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                running_loss += loss.item()
            print(f"[Epoch {ep + 1}/{epochs}]  loss = {running_loss / len(loader):.4f}")

        test_adapter.data = F.normalize(test_adapter.data, dim=1)
        self.global_semantic_prototypes = test_adapter.clone().detach().to(self.device)

    def run(self):
        global_prototypes, global_prototypes_labels = self.compute_prototypes_from_clients()
        self.total_training_time = 0.0

        client_mean_cosine_values = None
        original_local_epochs = self.args.local_epochs

        all_cosine_values = []
        min_cosine_per_iteration = []
        global_acc_history = []

        for i in range(self.args.global_epochs):
            if i == (self.args.global_epochs - 1):
                if hasattr(self.args, "local_epochs_last") and self.args.local_epochs_last > 0:
                    self.args.local_epochs = self.args.local_epochs_last
                else:
                    self.args.local_epochs = original_local_epochs
            else:
                self.args.local_epochs = original_local_epochs

            global_prototypes_dict = {}
            client_prototypes_list = []
            low_cosine_clients = None
            clients_local_prototypes = None
            clients_local_prototypes_labels = None
            clients_acc = []

            if self.args.cosine_compute is True:
                epsilon = 0.5
                if client_mean_cosine_values is not None:
                    low_cosine_clients = set()
                    for item in client_mean_cosine_values:
                        client_id, cos_value = item
                        if cos_value < epsilon:
                            low_cosine_clients.add(client_id)

            for client in self.clients:
                if i == 0:
                    local_prototypes, local_prototypes_labels, acc = client.update_prototypes(
                        global_prototypes,
                        global_prototypes_labels,
                        self.global_semantic_prototypes,
                    )
                else:
                    if self.args.cosine_compute is True and low_cosine_clients is not None:
                        if client.client_id in low_cosine_clients:
                            local_prototypes, local_prototypes_labels, acc = client.update_prototypes(
                                None, None, self.text_features
                            )
                        else:
                            local_prototypes, local_prototypes_labels, acc = client.update_prototypes(
                                None, None, self.global_semantic_prototypes
                            )
                    else:
                        local_prototypes, local_prototypes_labels, acc = client.update_prototypes(
                            None, None, self.global_semantic_prototypes
                        )

                if self.args.cosine_compute is True:
                    client_prototypes = {}
                    for label, prototype in zip(local_prototypes_labels, local_prototypes):
                        client_prototypes[label.item()] = prototype
                        if label.item() not in global_prototypes_dict:
                            global_prototypes_dict[label.item()] = []
                        global_prototypes_dict[label.item()].append(prototype.clone())
                    client_prototypes_list.append(client_prototypes)

                print(f"Train local prototypes of client {client.client_id}")

                if clients_local_prototypes is None:
                    clients_local_prototypes = local_prototypes
                    clients_local_prototypes_labels = local_prototypes_labels
                else:
                    clients_local_prototypes = torch.cat(
                        (clients_local_prototypes, local_prototypes), dim=0
                    )
                    clients_local_prototypes_labels = torch.cat(
                        (clients_local_prototypes_labels, local_prototypes_labels), dim=0
                    )

                clients_acc.append(acc)

            if self.args.cosine_compute is True:
                avg_distance, _ = calculate_intra_class_cross_client_distance(global_prototypes_dict)
                print(
                    f"-> Mean intra-class cross-client cosine distance: {avg_distance:.4f}"
                )

            global_acc = sum(clients_acc) / len(clients_acc)
            global_acc_history.append(global_acc)
            print(f"epoch {i + 1}, global accuracy = {global_acc:.2f}%")
            with open(self.output_file, "a", encoding="utf-8") as f, redirect_stdout(f):
                print(f"epoch {i + 1}, global accuracy = {global_acc:.2f}%")

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            self.update_global_text_features(
                clients_local_prototypes, clients_local_prototypes_labels
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            self.total_training_time += time.perf_counter() - start_time

        if len(global_acc_history) > 0:
            tail_k = min(10, len(global_acc_history))
            tail_acc = global_acc_history[-tail_k:]
            tail_acc_mean = float(np.mean(tail_acc))
            tail_acc_std = float(np.std(tail_acc, ddof=0))
            print(f"Last {tail_k} rounds: mean global_acc = {tail_acc_mean:.2f}%")
            print(f"Last {tail_k} rounds: global_acc std (percentage points) = {tail_acc_std:.4f}")
            with open(self.output_file, "a", encoding="utf-8") as f, redirect_stdout(f):
                print(f"Last {tail_k} rounds: mean global_acc = {tail_acc_mean:.2f}%")
                print(f"Last {tail_k} rounds: global_acc std (percentage points) = {tail_acc_std:.4f}")

        if self.args.cosine_compute is True:
            with open(self.output_file, "a", encoding="utf-8") as f, redirect_stdout(f):
                print(
                    f"-> Mean intra-class cross-client cosine distance: {avg_distance:.4f}"
                )

        global_correct = 0
        global_total = 0
        for client in self.clients:
            clip_logits = 100.0 * client.image_features @ self.global_semantic_prototypes.t()
            correct, total = cls_acc_2(clip_logits, client.gts)
            global_correct += correct
            global_total += total

        global_acc = global_correct / global_total * 100
        print(f"Global accuracy based on text features: {global_acc:.2f}%")
        with open(self.output_file, "a", encoding="utf-8") as f, redirect_stdout(f):
            print(f"Global accuracy based on text features: {global_acc:.2f}%")

        print("\n" + "=" * 40)
        print("Training Time Statistics:")
        print("=" * 40)
        total_all_clients = 0.0
        for client in self.clients:
            client_time = client.total_training_time
            total_all_clients += client_time
            print(f"Client {client.client_id}: {client_time:.4f} seconds")
        avg_time = total_all_clients / len(self.clients) if len(self.clients) > 0 else 0.0
        print(f"Average training time: {avg_time:.4f} seconds")
        print(f"Server text feature update time: {self.total_training_time:.4f} seconds")
        print("=" * 40 + "\n")

        with open(self.output_file, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 40 + "\n")
            f.write("Training Time Statistics:\n")
            f.write("=" * 40 + "\n")
            for client in self.clients:
                f.write(f"Client {client.client_id}: {client.total_training_time:.4f} seconds\n")
            f.write(f"Average training time: {avg_time:.4f} seconds\n")
            f.write(f"Server text feature update time: {self.total_training_time:.4f} seconds\n")
            f.write("=" * 40 + "\n\n")
