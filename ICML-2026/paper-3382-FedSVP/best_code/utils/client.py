import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from utils.utils import cls_acc


class Client:
    """FedSPA federated client."""

    def __init__(
        self,
        client_id: int,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        gts: torch.Tensor,
        args,
        few_shot_image_features: torch.Tensor,
        few_shot_labels: torch.Tensor,
        device: Optional[torch.device] = None,
        output_file: Optional[str] = None,
    ):
        self.client_id = client_id
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.image_features = image_features.to(self.device)
        self.text_features = text_features.to(self.device)
        self.gts = gts.to(self.device)

        self.few_shot_image_features = few_shot_image_features.to(self.device)
        self.few_shot_labels = few_shot_labels.to(self.device)

        self.args = args
        self.output_file = output_file

        self.global_semantic_prototypes = None
        self.prototypes = None
        self.prototypes_labels = None

        self.adapter = None
        self.total_training_time = 0.0
        self.round_times = []

    def update_prototypes(self, prototypes=None, labels=None, global_semantic_prototypes=None):
        round_start_time = time.time()

        if prototypes is not None:
            self.prototypes = prototypes
        if labels is not None:
            self.prototypes_labels = labels
        if global_semantic_prototypes is not None:
            self.global_semantic_prototypes = global_semantic_prototypes

        if self.global_semantic_prototypes is None:
            clip_weights = self.text_features.t()
        else:
            clip_weights = self.global_semantic_prototypes.t()

        dataset = TensorDataset(self.few_shot_image_features, self.few_shot_labels)
        cache_keys = self.prototypes.t().to(device=self.device)
        cache_values = self.prototypes_labels.to(device=self.device)

        adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(self.device)
        adapter.weight = nn.Parameter(cache_keys.t())
        self.adapter = adapter
        beta, alpha = self.args.beta, self.args.alpha

        if len(dataset) > 0:
            batch_size = min(self.args.local_batch_size, len(dataset))
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            optimizer = torch.optim.AdamW(adapter.parameters(), lr=self.args.local_lr, eps=1e-4)
            # CODE-03: Linear warmup (10% of steps) + CosineAnnealingLR
            total_steps = self.args.local_epochs * len(train_loader)
            warmup_steps = max(1, int(0.1 * total_steps))
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=warmup_steps
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, total_steps - warmup_steps
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )

            for train_idx in range(self.args.local_epochs):
                print(f"Train Epoch: {train_idx} / {self.args.local_epochs}")
                for image_features, target in train_loader:
                    affinity = adapter(image_features)
                    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                    clip_logits = 100.0 * image_features @ clip_weights
                    tip_logits = clip_logits + cache_logits * alpha
                    loss = F.cross_entropy(tip_logits, target)

                    optimizer.zero_grad()
                    loss.backward()
                    # CODE-01: Gradient clipping for training stability
                    torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()

                adapter.eval()
                affinity = adapter(self.image_features)
                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                clip_logits = 100.0 * self.image_features @ clip_weights
                tip_logits = clip_logits + cache_logits * alpha
                acc = cls_acc(tip_logits, self.gts)
                print(f"**** test accuracy: {acc:.2f}. ****\n")
        else:
            affinity = adapter(self.image_features)
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            clip_logits = 100.0 * self.image_features @ clip_weights
            tip_logits = clip_logits + cache_logits * alpha
            acc = cls_acc(tip_logits, self.gts)
            print(f"**** test accuracy: {acc:.2f}. ****\n")

        self.prototypes = adapter.weight.detach().clone()
        labels = self.prototypes_labels.argmax(dim=1)

        round_time = time.time() - round_start_time
        self.total_training_time += round_time
        self.round_times.append(round_time)

        if self.output_file is not None:
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(f"[Client {self.client_id}] Round training time: {round_time:.4f} seconds\n")

        return self.prototypes.detach().clone(), labels, acc
