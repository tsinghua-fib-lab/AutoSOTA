from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler


class ProtoClassifier:
    def __init__(self, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.prototypes = None
        self.classes = None

    def fit(self, x_support: torch.Tensor, y_support: torch.Tensor) -> None:
        x_support = x_support.to(self.device)
        y_support = y_support.to(self.device)
        self.classes = torch.unique(y_support)
        self.prototypes = torch.stack([x_support[y_support == c].mean(dim=0) for c in self.classes])

    def predict(self, x_query: torch.Tensor) -> torch.Tensor:
        dists = torch.cdist(x_query.to(self.device), self.prototypes, p=2)
        return self.classes[dists.argmin(dim=1)]


def create_fewshot_split(y: np.ndarray, k_shot: int, n_query: int, seed: int):
    rng = np.random.default_rng(seed)
    support_indices = []
    query_indices = []
    for cls in np.unique(y):
        cls_indices = np.where(y == cls)[0]
        rng.shuffle(cls_indices)
        if len(cls_indices) < k_shot:
            cut = max(1, len(cls_indices) - 1)
            support = cls_indices[:cut]
            query_source = cls_indices[cut:]
        else:
            support = cls_indices[:k_shot]
            query_source = cls_indices[k_shot:]
        query = query_source if n_query == -1 else query_source[:n_query]
        support_indices.extend(support.tolist())
        query_indices.extend(query.tolist())
    return support_indices, query_indices


def evaluate_fewshot(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    k_shot: int,
    n_query: int,
    n_runs: int,
    seed: int,
    device: str | torch.device,
    pca_components: int = 0,
    use_adapter: bool = False,
    adapter_dim: int = 64,
) -> tuple[dict, list[dict]]:
    x_np = embeddings.numpy()
    y_np = labels.numpy()
    rows = []
    accuracies = []
    balanced_accuracies = []
    for run in range(n_runs):
        support_idx, query_idx = create_fewshot_split(y_np, k_shot, n_query, seed + run)
        if not query_idx:
            continue

        scaler = StandardScaler()
        x_support_np = scaler.fit_transform(x_np[support_idx])
        x_query_np = scaler.transform(x_np[query_idx])

        # Apply PCA to reduce dimensionality if requested
        if pca_components > 0:
            n_comp = min(pca_components, len(support_idx) - 1)
            pca = TruncatedSVD(n_components=n_comp, random_state=seed + run)
            x_support_np = pca.fit_transform(x_support_np)
            x_query_np = pca.transform(x_query_np)

        x_support = torch.from_numpy(x_support_np).float()
        x_query = torch.from_numpy(x_query_np).float()
        y_support = torch.tensor(y_np[support_idx]).long()
        y_query = torch.tensor(y_np[query_idx]).long()

        # Train linear adapter on support set (identity-initialized projection)
        if use_adapter:
            in_dim = x_support.shape[1]
            proj = nn.Linear(in_dim, in_dim, bias=False).to(device)
            nn.init.eye_(proj.weight)
            opt = optim.Adam(proj.parameters(), lr=0.001, weight_decay=0.001)
            x_support_dev = x_support.to(device)
            y_support_dev = y_support.to(device)

            for _ in range(20):
                opt.zero_grad()
                adapted = proj(x_support_dev)
                classes = torch.unique(y_support_dev)
                prototypes = torch.stack([adapted[y_support_dev == c].mean(dim=0) for c in classes])
                dists = torch.cdist(adapted, prototypes, p=2)
                logits = -dists / 0.1
                loss = nn.CrossEntropyLoss()(logits, torch.searchsorted(classes, y_support_dev))
                loss.backward()
                opt.step()

            proj.eval()
            with torch.no_grad():
                x_support = proj(x_support.to(device)).cpu()
                x_query = proj(x_query.to(device)).cpu()
            del proj, opt

        classifier = ProtoClassifier(device=device)
        classifier.fit(x_support, y_support)
        preds = classifier.predict(x_query)
        acc = (preds.cpu() == y_query).float().mean().item()
        bal_acc = balanced_accuracy_score(y_query.numpy(), preds.cpu().numpy())
        accuracies.append(acc)
        balanced_accuracies.append(bal_acc)
        rows.append(
            {
                "run": run + 1,
                "accuracy": acc,
                "balanced_accuracy": bal_acc,
                "query_size": len(query_idx),
            }
        )

    metrics = {
        "k_shot": k_shot,
        "n_query": n_query,
        "n_runs": len(rows),
        "accuracy_mean": float(np.mean(accuracies)) if accuracies else 0.0,
        "accuracy_std": float(np.std(accuracies)) if accuracies else 0.0,
        "balanced_accuracy_mean": float(np.mean(balanced_accuracies)) if balanced_accuracies else 0.0,
        "balanced_accuracy_std": float(np.std(balanced_accuracies)) if balanced_accuracies else 0.0,
    }
    return metrics, rows

