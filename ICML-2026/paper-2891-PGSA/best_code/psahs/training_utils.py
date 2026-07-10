"""Training utilities: optimization, metrics, logging, and visualization helpers."""
from __future__ import annotations

import os
import random
import sys
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn

RUN_LOG_BASENAME = "train.log"


def set_seed(seed: int) -> None:
    """Best-effort reproducibility for one run."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    use_det = getattr(torch, "use_deterministic_algorithms", None)
    if callable(use_det):
        try:
            use_det(True, warn_only=True)
        except TypeError:
            use_det(True)


def build_optimizer(args, params):
    weight_decay = args.weight_decay
    params = filter(lambda p: p.requires_grad, params)
    if args.opt == "adam":
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == "sgd":
        optimizer = optim.SGD(params, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == "rmsprop":
        optimizer = optim.RMSprop(params, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == "adagrad":
        optimizer = optim.Adagrad(params, lr=args.lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.opt}")

    if args.opt_scheduler == "none":
        return None, optimizer
    if args.opt_scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate
        )
    elif args.opt_scheduler == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[100, 300, 500, 700, 900],
            gamma=args.opt_decay_rate,
        )
    elif args.opt_scheduler == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.opt_restart
        )
    else:
        raise ValueError(f"Unknown scheduler: {args.opt_scheduler}")
    return scheduler, optimizer


class _TeeLogger:
    def __init__(self, dir_path, log_basename=RUN_LOG_BASENAME):
        self.terminal = sys.stdout
        self.log = open(os.path.join(dir_path, log_basename), "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        try:
            if self.terminal is not None and hasattr(self.terminal, "flush"):
                self.terminal.flush()
            if getattr(self, "log", None) is not None and not self.log.closed:
                self.log.flush()
        except Exception:
            pass

    def close(self):
        self.flush()
        try:
            if getattr(self, "log", None) is not None and not self.log.closed:
                self.log.close()
        except Exception:
            pass


@contextmanager
def redirect_stdout_logger(dir_path):
    saved = sys.stdout
    logger = _TeeLogger(dir_path)
    sys.stdout = logger
    try:
        yield
    finally:
        sys.stdout = saved
        logger.close()


def ce_loss(pred, label):
    return nn.CrossEntropyLoss()(pred, label.type(torch.int64))


def bce_loss(pred, label):
    return nn.BCELoss()(pred.view(-1), label.view(-1).type(torch.float32))


def accuracy(pred, label):
    pred_label = pred.argmax(dim=1)
    return accuracy_score(label.cpu().numpy(), pred_label.cpu().numpy())


def auc_score(pred, label):
    try:
        prob = F.softmax(pred, dim=1)
        y = label.cpu().numpy()
        if torch.unique(label).numel() > 2:
            return roc_auc_score(y, prob.cpu().numpy(), multi_class="ovr")
        return roc_auc_score(y, prob[:, 1].cpu().numpy())
    except ValueError:
        return None


def classification_scores(pred, label):
    return accuracy(pred, label), auc_score(pred, label)


def aggregate_reports(reports):
    keys = reports[0].keys()
    summary = {}
    for key in keys:
        values = [r[key] if r[key] else 0 for r in reports]
        summary[key] = f"{np.mean(values):.5f} +/- {np.std(values):.5f}"
    return summary


def plot_domain_scatter(source_x, source_y, target_x, target_y, title, out_dir, method="pca"):
    """Plot source/target node embeddings colored by class (PCA or t-SNE)."""
    os.makedirs(out_dir, exist_ok=True)
    source_x = source_x.detach().cpu()
    target_x = target_x.detach().cpu()
    source_y = source_y.cpu()
    target_y = target_y.cpu()

    if method == "pca":
        pca = PCA(n_components=2)
        pca.fit(source_x)
        source_2d = pca.transform(source_x)
        target_2d = pca.transform(target_x)
    else:
        all_x = np.concatenate((source_x.numpy(), target_x.numpy()), axis=0)
        tsne = TSNE(n_components=2, perplexity=40, max_iter=300, init="pca")
        all_2d = tsne.fit_transform(all_x)
        source_2d = all_2d[: len(source_x)]
        target_2d = all_2d[len(source_x) :]

    classes = torch.unique(source_y)
    fig, ax = plt.subplots(figsize=(6, 5))
    for cls in classes:
        cls = cls.item()
        ax.scatter(
            target_2d[target_y == cls, 0],
            target_2d[target_y == cls, 1],
            label=f"class {cls} (target)",
            alpha=0.7,
        )
        ax.scatter(
            source_2d[source_y == cls, 0],
            source_2d[source_y == cls, 1],
            label=f"class {cls} (source)",
            alpha=0.7,
        )
    ax.legend(fontsize=8)
    ax.set_title(title)
    path = os.path.join(out_dir, f"{title.replace(' ', '_')}.pdf")
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def plot_training_curves(epochs, series, labels, title, out_path):
    """Generic multi-series training curve plot."""
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for values, label in zip(series, labels):
        ax.plot(epochs, values, label=label, linewidth=2.0)
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Value", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize=11)
    ax.set_title(title)
    fig.savefig(out_path, format="pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
