"""
Generate paper-style training metric plots (target test acc, agreement ratio, pseudo-label acc).

This script runs PSAHS training while recording per-epoch metrics, then writes PDFs under
``figures/<dataset>/``. It mirrors the plotting workflow from the research codebase without
the noise-injection ablation used for robustness experiments.

Example::

    python scripts/plot_training_metrics.py -d Blog --src_name Blog1 --tgt_name Blog2 --seeds 1
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from main.args import apply_psahs_defaults, build_parser
from main.data_loader import load_domain_pair
import main.models as models
from psahs.data import datasets
from psahs.paths import checkpoint_suffix, figures_dir, output_dir
from psahs import training_utils as utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_pseudo_label_accuracy(tgt_data, output_dim, agreed_mask):
    if agreed_mask is None:
        return None
    valid = agreed_mask & (tgt_data.y < output_dim)
    n_agreed = int(valid.sum().item())
    if n_agreed == 0:
        return None
    correct = (tgt_data.y_hat[valid] == tgt_data.y[valid]).sum().item()
    return correct / n_agreed


def train_one_epoch_with_metrics(model, mlp_model, src_data, tgt_data, args, opt, scheduler, epoch):
    src_data = src_data.to(device)
    tgt_data = tgt_data.to(device)
    agreed_mask = None
    mlp_model.eval()
    with torch.no_grad():
        _, tgt_mlp_pred = mlp_model(tgt_data)
    _, mlp_pred_tgt = tgt_mlp_pred.max(dim=1)

    if args.reweight and epoch >= (args.start_epoch - 1):
        _, [pred_tgt, _] = model.forward(tgt_data, 1)
        _, pred_tgt = pred_tgt.max(dim=1)
        agreed_mask = pred_tgt == mlp_pred_tgt
        tgt_data.y_hat[agreed_mask] = mlp_pred_tgt[agreed_mask]
        if (epoch - 1) % args.rw_freq == 0:
            datasets.adjust_graph_structure_fast_target_Plabel(tgt_data, h_thresh=args.h_threshold)
            tgt_data = tgt_data.to(device)

    da_alpha = min((args.alphatimes * (epoch + 1) / args.epochs), args.alphamin)
    [_, _], [pred_src, pred_domain_src] = model.forward(src_data, da_alpha)
    [_, _], [_, pred_domain_tgt] = model.forward(tgt_data, da_alpha)

    mask_src = src_data.source_training_mask
    loss = utils.ce_loss(pred_src[mask_src], src_data.y[mask_src])
    loss += utils.bce_loss(pred_domain_src[src_data.source_mask], torch.zeros_like(pred_domain_src[src_data.source_mask]))
    loss += utils.bce_loss(pred_domain_tgt[tgt_data.target_mask], torch.ones_like(pred_domain_tgt[tgt_data.target_mask]))
    opt.zero_grad()
    loss.backward()
    opt.step()
    scheduler.step()
    return loss.item(), agreed_mask


def evaluate_tgt_test(source_dataset, target_dataset, model):
    model.eval()
    with torch.no_grad():
        target_dataset = target_dataset.to(device)
        [_, _], [pred, _] = model.forward(target_dataset, 1)
        mask = target_dataset.target_testing_mask
        acc, _ = utils.classification_scores(pred[mask], target_dataset.y[mask])
    return acc


def save_metrics_plot(fig_dir, stem, epochs, tgt_test, agreed_ratio, pseudo_acc):
    os.makedirs(fig_dir, exist_ok=True)
    best_idx = int(np.argmax(tgt_test))
    sl = slice(0, best_idx + 1)
    epochs = np.asarray(epochs[sl], dtype=float)
    y_test = np.asarray(tgt_test[sl], dtype=float)
    y_agreed = np.ma.masked_invalid(
        np.array([np.nan if x is None else float(x) for x in agreed_ratio[sl]], dtype=float)
    )
    y_pseudo = np.ma.masked_invalid(
        np.array([np.nan if x is None else float(x) for x in pseudo_acc[sl]], dtype=float)
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(epochs, y_test, label="target test acc", color="C2", linewidth=2.0)
    ax.plot(epochs, y_agreed, label="agreed ratio", color="C1", linewidth=2.0)
    ax.plot(epochs, y_pseudo, label="pseudo-label acc", color="C0", linewidth=2.0)
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Accuracy / Agreement Ratio", fontsize=14)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=11)
    out_path = os.path.join(fig_dir, f"{stem}_training_metrics_to_best.pdf")
    fig.savefig(out_path, format="pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_with_plotting(args, seed):
    path = output_dir(args)
    ck = checkpoint_suffix(args)
    source, target = load_domain_pair(args, adjust_source=True)
    input_dim = source.num_node_features
    output_dim = source.num_classes

    model = models.directed_GNN_adv(input_dim, output_dim, args).to(device)
    mlp_model = models.MLPWithMLPClassifier(input_dim, output_dim, args).to(device)
    mlp_ckpt = os.path.join(path, f"best_mlp_model_seed{seed}{ck}.pt")
    if not os.path.isfile(mlp_ckpt):
        raise FileNotFoundError(f"Missing MLP checkpoint: {mlp_ckpt}")
    mlp_model.load_state_dict(torch.load(mlp_ckpt, map_location=device))
    scheduler, opt = utils.build_optimizer(args, model.parameters())

    epochs_plot, tgt_test_list, agreed_list, pseudo_list = [], [], [], []
    best_valid = 0.0
    if args.reweight:
        target.rw_stats_history = []

    for epoch in range(args.epochs):
        model.train()
        _, agreed_mask = train_one_epoch_with_metrics(
            model, mlp_model, source, target, args, opt, scheduler, epoch
        )
        epochs_plot.append(epoch)
        tgt_test_list.append(evaluate_tgt_test(source, target, model))
        pseudo_list.append(compute_pseudo_label_accuracy(target, output_dim, agreed_mask))
        agreed_list.append(None if agreed_mask is None else agreed_mask.float().mean().item())

        with torch.no_grad():
            source_dev = source.to(device)
            target_dev = target.to(device)
            [_, _], [pred_s, _] = model.forward(source_dev, 1)
            [_, _], [pred_t, _] = model.forward(target_dev, 1)
            valid_s = utils.accuracy(pred_s[source_dev.source_validation_mask], source_dev.y[source_dev.source_validation_mask])
            valid_t = utils.accuracy(pred_t[target_dev.target_validation_mask], target_dev.y[target_dev.target_validation_mask])
        valid_score = valid_s if args.valid_data == "src" else valid_t
        if valid_score > best_valid:
            best_valid = valid_score
            torch.save(model.state_dict(), os.path.join(path, f"best_valid_model_{seed}{ck}.pt"))

    fig_dir = figures_dir(args)
    if args.dataset == "Noncircle":
        stem = f"metrics_{args.num_nodes}_seed{seed}"
    else:
        stem = f"metrics_{args.src_name}_{args.tgt_name}_seed{seed}"
    out_path = save_metrics_plot(fig_dir, stem, epochs_plot, tgt_test_list, agreed_list, pseudo_list)
    print(f"Saved figure: {out_path}")
    return out_path


def main():
    parser = build_parser("Train PSAHS and save training metric figures")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1],
        help="Seeds to plot (default: 1).",
    )
    args = apply_psahs_defaults(parser.parse_args())

    for seed in args.seeds:
        utils.set_seed(seed)
        run_with_plotting(args, seed)


if __name__ == "__main__":
    main()
