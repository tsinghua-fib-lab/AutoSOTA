#!/usr/bin/env python3
"""Multi-tau evaluation: sweeps PGCM temperatures in a single inference pass.

Instead of running N separate evaluations (one per tau configuration), this
script runs the expensive backbone once per batch, then applies multiple tau
values to the PGCM peak aggregation.  This reduces wall-clock time from
N × 11 min to roughly 11 min + epsilon.
"""

import argparse
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data.eval_dataset import UniversalFakeDetectDataset
from data.transforms import MEAN, STD, create_eval_transforms
from models.pgc import PGCNetwork
from utils.cli import DINO_CHOICES, parse_devices
from utils.logging_utils import setup_logging
from utils.metrics import compute_all_metrics, compute_mean_metrics, log_metrics


logger = logging.getLogger(__name__)

# Default tau sweep grid: (tau_rgb, tau_res) pairs.
DEFAULT_TAU_GRID = [
    (0.1, 0.1),
    (0.2, 0.2),
    (0.35, 0.35),
    (0.5, 0.5),   # baseline
    (0.75, 0.75),
    (1.0, 1.0),
    (1.5, 1.5),
    (2.0, 2.0),
]


def peak_aggregation(scores, tau, log_n):
    """PGCM peak aggregation (paper Eq. 3)."""
    if tau <= 0.0:
        raise ValueError(f"tau must be > 0, got {tau}")
    z = tau * (torch.logsumexp(scores / tau, dim=1) - log_n)
    return z.unsqueeze(1)  # [B] → [B, 1]


def compute_z_local(s_rgb, s_res, lambda_rgb, tau_rgb, tau_res, log_n_rgb, log_n_res):
    """Compute PGCM local calibration bias for a given tau pair."""
    z_rgb = peak_aggregation(s_rgb, tau_rgb, log_n_rgb)
    z_res = peak_aggregation(s_res, tau_res, log_n_res)
    z_local = z_res + lambda_rgb * z_rgb
    return z_local


def build_parser():
    p = argparse.ArgumentParser(description="Multi-tau PGCM evaluation")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--test_root", type=str, required=True)
    p.add_argument("--name", type=str, default="multi_tau")
    p.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--devices", type=str, default="0")
    p.add_argument("--cropSize", type=int, default=224)
    p.add_argument("--dino_variant", type=str, default="dinov2-large", choices=DINO_CHOICES)
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=1.0)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--dino_pretrained_root", type=str, default=None)
    # Single-pass vs. separate runs.
    p.add_argument("--tau_list", type=str, default="0.1,0.2,0.35,0.5,0.75,1.0,1.5,2.0",
                   help="Comma-separated tau values to sweep (applied to both RGB and residual).")
    return p


def main():
    args = build_parser().parse_args()
    devices, device = parse_devices(args.devices)

    logger_inst, log_path = setup_logging(args.checkpoints_dir, args.name, log_type='test')
    logger_inst.info("Multi-tau evaluation — log: %s", log_path)

    # Parse tau sweep.
    tau_values = [float(x.strip()) for x in args.tau_list.split(",")]
    logger_inst.info("Tau sweep: %s", tau_values)

    # Build dataset.
    transform = create_eval_transforms(
        image_size=args.cropSize, mean=MEAN['dino'], std=STD['dino'],
    )
    dataset = UniversalFakeDetectDataset(args.test_root, transform=transform)

    # Build model.
    model = PGCNetwork(
        dino_variant=args.dino_variant,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
        pretrained_root=args.dino_pretrained_root,
        tau_rgb=0.5, tau_res=0.5,  # placeholder; we bypass PGCM forward
    )
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state['model'])
    model.to(device).eval()

    # Per-tau accumulators.
    all_metrics: dict[float, dict] = {t: {} for t in tau_values}

    for subset_name in dataset.get_subset_names():
        subset_indices = dataset.get_subset_indices(subset_name)
        subset_dataset = Subset(dataset, subset_indices)
        subset_loader = DataLoader(
            subset_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )

        # Per-tau score + label accumulators for this subset.
        tau_labels: dict[float, list] = {t: [] for t in tau_values}
        tau_scores: dict[float, list] = {t: [] for t in tau_values}

        # Pre-compute log_n for RGB (patches) and residual (spatial map).
        # These are constant per model architecture, not per batch.
        log_n_rgb = None
        log_n_res = None

        with torch.no_grad():
            for batch in subset_loader:
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch
                images = images.to(device)
                labels = labels.to(device)

                rgb = images[:, :3, :, :]
                residual = images[:, 3:, :, :]

                # --- RGB stream ---
                f_rgb_cls, f_rgb_tokens = model.rgb_stream(rgb)  # [B,D],[B,N,D]

                # --- Residual stream ---
                f_residual_map = model.residual_stream(residual)  # [B,C,Hr,Wr]
                f_res_pooled = torch.flatten(
                    torch.nn.functional.adaptive_avg_pool2d(f_residual_map, 1), 1
                )  # [B, C]

                # --- PGCM raw scores (single pass) ---
                s_rgb = model.pgcm.rgb_score_head(f_rgb_tokens).squeeze(-1)  # [B, N]
                s_res = model.pgcm.residual_score_head(f_residual_map).flatten(1)  # [B, Hr*Wr]

                # Cache log_n values.
                if log_n_rgb is None:
                    log_n_rgb = s_rgb.new_tensor(float(s_rgb.size(1))).log()
                    log_n_res = s_res.new_tensor(float(s_res.size(1))).log()

                lambda_rgb = model.pgcm.lambda_rgb

                # --- Global feature + classifier ---
                f_global = torch.cat([f_rgb_cls, f_res_pooled], dim=1)
                f_global_normed = torch.nn.functional.normalize(f_global, p=2, dim=1)
                z_global = model.classifier(f_global_normed)  # [B, 1]

                # --- Evaluate each tau ---
                for tau in tau_values:
                    z_local = compute_z_local(
                        s_rgb, s_res, lambda_rgb, tau, tau,
                        log_n_rgb, log_n_res,
                    )
                    y_pred = z_global + z_local
                    logits = y_pred.view(-1)
                    probs = torch.sigmoid(logits)

                    tau_labels[tau].extend(labels.cpu().numpy())
                    tau_scores[tau].extend(probs.cpu().numpy())

        # Per-tau per-subset metrics.
        for tau in tau_values:
            y_true = np.array(tau_labels[tau])
            y_score = np.array(tau_scores[tau])
            metrics = compute_all_metrics(y_true, y_score)
            all_metrics[tau][subset_name] = metrics

    # --- Summary ---
    logger_inst.info("=" * 80)
    logger_inst.info("MULTI-TAU RESULTS")
    logger_inst.info("=" * 80)
    best_tau = None
    best_acc = -1.0
    for tau in tau_values:
        mean_m = compute_mean_metrics(all_metrics[tau])
        acc = mean_m.get('acc', 0)
        ap = mean_m.get('ap', 0)
        auc = mean_m.get('auc', 0)
        logger_inst.info(
            "tau=%.2f | ACC: %.4f | AP: %.4f | AUC: %.4f",
            tau, acc, ap, auc,
        )
        if acc > best_acc:
            best_acc = acc
            best_tau = tau

    logger_inst.info("Best tau: %.2f (ACC=%.4f)", best_tau, best_acc)

    # Per-subset breakdown for best tau.
    logger_inst.info("--- Per-subset breakdown for best tau=%.2f ---", best_tau)
    for subset_name in sorted(all_metrics[best_tau].keys()):
        m = all_metrics[best_tau][subset_name]
        log_metrics(m, f"  {subset_name}")

    logger_inst.info("Multi-tau evaluation completed.")


if __name__ == '__main__':
    main()
