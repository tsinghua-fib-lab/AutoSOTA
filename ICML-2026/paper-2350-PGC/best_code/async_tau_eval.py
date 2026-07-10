#!/usr/bin/env python3
"""Asymmetric tau evaluation: sweep tau_rgb and tau_res independently.

Tests different combinations of RGB and residual stream temperatures
in a single inference pass. The RGB stream captures semantic features
while the residual stream captures pixel-level artifacts — they may
benefit from different peak aggregation temperatures.
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


# Asymmetric tau grid: tau_rgb × tau_res pairs.
ASYNC_TAU_GRID = [
    (0.15, 0.15),
    (0.15, 0.20),
    (0.15, 0.25),
    (0.18, 0.18),
    (0.18, 0.20),
    (0.18, 0.22),
    (0.20, 0.15),
    (0.20, 0.18),
    (0.20, 0.20),   # symmetric baseline
    (0.20, 0.22),
    (0.20, 0.25),
    (0.22, 0.20),
    (0.25, 0.20),
    (0.25, 0.25),
]


def peak_aggregation(scores, tau, log_n):
    if tau <= 0.0:
        raise ValueError(f"tau must be > 0, got {tau}")
    z = tau * (torch.logsumexp(scores / tau, dim=1) - log_n)
    return z.unsqueeze(1)


def compute_z_local(s_rgb, s_res, lambda_rgb, tau_rgb, tau_res, log_n_rgb, log_n_res):
    z_rgb = peak_aggregation(s_rgb, tau_rgb, log_n_rgb)
    z_res = peak_aggregation(s_res, tau_res, log_n_res)
    z_local = z_res + lambda_rgb * z_rgb
    return z_local


def build_parser():
    p = argparse.ArgumentParser(description="Asymmetric tau evaluation")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--test_root", type=str, required=True)
    p.add_argument("--name", type=str, default="async_tau")
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
    return p


def main():
    args = build_parser().parse_args()
    devices, device = parse_devices(args.devices)

    logger_inst, log_path = setup_logging(args.checkpoints_dir, args.name, log_type='test')
    logger_inst.info("Asymmetric tau evaluation — log: %s", log_path)
    logger_inst.info("Testing %d (tau_rgb, tau_res) pairs", len(ASYNC_TAU_GRID))

    transform = create_eval_transforms(
        image_size=args.cropSize, mean=MEAN['dino'], std=STD['dino'],
    )
    dataset = UniversalFakeDetectDataset(args.test_root, transform=transform)

    model = PGCNetwork(
        dino_variant=args.dino_variant,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
        pretrained_root=args.dino_pretrained_root,
        tau_rgb=0.5, tau_res=0.5,
    )
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state['model'])
    model.to(device).eval()

    # Per-(tau_rgb, tau_res) accumulators.
    tau_key = lambda tr, ts: f"rgb{tr:.2f}_res{ts:.2f}"
    all_metrics: dict[str, dict] = {}

    for subset_name in dataset.get_subset_names():
        subset_indices = dataset.get_subset_indices(subset_name)
        subset_dataset = Subset(dataset, subset_indices)
        subset_loader = DataLoader(
            subset_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )

        per_pair_labels: dict[str, list] = {}
        per_pair_scores: dict[str, list] = {}
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

                f_rgb_cls, f_rgb_tokens = model.rgb_stream(rgb)
                f_residual_map = model.residual_stream(residual)
                f_res_pooled = torch.flatten(
                    torch.nn.functional.adaptive_avg_pool2d(f_residual_map, 1), 1
                )

                s_rgb = model.pgcm.rgb_score_head(f_rgb_tokens).squeeze(-1)
                s_res = model.pgcm.residual_score_head(f_residual_map).flatten(1)

                if log_n_rgb is None:
                    log_n_rgb = s_rgb.new_tensor(float(s_rgb.size(1))).log()
                    log_n_res = s_res.new_tensor(float(s_res.size(1))).log()

                lambda_rgb = model.pgcm.lambda_rgb

                f_global = torch.cat([f_rgb_cls, f_res_pooled], dim=1)
                f_global_normed = torch.nn.functional.normalize(f_global, p=2, dim=1)
                z_global = model.classifier(f_global_normed)

                for tau_rgb, tau_res in ASYNC_TAU_GRID:
                    key = tau_key(tau_rgb, tau_res)
                    z_local = compute_z_local(
                        s_rgb, s_res, lambda_rgb, tau_rgb, tau_res,
                        log_n_rgb, log_n_res,
                    )
                    y_pred = z_global + z_local
                    probs = torch.sigmoid(y_pred.view(-1))

                    if key not in per_pair_labels:
                        per_pair_labels[key] = []
                        per_pair_scores[key] = []
                    per_pair_labels[key].extend(labels.cpu().numpy())
                    per_pair_scores[key].extend(probs.cpu().numpy())

        for tau_rgb, tau_res in ASYNC_TAU_GRID:
            key = tau_key(tau_rgb, tau_res)
            y_true = np.array(per_pair_labels[key])
            y_score = np.array(per_pair_scores[key])
            metrics = compute_all_metrics(y_true, y_score)
            if key not in all_metrics:
                all_metrics[key] = {}
            all_metrics[key][subset_name] = metrics

    # Summary.
    logger_inst.info("=" * 80)
    logger_inst.info("ASYMMETRIC TAU RESULTS (sorted by ACC)")
    logger_inst.info("=" * 80)
    results = []
    for tau_rgb, tau_res in ASYNC_TAU_GRID:
        key = tau_key(tau_rgb, tau_res)
        mean_m = compute_mean_metrics(all_metrics[key])
        results.append((tau_rgb, tau_res, mean_m))
    results.sort(key=lambda x: x[2].get('acc', 0), reverse=True)

    for tau_rgb, tau_res, mean_m in results:
        logger_inst.info(
            "rgb=%.2f res=%.2f | ACC: %.4f | AP: %.4f | AUC: %.4f",
            tau_rgb, tau_res,
            mean_m.get('acc', 0), mean_m.get('ap', 0), mean_m.get('auc', 0),
        )

    best = results[0]
    logger_inst.info("Best: rgb=%.2f res=%.2f (ACC=%.4f)", best[0], best[1], best[2].get('acc', 0))

    # Per-subset for best.
    best_key = tau_key(best[0], best[1])
    logger_inst.info("--- Per-subset for best (rgb=%.2f, res=%.2f) ---", best[0], best[1])
    for sn in sorted(all_metrics[best_key].keys()):
        log_metrics(all_metrics[best_key][sn], f"  {sn}")

    logger_inst.info("Asymmetric tau evaluation completed.")


if __name__ == '__main__':
    main()
