#!/usr/bin/env python3
"""Checkpoint ensemble evaluation.

Averages predictions from two PGC checkpoints:
- ProGAN-only: PGC_train_progan_ckpt.pth
- ProGAN+SDv1.4: PGC_train_progan_sdv1_4_ckpt.pth
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


def build_model(ckpt_path, args, device):
    model = PGCNetwork(
        dino_variant=args.dino_variant,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
        pretrained_root=args.dino_pretrained_root,
        tau_rgb=args.tau_rgb,
        tau_res=args.tau_res,
    )
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['model'])
    model.to(device).eval()
    return model


def build_parser():
    p = argparse.ArgumentParser(description="Checkpoint ensemble evaluation")
    p.add_argument("--ckpt_a", type=str, required=True, help="First checkpoint")
    p.add_argument("--ckpt_b", type=str, required=True, help="Second checkpoint")
    p.add_argument("--test_root", type=str, required=True)
    p.add_argument("--name", type=str, default="ensemble")
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
    p.add_argument("--tau_rgb", type=float, default=0.5)
    p.add_argument("--tau_res", type=float, default=0.5)
    return p


def main():
    args = build_parser().parse_args()
    devices, device = parse_devices(args.devices)

    logger_inst, log_path = setup_logging(args.checkpoints_dir, args.name, log_type='test')
    logger_inst.info("Checkpoint ensemble evaluation — log: %s", log_path)
    logger_inst.info("Model A: %s", args.ckpt_a)
    logger_inst.info("Model B: %s", args.ckpt_b)

    # Build dataset.
    transform = create_eval_transforms(
        image_size=args.cropSize, mean=MEAN['dino'], std=STD['dino'],
    )
    dataset = UniversalFakeDetectDataset(args.test_root, transform=transform)

    # Build both models.
    logger_inst.info("Loading model A...")
    model_a = build_model(args.ckpt_a, args, device)
    logger_inst.info("Loading model B...")
    model_b = build_model(args.ckpt_b, args, device)

    # Evaluate each model separately AND ensemble.
    for model, label in [(model_a, "ckpt_a (ProGAN)"), (model_b, "ckpt_b (ProGAN+SDv1.4)")]:
        logger_inst.info("=" * 60)
        logger_inst.info("Evaluating: %s", label)
        subset_metrics = {}
        for subset_name in dataset.get_subset_names():
            subset_indices = dataset.get_subset_indices(subset_name)
            subset_dataset = Subset(dataset, subset_indices)
            subset_loader = DataLoader(
                subset_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True,
            )
            labels_list, scores_list = [], []
            with torch.no_grad():
                for batch in subset_loader:
                    if len(batch) == 3:
                        images, lbls, _ = batch
                    else:
                        images, lbls = batch
                    images = images.to(device)
                    lbls = lbls.to(device)
                    _feat, logits = model(images, return_feature=True)
                    probs = torch.sigmoid(logits.view(-1))
                    labels_list.extend(lbls.cpu().numpy())
                    scores_list.extend(probs.cpu().numpy())
            m = compute_all_metrics(np.array(labels_list), np.array(scores_list))
            subset_metrics[subset_name] = m
            log_metrics(m, f"  [{label}] {subset_name}")
        mean_m = compute_mean_metrics(subset_metrics)
        log_metrics(mean_m, f"[{label}] OVERALL")

    # Ensemble: average predictions.
    logger_inst.info("=" * 60)
    logger_inst.info("ENSEMBLE: averaging predictions from both checkpoints")
    subset_metrics_ens = {}
    for subset_name in dataset.get_subset_names():
        subset_indices = dataset.get_subset_indices(subset_name)
        subset_dataset = Subset(dataset, subset_indices)
        subset_loader = DataLoader(
            subset_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )
        labels_list, scores_list = [], []
        with torch.no_grad():
            for batch in subset_loader:
                if len(batch) == 3:
                    images, lbls, _ = batch
                else:
                    images, lbls = batch
                images = images.to(device)
                lbls = lbls.to(device)

                _feat_a, logits_a = model_a(images, return_feature=True)
                _feat_b, logits_b = model_b(images, return_feature=True)
                logits_avg = (logits_a + logits_b) / 2.0
                probs = torch.sigmoid(logits_avg.view(-1))

                labels_list.extend(lbls.cpu().numpy())
                scores_list.extend(probs.cpu().numpy())

        m = compute_all_metrics(np.array(labels_list), np.array(scores_list))
        subset_metrics_ens[subset_name] = m
        log_metrics(m, f"  [ENSEMBLE] {subset_name}")

    mean_ens = compute_mean_metrics(subset_metrics_ens)
    log_metrics(mean_ens, "[ENSEMBLE] OVERALL")
    logger_inst.info("Ensemble evaluation completed.")


if __name__ == '__main__':
    main()
