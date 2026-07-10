#!/usr/bin/env python3
"""Monte Carlo Dropout evaluation.

Runs N stochastic forward passes with LoRA dropout enabled at inference time
and averages predictions. Provides uncertainty-aware prediction averaging.
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


def build_parser():
    p = argparse.ArgumentParser(description="Monte Carlo Dropout evaluation")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--test_root", type=str, required=True)
    p.add_argument("--name", type=str, default="mc_dropout")
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
    p.add_argument("--mc_samples", type=int, default=5,
                   help="Number of MC dropout forward passes per image.")
    return p


def enable_dropout_in_lora(model):
    """Force all LoRA dropout layers into train mode (dropout active)."""
    from models.lora.lora import LoRALinear
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.dropout.train()
    logger.info("[MC Dropout] Enabled dropout in %d LoRA layers",
                sum(1 for m in model.modules() if hasattr(m, 'dropout') and isinstance(m, torch.nn.Module)))


def mc_forward(model, images, n_samples):
    """Run n_samples forward passes with dropout active, return mean logits."""
    all_logits = []
    for _ in range(n_samples):
        _feat, logits = model(images, return_feature=True)
        all_logits.append(logits)
    # Average logits across MC samples.
    stacked = torch.stack(all_logits, dim=0)  # [N, B, 1]
    mean_logits = stacked.mean(dim=0)  # [B, 1]
    return mean_logits


def main():
    args = build_parser().parse_args()
    devices, device = parse_devices(args.devices)

    logger_inst, log_path = setup_logging(args.checkpoints_dir, args.name, log_type='test')
    logger_inst.info("MC Dropout evaluation — log: %s", log_path)
    logger_inst.info("MC samples: %d", args.mc_samples)

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
        tau_rgb=args.tau_rgb,
        tau_res=args.tau_res,
    )
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state['model'])
    model.to(device).eval()

    # Enable dropout for MC sampling.
    enable_dropout_in_lora(model)

    # Evaluate.
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

                logits = mc_forward(model, images, args.mc_samples)
                probs = torch.sigmoid(logits.view(-1))

                labels_list.extend(lbls.cpu().numpy())
                scores_list.extend(probs.cpu().numpy())

        m = compute_all_metrics(np.array(labels_list), np.array(scores_list))
        subset_metrics[subset_name] = m
        log_metrics(m, f"  [MC-{args.mc_samples}] {subset_name}")

    mean_m = compute_mean_metrics(subset_metrics)
    log_metrics(mean_m, f"[MC-{args.mc_samples}] OVERALL")
    logger_inst.info("MC Dropout evaluation completed.")


if __name__ == '__main__':
    main()
