"""
EnergyPG evaluation for DAVE (fast version with precomputed bboxes).
Based on protocol from Appendix B.2 of the DAVE paper.

Paper: DAVE 82.43% EnergyPG on DeiT-III-B-16 (Table 1, 50k images, 50 steps)
"""

import os, sys, json, argparse, pickle
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

import torch
from torchvision.datasets import ImageFolder

for k in list(os.environ.keys()):
    if "proxy" in k.lower() and k.lower() not in ("no_proxy",):
        os.environ.pop(k, None)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from core.explainer import DAVEExplainer
from core.utils.utils import set_seed, get_device


def build_argparser():
    p = argparse.ArgumentParser("EnergyPG evaluation for DAVE")
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--bbox-path", type=str, required=True,
                   help="Pickle with precomputed mapped bboxes {filename: [(xmin,ymin,xmax,ymax), ...]}")
    p.add_argument("--out-dir", type=str, default="evaluation/energypg_results")
    p.add_argument("--model-cfg-path", type=str, required=True)
    p.add_argument("--num-images", type=int, default=-1)
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--post-proc", action="store_true", default=True)
    p.add_argument("--no-post-proc", dest="post_proc", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--detach-layers", type=str, default=None,
                   help="Layer range for detach, e.g. 0-7 for layers 1-8")
    return p


def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device}")

    # Load precomputed bbox (already mapped to 224x224 crop)
    print(f"Loading mapped bbox from {args.bbox_path}...")
    with open(args.bbox_path, "rb") as f:
        bbox_data = pickle.load(f)
    print(f"Loaded {len(bbox_data)} mapped bbox entries")

    # Load DAVE
    print("Loading DAVE explainer...")
    explainer = DAVEExplainer(model_cfg_path=args.model_cfg_path, device=device)
    model_name = explainer.model_name
    print(f"Model: {model_name}")

    transform = explainer.input_transform
    dataset = ImageFolder(args.data_dir, transform=transform)
    print(f"Dataset: {len(dataset)} images")

    out_dir = Path(args.out_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare sample order
    samples = list(dataset.samples)
    if args.shuffle:
        import random
        random.Random(args.seed).shuffle(samples)

    n_imgs = args.num_images if args.num_images > 0 else len(samples)
    print(f"\n=== Computing EnergyPG ({n_imgs} images, {args.num_steps} steps) ===")

    scores = []
    num_dropped = 0
    num_processed = 0
    num_skipped = 0

    for idx, (img_path, target_cls) in enumerate(samples):
        if num_processed >= n_imgs:
            break

        fname = os.path.basename(img_path)
        if fname not in bbox_data:
            num_skipped += 1
            continue

        boxes = bbox_data[fname]

        x, _ = dataset[idx]
        x = x.unsqueeze(0).to(device)
        y = torch.tensor([target_cls], device=device)

        img_seed = args.seed * 10000 + num_processed
        detach_range = None
        if args.detach_layers is not None:
            parts = args.detach_layers.split('-')
            detach_range = (int(parts[0]), int(parts[1]))
        attr = explainer.explain(x=x, y=y, num_steps=args.num_steps, post_proc=args.post_proc, seed=img_seed, detach_layer_range=detach_range)

        if attr.shape[1] != 1:
            attr = attr.sum(dim=1, keepdim=True)

        pos_attr = torch.clamp(attr[0, 0], min=0)
        total_mass = pos_attr.sum().item()

        if total_mass <= 0:
            num_dropped += 1
            num_processed += 1
            continue

        H, W = pos_attr.shape
        bbox_mass = 0.0
        for (xmin, ymin, xmax, ymax) in boxes:
            bbox_mass += pos_attr[ymin:ymax, xmin:xmax].sum().item()

        score = bbox_mass / total_mass
        scores.append(score)
        num_processed += 1

        if num_processed % args.log_every == 0:
            avg = np.mean(scores) * 100 if scores else 0
            print(f"[{num_processed}/{n_imgs}] EnergyPG={avg:.2f}% (n={len(scores)}, dropped={num_dropped}, skipped={num_skipped})")

    energy_pg = np.mean(scores) * 100 if scores else 0.0
    energy_pg_std = np.std(scores) * 100 if scores else 0.0

    results = {
        "energypg_pct": float(energy_pg),
        "energypg_std_pct": float(energy_pg_std),
        "num_processed": num_processed,
        "num_valid_scores": len(scores),
        "num_dropped": num_dropped,
        "num_skipped": num_skipped,
        "dropped_pct": num_dropped / max(num_processed, 1) * 100,
    }

    summary = {**results, "args": vars(args), "model_name": model_name}
    with open(out_dir / "energypg_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== EnergyPG RESULTS ===")
    print(f"Model: {model_name}")
    print(f"EnergyPG: {results['energypg_pct']:.2f}%")
    print(f"EnergyPG std: {results['energypg_std_pct']:.2f}%")
    print(f"Images: {num_processed}, Valid: {len(scores)}, Dropped: {num_dropped}, Skipped: {num_skipped}")
    print(f"Saved to: {out_dir}/energypg_summary.json")


if __name__ == "__main__":
    main()
