"""
GridPG (Grid Pointing Game) evaluation for DAVE.
Based on protocol from Appendix B.2 of the DAVE paper.

Constructs 2x2 composite grid images (448x448) from 4 ImageNet validation 
images (224x224 each), each from a different class. Interpolates ViT position 
embeddings to handle the larger input. Computes DAVE attribution for each class
and measures the fraction of positive attribution mass in the target cell.

Paper: DAVE 65.76% GridPG on DeiT-III-B-16 (Table 1, 500 grids, 4000 images, 50 steps)
"""

import os, sys, json, argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.datasets import ImageFolder

# Clear proxy env vars for HF
for k in list(os.environ.keys()):
    if "proxy" in k.lower() and k.lower() not in ("no_proxy",):
        os.environ.pop(k, None)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from core.explainer import DAVEExplainer
from core.utils.utils import set_seed, get_device


def build_argparser():
    p = argparse.ArgumentParser("GridPG evaluation for DAVE")
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="evaluation/gridpg_results")
    p.add_argument("--model-cfg-path", type=str, required=True)
    p.add_argument("--num-grids", type=int, default=500)
    p.add_argument("--num-images", type=int, default=4000)
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--post-proc", action="store_true", default=True)
    p.add_argument("--no-post-proc", dest="post_proc", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--min-conf", type=float, default=0.90)
    return p


def interpolate_pos_embed(model, new_grid_size: Tuple[int, int]):
    """Interpolate position embeddings for larger input."""
    pos_embed = model.pos_embed  # (1, N, C)
    B, N, C = pos_embed.shape
    H = W = int(N ** 0.5)

    pos_2d = pos_embed.permute(0, 2, 1).reshape(1, C, H, W)
    pos_2d_new = F.interpolate(
        pos_2d, size=new_grid_size, mode="bicubic", align_corners=False
    )
    new_pos = pos_2d_new.reshape(1, C, -1).permute(0, 2, 1)
    model.pos_embed = torch.nn.Parameter(new_pos.to(
        device=pos_embed.device, dtype=pos_embed.dtype
    ))
    # Allow dynamic input size
    model.patch_embed.strict_img_size = False


def select_high_conf_images(
    explainer: DAVEExplainer, dataset: ImageFolder,
    device: torch.device, num_images: int,
    min_conf: float, max_per_class: int = 8,
) -> List[Tuple[int, int, float]]:
    """Select correctly classified high-confidence images."""
    model = explainer.model
    model.eval()

    class_counts = {}
    selected = []
    indices = torch.randperm(len(dataset)).tolist()

    with torch.no_grad():
        for idx in indices:
            if len(selected) >= num_images:
                break
            x, y = dataset[idx]
            x = x.unsqueeze(0).to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1).item()
            conf = probs.max(dim=1).values.item()

            if pred != y or conf < min_conf:
                continue

            cls = int(y)
            if class_counts.get(cls, 0) >= max_per_class:
                continue
            class_counts[cls] = class_counts.get(cls, 0) + 1
            selected.append((idx, cls, conf))

    print(f"Selected {len(selected)} images from {len(class_counts)} classes")
    return selected


def build_grids(
    selected: List[Tuple[int, int, float]],
    num_grids: int, seed: int,
) -> List[List[Tuple[int, int]]]:
    """Build 2x2 grids from selected images."""
    rng = np.random.RandomState(seed)
    by_class: Dict[int, List[int]] = {}
    for idx, cls, _ in selected:
        by_class.setdefault(cls, []).append(idx)

    available_classes = sorted(by_class.keys())
    grids = []
    class_usage = {c: 0 for c in available_classes}

    for _ in range(num_grids * 10):
        if len(grids) >= num_grids:
            break
        if len(available_classes) < 4:
            break
        chosen = list(rng.choice(available_classes, size=4, replace=False))
        grid = []
        valid = True
        for cls in chosen:
            imgs = by_class[cls]
            if not imgs:
                valid = False; break
            img_idx = imgs[class_usage[cls] % len(imgs)]
            class_usage[cls] += 1
            grid.append((img_idx, cls))
        if valid and len(grid) == 4:
            grids.append(grid)

    print(f"Built {len(grids)} grids")
    return grids


def make_grid_image(images: List[Tensor]) -> Tensor:
    """Combine 4 images (3,224,224) into a 2x2 grid (3,448,448)."""
    top = torch.cat([images[0], images[1]], dim=2)
    bottom = torch.cat([images[2], images[3]], dim=2)
    return torch.cat([top, bottom], dim=1)


def compute_gridpg(
    explainer: DAVEExplainer, dataset: ImageFolder,
    grids: List[List[Tuple[int, int]]],
    device: torch.device, num_steps: int, post_proc: bool,
) -> Dict:
    """
    Compute GridPG metric.
    
    For each grid (4 images, 4 classes):
      1. Create 448x448 composite image
      2. Interpolate pos embeddings for 28x28 grid
      3. For each class, compute DAVE attribution on the composite
      4. L_i = positive_attributions_in_cell_i / total_positive_attributions
      5. Drop samples with no positive attributions (standard practice)
    """
    all_cell_scores = []
    num_dropped = 0
    model = explainer.model

    for grid_idx, grid in enumerate(grids):
        # Get raw images (before DAVE transform) for compositing
        imgs_raw = []
        classes = []
        for ds_idx, cls in grid:
            # Get image WITHOUT the DAVE input_transform (raw PIL/normalized differently)
            x, _ = dataset[ds_idx]
            imgs_raw.append(x)
            classes.append(cls)

        # Create composite grid image (already normalized by dataset transform)
        grid_img = make_grid_image(imgs_raw).unsqueeze(0).to(device)  # (1, 3, 448, 448)

        # Interpolate pos embeddings for 448x448 -> 28x28 patches
        interpolate_pos_embed(model, new_grid_size=(28, 28))

        grid_scores = []
        for cell_idx, cls in enumerate(classes):
            y = torch.tensor([cls], device=device)

            attr_map = explainer.explain(
                x=grid_img, y=y, num_steps=num_steps, post_proc=post_proc,
            )  # (1, 1, 448, 448)

            if attr_map.shape[1] != 1:
                attr_map = attr_map.sum(dim=1, keepdim=True)

            pos_attr = torch.clamp(attr_map[0, 0], min=0)
            H, W = pos_attr.shape
            h_half, w_half = H // 2, W // 2

            cells = [
                pos_attr[:h_half, :w_half],
                pos_attr[:h_half, w_half:],
                pos_attr[h_half:, :w_half],
                pos_attr[h_half:, w_half:],
            ]

            cell_mass = torch.tensor([c.sum().item() for c in cells])
            total_mass = cell_mass.sum().item()

            if total_mass <= 0:
                num_dropped += 1
                continue

            score = (cell_mass[cell_idx] / total_mass).item()
            grid_scores.append(score)
            all_cell_scores.append(score)

        # Restore original pos embedding for next grid (224 input)
        interpolate_pos_embed(model, new_grid_size=(14, 14))
        model.patch_embed.strict_img_size = True

        if (grid_idx + 1) % 25 == 0:
            avg = np.mean(all_cell_scores) * 100 if all_cell_scores else 0
            print(f"[grid {grid_idx+1}/{len(grids)}] running GridPG={avg:.2f}% "
                  f"(n={len(all_cell_scores)}, dropped={num_dropped})")

    gridpg = np.mean(all_cell_scores) * 100 if all_cell_scores else 0.0
    gridpg_std = np.std(all_cell_scores) * 100 if all_cell_scores else 0.0

    return {
        "gridpg_pct": float(gridpg),
        "gridpg_std_pct": float(gridpg_std),
        "num_grids": len(grids),
        "num_total_scores": len(all_cell_scores),
        "num_dropped": num_dropped,
        "dropped_pct": num_dropped / max(len(grids) * 4, 1) * 100,
    }


def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device}")

    print("Loading DAVE explainer...")
    explainer = DAVEExplainer(model_cfg_path=args.model_cfg_path, device=device)
    model_name = explainer.model_name
    print(f"Model: {model_name}")

    transform = explainer.input_transform
    dataset = ImageFolder(args.data_dir, transform=transform)
    print(f"Dataset: {len(dataset)} images, {len(dataset.classes)} classes")

    out_dir = Path(args.out_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Select images
    print("\n=== Stage 1: Selecting high-confidence images ===")
    selected = select_high_conf_images(
        explainer=explainer, dataset=dataset, device=device,
        num_images=args.num_images, min_conf=args.min_conf,
        max_per_class=max(4, args.num_images // 500),
    )
    if len(selected) < 4:
        print("ERROR: Not enough valid images!"); return

    # Stage 2: Build grids
    print("\n=== Stage 2: Building grids ===")
    grids = build_grids(selected=selected, num_grids=args.num_grids, seed=args.seed)

    # Stage 3: Compute GridPG
    print(f"\n=== Stage 3: Computing GridPG ({len(grids)} grids, {args.num_steps} steps) ===")
    results = compute_gridpg(
        explainer=explainer, dataset=dataset, grids=grids,
        device=device, num_steps=args.num_steps, post_proc=args.post_proc,
    )

    # Save
    summary = {**results, "args": vars(args), "model_name": model_name}
    with open(out_dir / "gridpg_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== GridPG RESULTS ===")
    print(f"Model: {model_name}")
    print(f"GridPG: {results['gridpg_pct']:.2f}%")
    print(f"GridPG std: {results['gridpg_std_pct']:.2f}%")
    print(f"Grids: {results['num_grids']}, Scores: {results['num_total_scores']}")
    print(f"Dropped: {results['num_dropped']} ({results['dropped_pct']:.1f}%)")
    print(f"Saved to: {out_dir}/gridpg_summary.json")


if __name__ == "__main__":
    main()
