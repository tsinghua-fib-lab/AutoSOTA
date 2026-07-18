import json
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from core.explainer import DAVEExplainer
from core.utils.utils import set_seed, get_device

from evaluation.utils.perturbation import (
    maybe_slice,
    maybe_mask,
    maybe_log,
    register_results,
    pixel_perturbation_curve,
)
from evaluation.utils.plotting import plot_curves
from evaluation.args.perturbation import build_argparser



def main():
    args = build_argparser().parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

    # DAVE;
    dave_explainer = DAVEExplainer(
        model_cfg_path=args.model_cfg_path,
        device=device,
    )
    model_name = dave_explainer.model_name

    # Data Loader;
    transform = dave_explainer.input_transform
    dataset = ImageFolder(
        args.data_dir, 
        transform=transform,
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
    )

    out_dir = Path(args.out_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Storage;
    curves_accum: Dict[str, List[np.ndarray]] = {}
    auc_accum:    Dict[str, List[float]] = {}
    
    num_seen: int = 0
    for batch_idx, (x, y) in enumerate(data_loader):

        if args.max_samples > 0:
            x, y = maybe_slice(
                x=x, 
                y=y, 
                num_seen=num_seen, 
                max_samples=args.max_samples,
            )
            if x is None or y is None:
                break

        num_smp = x.shape[0]
        num_seen += num_smp
        
        index = torch.arange(num_smp) + (batch_idx * args.batch_size)
        tags = [
            f"{int(index[i].item())}_y{int(y[i].item())}" 
            for i in range(num_smp)
        ]
        
        x = x.to(device)
        y = y.to(device)

        if args.only_correct:
            x, y = maybe_mask(
                x=x,
                y=y,
                model=dave_explainer.model,
                conf_on=args.conf_on,
                min_conf=args.min_conf,
            )
            if x is None or y is None:
                continue

        attr_maps = dave_explainer.explain(
            x=x, 
            y=y,
            num_steps=args.num_steps,
            post_proc=args.post_proc,
        )
        
        if attr_maps.shape[1] != 1:
            attr_maps = attr_maps.sum(dim=1, keepdim=True)

        curves = pixel_perturbation_curve(
            model=dave_explainer.model,
            model_name=model_name,
            x=x,
            y=y,
            attr_map=attr_maps,
            direction=args.direction,
            num_pixels_frac=float(args.num_pixels),
            sample_points=int(args.sample_points),
            mask_baseline=float(args.mask_baseline),
            mask_batch_size=int(args.mask_batch_size),
            mask_mode=str(args.mask_mode),
            blur_kernel=int(args.blur_kernel),
            blur_sigma=float(args.blur_sigma),
        )

        register_results(
            key=f"DAVE_{model_name}", 
            curves_bt=curves,
            curves_accum=curves_accum,
            auc_accum=auc_accum,
            tags_local=tags,
            num_pixels=args.num_pixels,
            save_curves=args.save_curves,
            save_every=args.save_every,
            out_dir=out_dir,
        )

        if args.log_every > 0:
            maybe_log(
                batch_idx=batch_idx,
                log_every=args.log_every,
                num_seen=num_seen,
                num_smp=num_smp,
                auc_accum=auc_accum,
            )

    summary = {
        "args": vars(args),
        "results": {},
    }

    curves_mean_for_plot: Dict[str, np.ndarray] = {}

    for key, curves_list in curves_accum.items():
        if len(curves_list) == 0:
            continue
        
        max_len = max(c.shape[0] for c in curves_list)
        curves_pad = np.stack([np.pad(c, (0, max_len - c.shape[0]), mode="edge") for c in curves_list], axis=0)
        mean_curve = curves_pad.mean(0)
        std_curve = curves_pad.std(0)

        curves_mean_for_plot[key] = mean_curve

        x_axis = np.linspace(0, 100 * float(args.num_pixels), len(mean_curve))
        aucs = np.array(auc_accum.get(key, []), dtype=np.float64)

        summary["results"][key] = {
            "n": int(len(curves_list)),
            "curve_len": int(len(mean_curve)),
            "x_percent_masked": x_axis.tolist(),
            "mean_curve": mean_curve.tolist(),
            "std_curve": std_curve.tolist(),
            "auc_mean": float(np.nanmean(aucs)) if aucs.size else float("nan"),
            "auc_std": float(np.nanstd(aucs)) if aucs.size else float("nan"),
        }

        res_dir = out_dir / "summary_arrays" / key
        res_dir.mkdir(parents=True, exist_ok=True)
        np.save(res_dir / "mean_curve.npy", mean_curve)
        np.save(res_dir / "std_curve.npy", std_curve)
        np.save(res_dir / "x_percent.npy", x_axis)
        if aucs.size:
            np.save(res_dir / "aucs.npy", aucs)

    with open(out_dir / "pixelperturbation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if len(curves_mean_for_plot) > 0:
        plot_curves(
            out_path=out_dir / "pixelperturbation_curves.png",
            curves_mean=curves_mean_for_plot,
            num_pixels_frac=float(args.num_pixels),
            title=f"{model_name}: Pixel Perturbation Metric",
        )

    print("\n=== FINAL SUMMARY (AUC) ===")
    for key in sorted(summary["results"].keys()):
        r = summary["results"][key]
        print(f"{key}: n={r['n']} auc_mean={r['auc_mean']:.6f} auc_std={r['auc_std']:.6f} curve_len={r['curve_len']}")

    print(f"\nWrote: {out_dir / 'pixelperturbation_summary.json'}")
    if (out_dir / "pixelperturbation_curves.png").exists():
        print(f"Wrote: {out_dir / 'pixelperturbation_curves.png'}")


if __name__ == "__main__":
    main()

