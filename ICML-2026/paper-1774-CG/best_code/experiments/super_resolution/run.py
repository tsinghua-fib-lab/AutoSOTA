"""Clean entry point for the 4x ImageNet super-resolution experiment (paper §6.3a).

Reconstructs 256x256 images from 64x64 low-resolution observations with a pixel
mean-flow (pMF) prior and the gradient-free Calibrated Bayesian Guidance
estimator:

    python run.py super-resolution --val-root <imagenet-val-256> \
        --num-particles 256 --num-outer-steps 32

Each denoising step draws K=``--num-particles`` candidate x0 reconstructions
from the one-step pMF posterior and re-weights them by the Gaussian SR
likelihood; no likelihood gradient is computed. The validated SR core (bicubic
4x downsampler, likelihood, pMF adapter, single-image guided sampler) is
vendored verbatim under ``pixel_space_inverse_problems``; this driver reuses it
and the shared ``calibrated_guidance`` library, and logs to the unified W&B
project. Mirrors the original ``benchmark_pmf_superresolution.run_benchmark``
orchestration with a clean, flag-driven, SR-only interface.
"""

from __future__ import annotations

import argparse
import csv
import os
import time

from experiments.common import seed_everything
from experiments.common.wandb_logging import WandbRun
from experiments.super_resolution import _engine

EXPERIMENT = "super_resolution"

_PER_IMAGE_CSV_HEADER = [
    "bench_index",
    "val_index_1based",
    "filename",
    "wnid",
    "pytorch_class_id",
    "psnr",
    "lpips",
    "seconds",
]


def _to_0_1(x_m1to1):
    return ((x_m1to1.clamp(-1.0, 1.0) + 1.0) / 2.0).clamp(0.0, 1.0)


def _append_csv_row(csv_path: str, row: dict) -> None:
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_PER_IMAGE_CSV_HEADER)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def _resolve_range(args, n_items: int):
    """Resolve the [start, end) bench-index range from the mutually-related flags."""
    start = int(args.start_index)
    if args.end_index is not None:
        end = int(args.end_index)
    elif args.num_images is not None:
        end = start + int(args.num_images)
    else:
        end = n_items
    if args.limit is not None:
        end = min(end, start + int(args.limit))
    end = min(end, n_items)
    if not (0 <= start < end <= n_items):
        raise SystemExit(
            f"invalid image range start={start} end={end} (dataset has {n_items} items)"
        )
    return start, end


def run(args) -> int:
    seed_everything(args.seed)

    # Heavy imports happen here, not at module load, so the driver imports on a
    # GPU-less / checkpoint-less box (e.g. for --help or arg parsing).
    import torch
    import piq

    from experiments.super_resolution import _engine as engine  # noqa: F811
    # _engine puts experiments/super_resolution on sys.path so the vendored core
    # package resolves; the pMF adapter then resolves external/pMF itself.
    from pixel_space_inverse_problems.imagenet_val_dataset import StratifiedValSubsample
    from pixel_space_inverse_problems.pixel_mean_flow_adapter import (
        PixelMeanFlowOneStepModel,
        build_pmf_module,
    )
    from pixel_space_inverse_problems.superresolution_likelihood import (
        downsample_pixels_by_bicubic_4x,
    )
    from pixel_space_inverse_problems.run_sibling_reinforce_pixel_mean_flow import (
        run_single_image_reinforce_guided_sampling,
        save_pixel_image_minus1_to_1,
    )

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA requested but not available. Pass --device cpu for a (very slow) CPU run."
        )
    torch_device = torch.device(device)

    if not os.path.exists(args.checkpoint):
        raise SystemExit(
            f"pMF checkpoint not found: {args.checkpoint}\n"
            "Run experiments/super_resolution/download.sh (or pass --checkpoint)."
        )

    # Build the one-step pMF model on the requested device.
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    module = build_pmf_module(args.model_name)
    module.load_state_dict(state_dict, strict=False)
    module = module.to(torch_device).eval()
    del state_dict
    pmf_one_step_model = PixelMeanFlowOneStepModel(
        pmf_module=module,
        cfg_omega=args.cfg_omega,
        interval_min=args.interval_min,
        interval_max=args.interval_max,
    )

    dataset = StratifiedValSubsample(
        val_root=args.val_root,
        val_tar_path=args.val_tar_path,
        manifest_csv=args.manifest_csv,
        seed=args.seed,
        subsample_size=args.subsample_size,
        sr3_top1k_path=args.sr3_top1k_path,
    )
    items = dataset.items()
    start_index, end_index = _resolve_range(args, len(items))
    n_total = end_index - start_index

    os.makedirs(args.output_dir, exist_ok=True)
    for sub in ("gt", "lr", "sr"):
        os.makedirs(os.path.join(args.output_dir, sub), exist_ok=True)
    per_image_csv = os.path.join(args.output_dir, "per_image_metrics.csv")

    lpips_metric = piq.LPIPS().to(torch_device).eval()

    config = {
        "estimator": args.estimator,
        "sampler": "renoise",
        "num_particles": args.num_particles,
        "num_outer_steps": args.num_outer_steps,
        "noise_std": args.noise_std,
        "cfg_omega": args.cfg_omega,
        "tile_count": args.tile_count,
        "guidance_scale": args.guidance_scale,
        "sibling_renoise_t": args.sibling_renoise_t,
        "model_name": args.model_name,
        "checkpoint": os.path.basename(args.checkpoint),
        "subsample_size": args.subsample_size,
        "sr3_top1k_path": args.sr3_top1k_path,
        "start_index": start_index,
        "end_index": end_index,
        "n_images": n_total,
        "seed": args.seed,
    }
    run_name = args.run_name or (
        f"sr_renoise_k{args.num_particles}_s{args.num_outer_steps}"
        f"_{start_index}-{end_index}"
    )

    # Paper SR conditions pMF on the classifier's top-1 prediction of the LR image,
    # not the ground-truth class. Load that mapping if a CSV was provided.
    predicted_labels = {}
    if args.predicted_labels_csv:
        import csv as _csv

        with open(args.predicted_labels_csv, newline="") as _f:
            for _row in _csv.DictReader(_f):
                predicted_labels[_row["filename"]] = int(_row["pred_class_id"])

    psnr_values = []
    lpips_values = []

    with WandbRun(EXPERIMENT, config, run_name=run_name) as wb:
        for bench_index in range(start_index, end_index):
            gt_cpu, _label_cpu, item = dataset[bench_index]
            gt = gt_cpu.unsqueeze(0).to(torch_device)
            lr = downsample_pixels_by_bicubic_4x(gt)
            cond_id = predicted_labels.get(item.filename, int(item.pytorch_class_id))
            class_label = torch.tensor([cond_id], dtype=torch.long, device=torch_device)

            t0 = time.time()
            with torch.no_grad():
                sr = run_single_image_reinforce_guided_sampling(
                    pmf_one_step_model=pmf_one_step_model,
                    observation_low_res=lr,
                    class_label=class_label,
                    num_outer_steps=args.num_outer_steps,
                    num_particles=args.num_particles,
                    noise_std=args.noise_std,
                    guidance_scale=args.guidance_scale,
                    sibling_renoise_t=args.sibling_renoise_t,
                    device=torch_device,
                    estimator=args.estimator,
                    tile_count=args.tile_count,
                    intermediate_results_dir=None,
                    image_index=bench_index,
                    save_intermediate_images=False,
                    task="superresolution",
                )
            elapsed = time.time() - t0

            save_pixel_image_minus1_to_1(
                gt[0], os.path.join(args.output_dir, "gt", f"{bench_index:04d}_gt.png")
            )
            save_pixel_image_minus1_to_1(
                lr[0], os.path.join(args.output_dir, "lr", f"{bench_index:04d}_lr.png")
            )
            save_pixel_image_minus1_to_1(
                sr[0], os.path.join(args.output_dir, "sr", f"{bench_index:04d}_sr.png")
            )

            gt_0_1 = _to_0_1(gt[0]).unsqueeze(0)
            sr_0_1 = _to_0_1(sr[0]).unsqueeze(0)
            with torch.no_grad():
                psnr_val = float(piq.psnr(sr_0_1, gt_0_1, data_range=1.0).item())
                lpips_val = float(lpips_metric(sr_0_1, gt_0_1).item())
            psnr_values.append(psnr_val)
            lpips_values.append(lpips_val)

            row = {
                "bench_index": bench_index,
                "val_index_1based": item.val_index_1based,
                "filename": item.filename,
                "wnid": item.wnid,
                "pytorch_class_id": item.pytorch_class_id,
                "psnr": psnr_val,
                "lpips": lpips_val,
                "seconds": elapsed,
            }
            _append_csv_row(per_image_csv, row)
            wb.log(
                {
                    "image/psnr": psnr_val,
                    "image/lpips": lpips_val,
                    "image/seconds": elapsed,
                    "image/index": bench_index,
                }
            )
            done = bench_index - start_index + 1
            print(
                f"[{done}/{n_total}] {item.filename} "
                f"psnr={psnr_val:.3f} lpips={lpips_val:.4f} ({elapsed:.1f}s)",
                flush=True,
            )

        final = {
            "psnr": sum(psnr_values) / len(psnr_values) if psnr_values else float("nan"),
            "lpips": sum(lpips_values) / len(lpips_values) if lpips_values else float("nan"),
        }
        wb.log({f"final/{k}": v for k, v in final.items()})
        print("Aggregate:", {k: round(float(v), 4) for k, v in final.items()})
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run.py super-resolution",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Sampling / estimator.
    p.add_argument("--num-particles", type=int, default=256,
                   help="K candidate x0 samples drawn from the pMF posterior per step.")
    p.add_argument("--num-outer-steps", type=int, default=8, help="outer flow-matching steps (paper: 8).")
    p.add_argument("--noise-std", type=float, default=0.01, help="Gaussian SR likelihood noise std (paper sigma_y: 0.01).")
    p.add_argument("--cfg-omega", type=float, default=3.0, help="pMF classifier-free guidance omega (paper: 3.0).")
    p.add_argument("--estimator", choices=["reinforce", "reparam"], default="reinforce",
                   help="gradient-free (reinforce) or gradient-based (reparam) CBG.")
    p.add_argument("--guidance-scale", type=float, default=None,
                   help="likelihood temperature; default None -> adaptive per-step scale.")
    p.add_argument("--tile-count", type=int, default=64 * 64,
                   help="reinforce tile grid (perfect square); 1 = whole-image. Default 4096.")
    p.add_argument("--sibling-renoise-t", type=float, default=None,
                   help="renoise level for sibling sampling (default None -> current t).")
    p.add_argument("--interval-min", type=float, default=0.1, help="pMF mean-flow interval min.")
    p.add_argument("--interval-max", type=float, default=0.8, help="pMF mean-flow interval max.")
    # Model / checkpoint.
    p.add_argument("--model-name", default="pmfDiT_B_16", help="pMF model variant.")
    p.add_argument("--checkpoint", default=_engine.DEFAULT_CHECKPOINT, help="pMF checkpoint path.")
    # Data.
    p.add_argument("--val-root", default=os.environ.get("SR_VAL_ROOT"),
                   help="ImageNet-val-256 dir (required for real runs).")
    p.add_argument("--val-tar-path", default=os.environ.get("IMAGENET_VAL_TAR") or None,
                   help="optional ImageNet-val tar for NFS-friendly random-access reads "
                        "(defaults to the IMAGENET_VAL_TAR env var).")
    p.add_argument("--manifest-csv", default=None, help="optional cached 50k val manifest CSV.")
    p.add_argument("--subsample-size", type=int, default=1000,
                   help="stratified 1-per-class subset size (default 1000).")
    p.add_argument("--sr3-top1k-path", default=None,
                   help="optional SR3 community 1000-image manifest (paper-comparable test set).")
    p.add_argument("--predicted-labels-csv", default=None,
                   help="CSV (filename,...,pred_class_id) to condition pMF on the classifier's "
                        "top-1 prediction instead of the ground-truth class (paper SR setup).")
    # Range selection.
    p.add_argument("--start-index", type=int, default=0, help="first bench index (inclusive).")
    p.add_argument("--end-index", type=int, default=None, help="last bench index (exclusive).")
    p.add_argument("--num-images", type=int, default=None,
                   help="process this many images from --start-index (alt to --end-index).")
    p.add_argument("--limit", type=int, default=None,
                   help="hard cap on images processed (smoke tests); first N of the range.")
    # I/O / misc.
    p.add_argument("--output-dir", default="results/super_resolution")
    p.add_argument("--run-name", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p


def cli(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.val_root is None:
        # Defer to the vendored dataset default only if explicitly set; otherwise
        # require it for real runs but allow --help / arg parsing without it.
        from pixel_space_inverse_problems.imagenet_val_dataset import DEFAULT_VAL_ROOT
        args.val_root = DEFAULT_VAL_ROOT
    return run(args)


if __name__ == "__main__":
    raise SystemExit(cli())
