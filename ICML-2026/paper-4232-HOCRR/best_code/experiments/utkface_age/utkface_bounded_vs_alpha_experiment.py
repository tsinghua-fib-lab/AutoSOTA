#!/usr/bin/env python3
"""
UTKFace certification comparison following the MNIST draft workflow.

This script compares, on the same sampled UTKFace test points:
1) Bounded certifier with mean constraint: (E, C, G) + M
2) Alpha-trimming certifier (Rekavandi et al. style probability certificate)

It also stores the unbounded variance-gradient radius for reference.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

from regression_certifiers.certify import BoundedCertifierWithMean  # noqa: E402
from regression_certifiers.certify.alpha_trimming_certifier import (  # noqa: E402
    clopper_pearson_lower,
    probability_success_from_alpha,
    radius_from_probabilities,
    within_eps,
)
from regression_certifiers.certify.variance_gradient_certifier import (  # noqa: E402
    VarianceGradientCertifier,
)


UTK_FILENAME_RE = re.compile(r"^(\d+)_(\d)_(\d)_(.+)\.(jpg|jpeg|png)$", re.IGNORECASE)


@dataclass
class UTKSample:
    path: Path
    age: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_utkface(utk_dir: Path) -> List[UTKSample]:
    files = sorted([p for p in utk_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    out: List[UTKSample] = []
    for p in files:
        m = UTK_FILENAME_RE.match(p.name)
        if m is None:
            continue
        out.append(UTKSample(path=p, age=float(m.group(1))))
    if not out:
        raise RuntimeError(f"No UTKFace files parsed under {utk_dir}")
    return out


def split_test_indices(n: int, train_ratio: float, val_ratio: float, seed: int) -> np.ndarray:
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return idx[n_train + n_val :]


def load_rgb01(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as img:
        rgb = img.convert("RGB").resize((image_size, image_size))
    return np.asarray(rgb, dtype=np.float32) / 255.0


def rgb01_to_bgr_u8(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).round().astype(np.uint8)[:, :, ::-1]


def load_model(model_dir: Path, device: torch.device):
    processor = AutoImageProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoModelForImageClassification.from_pretrained(
        str(model_dir), trust_remote_code=True, dtype=torch.float32
    ).to(device)
    model.eval()
    return model, processor


def resolve_amp_dtype(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "none":
        return None
    if amp_dtype == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if amp_dtype == "bfloat16":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if amp_dtype == "float16":
        return torch.float16
    raise ValueError(f"Unsupported amp_dtype: {amp_dtype}")


def preprocess_rgb01_batch_torch(
    batch_rgb01: np.ndarray,
    *,
    device: torch.device,
    input_size: int,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
) -> torch.Tensor:
    x = torch.from_numpy(batch_rgb01).to(device=device, dtype=torch.float32, non_blocking=True)
    x = x.permute(0, 3, 1, 2).contiguous()
    if x.shape[-1] != input_size or x.shape[-2] != input_size:
        x = F.interpolate(x, size=(input_size, input_size), mode="bilinear", align_corners=False)
    x = (x - mean_t) / std_t
    return x


def predict_many(
    model,
    device: torch.device,
    images_rgb01: np.ndarray,
    batch_size: int = 256,
    input_size: int = 384,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
    amp_dtype: str = "auto",
) -> np.ndarray:
    with_persons = bool(getattr(model.config, "with_persons_model", False))
    n = images_rgb01.shape[0]
    preds = np.zeros(n, dtype=np.float64)
    cur_bs = max(1, int(batch_size))
    amp_t = resolve_amp_dtype(device, amp_dtype)
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    with torch.inference_mode():
        i = 0
        while i < n:
            j = min(i + cur_bs, n)
            batch = images_rgb01[i:j]
            try:
                x = preprocess_rgb01_batch_torch(
                    batch,
                    device=device,
                    input_size=int(input_size),
                    mean_t=mean_t,
                    std_t=std_t,
                )
                amp_ctx = (
                    torch.autocast(device_type="cuda", dtype=amp_t)
                    if amp_t is not None
                    else nullcontext()
                )
                with amp_ctx:
                    if with_persons:
                        out = model(faces_input=x, body_input=x)
                    else:
                        out = model(faces_input=x)
                preds[i:j] = (
                    out.age_output.squeeze(1).detach().float().cpu().numpy().astype(np.float64)
                )
                i = j
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and cur_bs > 1:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    cur_bs = max(1, cur_bs // 2)
                    continue
                raise
    return preds


def summarize(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "fraction_positive": float(np.mean(arr > 0.0)),
    }


def summarize_abs_error(abs_errors: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(abs_errors, dtype=float)
    return {
        "mae": float(np.mean(arr)),
        "rmse": float(np.sqrt(np.mean(arr**2))),
        "medae": float(np.median(arr)),
        "p90ae": float(np.percentile(arr, 90)),
        "maxae": float(np.max(arr)),
    }


def estimate_ecg_stats_for_point(
    *,
    f_values: np.ndarray,
    eta_flat: np.ndarray,
    vg_certifier: VarianceGradientCertifier,
    bounded_certifier: BoundedCertifierWithMean,
    eps_y: float,
    age_min: float,
    age_max: float,
    compute_certificates: bool = True,
) -> Dict[str, float]:
    f_values = np.clip(f_values, age_min, age_max).astype(np.float64)

    # Use the bounded certifier estimators for (E, C, G):
    # they allocate failure probability across THREE estimated quantities.
    C_hat, C_lcb, C_ucb = bounded_certifier.u_statistic_variance_estimator_alpha_half(f_values)
    G_hat, G_lcb, G_ucb = bounded_certifier.u_statistic_gradient_norm_estimator_alpha_half(
        f_values, eta_flat.astype(np.float64)
    )
    E_hat, E_lcb, E_ucb = bounded_certifier.u_statistic_mean_estimator_alpha_third(f_values)

    r_unbounded = float("nan")
    r_bounded_candidates: List[float] = [float("nan"), float("nan"), float("nan")]
    r_bounded_ecg = float("nan")
    if compute_certificates:
        r_unbounded = float(vg_certifier.variance_gradient_certificate(float(C_ucb), float(G_ucb), eps_y))
        # Conservative treatment for mean-estimation uncertainty:
        # evaluate the bounded certificate at E in {E_hat, E_lcb, E_ucb} and keep the minimum radius.
        r_bounded_candidates = [
            bounded_certifier.certify_point_from_estimates(
                C_ucb=float(C_ucb), G_ucb=float(G_ucb), E_est=float(E_hat)
            ),
            bounded_certifier.certify_point_from_estimates(
                C_ucb=float(C_ucb), G_ucb=float(G_ucb), E_est=float(E_lcb)
            ),
            bounded_certifier.certify_point_from_estimates(
                C_ucb=float(C_ucb), G_ucb=float(G_ucb), E_est=float(E_ucb)
            ),
        ]
        r_bounded_ecg = float(min(r_bounded_candidates))

    return {
        "N_samples": int(len(f_values)),
        "C_hat": float(C_hat),
        "C_lcb": float(C_lcb),
        "C_ucb": float(C_ucb),
        "G_hat": float(G_hat),
        "G_lcb": float(G_lcb),
        "G_ucb": float(G_ucb),
        "E_hat": float(E_hat),
        "E_lcb": float(E_lcb),
        "E_ucb": float(E_ucb),
        "radius_unbounded_vg": r_unbounded,
        "radius_bounded_ecg_candidates": [float(r) for r in r_bounded_candidates],
        "radius_bounded_ecg": r_bounded_ecg,
    }


def alpha_radius_batched(
    *,
    preds_noisy: np.ndarray,
    clean_pred: float,
    sigma: float,
    eps_y: float,
    alpha: float,
    n_tr: int,
    n_sample: int,
    P: float,
    confidence: float,
    seed: int,
    age_min: float,
    age_max: float,
) -> Dict[str, float]:
    _ = seed  # keep argument for stable API
    preds = np.clip(preds_noisy[:n_tr], age_min, age_max)

    center = float(np.clip(clean_pred, age_min, age_max))
    k = int(sum(within_eps(float(pred), center, eps_y, circular=False) for pred in preds))
    delta = 1.0 - float(confidence)
    pA_lcb = float(clopper_pearson_lower(k, n_tr, delta))
    q = float(probability_success_from_alpha(alpha, n_sample, P))
    radius = float(radius_from_probabilities(pA_lcb, q, sigma))

    return {
        "radius_alpha": radius,
        "n_tr": int(n_tr),
        "n_sample": int(n_sample),
        "k_success": int(k),
        "pA_lcb": pA_lcb,
        "q": q,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="UTKFace comparison: bounded (E,C,G)+M vs alpha-trimming."
    )
    p.add_argument("--utk_dir", type=str, required=True)
    p.add_argument("--model_dir", type=str, default="models/mivolo_v2_hf")
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--max_test_samples", type=int, default=2000)
    p.add_argument("--n_points", type=int, default=100)
    p.add_argument(
        "--selected_indices_file",
        type=str,
        default="",
        help=(
            "Optional JSON file containing explicit selected test dataset indices "
            "(list[int] or {'selected_test_indices': [...]}); overrides random selection."
        ),
    )
    p.add_argument(
        "--point_start",
        type=int,
        default=0,
        help="Inclusive start index in the globally selected n_points set (for sharding).",
    )
    p.add_argument(
        "--point_end",
        type=int,
        default=-1,
        help="Exclusive end index in the globally selected n_points set (-1 means n_points).",
    )
    p.add_argument(
        "--shard_id",
        type=int,
        default=-1,
        help="Optional shard id (0-based). If set with --num_shards, overrides point_start/end.",
    )
    p.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Optional total number of shards. Used with --shard_id.",
    )
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--sigma", type=float, default=0.20)
    p.add_argument("--eps_y", type=float, default=6.0, help="Output tolerance in years.")
    p.add_argument("--confidence", type=float, default=0.90)

    p.add_argument("--N", type=int, default=10000, help="MC samples for (E,C,G) estimation.")
    p.add_argument("--n_trials", type=int, default=1, help="Independent trials for (E,C,G).")

    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--alpha_n_tr", type=int, default=10000, help="MC samples for alpha p_A.")
    p.add_argument(
        "--alpha_n_sample",
        type=int,
        default=200,
        help="Samples for alpha-to-q binomial mapping (MNIST draft style: 200 for alpha=0.35).",
    )
    p.add_argument("--alpha_P", type=float, default=0.9)

    p.add_argument("--age_min", type=float, default=0.0)
    p.add_argument("--age_max", type=float, default=116.0)
    p.add_argument(
        "--M",
        type=float,
        default=116.0,
        help="Absolute output bound for bounded certifier (|f|<=M).",
    )

    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument(
        "--mode",
        type=str,
        choices=["both", "ours", "alpha"],
        default="both",
        help="Compute both methods or only one (for faster sweeps).",
    )
    p.add_argument(
        "--amp_dtype",
        type=str,
        choices=["auto", "bfloat16", "float16", "none"],
        default="auto",
        help="CUDA mixed-precision dtype for model forward.",
    )
    p.add_argument(
        "--output",
        type=str,
        default="outputs/utkface_bounded_vs_alpha.json",
    )
    p.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="Write partial checkpoint every K processed points.",
    )
    p.add_argument(
        "--preflight_only",
        action="store_true",
        help=(
            "Run a fast sanity check (data parse, model forward, noisy batch forward, "
            "alpha/ecg estimator calls) on one point, then exit."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    utk_dir = Path(args.utk_dir)
    model_dir = Path(args.model_dir)
    if not utk_dir.exists():
        raise FileNotFoundError(f"UTKFace directory not found: {utk_dir}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    samples = parse_utkface(utk_dir)
    idx_test = split_test_indices(len(samples), args.train_ratio, args.val_ratio, args.seed)
    if args.max_test_samples > 0:
        idx_test = idx_test[: min(len(idx_test), args.max_test_samples)]
    if len(idx_test) < args.n_points:
        raise ValueError(
            f"Not enough test samples ({len(idx_test)}) for n_points={args.n_points}. "
            "Increase --max_test_samples or reduce --n_points."
        )

    if args.selected_indices_file:
        selected_path = Path(args.selected_indices_file)
        if not selected_path.exists():
            raise FileNotFoundError(f"selected_indices_file not found: {selected_path}")
        with selected_path.open("r", encoding="utf-8") as f:
            selected_obj = json.load(f)
        if isinstance(selected_obj, dict):
            if "selected_test_indices" not in selected_obj:
                raise ValueError(
                    "selected_indices_file dict must contain key 'selected_test_indices'."
                )
            selected_list = selected_obj["selected_test_indices"]
        elif isinstance(selected_obj, list):
            selected_list = selected_obj
        else:
            raise ValueError(
                "selected_indices_file must be either a JSON list or dict with selected_test_indices."
            )
        chosen_dataset_idx = [int(x) for x in selected_list]
        if len(chosen_dataset_idx) == 0:
            raise ValueError("selected_indices_file contains no indices.")
        if len(chosen_dataset_idx) != len(set(chosen_dataset_idx)):
            raise ValueError("selected_indices_file contains duplicate indices.")
        idx_test_set = set(int(x) for x in idx_test.tolist())
        bad = [x for x in chosen_dataset_idx if x not in idx_test_set]
        if bad:
            raise ValueError(
                "selected_indices_file contains indices outside test split. "
                f"First few invalid: {bad[:5]}"
            )
        if int(args.n_points) != len(chosen_dataset_idx):
            print(
                f"[INFO] overriding n_points from {args.n_points} to {len(chosen_dataset_idx)} "
                "based on selected_indices_file"
            )
            args.n_points = int(len(chosen_dataset_idx))
    else:
        rng = np.random.default_rng(args.seed)
        chosen_local = rng.choice(len(idx_test), size=args.n_points, replace=False)
        chosen_dataset_idx = [int(idx_test[int(i)]) for i in chosen_local]
    chosen_samples = [samples[i] for i in chosen_dataset_idx]

    # Resolve shard range in the globally selected n_points set.
    point_start = int(args.point_start)
    point_end = int(args.n_points if args.point_end < 0 else args.point_end)
    if args.shard_id >= 0:
        if args.num_shards <= 0:
            raise ValueError("--num_shards must be >= 1 when --shard_id is set.")
        if args.shard_id >= args.num_shards:
            raise ValueError("--shard_id must be < --num_shards.")
        # Balanced contiguous split.
        boundaries = np.linspace(0, args.n_points, num=args.num_shards + 1, dtype=int)
        point_start = int(boundaries[args.shard_id])
        point_end = int(boundaries[args.shard_id + 1])
    if point_start < 0 or point_start > args.n_points:
        raise ValueError(f"Invalid point_start={point_start} for n_points={args.n_points}.")
    if point_end < point_start or point_end > args.n_points:
        raise ValueError(f"Invalid point_end={point_end} for n_points={args.n_points}.")

    shard_global_indices = list(range(point_start, point_end))
    shard_dataset_indices = [chosen_dataset_idx[i] for i in shard_global_indices]
    shard_samples = [chosen_samples[i] for i in shard_global_indices]

    model, processor = load_model(model_dir, device)
    processor_input_size = int(getattr(processor, "input_size", 384))
    processor_mean = tuple(getattr(processor, "mean", [0.485, 0.456, 0.406]))
    processor_std = tuple(getattr(processor, "std", [0.229, 0.224, 0.225]))
    vg = VarianceGradientCertifier(sigma=args.sigma, eps_y=args.eps_y, confidence=args.confidence)
    bounded_ecg = BoundedCertifierWithMean(
        sigma=args.sigma,
        M=args.M,
        eps_y=args.eps_y,
        confidence=args.confidence,
        quadrature_points=60,
    )

    if args.preflight_only:
        if len(shard_samples) == 0:
            raise ValueError("Preflight requested but shard has zero points.")
        sample = shard_samples[0]
        global_i = shard_global_indices[0]
        print("[PRECHECK] Running one-point smoke validation...")
        x0 = load_rgb01(sample.path, args.image_size)
        clean_pred = float(
            np.clip(
                predict_many(
                    model,
                    device,
                    x0[None, ...],
                    batch_size=1,
                    input_size=processor_input_size,
                    mean=processor_mean,
                    std=processor_std,
                    amp_dtype=args.amp_dtype,
                )[0],
                args.age_min,
                args.age_max,
            )
        )
        h, w, c = x0.shape
        d = h * w * c
        if args.mode == "both":
            n_debug = int(max(4, min(args.N, args.alpha_n_tr, 64)))
        elif args.mode == "ours":
            n_debug = int(max(4, min(args.N, 64)))
        else:
            n_debug = int(max(4, min(args.alpha_n_tr, 64)))
        trial_seed = int(args.seed + global_i * 100003 + 1009)
        rng_debug = np.random.default_rng(trial_seed)
        eta_debug = rng_debug.normal(0.0, args.sigma, size=(n_debug, d)).astype(np.float32)
        x_noisy_debug = np.clip(x0.reshape(1, -1) + eta_debug, 0.0, 1.0).reshape(n_debug, h, w, c)
        preds_debug = predict_many(
            model,
            device,
            x_noisy_debug,
            batch_size=min(args.batch_size, 64),
            input_size=processor_input_size,
            mean=processor_mean,
            std=processor_std,
            amp_dtype=args.amp_dtype,
        )
        preds_debug = np.clip(preds_debug, args.age_min, args.age_max)

        # Touch selected methods' core calls so interface/runtime regressions surface quickly.
        if args.mode in {"both", "ours"}:
            _ = estimate_ecg_stats_for_point(
                f_values=preds_debug[: min(len(preds_debug), int(args.N))],
                eta_flat=eta_debug[: min(len(eta_debug), int(args.N))].astype(np.float64),
                vg_certifier=vg,
                bounded_certifier=bounded_ecg,
                eps_y=args.eps_y,
                age_min=args.age_min,
                age_max=args.age_max,
                compute_certificates=False,
            )
        if args.mode in {"both", "alpha"}:
            _ = alpha_radius_batched(
                preds_noisy=preds_debug,
                clean_pred=clean_pred,
                sigma=args.sigma,
                eps_y=args.eps_y,
                alpha=args.alpha,
                n_tr=min(n_debug, int(args.alpha_n_tr)),
                n_sample=min(args.alpha_n_sample, n_debug),
                P=args.alpha_P,
                confidence=args.confidence,
                seed=args.seed + 17,
                age_min=args.age_min,
                age_max=args.age_max,
            )
        preflight_out = {
            "timestamp": datetime.now().isoformat(),
            "experiment_type": "utkface_bounded_ecg_vs_alpha_preflight",
            "preflight_passed": True,
            "checked_sample_path": str(sample.path),
            "checked_sample_global_idx": int(global_i),
            "n_debug": int(n_debug),
            "config": vars(args),
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(preflight_out, f, indent=2)
        print("[PRECHECK] Passed.")
        print("Saved:", out_path)
        return

    print("=" * 80)
    print("UTKFACE: BOUNDED (E,C,G)+M VS ALPHA-TRIMMING")
    print("=" * 80)
    print(f"Selected points (global): {len(chosen_samples)}")
    print(f"Shard range in selected set: [{point_start}, {point_end}) -> {len(shard_samples)} points")
    print(f"Sigma={args.sigma}, eps_y={args.eps_y}, confidence={args.confidence}")
    print(f"Bounded certifier M={args.M}, age clipping=[{args.age_min}, {args.age_max}]")
    print(f"Mode={args.mode}, amp_dtype={args.amp_dtype}, model_input_size={processor_input_size}")
    print(f"(E,C,G) N={args.N}, trials={args.n_trials}")
    print(f"Alpha n_tr={args.alpha_n_tr}, n_sample={args.alpha_n_sample}, alpha={args.alpha}, P={args.alpha_P}")
    print("=" * 80)

    point_rows: List[Dict[str, object]] = []
    clean_abs_errors: List[float] = []
    smoothed_abs_errors: List[float] = []
    bounded_primary: List[float] = []
    alpha_primary: List[float] = []
    unbounded_primary: List[float] = []

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = out_path.with_suffix(out_path.suffix + ".partial")

    def write_checkpoint() -> None:
        if len(point_rows) == 0:
            return
        clean_err_summary = summarize_abs_error(clean_abs_errors)
        smoothed_err_summary = summarize_abs_error(smoothed_abs_errors)
        ckpt = {
            "timestamp": datetime.now().isoformat(),
            "experiment_type": "utkface_bounded_ecg_vs_alpha_partial",
            "dataset": {
                "name": "UTKFace",
                "utk_dir": str(utk_dir.resolve()),
                "n_total_parsed": int(len(samples)),
                "n_test_pool": int(len(idx_test)),
                "n_points": int(args.n_points),
                "selected_test_indices": chosen_dataset_idx,
                "shard": {
                    "point_start": int(point_start),
                    "point_end": int(point_end),
                    "n_points_in_shard": int(len(shard_samples)),
                    "selected_test_indices_shard": shard_dataset_indices,
                    "processed_points": int(len(point_rows)),
                },
            },
            "model": {
                "name": "iitolstykh/mivolo_v2",
                "model_dir": str(model_dir.resolve()),
                "with_persons_model": bool(getattr(model.config, "with_persons_model", False)),
                "processor_input_size": getattr(processor, "input_size", None),
            },
            "config": vars(args),
            "summary_so_far": {
                "clean_abs_error": clean_err_summary,
                "smoothed_abs_error": smoothed_err_summary,
                "smoothed_minus_clean_mae": float(
                    smoothed_err_summary["mae"] - clean_err_summary["mae"]
                ),
                "bounded_ecg_radius": summarize(bounded_primary) if bounded_primary else None,
                "alpha_radius": summarize(alpha_primary) if alpha_primary else None,
                "unbounded_vg_radius": summarize(unbounded_primary) if unbounded_primary else None,
            },
            "samples": point_rows,
        }
        with partial_path.open("w", encoding="utf-8") as f:
            json.dump(ckpt, f, indent=2)

    for local_i, (global_i, dataset_idx, sample) in enumerate(
        zip(shard_global_indices, shard_dataset_indices, shard_samples)
    ):
        print(f"[{local_i+1}/{len(shard_samples)}] global#{global_i} {sample.path.name}")
        x0 = load_rgb01(sample.path, args.image_size)
        clean_pred = float(
            np.clip(
                predict_many(
                    model,
                    device,
                    x0[None, ...],
                    batch_size=1,
                    input_size=processor_input_size,
                    mean=processor_mean,
                    std=processor_std,
                    amp_dtype=args.amp_dtype,
                )[0],
                args.age_min,
                args.age_max,
            )
        )
        abs_err = float(abs(clean_pred - sample.age))
        clean_abs_errors.append(abs_err)

        ecg_trials: List[Dict[str, float]] = []
        alpha_trials: List[Dict[str, float]] = []
        g_hat_trials: List[float] = []
        h, w, c = x0.shape
        d = h * w * c
        z_flat = x0.reshape(-1).astype(np.float32)
        if args.mode == "both":
            n_common = max(int(args.N), int(args.alpha_n_tr))
        elif args.mode == "ours":
            n_common = int(args.N)
        else:
            n_common = int(args.alpha_n_tr)
        for t in range(args.n_trials):
            trial_seed = int(args.seed + global_i * 100003 + t * 1009)
            rng_common = np.random.default_rng(trial_seed)
            eta_common = rng_common.normal(0.0, args.sigma, size=(n_common, d)).astype(np.float32)
            x_noisy_common = np.clip(z_flat[None, :] + eta_common, 0.0, 1.0).reshape(n_common, h, w, c)
            preds_common = predict_many(
                model,
                device,
                x_noisy_common,
                batch_size=args.batch_size,
                input_size=processor_input_size,
                mean=processor_mean,
                std=processor_std,
                amp_dtype=args.amp_dtype,
            )
            preds_common = np.clip(preds_common, args.age_min, args.age_max)
            g_hat_trials.append(float(np.mean(preds_common)))

            if args.mode in {"both", "ours"}:
                est = estimate_ecg_stats_for_point(
                    f_values=preds_common[: args.N],
                    eta_flat=eta_common[: args.N].astype(np.float64),
                    vg_certifier=vg,
                    bounded_certifier=bounded_ecg,
                    eps_y=args.eps_y,
                    age_min=args.age_min,
                    age_max=args.age_max,
                )
                ecg_trials.append(est)

            if args.mode in {"both", "alpha"}:
                alpha_seed = int(args.seed + global_i * 700001 + 17 + t)
                alpha_out_t = alpha_radius_batched(
                    preds_noisy=preds_common,
                    clean_pred=clean_pred,
                    sigma=args.sigma,
                    eps_y=args.eps_y,
                    alpha=args.alpha,
                    n_tr=args.alpha_n_tr,
                    n_sample=args.alpha_n_sample,
                    P=args.alpha_P,
                    confidence=args.confidence,
                    seed=alpha_seed,
                    age_min=args.age_min,
                    age_max=args.age_max,
                )
                alpha_trials.append(alpha_out_t)

        # Smoothed regressor estimate g(x)=E[f(x+eta)] via Monte Carlo sample mean.
        # We estimate this per trial and then average across trials for stability.
        smoothed_pred = float(np.mean(np.asarray(g_hat_trials, dtype=float))) if g_hat_trials else float("nan")
        smoothed_abs_err = float(abs(smoothed_pred - sample.age))
        smoothed_abs_errors.append(smoothed_abs_err)

        alpha_out = None
        if len(alpha_trials) > 0:
            alpha_radii = np.asarray([x["radius_alpha"] for x in alpha_trials], dtype=float)
            alpha_out = dict(alpha_trials[0])
            alpha_out["radius_alpha_trials"] = [float(r) for r in alpha_radii.tolist()]
            alpha_out["radius_alpha_mean_over_trials"] = float(np.mean(alpha_radii))
            alpha_out["radius_alpha_std_over_trials"] = float(np.std(alpha_radii))
            alpha_out["n_trials"] = int(args.n_trials)

        bounded_mean = None
        unbounded_mean = None
        if len(ecg_trials) > 0:
            bounded_mean = float(np.mean([r["radius_bounded_ecg"] for r in ecg_trials]))
            unbounded_mean = float(np.mean([r["radius_unbounded_vg"] for r in ecg_trials]))
            bounded_primary.append(bounded_mean)
            unbounded_primary.append(unbounded_mean)
        if alpha_out is not None:
            alpha_primary.append(float(alpha_out["radius_alpha_mean_over_trials"]))

        point_rows.append(
            {
                "sample_local_idx": int(local_i),
                "sample_global_idx": int(global_i),
                "test_dataset_idx": int(dataset_idx),
                "path": str(sample.path),
                "age_true": float(sample.age),
                "clean_pred": float(clean_pred),
                "clean_abs_error": abs_err,
                "smoothed_pred_mean_over_trials": smoothed_pred,
                "smoothed_pred_trials": [float(v) for v in g_hat_trials],
                "smoothed_abs_error_mean_over_trials": smoothed_abs_err,
                "ecg_trials": ecg_trials,
                "ecg_radius_mean_over_trials": bounded_mean,
                "unbounded_radius_mean_over_trials": unbounded_mean,
                "alpha_result": alpha_out,
            }
        )
        if args.save_every > 0 and ((local_i + 1) % args.save_every == 0):
            write_checkpoint()

    clean_metrics = summarize_abs_error(clean_abs_errors)
    smoothed_metrics = summarize_abs_error(smoothed_abs_errors)

    output = {
        "timestamp": datetime.now().isoformat(),
        "experiment_type": "utkface_bounded_ecg_vs_alpha",
        "dataset": {
            "name": "UTKFace",
            "utk_dir": str(utk_dir.resolve()),
            "n_total_parsed": int(len(samples)),
            "n_test_pool": int(len(idx_test)),
            "n_points": int(args.n_points),
            "selected_test_indices": chosen_dataset_idx,
            "shard": {
                "point_start": int(point_start),
                "point_end": int(point_end),
                "n_points_in_shard": int(len(shard_samples)),
                "selected_test_indices_shard": shard_dataset_indices,
            },
        },
        "model": {
            "name": "iitolstykh/mivolo_v2",
            "model_dir": str(model_dir.resolve()),
            "with_persons_model": bool(getattr(model.config, "with_persons_model", False)),
            "processor_input_size": getattr(processor, "input_size", None),
        },
        "config": vars(args),
        "summary": {
            "clean_abs_error": clean_metrics,
            "smoothed_abs_error": smoothed_metrics,
            "smoothed_minus_clean_mae": float(smoothed_metrics["mae"] - clean_metrics["mae"]),
            "bounded_ecg_radius": summarize(bounded_primary) if bounded_primary else None,
            "alpha_radius": summarize(alpha_primary) if alpha_primary else None,
            "unbounded_vg_radius": summarize(unbounded_primary) if unbounded_primary else None,
        },
        "samples": point_rows,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    if partial_path.exists():
        partial_path.unlink()

    print("\nSaved:", out_path)
    print("Done.")


if __name__ == "__main__":
    main()

