#!/usr/bin/env python3
"""
UTKFace single-point convergence/coverage analysis (appendix-style).

This mirrors the MNIST appendix-style structure:
- Part 1: convergence + CI coverage for estimator quantities (C, theta, G, g_z)
- Part 2: convergence of certified radius estimates to a high-N reference radius
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from scipy.optimize import brentq
from scipy.stats import norm
from scipy.stats import t
from transformers import AutoImageProcessor, AutoModelForImageClassification


UTK_FILENAME_RE = re.compile(r"^(\d+)_(\d)_(\d)_(.+)\.(jpg|jpeg|png)$", re.IGNORECASE)


class ConvergenceValidator:
    """Self-contained estimator + certificate utilities for convergence analysis."""

    def __init__(self, sigma: float, eps_y: float, confidence: float):
        self.sigma = float(sigma)
        self.eps_y = float(eps_y)
        self.confidence = float(confidence)

    def u_statistic_variance_estimator_alpha_half(self, samples: np.ndarray) -> tuple:
        n = len(samples)
        if n < 2:
            return 0.0, 0.0, 0.0
        theta_hat = np.var(samples, ddof=1)
        mean_val = np.mean(samples)
        fourth_moment = np.mean((samples - mean_val) ** 4)
        asymptotic_var = max(0.0, fourth_moment - theta_hat**2)
        alpha_total = 1 - self.confidence
        alpha_split = alpha_total / 2.0
        z_critical = norm.ppf(1 - alpha_split)  # One-sided UCB for certification
        se = np.sqrt(asymptotic_var / n)
        return theta_hat, theta_hat - z_critical * se, theta_hat + z_critical * se

    def _theta_hat_and_asymptotic_var(self, f_values: np.ndarray, eta_samples: np.ndarray) -> tuple:
        n = len(f_values)
        if n < 2:
            return 0.0, 0.0
        W = (1 / self.sigma**2) * eta_samples * f_values[:, np.newaxis]
        sum_W = np.sum(W, axis=0)
        sum_W_sq_norm = np.dot(sum_W, sum_W)
        sum_sq_norm_W = np.sum(np.linalg.norm(W, axis=1) ** 2)
        off_diag = 0.5 * (sum_W_sq_norm - sum_sq_norm_W)
        num_pairs = n * (n - 1) / 2
        theta_hat = off_diag / num_pairs if num_pairs > 0 else 0.0
        mu_hat = np.mean(W, axis=0)
        centered = W - mu_hat
        Sigma_hat = np.cov(centered, rowvar=False, ddof=1)
        asymptotic_var = max(0.0, 4 * np.dot(mu_hat, np.dot(Sigma_hat, mu_hat)))
        return float(theta_hat), float(asymptotic_var)

    def compute_theta_ci_with_z_critical(
        self, f_values: np.ndarray, eta_samples: np.ndarray, confidence: float | None = None
    ) -> tuple:
        n = len(f_values)
        if n < 2:
            return 0.0, 0.0, 0.0
        conf = self.confidence if confidence is None else float(confidence)
        theta_hat, asymptotic_var = self._theta_hat_and_asymptotic_var(f_values, eta_samples)
        alpha_total = 1 - conf
        alpha_split = alpha_total / 2.0
        z_critical = norm.ppf(1 - alpha_split)  # One-sided UCB for certification
        se = np.sqrt(asymptotic_var / n)
        return theta_hat, theta_hat - z_critical * se, theta_hat + z_critical * se

    def u_statistic_gradient_norm_estimator_alpha_half(
        self, f_values: np.ndarray, eta_samples: np.ndarray
    ) -> tuple:
        n = len(f_values)
        if n < 2:
            return 0.0, 0.0, 0.0
        theta_hat, asymptotic_var = self._theta_hat_and_asymptotic_var(f_values, eta_samples)
        alpha_total = 1 - self.confidence
        alpha_split = alpha_total / 2.0
        z_critical = norm.ppf(1 - alpha_split)  # One-sided UCB for certification
        se = np.sqrt(asymptotic_var / n)
        theta_lower = theta_hat - z_critical * se
        theta_upper = theta_hat + z_critical * se
        return (
            float(np.sqrt(max(0.0, theta_hat))),
            float(np.sqrt(max(0.0, theta_lower))),
            float(np.sqrt(max(0.0, theta_upper))),
        )

    def variance_gradient_certificate(self, C_ucb: float, G_ucb: float, eps_y: float) -> float:
        def get_max_harm_at_r(r: float) -> float:
            if r < 0:
                return -float("inf")
            V_arg = r**2 / self.sigma**2
            V_r = np.exp(V_arg) - 1 - V_arg
            if V_r <= 0:
                return r * G_ucb
            g_max = min(G_ucb, np.sqrt(C_ucb) / self.sigma if C_ucb > 0 else 0.0)
            numerator = r * np.sqrt(C_ucb)
            denominator = np.sqrt(self.sigma**4 * V_r + self.sigma**2 * r**2)
            g_star = numerator / denominator if denominator > 1e-9 else 0.0
            g_opt = min(g_star, g_max)
            return np.sqrt(max(0, C_ucb - self.sigma**2 * g_opt**2)) * np.sqrt(V_r) + r * g_opt

        try:
            r_high = 20.0 * self.sigma
            if get_max_harm_at_r(r_high) < eps_y:
                return r_high
            r = brentq(lambda rr: get_max_harm_at_r(rr) - eps_y, 0.0, r_high, xtol=1e-7, rtol=1e-7)
            return max(0.0, float(r))
        except (ValueError, RuntimeError):
            return 0.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_utkface(utk_dir: Path) -> List[Tuple[Path, float]]:
    files = sorted([p for p in utk_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    out: List[Tuple[Path, float]] = []
    for p in files:
        m = UTK_FILENAME_RE.match(p.name)
        if m is None:
            continue
        out.append((p, float(m.group(1))))
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


def predict_many(
    model,
    processor,
    device: torch.device,
    images_rgb01: np.ndarray,
    batch_size: int = 64,
) -> np.ndarray:
    with_persons = bool(getattr(model.config, "with_persons_model", False))
    n = images_rgb01.shape[0]
    preds = np.zeros(n, dtype=np.float64)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            bgr_batch = [rgb01_to_bgr_u8(images_rgb01[k]) for k in range(i, j)]
            x = processor(images=bgr_batch)["pixel_values"].to(device)
            if with_persons:
                out = model(faces_input=x, body_input=x)
            else:
                out = model(faces_input=x)
            preds[i:j] = out.age_output.squeeze(1).detach().cpu().numpy().astype(np.float64)
    return preds


def run_convergence(args: argparse.Namespace) -> Dict:
    utk_dir = Path(args.utk_dir)
    model_dir = Path(args.model_dir)
    device = torch.device(args.device)

    samples = parse_utkface(utk_dir)
    idx_test = split_test_indices(len(samples), args.train_ratio, args.val_ratio, args.seed)
    if args.max_test_samples > 0:
        idx_test = idx_test[: min(len(idx_test), args.max_test_samples)]
    if len(idx_test) == 0:
        raise RuntimeError("Test split is empty.")

    local_idx = int(args.image_idx)
    if local_idx < 0 or local_idx >= len(idx_test):
        raise ValueError(f"image_idx={local_idx} out of range for test size {len(idx_test)}")

    sample_path, y_true = samples[int(idx_test[local_idx])]
    x0 = load_rgb01(sample_path, args.image_size)  # [H,W,3], float32 [0,1]
    h, w, c = x0.shape
    d = h * w * c

    model, processor = load_model(model_dir, device)
    clean_pred = float(predict_many(model, processor, device, x0[None, ...], batch_size=1)[0])

    validator = ConvergenceValidator(
        sigma=args.sigma,
        eps_y=args.eps_y,
        confidence=args.confidence,
    )

    print("=" * 80)
    print("UTKFACE SINGLE-POINT CONVERGENCE (APPENDIX STYLE)")
    print("=" * 80)
    print(f"Sample path: {sample_path}")
    print(f"True age: {y_true:.2f}")
    print(f"Clean prediction: {clean_pred:.4f}")
    print(f"image_size: {args.image_size}, dimension d: {d}")
    print(f"N values: {args.N_values}, n_trials: {args.n_trials}")
    print(f"ground_truth_N: {args.ground_truth_N}")

    # Ground truth / reference estimates from high N
    print("\nEstimating high-N reference quantities...")
    rng_gt = np.random.default_rng(args.seed)
    eta_gt = rng_gt.normal(0.0, args.sigma, size=(args.ground_truth_N, h, w, c)).astype(np.float32)
    x_gt = np.clip(x0[None, ...] + eta_gt, 0.0, 1.0)
    f_gt = predict_many(model, processor, device, x_gt, batch_size=args.batch_size)
    eta_gt_flat = eta_gt.reshape(args.ground_truth_N, -1).astype(np.float64)

    g_true = float(np.mean(f_gt))
    C_true = float(np.var(f_gt, ddof=0))
    theta_true, _, _ = validator.compute_theta_ci_with_z_critical(
        f_gt, eta_gt_flat, confidence=validator.confidence
    )
    G_true, _, _ = validator.u_statistic_gradient_norm_estimator_alpha_half(f_gt, eta_gt_flat)
    theta_true = float(theta_true)
    G_true = float(G_true)
    r_theoretical = float(validator.variance_gradient_certificate(C_true, G_true, args.eps_y))

    print(
        f"Reference: g={g_true:.6f}, C={C_true:.6f}, theta={theta_true:.6f}, "
        f"G={G_true:.6f}, R={r_theoretical:.6f}"
    )

    part1 = {
        "sigma": float(args.sigma),
        "eps_y": float(args.eps_y),
        "N_values": list(map(int, args.N_values)),
        "n_trials": int(args.n_trials),
        "ground_truth_N": int(args.ground_truth_N),
        "ground_truth": {
            "g_z": g_true,
            "C": C_true,
            "theta": theta_true,
            "G_norm": G_true,
        },
        "results_by_N": {str(int(N)): [] for N in args.N_values},
    }
    part2 = {
        "sigma": float(args.sigma),
        "eps_y": float(args.eps_y),
        "N_values": list(map(int, args.N_values)),
        "n_trials": int(args.n_trials),
        "theoretical_radius": r_theoretical,
        "results_by_N": {str(int(N)): [] for N in args.N_values},
    }

    trial_counter = 0
    alpha_total = 1.0 - args.confidence
    alpha_split_g = alpha_total / 3.0

    for N in args.N_values:
        print(f"\nN={N}: ", end="", flush=True)
        for t_idx in range(args.n_trials):
            rng = np.random.default_rng(args.seed + trial_counter)
            eta = rng.normal(0.0, args.sigma, size=(N, h, w, c)).astype(np.float32)
            x_noisy = np.clip(x0[None, ...] + eta, 0.0, 1.0)
            f = predict_many(model, processor, device, x_noisy, batch_size=args.batch_size)
            eta_flat = eta.reshape(N, -1).astype(np.float64)

            C_hat, C_low, C_up = validator.u_statistic_variance_estimator_alpha_half(f)
            theta_hat, theta_low, theta_up = validator.compute_theta_ci_with_z_critical(
                f, eta_flat, confidence=validator.confidence
            )
            G_hat, G_low, G_up = validator.u_statistic_gradient_norm_estimator_alpha_half(f, eta_flat)

            g_hat = float(np.mean(f))
            g_std = float(np.std(f, ddof=1)) if N > 1 else 0.0
            t_critical = t.ppf(1 - alpha_split_g / 2, df=max(1, N - 1))
            g_se = g_std / np.sqrt(max(1, N))
            g_low = g_hat - t_critical * g_se
            g_up = g_hat + t_critical * g_se

            r_emp = float(
                validator.variance_gradient_certificate(
                    float(C_up), float(G_up), args.eps_y
                )
            )

            part1["results_by_N"][str(int(N))].append(
                {
                    "trial": int(t_idx),
                    "N_samples": int(N),
                    "C_hat": float(C_hat),
                    "C_lower": float(C_low),
                    "C_upper": float(C_up),
                    "theta_hat": float(theta_hat),
                    "theta_lower": float(theta_low),
                    "theta_upper": float(theta_up),
                    "G_norm_hat": float(G_hat),
                    "G_norm_lower": float(G_low),
                    "G_norm_upper": float(G_up),
                    "g_z_hat": float(g_hat),
                    "g_z_lower": float(g_low),
                    "g_z_upper": float(g_up),
                }
            )
            part2["results_by_N"][str(int(N))].append(
                {
                    "trial": int(t_idx),
                    "N_samples": int(N),
                    "C_ucb": float(C_up),
                    "G_ucb": float(G_up),
                    "r_empirical": float(r_emp),
                }
            )

            trial_counter += 1
            if (t_idx + 1) % 5 == 0 or (t_idx + 1) == args.n_trials:
                print(f"{t_idx+1} ", end="", flush=True)
        print("")

    return {
        "timestamp": datetime.now().isoformat(),
        "experiment_type": "utkface_single_point_convergence_appendix_style",
        "model": {
            "name": "iitolstykh/mivolo_v2",
            "model_dir": str(model_dir.resolve()),
            "processor_input_size": getattr(processor, "input_size", None),
            "with_persons_model": bool(getattr(model.config, "with_persons_model", False)),
        },
        "dataset": {
            "utk_dir": str(utk_dir.resolve()),
            "test_pool_size": int(len(idx_test)),
            "image_idx_local_test": int(local_idx),
            "sample_path": str(sample_path),
            "true_age": float(y_true),
            "clean_pred": float(clean_pred),
            "image_size": int(args.image_size),
            "dimension": int(d),
        },
        "part1": part1,
        "part2": part2,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UTKFace appendix-style single-point convergence analysis.")
    p.add_argument("--utk_dir", type=str, required=True)
    p.add_argument("--model_dir", type=str, default="models/mivolo_v2_hf")
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--image_idx", type=int, default=0)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--max_test_samples", type=int, default=500)
    p.add_argument("--sigma", type=float, default=0.06)
    p.add_argument("--eps_y", type=float, default=5.0)
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--N_values", nargs="+", type=int, default=[100, 500, 1000, 2000, 5000])
    p.add_argument("--n_trials", type=int, default=10)
    p.add_argument("--ground_truth_N", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument(
        "--output",
        type=str,
        default="outputs/utkface_single_point_convergence.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out = run_convergence(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()

