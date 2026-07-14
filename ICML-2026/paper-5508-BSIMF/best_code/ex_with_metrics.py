import argparse
import csv
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "core"))
from typing import Dict, List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

from make_data import get_kfold_loaders, get_train_val_loaders
from model import ContentUncertaintyDAG
from torch.utils.data import DataLoader, Subset

from train import (
    train,
    validate,
    em_update_model_mixture_from_loader,
    fit_x_pred_sigma_scale,
)


def _write_csv(rows: List[Dict[str, float]], out_path: str) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    if "epoch" in keys:
        keys = ["epoch"] + [k for k in keys if k != "epoch"]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def _plot_metric(history: Dict[str, List[Dict[str, float]]], key: str, out_path: str) -> None:
    train_rows = history.get("train", [])
    val_rows = history.get("val", [])
    if not train_rows and not val_rows:
        return

    def series(rows):
        xs, ys = [], []
        for r in rows:
            if key in r and r.get(key) is not None:
                xs.append(r.get("epoch", len(xs) + 1))
                ys.append(r[key])
        return xs, ys

    tx, ty = series(train_rows)
    vx, vy = series(val_rows)

    if not ty and not vy:
        return

    plt.figure()
    if ty:
        plt.plot(tx, ty, label="train")
    if vy:
        plt.plot(vx, vy, label="val")
    plt.xlabel("epoch")
    plt.ylabel(key)
    plt.title(key)
    plt.legend()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _choose_patch_size(image_size: int, max_patches: int = 1024) -> int:
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")

    divisors = set()
    i = 1
    while i * i <= image_size:
        if image_size % i == 0:
            divisors.add(i)
            divisors.add(image_size // i)
        i += 1

    candidates = sorted(divisors, reverse=True)
    for p in candidates:
        n = (image_size // p) ** 2
        if n <= max_patches:
            return p
    return 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, help="Output directory")
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--weight_decay", type=float)
    ap.add_argument("--seed", type=int)

    ap.add_argument("--n_splits", type=int, help="K for K-fold. If <2, uses holdout split.")
    ap.add_argument("--fold", type=int, help="Fold index for K-fold.")
    ap.add_argument("--val_frac", type=float, help="Holdout fraction if n_splits < 2")

    ap.add_argument("--num_workers", type=int)
    ap.add_argument("--flatten_conn", action="store_true", help="(Not supported) Flatten y. Model expects image y.")
    ap.add_argument("--upper_triangle_only", action="store_true", help="Only for flattened conn (ignored otherwise).")

    ap.add_argument("--z_dim", type=int)
    ap.add_argument("--u_dim", type=int, help="If -1, set u_dim = x_dim (required for continuous x).")
    ap.add_argument("--num_components", type=int)
    ap.add_argument("--vit_embed_dim", type=int)
    ap.add_argument("--vit_depth", type=int)
    ap.add_argument("--vit_num_heads", type=int)
    ap.add_argument("--vit_patch_size", type=int, help="If 0, auto-choose based on image size.")
    ap.add_argument("--max_patches", type=int, help="Used when auto-choosing vit_patch_size.")
    ap.add_argument("--rank_r", type=int)

    ap.add_argument(
        "--mask_prior_rho",
        type=float,
        help="Prior probability rho for m=1 (background/corruption). Smaller => sparser mask.",
    )
    ap.add_argument(
        "--learn_mask_prior",
        action="store_true",
        help="If set, learn per-pixel mask prior logits (overrides fixed --mask_prior_rho).",
    )
    ap.add_argument(
        "--mask_tv_lambda",
        type=float,
        help="Total-variation regularizer weight on mask probabilities g(y,z).",
    )
    ap.add_argument(
        "--mask_tv_samples",
        type=int,
        help="Number of z samples used to estimate the TV penalty (1 is usually enough).",
    )

    ap.add_argument("--sparse_m_lambda", type=float)
    ap.add_argument("--sparse_m_target", type=float)
    ap.add_argument(
        "--sparse_m_on",
        type=str,
        choices=["content", "mask", "background", "prior"],
        help="Where to apply sparsity penalty. 'prior' is an alias for 'mask'.",
    )

    ap.add_argument(
        "--drop_all_x_prob",
        type=float,
        help=(
            "During training, with this probability per subject, hide all of X from the encoder "
            "input (q(z,u|y)) but still score the X-likelihood on observed entries. "
            "This trains a meaningful X|Y predictive pathway."
        ),
    )

    ap.add_argument(
        "--sigma_calib_frac",
        type=float,
        help=(
            "Fraction of the TRAIN-FOLD used to fit a single scalar sigma scale for predictive intervals p(x|y). "
            "This subset is STILL used in normal training; it is only used after training to calibrate uncertainty."
        ),
    )
    ap.add_argument(
        "--sigma_calib_every",
        type=int,
        help=(
            "Re-fit sigma scale every N epochs and use it for validation-time x_pred_* interval metrics. "
            "Set to 0 to disable calibration (uses sigma_scale=1.0)."
        ),
    )
    ap.add_argument(
        "--sigma_calib_max_points",
        type=int,
        help="Max number of observed x entries used to fit the sigma scale (subsamples if larger).",
    )
    ap.add_argument(
        "--sigma_calib_target",
        type=float,
        help="Target central coverage for calibration (e.g., 0.90 for PICP90).",
    )

    ap.add_argument(
        "--refresh_mixture_every",
        type=int,
        help=(
            "If >0, run an EM-style mixture prior update every N epochs. "
            "(Set to 1 for true EM updates every epoch.)"
        ),
    )

    args = ap.parse_args()

    if args.flatten_conn:
        raise ValueError(
            "--flatten_conn is not supported for this model. "
            "The ViT encoder expects y as (B, 1, H, W)."
        )

    if args.sparse_m_on == "prior":
        args.sparse_m_on = "mask"
    if not (0.0 < float(args.mask_prior_rho) < 1.0):
        raise ValueError(f"--mask_prior_rho must be in (0,1), got {args.mask_prior_rho}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.n_splits >= 2:
        train_loader, val_loader, train_ds, val_ds, combined = get_kfold_loaders(
            batch_size=args.batch_size,
            n_splits=args.n_splits,
            fold_idx=args.fold,
            seed=args.seed,
            shuffle=True,
            num_workers=args.num_workers,
            flatten_conn=False,
            upper_triangle_only=False,
            return_subject_id=False,
            add_channel_dim=True,
        )
        split_name = f"kfold_{args.n_splits}_fold_{args.fold}"
    else:
        train_loader, val_loader, train_ds, val_ds, combined = get_train_val_loaders(
            batch_size=args.batch_size,
            val_frac=args.val_frac,
            seed=args.seed,
            shuffle=True,
            num_workers=args.num_workers,
            flatten_conn=False,
            upper_triangle_only=False,
            return_subject_id=False,
            add_channel_dim=True,
        )
        split_name = f"holdout_val_{args.val_frac:.3f}"

    out_dir = os.path.join(args.out_dir, split_name)
    os.makedirs(out_dir, exist_ok=True)

    calib_loader = None
    n_sigma_calib = 0
    if args.sigma_calib_every and args.sigma_calib_every > 0 and args.sigma_calib_frac > 0:
        n_train = int(len(train_ds))
        n_sigma_calib = max(1, int(round(float(args.sigma_calib_frac) * n_train)))
        rng = np.random.RandomState(int(args.seed) + 10_000 + int(args.fold))
        idx = rng.choice(np.arange(n_train), size=n_sigma_calib, replace=False)
        calib_ds = Subset(train_ds, idx.tolist())
        calib_loader = DataLoader(
            calib_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False,
            drop_last=False,
        )

    first_batch = next(iter(train_loader))
    if len(first_batch) == 3:
        x0, y0, _group0 = first_batch
    elif len(first_batch) == 2:
        x0, y0 = first_batch
    else:
        raise ValueError(f"Unexpected batch length from dataset: {len(first_batch)}")

    x_dim = int(x0.shape[1])
    if y0.ndim != 4:
        raise ValueError(f"Expected y to be (B, 1, H, W), got shape {tuple(y0.shape)}")
    if y0.shape[2] != y0.shape[3]:
        raise ValueError(f"Expected square y, got H={y0.shape[2]}, W={y0.shape[3]}")
    image_size = int(y0.shape[2])

    u_dim = args.u_dim if args.u_dim != -1 else x_dim
    if u_dim != x_dim:
        raise ValueError(
            f"For continuous x, the model requires u_dim == x_dim. Got u_dim={u_dim}, x_dim={x_dim}."
        )

    vit_patch_size = args.vit_patch_size
    if vit_patch_size == 0:
        vit_patch_size = _choose_patch_size(image_size, max_patches=args.max_patches)
    if image_size % vit_patch_size != 0:
        raise ValueError(f"vit_patch_size={vit_patch_size} must divide image_size={image_size}")

    model = ContentUncertaintyDAG(
        x_dim=x_dim,
        image_size=image_size,
        z_dim=args.z_dim,
        u_dim=u_dim,
        num_components=args.num_components,
        x_distribution="continuous",
        x_num_classes=None,
        vit_embed_dim=args.vit_embed_dim,
        vit_depth=args.vit_depth,
        vit_num_heads=args.vit_num_heads,
        vit_mlp_ratio=4.0,
        vit_dropout=0.0,
        mlp_hidden_dims=(128, 128),
        rank_r=args.rank_r,
        vit_patch_size=vit_patch_size,
        mask_prior_rho=args.mask_prior_rho,
        learn_mask_prior=args.learn_mask_prior,
        base_mean_mode="learnable_scalar",
        base_init_mean=0.1,
        base_var_mode="learnable_scalar",
        base_init_var=0.25**2,
        use_mask_x_in_encoder=True,
        u_missing_fallback_to_prior=True,
    ).to(device)

    mixture_param_names = {"pi_logits", "mu_components", "log_var_components"}
    for n, p in model.named_parameters():
        if n in mixture_param_names:
            p.requires_grad_(False)
    optim_params = [p for n, p in model.named_parameters() if n not in mixture_param_names]
    optimizer = torch.optim.AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay)

    history: Dict[str, List[Dict[str, float]]] = {"train": [], "val": []}

    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(
            {
                "split": split_name,
                "n_subjects_total": len(combined),
                "n_train": len(train_ds),
                "n_val": len(val_ds),
                "seed": args.seed,
                "image_size": image_size,
                "x_dim": x_dim,
                "z_dim": args.z_dim,
                "u_dim": u_dim,
                "num_components": args.num_components,
                "vit_patch_size": vit_patch_size,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "sigma_calib_frac": args.sigma_calib_frac,
                "sigma_calib_every": args.sigma_calib_every,
                "sigma_calib_max_points": args.sigma_calib_max_points,
                "sigma_calib_target": args.sigma_calib_target,
                "n_sigma_calib": n_sigma_calib,
            },
            f,
            indent=2,
        )

    best_val_elbo: Optional[float] = None

    for epoch in range(1, args.epochs + 1):
        tr = train(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            device=device,
            num_samples_z=4,
            num_samples_u=4,
            sparse_m_lambda=args.sparse_m_lambda,
            sparse_m_target=args.sparse_m_target,
            sparse_m_on=args.sparse_m_on,
            mask_tv_lambda=args.mask_tv_lambda,
            mask_tv_samples=args.mask_tv_samples,
            drop_all_x_prob=args.drop_all_x_prob,
        )

        if args.refresh_mixture_every and (epoch % args.refresh_mixture_every == 0):
            em_update_model_mixture_from_loader(
                model=model,
                loader=train_loader,
                device=device,
                n_steps=1,
                min_pi=0.05,
                var_floor=1e-4,
                use_q_var=True,
                reinit_dead=True,
                random_state=args.seed,
            )

        sigma_scale_info: Dict[str, float] = {}
        sigma_scale = 1.0
        if calib_loader is not None and args.sigma_calib_every and args.sigma_calib_every > 0:
            if epoch % int(args.sigma_calib_every) == 0:
                sigma_scale_info = fit_x_pred_sigma_scale(
                    model=model,
                    calib_loader=calib_loader,
                    device=device,
                    target_coverage=float(args.sigma_calib_target),
                    max_points=int(args.sigma_calib_max_points),
                    seed=int(args.seed) + 1_000_000 * int(args.fold) + int(epoch),
                )
                sigma_scale = float(sigma_scale_info.get("x_pred_sigma_scale", 1.0))
            else:
                if history["val"] and ("x_pred_sigma_scale" in history["val"][-1]):
                    sigma_scale = float(history["val"][-1]["x_pred_sigma_scale"])

        va = validate(
            epoch=epoch,
            model=model,
            val_loader=val_loader,
            device=device,
            num_samples_z=4,
            num_samples_u=4,
            sparse_m_lambda=args.sparse_m_lambda,
            sparse_m_target=args.sparse_m_target,
            sparse_m_on=args.sparse_m_on,
            mask_tv_lambda=args.mask_tv_lambda,
            mask_tv_samples=args.mask_tv_samples,
            drop_all_x_prob=args.drop_all_x_prob,
            x_pred_sigma_scale=sigma_scale,
        )

        if sigma_scale_info:
            va.update(sigma_scale_info)
        history["train"].append(tr)
        history["val"].append(va)

        val_elbo = va.get("elbo", None)
        if val_elbo is not None and np.isfinite(val_elbo):
            if best_val_elbo is None or val_elbo > best_val_elbo:
                best_val_elbo = float(val_elbo)
                torch.save(model.state_dict(), os.path.join(out_dir, "model_best.pt"))

        _write_csv(history["train"], os.path.join(out_dir, "metrics_train.csv"))
        _write_csv(history["val"], os.path.join(out_dir, "metrics_val.csv"))
        with open(os.path.join(out_dir, "metrics_history.json"), "w") as f:
            json.dump(history, f, indent=2)

    torch.save(model.state_dict(), os.path.join(out_dir, "model_final.pt"))

    keys_to_plot = [
        "elbo", "L_Y", "L_X", "KL_zc", "KL_u", "KL_m", "Sparse_M",
        "x_rmse", "x_r2", "x_ll_per_dim",
        "y_rmse_all", "y_rmse_weighted", "y_ll_mix_per_pixel",
        "cluster_acc", "cluster_ari", "cluster_acc_major", "cluster_bal_acc_major",
    ]
    for k in keys_to_plot:
        _plot_metric(history, k, os.path.join(out_dir, f"plot_{k}.png"))


if __name__ == "__main__":
    main()
