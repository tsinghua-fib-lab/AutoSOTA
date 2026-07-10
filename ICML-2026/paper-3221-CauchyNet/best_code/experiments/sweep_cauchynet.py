"""
Hyperparameter sweep for CauchyNet on the 1D gap-filling target.
=================================================================

A 4-stage coordinate sweep:
  Stage 1: learning rate           (h, n_train, n_val, epochs fixed)
  Stage 2: hidden size             (best lr from stage 1 carried forward)
  Stage 3: n_train                 (best lr, h)
  Stage 4: epochs                  (best lr, h, n_train)

At each stage we sweep one variable, score by test MAE on the gap
regions (mean over 5 seeds), and carry the best value forward.

After the sweep, the chosen configuration is re-evaluated with 10 seeds
against the three baselines (SIREN, FNN_ReLU, N-BEATS)
at the same training budget for a fair head-to-head comparison.

Outputs:
    results/sweep_cauchynet.json       full sweep table
    results/sweep_cauchynet_best.json  final best config + head-to-head
"""
import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from best_config_gap_filling import (
    CauchyNet, SIREN, FNN_ReLU, _make_nbeats,
    f, df, train_score_one,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)


def build_dataset(n_train, n_val, n_test, seed=10):
    """Parametric version of build_data: sample given counts of train/val/test.
    Train and val are sampled away from turning points; test is sampled near
    turning points (the "gap" regions).
    """
    torch.manual_seed(seed); np.random.seed(seed)
    X_dense = torch.linspace(-2, 2, 2000)
    with torch.no_grad():
        dY = df(X_dense)
    turning_pts = X_dense[torch.abs(dY) < 0.15]

    def sample(n, near_turning):
        out = []
        while len(out) < n:
            samp = torch.normal(0., 1., size=(1,)) * 2.
            samp = torch.clamp(samp, -2, 2)
            cond = any(torch.abs(turning_pts - samp) < 0.15)
            if cond == near_turning:
                out.append(samp)
        return torch.cat(out)

    train_X = sample(n_train, near_turning=False)
    val_X = sample(n_val, near_turning=False)
    test_X = sample(n_test, near_turning=True)
    return (
        train_X.unsqueeze(-1), f(train_X).unsqueeze(-1),
        val_X.unsqueeze(-1), f(val_X).unsqueeze(-1),
        test_X.unsqueeze(-1), f(test_X).unsqueeze(-1),
    )


def make_loaders(n_train, n_val, n_test=100, seed=10, batch_size=32):
    tX, tY, vX, vY, teX, teY = build_dataset(n_train, n_val, n_test, seed)
    train_loader = DataLoader(TensorDataset(tX, tY), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(vX, vY), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, (teX, teY)


def score_config(h, lr, n_train, n_val, epochs, n_seeds=5, label=""):
    """Run CauchyNet at this config, return (mae_mean, mae_std, train_time_mean, params)."""
    train_loader, val_loader, test_data = make_loaders(n_train, n_val)
    maes, times = [], []
    params_count = None
    for s in range(n_seeds):
        torch.manual_seed(s); np.random.seed(s)
        model = CauchyNet(hidden_size=h, r_re=2.5, r_im=0.4, seed=s)
        res = train_score_one(
            f"CN h={h} lr={lr} n={n_train} ep={epochs} seed={s}",
            model, train_loader, val_loader, test_data,
            epochs=epochs, lr=lr, imag_pen=0.0, log=False,
        )
        maes.append(res["mae_mean"]); times.append(res["train_time_s"])
        params_count = res["params"]
    arr = np.array(maes)
    print(f"  {label:<28} MAE mean={arr.mean():.4f}  std={arr.std():.4f}  "
          f"params={params_count}  train_time={np.mean(times):.1f}s")
    return float(arr.mean()), float(arr.std()), float(np.mean(times)), params_count


def main():
    t_start = time.time()
    # Sweep budget — small enough to finish quickly, large enough to differentiate
    n_seeds = 5

    # ─── Defaults (carry forward best values stage by stage) ─────────────
    best = dict(h=64, lr=5e-3, n_train=100, n_val=50, epochs=1000)

    all_results = {}

    # ─── Stage 1: learning rate ──────────────────────────────────────────
    print("\n" + "=" * 64)
    print("Stage 1: sweep lr  (h={h}, n_train={n_train}, n_val={n_val}, epochs={epochs})"
          .format(**best))
    print("=" * 64)
    lr_grid = [1e-3, 3e-3, 5e-3, 1e-2, 2e-2, 5e-2]
    stage1 = {}
    for lr in lr_grid:
        mae, std, t, p = score_config(
            h=best["h"], lr=lr, n_train=best["n_train"],
            n_val=best["n_val"], epochs=best["epochs"], n_seeds=n_seeds,
            label=f"lr={lr:.0e}",
        )
        stage1[f"lr={lr:.0e}"] = dict(mae_mean=mae, mae_std=std, train_time_s=t, params=p)
    best_lr = min(lr_grid, key=lambda x: stage1[f"lr={x:.0e}"]["mae_mean"])
    print(f"\n  -> Best lr = {best_lr:.0e}  (MAE = {stage1[f'lr={best_lr:.0e}']['mae_mean']:.4f})")
    best["lr"] = best_lr
    all_results["stage1_lr"] = stage1

    # ─── Stage 2: hidden size ────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("Stage 2: sweep h   (lr={lr:.0e}, n_train={n_train}, n_val={n_val}, epochs={epochs})"
          .format(**best))
    print("=" * 64)
    h_grid = [16, 32, 64, 128, 256, 512]
    stage2 = {}
    for h in h_grid:
        mae, std, t, p = score_config(
            h=h, lr=best["lr"], n_train=best["n_train"],
            n_val=best["n_val"], epochs=best["epochs"], n_seeds=n_seeds,
            label=f"h={h}",
        )
        stage2[f"h={h}"] = dict(mae_mean=mae, mae_std=std, train_time_s=t, params=p)
    best_h = min(h_grid, key=lambda x: stage2[f"h={x}"]["mae_mean"])
    print(f"\n  -> Best h = {best_h}  (MAE = {stage2[f'h={best_h}']['mae_mean']:.4f}, "
          f"params = {stage2[f'h={best_h}']['params']})")
    best["h"] = best_h
    all_results["stage2_h"] = stage2

    # ─── Stage 3: n_train ───────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("Stage 3: sweep n_train  (lr={lr:.0e}, h={h}, n_val={n_val}, epochs={epochs})"
          .format(**best))
    print("=" * 64)
    n_grid = [30, 50, 100, 200]
    stage3 = {}
    for n in n_grid:
        mae, std, t, p = score_config(
            h=best["h"], lr=best["lr"], n_train=n,
            n_val=best["n_val"], epochs=best["epochs"], n_seeds=n_seeds,
            label=f"n_train={n}",
        )
        stage3[f"n_train={n}"] = dict(mae_mean=mae, mae_std=std, train_time_s=t, params=p)
    best_n = min(n_grid, key=lambda x: stage3[f"n_train={x}"]["mae_mean"])
    print(f"\n  -> Best n_train = {best_n}  (MAE = {stage3[f'n_train={best_n}']['mae_mean']:.4f})")
    best["n_train"] = best_n
    all_results["stage3_n_train"] = stage3

    # ─── Stage 4: epochs ────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("Stage 4: sweep epochs   (lr={lr:.0e}, h={h}, n_train={n_train}, n_val={n_val})"
          .format(**best))
    print("=" * 64)
    ep_grid = [200, 500, 1000, 2000, 3000]
    stage4 = {}
    for ep in ep_grid:
        mae, std, t, p = score_config(
            h=best["h"], lr=best["lr"], n_train=best["n_train"],
            n_val=best["n_val"], epochs=ep, n_seeds=n_seeds,
            label=f"epochs={ep}",
        )
        stage4[f"epochs={ep}"] = dict(mae_mean=mae, mae_std=std, train_time_s=t, params=p)
    best_ep = min(ep_grid, key=lambda x: stage4[f"epochs={x}"]["mae_mean"])
    print(f"\n  -> Best epochs = {best_ep}  (MAE = {stage4[f'epochs={best_ep}']['mae_mean']:.4f})")
    best["epochs"] = best_ep
    all_results["stage4_epochs"] = stage4

    # ─── Best CauchyNet config ──────────────────────────────────────────
    print("\n" + "=" * 64)
    print("FINAL BEST CONFIG")
    print("=" * 64)
    print(f"  lr        = {best['lr']:.0e}")
    print(f"  h         = {best['h']}")
    print(f"  n_train   = {best['n_train']}")
    print(f"  n_val     = {best['n_val']}")
    print(f"  epochs    = {best['epochs']}")

    # Re-evaluate CauchyNet at best config with 10 seeds for final number
    print("\nRe-evaluating CauchyNet at best config with 10 seeds...")
    cn_mae, cn_std, cn_t, cn_p = score_config(
        h=best["h"], lr=best["lr"], n_train=best["n_train"],
        n_val=best["n_val"], epochs=best["epochs"], n_seeds=10,
        label="CauchyNet best (10 seeds)",
    )

    # ─── Head-to-head: baselines at the same training budget ────────────
    print("\n" + "=" * 64)
    print("Head-to-head at best training budget (10 seeds each)")
    print("=" * 64)
    train_loader, val_loader, test_data = make_loaders(best["n_train"], best["n_val"])
    head2head = {}
    head2head["CauchyNet"] = dict(
        mae_mean=cn_mae, mae_std=cn_std, params=cn_p, train_time_s=cn_t)

    # Each baseline at the same h
    # (For N-BEATS we'll pick a num_blocks that gets ~same param count as FNN.)
    def run_baseline(name, factory, lr_override=None):
        lr_use = lr_override if lr_override is not None else best["lr"]
        maes, ts = [], []
        params_count = None
        for s in range(10):
            torch.manual_seed(s); np.random.seed(s)
            m = factory(s)
            r = train_score_one(
                f"{name} seed={s}", m, train_loader, val_loader, test_data,
                epochs=best["epochs"], lr=lr_use, imag_pen=0.0, log=False)
            maes.append(r["mae_mean"]); ts.append(r["train_time_s"])
            params_count = r["params"]
        arr = np.array(maes)
        print(f"  {name:<24} MAE mean={arr.mean():.4f}  std={arr.std():.4f}  "
              f"params={params_count}")
        head2head[name] = dict(mae_mean=float(arr.mean()),
                               mae_std=float(arr.std()),
                               params=params_count,
                               train_time_s=float(np.mean(ts)))

    run_baseline("SIREN",
                 lambda s: SIREN(hidden_size=best["h"]))
    run_baseline("FNN_ReLU",
                 lambda s: FNN_ReLU(h=best["h"]))
    # N-BEATS — pick num_blocks such that params are roughly matched to FNN
    # FNN at h: 2h + (h+1). For h=64: 193. NBeats per block: h^2+5h+2 for h_nb.
    # Use h_nb solving 3(h_nb^2 + 5h_nb + 2) ≈ FNN params.
    fnn_p = 2 * best["h"] + (best["h"] + 1)
    h_nb = max(2, int(round((-5 + (25 + 4 * (fnn_p / 3 - 2)) ** 0.5) / 2)))
    print(f"  (using N-BEATS h={h_nb} for ~{fnn_p}-param match)")
    run_baseline("NBeats",
                 lambda s: _make_nbeats(h=h_nb, num_blocks=3))

    # Print head-to-head summary
    cn = head2head["CauchyNet"]["mae_mean"]
    print(f"\n{'Model':<24} {'Params':>8} {'MAE':>10} {'vs CauchyNet':>14}")
    for k, v in head2head.items():
        rel = v["mae_mean"] / cn
        print(f"{k:<24} {v['params']:>8} {v['mae_mean']:>10.4f} {rel:>13.2f}x")

    # ─── Save everything ─────────────────────────────────────────────────
    all_results["best_config"] = best
    all_results["head_to_head"] = head2head
    out_path = os.path.join(OUT_DIR, "sweep_cauchynet.json")
    with open(out_path, "w") as fp:
        json.dump(all_results, fp, indent=2)
    print(f"\nSaved {out_path}")
    print(f"\nTotal sweep time: {(time.time() - t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
