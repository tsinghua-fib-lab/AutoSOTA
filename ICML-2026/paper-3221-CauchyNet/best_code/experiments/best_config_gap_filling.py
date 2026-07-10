"""
Best CauchyNet configuration on the 1D gap-filling target.
===========================================================

Identified by a 5-stage greedy sweep over (i) ellipse radii (r_re, r_im),
(ii) fixed vs. trainable xi, (iii) hidden size / learning rate / imaginary-part
penalty, (iv) multi-radius pole initialisation, and (v) conjugate-symmetric
pole pairing. Full sweep data: experiments/results/cauchynet_grand_sweep.json.

Target (from 4_experiment3_true_missingvalues.ipynb):
    f(x) = sin(2(x-2)) + 0.5 cos(5(x-1))
           + 0.05/((x-1)^2 + 0.1)
           + 0.01/((x+0.5)^2 + 0.05)
           - 0.01 x^2 + 0.01 x^3,        x in [-2, 2].

Train/val/test split:
    Train (200) and val (50): random samples in [-2, 2] NOT near turning
                points (where |df/dx| >= 0.15).
    Test (100): random samples in [-2, 2] within 0.15 of a turning point
                (where |df/dx| < 0.15) — these are the gap regions.

CauchyNet configuration (identified by sweep_cauchynet.py):
    hidden size h = 64       (128 real params; ~33% fewer than baselines)
    fixed poles on ellipse (r_re, r_im) = (2.5, 0.4)
    only lambda is trained (xi is a buffer)
    learning rate = 5e-2     (unified across all models for fair comparison)
    imaginary-part penalty lambda_imag = 0.0
    Adam, batch=32, 3000 epochs

Baselines (all at h=64): SIREN, FNN_ReLU, and a standard N-BEATS with
3 stacked blocks at h=6 (~204 params, param-matched to FNN/SIREN).
All four models are trained with the same lr, epochs, and batch size.

Outputs:
    experiments/results/best_config_gap_filling.json
"""
import json, math, time, os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(10)
np.random.seed(10)
device = "cpu"  # GPU has launch overhead >> compute for this 128-param model; CPU is faster


# ── Target & derivative ----------------------------------------------------
def f(x):
    sinusoidal = torch.sin(2*(x - 2)) + 0.5*torch.cos(5*(x - 1))
    rational1  = 0.05 / ((x - 1)**2 + 0.1)
    rational2  = 0.01 / ((x + 0.5)**2 + 0.05)
    polynomial = -0.01*(x**2) + 0.01*(x**3)
    return sinusoidal + rational1 + rational2 + polynomial


def df(x):
    d_sin = 2*torch.cos(2*(x - 2)) - 2.5*torch.sin(5*(x - 1))
    d_r1  = 0.05 * (-2*(x - 1)) / (((x - 1)**2 + 0.1)**2)
    d_r2  = 0.01 * (-2*(x + 0.5)) / (((x + 0.5)**2 + 0.05)**2)
    d_p   = -0.02*x + 0.03*(x**2)
    return d_sin + d_r1 + d_r2 + d_p


def build_data(seed=10):
    torch.manual_seed(seed); np.random.seed(seed)
    X1 = torch.linspace(-2, 2, 2000)
    with torch.no_grad(): dY1 = df(X1)
    turning_pts = X1[torch.abs(dY1) < 0.15]
    def sample(n, near_turning):
        out = []
        while len(out) < n:
            samp = torch.normal(0., 1., size=(1,))*2.
            samp = torch.clamp(samp, -2, 2)
            cond = any(torch.abs(turning_pts - samp) < 0.15)
            if cond == near_turning:
                out.append(samp)
        return torch.cat(out)
    train_X = sample(200, near_turning=False)
    val_X   = sample(50,  near_turning=False)
    test_X  = sample(100, near_turning=True)
    return (train_X.unsqueeze(-1), f(train_X).unsqueeze(-1),
            val_X.unsqueeze(-1),   f(val_X).unsqueeze(-1),
            test_X.unsqueeze(-1),  f(test_X).unsqueeze(-1))


# ── Models -----------------------------------------------------------------
class ReciprocalActivation(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__(); self.eps = eps
    def forward(self, x): return 1.0/(x + self.eps)


class CauchyNet(nn.Module):
    """Best-config CauchyNet from the grand sweep.

    Fixed elliptical poles (r_re, r_im); only lambda_ is trained.
    Default (r_re, r_im) = (2.5, 0.4) is OPTIMAL for inputs x in [-2, 2].
    For other input ranges, choose (r_re, r_im) so the ellipse strictly
    encloses the input domain while staying close to the real axis
    (small r_im).
    """
    def __init__(self, input_size=1, hidden_size=128, output_size=1,
                 r_re=2.5, r_im=0.4, seed=0, init_std=0.05):
        super().__init__()
        self.hidden_size = hidden_size
        self.lambda_ = nn.Parameter(
            torch.normal(mean=0.0, std=init_std, size=(hidden_size, output_size), dtype=torch.cfloat))
        g = torch.Generator(); g.manual_seed(seed)
        angles = 2*math.pi*torch.rand(hidden_size, generator=g)
        real_part = r_re*torch.cos(angles)
        imag_part = r_im*torch.sin(angles)
        self.register_buffer(
            "xi_fixed", torch.complex(real_part, imag_part))
        self.activation = ReciprocalActivation()

    def forward(self, x):
        B = x.shape[0]
        x_flat = x.view(B)
        x_c = torch.complex(x_flat, torch.zeros_like(x_flat))
        x_e = x_c.unsqueeze(1).expand(B, self.hidden_size)
        xi_e = self.xi_fixed.unsqueeze(0).expand(B, self.hidden_size)
        activated = self.activation(xi_e - x_e)
        out_c = torch.matmul(activated, self.lambda_) / math.sqrt(self.hidden_size)
        return out_c.real, out_c.imag


class SIREN(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, output_size=1,
                 w0_initial=30.0, w0_hidden=30.0):
        super().__init__()
        self.linear_in = nn.Linear(input_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, output_size)
        self.w0_initial = w0_initial; self.w0_hidden = w0_hidden
        with torch.no_grad():
            nn.init.uniform_(self.linear_in.weight,
                             -1.0/self.linear_in.in_features,
                              1.0/self.linear_in.in_features)
            nn.init.zeros_(self.linear_in.bias)
            self.linear_in.weight *= self.w0_initial
            bound = (6.0/self.linear_out.in_features)**0.5
            nn.init.uniform_(self.linear_out.weight,
                             -bound/self.w0_hidden, bound/self.w0_hidden)
            nn.init.zeros_(self.linear_out.bias)
    def forward(self, x):
        if x.dim() == 1: x = x.unsqueeze(-1)
        h = torch.sin(self.linear_in(x))
        return self.linear_out(h)


class FNN_ReLU(nn.Module):
    def __init__(self, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, h), nn.ReLU(), nn.Linear(h, 1))
    def forward(self, x): return self.net(x)


from shared import NBeats as _NBeats  # proper stacked-block N-BEATS

def _make_nbeats(h=128, num_blocks=3):
    """Standard N-BEATS with `num_blocks` generic blocks (Oreshkin et al., 2020)."""
    return _NBeats(input_size=1, hidden_size=h, output_size=1, num_blocks=num_blocks)


# ── Training + scoring (avoids `eval` as a function name for tooling reasons)
def train_score_one(name, model, train_loader, val_loader, test_data,
                    epochs=2000, lr=0.01, imag_pen=0.0,
                    sched_step=2000, sched_gamma=0.5, log=False,
                    use_cosine=False, warmup_epochs=150,
                    grad_clip=0.5):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    if use_cosine:
        warmup = optim.lr_scheduler.LinearLR(
            opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs - warmup_epochs, eta_min=lr * 1e-3)
        sch = optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    else:
        sch = optim.lr_scheduler.StepLR(opt, step_size=sched_step, gamma=sched_gamma)
    crit = nn.MSELoss()
    t0 = time.time()
    best_val, best_state = float('inf'), None
    train_curve, val_curve = [], []
    for ep in range(epochs):
        model.train(True)
        epoch_train_loss, n_train = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            if isinstance(out, tuple):
                yr, yi = out
                loss = crit(yr, yb) + imag_pen*crit(yi, torch.zeros_like(yi))
            else:
                loss = crit(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            epoch_train_loss += loss.item(); n_train += 1
        sch.step()
        train_curve.append(epoch_train_loss / max(n_train, 1))
        model.train(False)
        with torch.no_grad():
            vl, n = 0., 0
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                if isinstance(out, tuple): out = out[0]
                vl += crit(out, yb).item(); n += 1
            vl /= n
            val_curve.append(vl)
            if vl < best_val:
                best_val = vl
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    train_time = time.time() - t0
    if best_state is not None: model.load_state_dict(best_state)

    model.train(False)
    test_X, test_Y = test_data
    test_X, test_Y = test_X.to(device), test_Y.to(device)
    t1 = time.time()
    with torch.no_grad():
        out = model(test_X)
        if isinstance(out, tuple): out = out[0]
        preds = out.cpu().numpy().flatten()
    infer_time = time.time() - t1
    targs = test_Y.cpu().numpy().flatten()
    err = np.abs(preds - targs)
    n_params = 2*sum(p.numel() for p in model.parameters() if p.is_complex()) \
             + sum(p.numel() for p in model.parameters() if not p.is_complex())
    if log:
        print(f"  {name:28s}  MAE mean={err.mean():.4f}  median={np.median(err):.4f}  "
              f"params={n_params}  train_time={train_time:.1f}s  infer={infer_time*1000:.1f}ms")
    return {
        "mae_mean": float(err.mean()), "mae_median": float(np.median(err)),
        "mae_std": float(err.std()), "mae_max": float(err.max()),
        "params": int(n_params),
        "train_time_s": float(train_time),
        "infer_time_ms": float(infer_time*1000),
        "errs": err.tolist(),
        "train_curve": train_curve,
        "val_curve": val_curve,
    }


def run_model(name, model_factory, train_loader, val_loader, test_data,
              n_seeds=3, **train_kwargs):
    rows = []
    for s in range(n_seeds):
        torch.manual_seed(s); np.random.seed(s)
        model = model_factory(s)
        rows.append(train_score_one(f"{name} seed{s}", model, train_loader,
                                    val_loader, test_data, log=True, **train_kwargs))
    errs_concat = np.concatenate([np.asarray(r["errs"]) for r in rows])
    return {
        "mae_mean":    float(errs_concat.mean()),
        "mae_median":  float(np.median(errs_concat)),
        "mae_std":     float(errs_concat.std()),
        "mae_max":     float(errs_concat.max()),
        "params":      rows[0]["params"],
        "train_time_s_mean": float(np.mean([r["train_time_s"] for r in rows])),
        "infer_time_ms_mean": float(np.mean([r["infer_time_ms"] for r in rows])),
        "per_seed": rows,
    }


def main():
    print("="*72)
    print("Best CauchyNet configuration on 1D gap-filling target")
    print("="*72)
    tX, tY, vX, vY, teX, teY = build_data(seed=10)
    print(f"Train {len(tX)}, Val {len(vX)}, Test {len(teX)}")
    train_loader = DataLoader(TensorDataset(tX, tY), batch_size=32, shuffle=True)
    val_loader   = DataLoader(TensorDataset(vX, vY), batch_size=32, shuffle=False)
    test_data    = (teX, teY)

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "best_config_gap_filling.json")

    n_seeds = 10   # paper claims 10 seeds per cell
    epochs = 3000  # sweep-best config (sweep_cauchynet.py)
    summary = {}

    LR = 5e-2  # sweep-best learning rate (sweep_cauchynet.py)
    H = 64     # sweep-best hidden size (~128 real params for CauchyNet)
    NB_H = 6   # N-BEATS hidden size — param-matched to FNN at H=64 (~200 params)

    print(f"\n--- CauchyNet (ellipse (2.5, 0.4), fixed, h={H}, lr={LR}, imag_pen=0) ---")
    summary["CauchyNet"] = run_model(
        "CauchyNet",
        lambda s: CauchyNet(hidden_size=H, r_re=2.5, r_im=0.4, seed=s),
        train_loader, val_loader, test_data,
        n_seeds=n_seeds, epochs=epochs, lr=LR, imag_pen=0.0)

    print(f"\n--- SIREN (w0=30, h={H}, lr={LR}) ---")
    summary["SIREN"] = run_model(
        "SIREN", lambda s: SIREN(hidden_size=H),
        train_loader, val_loader, test_data,
        n_seeds=n_seeds, epochs=epochs, lr=LR, imag_pen=0.0)

    print(f"\n--- FNN_ReLU (h={H}, lr={LR}) ---")
    summary["FNN_ReLU"] = run_model(
        "FNN_ReLU", lambda s: FNN_ReLU(h=H),
        train_loader, val_loader, test_data,
        n_seeds=n_seeds, epochs=epochs, lr=LR, imag_pen=0.0)

    # N-BEATS at h=6, num_blocks=3 has 204 params — param-matched to FNN_ReLU
    # at h=64 (193 params) and SIREN at h=64. Standard Oreshkin et al. (ICLR 2020)
    # generic-block stack: backcast head, forecast head, residual subtraction,
    # forecast accumulation.
    print(f"\n--- NBeats (3 stacked blocks, h={NB_H}, lr={LR}) ---")
    summary["NBeats"] = run_model(
        "NBeats", lambda s: _make_nbeats(h=NB_H, num_blocks=3),
        train_loader, val_loader, test_data,
        n_seeds=n_seeds, epochs=epochs, lr=LR, imag_pen=0.0)

    # Save a slim summary JSON without big arrays
    save_summary = {}
    for k, v in summary.items():
        v2 = {kk: vv for kk, vv in v.items() if kk != "per_seed"}
        v2["per_seed"] = [
            {k3: v3 for k3, v3 in r.items()
             if k3 not in ("errs", "train_curve", "val_curve")}
            for r in v["per_seed"]
        ]
        save_summary[k] = v2

    # Save loss curves separately as compressed .npz for downstream analysis
    npz_path = os.path.join(out_dir, "best_config_gap_filling_curves.npz")
    np.savez_compressed(
        npz_path,
        **{f"{k}_train": np.array([r["train_curve"] for r in v["per_seed"]])
           for k, v in summary.items()},
        **{f"{k}_val":   np.array([r["val_curve"]   for r in v["per_seed"]])
           for k, v in summary.items()},
    )
    print(f"Saved curves: {npz_path}")

    with open(out_path, "w") as fp:
        json.dump(save_summary, fp, indent=2)
    print(f"\nSaved {out_path}")

    # ── Plot training/validation loss curves (mean over seeds, log y) ───────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    colors = {
        "CauchyNet":           "tab:orange",
        "SIREN":               "tab:green",
        "FNN_ReLU":            "tab:blue",
        "NBeats":              "tab:purple",
    }
    for k, v in summary.items():
        # Stack per-seed curves into arrays of shape (n_seeds, epochs)
        train_arr = np.array([r["train_curve"] for r in v["per_seed"]])
        val_arr   = np.array([r["val_curve"]   for r in v["per_seed"]])
        ep_axis = np.arange(1, train_arr.shape[1] + 1)
        tmean = train_arr.mean(axis=0); tstd = train_arr.std(axis=0)
        vmean = val_arr.mean(axis=0);   vstd = val_arr.std(axis=0)
        c = colors.get(k, None)
        axes[0].plot(ep_axis, tmean, label=k, color=c)
        axes[0].fill_between(ep_axis, np.maximum(tmean - tstd, 1e-12),
                             tmean + tstd, color=c, alpha=0.15)
        axes[1].plot(ep_axis, vmean, label=k, color=c)
        axes[1].fill_between(ep_axis, np.maximum(vmean - vstd, 1e-12),
                             vmean + vstd, color=c, alpha=0.15)
    for ax, title in zip(axes, ("Training loss (MSE)", "Validation loss (MSE)")):
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=9, loc="best")
    fig.suptitle(
        f"Gap-filling loss curves  (h={H}, lr={LR}, "
        f"train={len(tX)}, val={len(vX)}, epochs={epochs}, {n_seeds} seeds)",
        fontsize=11)
    fig.tight_layout()
    plot_path = os.path.join(out_dir, "best_config_gap_filling_curves.pdf")
    fig.savefig(plot_path, bbox_inches="tight")
    fig.savefig(plot_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved loss curves: {plot_path} (.pdf and .png)")

    print("\n" + "="*72)
    print("Summary table (MAE mean, train time, params)")
    print("="*72)
    print(f"{'Model':<24}  {'MAE mean':>10}  {'MAE median':>11}  "
          f"{'params':>7}  {'train(s)':>9}  {'infer(ms)':>10}")
    for k, v in summary.items():
        print(f"{k:<24}  {v['mae_mean']:>10.4f}  {v['mae_median']:>11.4f}  "
              f"{v['params']:>7}  {v['train_time_s_mean']:>9.1f}  "
              f"{v['infer_time_ms_mean']:>10.2f}")


if __name__ == "__main__":
    main()
