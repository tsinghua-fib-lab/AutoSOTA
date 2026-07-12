# run_grid_penalty.py  ― now with invariance‐error evaluation
import math, random
import numpy as np
import matplotlib.pyplot as plt
import torch, torch.optim as optim

from approx_so2_equiv_nn import (
    ApproxHarmonicInvariantMLP,
)

# ─────────────────────── deterministic setup ───────────────────────
import os

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)


# ───────────────────────── data ─────────────────────────
def generate_data(n_inner=100, n_outer=100, seed=1):
    rng = np.random.default_rng(seed)
    r_inner, r_outer_min, r_outer_max = 1.0, 2.3, 3.0
    theta_in = 2 * np.pi * rng.random(n_inner)
    rad_in = r_inner * np.sqrt(rng.random(n_inner))
    X_in = np.c_[rad_in * np.cos(theta_in), rad_in * np.sin(theta_in)]
    y_in = np.ones(n_inner)

    theta_out = 0.5 * np.pi * rng.random(n_outer) - 0.25 * np.pi
    rad_out = rng.uniform(r_outer_min, r_outer_max, n_outer)
    X_out = np.c_[rad_out * np.cos(theta_out), rad_out * np.sin(theta_out)]
    y_out = -np.ones(n_outer)

    X = np.vstack([X_in, X_out]).astype(np.float32)
    y = np.hstack([y_in, y_out]).astype(np.int64)
    return X, y


# ─────────────────────── training loop ───────────────────────
def train_once(model, X, y, λ_eq, λ_ne, *, epochs=20, lr=3e-3, device="cuda:0"):
    model.to(device)
    X_t = torch.tensor(X, device=device)
    y_t = torch.tensor((y > 0).astype(np.float32), device=device)[:, None]

    opt = optim.Adam(model.parameters(), lr=lr)
    bce = torch.nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        opt.zero_grad()
        logits = model(X_t).real
        loss = bce(logits, y_t)

        pen = model.compute_non_equivariance_penalty()
        loss = loss + λ_eq * pen["equivariant_part"] + λ_ne * pen["nonequiv_part"]
        loss.backward()
        opt.step()
    return model.eval()


# ─────────────────── empirical invariance test ───────────────────
@torch.no_grad()
def invariance_error(
    model, X: np.ndarray, *, n_rot: int = 30, n_batch: int = 256, device: str = "cuda:0"
):
    """
    Returns mean & max |f(Rα·x) − f(x)| over `n_rot` random rotations
    and `n_batch` random data points.
    """
    sel = np.random.choice(len(X), size=n_batch, replace=False)
    pts = torch.tensor(X[sel], device=device)
    base = model(pts).real  # (B,1)

    errs = []
    for _ in range(n_rot):
        α = 2 * math.pi * random.random()
        R = torch.tensor(
            [[math.cos(α), -math.sin(α)], [math.sin(α), math.cos(α)]], device=device
        )
        pts_rot = pts @ R.T
        out_rot = model(pts_rot).real
        errs.append((out_rot - base).abs())

    errs = torch.stack(errs)  # (n_rot,B,1)
    return errs.mean().item(), errs.max().item()


# ───────────────────────── plotting ─────────────────────────
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def plot_grid(models, X, y, titles, errors, lim=4.0, grid=300, n_lvls=8):
    xs = np.linspace(-lim, lim, grid)
    xx, yy = np.meshgrid(xs, xs)
    pts = torch.tensor(np.c_[xx.ravel(), yy.ravel()].astype(np.float32)).to("cuda:0")

    n_models = len(models)
    fig, axes = plt.subplots(
        1, n_models, figsize=(4.5 * n_models, 4.5), constrained_layout=True
    )
    axes = np.atleast_1d(axes)

    cmap = cm.get_cmap("coolwarm")

    for ax, net, ttl, (mean_err, max_err) in zip(axes, models, titles, errors):
        with torch.no_grad():
            score = net(pts).cpu().real.numpy().reshape(xx.shape)

        # ───────────────── Background region ─────────────────
        ax.contourf(
            xx,
            yy,
            np.sign(score),
            levels=[-1, 0, 1],
            colors=["#ffdddd", "#ddddff"],
            alpha=0.5,
        )

        # ───────────────── Decision boundary ─────────────────
        ax.contour(xx, yy, score, levels=[0], colors="black", linewidths=1.5)

        # ─────────────── Adaptive level sets ────────────────
        vmin, vmax = np.percentile(score, [5, 95])
        if abs(vmax - vmin) < 1e-2:  # handle near-constant outputs
            vmax = vmin + 1e-2

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        lvls = np.linspace(vmin, vmax, 2 * n_lvls + 1)[1:-1]
        ax.contour(
            xx,
            yy,
            score,
            levels=lvls,
            linewidths=1.0,
            cmap=cmap,
            norm=norm,
            alpha=0.9,
            zorder=1,
        )

        # ─────────────────── Data points ────────────────────
        pos, neg = y > 0, y < 0
        ax.scatter(X[pos, 0], X[pos, 1], c="blue", edgecolors="k", s=20, zorder=10)
        ax.scatter(
            X[neg, 0], X[neg, 1], c="red", edgecolors="k", marker="X", s=30, zorder=10
        )

        # ──────────────────── Styling ───────────────────────
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(ttl, fontsize=17, pad=10, y=-0.125)

        # ───── NEW: annotate invariance error + exponential ─────
        ax.text(
            0.5,
            -0.125,
            "$\mathcal{E}(T)$" + f" = {max_err:.2e}",
            ha="center",
            va="top",
            fontsize=17,
            transform=ax.transAxes,
        )

    plt.savefig(
        "figures/nn_approx_invariance.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(
        "figures/nn_approx_invariance.pdf", dpi=300, bbox_inches="tight"
    )
    plt.show()


# ───────────────────────── main ─────────────────────────
if __name__ == "__main__":
    X, y = generate_data(seed=3)

    # (label , λ_eq , λ_ne)
    combos = [
        ("$\lambda_G=1.0$, $λ_\perp=0.0001$", 1e0, 1e-3),
        ("$\lambda_G=0.001$, $λ_\perp=0.0001$", 1e-2, 1e-3),
        ("$\lambda_G=0.0001$, $λ_\perp=0.0001$", 1e-3, 1e-3),
        ("$\lambda_G=0.0001$, $λ_\perp=0.001$", 1e-3, 1e-2),
        ("$\lambda_G=0.0001$, $λ_\perp=1.0$", 1e-3, 1e0),
    ]

    trained, titles, errors = [], [], []
    for ttl, λeq, λne in combos:
        net = ApproxHarmonicInvariantMLP(M=4, C=4, hidden_c=8)
        net = train_once(net, X, y, λeq, λne, epochs=200).to("cuda:0")
        trained.append(net)
        titles.append(ttl)

        mean_err, max_err = invariance_error(net, X, n_rot=40, n_batch=len(X))
        errors.append((mean_err, max_err))
        print(f"{ttl:>20s}  mean|Δ|={mean_err:.3e}   max|Δ|={max_err:.3e}")

    # plot with errors
    plot_grid(trained, X, y, titles, errors)
