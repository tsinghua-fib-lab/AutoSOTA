import os
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Environment: piecewise mean
# -----------------------------
def simulate_env(T, taus, mus, sigma=1.0, seed=1):
    """
    taus: [tau0=1, tau1, ..., tauS, tau_{S+1}>=T+1]
    mus:  [mu0, mu1, ..., muS]  (S+1 means, one per segment)

    Output:
      X    : length-T array, X[t-1] corresponds to time t
      mu_t : length-T array, true mean at each time t
    """
    rng = np.random.default_rng(seed)
    mu_full = np.zeros(T + 1)  # use indices 1..T

    for j in range(len(mus)):
        start, end = taus[j], taus[j + 1]
        # numpy slice truncates automatically if end > T+1
        mu_full[start:end] = mus[j]

    mu_t = mu_full[1:]
    X = mu_t + sigma * rng.standard_normal(T)
    return X, mu_t


# -----------------------------
# Mixture illustration utilities
# -----------------------------
def build_piecewise_mean(T, taus, mus):
    mu_full = np.zeros(T + 1)
    for j in range(len(mus)):
        start, end = taus[j], taus[j + 1]
        mu_full[start:end] = mus[j]
    return mu_full[1:]
def normal_pdf(x, mu, sigma):
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def mixture_pdf(x, mus, weights, sigma):
    pdf = np.zeros_like(x, dtype=float)
    for m, w in zip(mus, weights):
        pdf += w * normal_pdf(x, m, sigma)
    return pdf

def plot_distribution_two_gaussians(ax, mu_a, mu_b, sigma):
    x_min = 0.0
    x_max = max(mu_a, mu_b) + 4.0 * sigma
    pad = 0.15 * (x_max - x_min + 1e-9)
    x_grid = np.linspace(x_min - pad, x_max + pad, 600)

    rng = np.random.default_rng(7)
    s1 = rng.normal(mu_a, sigma, size=4000)
    s2 = rng.normal(mu_b, sigma, size=4000)
    ax.hist(s1, bins=30, density=True, histtype="stepfilled", alpha=0.25, color="tab:green")
    ax.hist(s2, bins=30, density=True, histtype="stepfilled", alpha=0.25, color="tab:blue")
    ax.plot(x_grid, normal_pdf(x_grid, mu_a, sigma), color="tab:green", linewidth=2.0)
    ax.plot(x_grid, normal_pdf(x_grid, mu_b, sigma), color="tab:blue", linewidth=2.0)

    ax.set_xlim(0.0, 15.5)
    ax.grid(True, alpha=0.25)
    if ax.get_legend() is not None:
        ax.get_legend().remove()


def plot_distribution_mixture_vs_pure(ax, mus_mix, weights, mu_pure, sigma):
    x_min = 0.0
    x_max = max(max(mus_mix), mu_pure) + 4.0 * sigma
    pad = 0.15 * (x_max - x_min + 1e-9)
    x_grid = np.linspace(x_min - pad, x_max + pad, 600)

    rng = np.random.default_rng(9)
    n = 6000
    mix_choices = rng.choice([0, 1], size=n, p=weights)
    mix_means = np.where(mix_choices == 0, mus_mix[0], mus_mix[1])
    s_mix = rng.normal(mix_means, sigma)
    s_pure = rng.normal(mu_pure, sigma, size=n)
    ax.hist(s_mix, bins=30, density=True, histtype="stepfilled", alpha=0.25, color="tab:green")
    ax.hist(s_pure, bins=30, density=True, histtype="stepfilled", alpha=0.25, color="tab:blue")
    ax.plot(x_grid, mixture_pdf(x_grid, mus_mix, weights, sigma), color="tab:green", linewidth=2.0)
    ax.plot(x_grid, normal_pdf(x_grid, mu_pure, sigma), color="tab:blue", linewidth=2.0)

    ax.set_xlim(0.0, 15.5)
    ax.grid(True, alpha=0.25)
    if ax.get_legend() is not None:
        ax.get_legend().remove()


def plot_environment(ax, t, X, mu_t, taus, mus, stat=None, show_legend=False):
    t_steps = []
    mu_steps = []
    for j in range(len(mus)):
        start = taus[j]
        end = taus[j + 1]
        t_steps.extend([start, end])
        mu_steps.extend([mus[j], mus[j]])
    ax.scatter(t, X, s=10, alpha=0.35, color="tab:gray", label="observations")
    ax.plot(t_steps, mu_steps, linewidth=2.4, color="tab:blue", label="means")
    if stat is not None:
        ax.plot(t, stat, linewidth=2.0, color="tab:purple", label="SNR")
    for tau in taus[1:-1]:
        ax.axvline(tau, color="tab:green", linestyle=":", linewidth=1.6, alpha=0.6)
    ax.set_xlim(t[0], t[-1])
    y_min = min(mu_t.min(), 0.0 if stat is None else stat.min()) - 0.5
    y_max = max(mu_t.max(), 0.0 if stat is None else stat.max()) + 0.5
    ax.set_ylim(bottom=y_min, top=y_max)
    ax.grid(True, alpha=0.25)
    if show_legend:
        ax.legend(
            loc="lower right",
            bbox_to_anchor=(0.98, 0.02),
            fontsize=16,
            frameon=True,
        )


def compute_statistic_increasing(t, start_t, rate=1.0):
    stat = np.zeros_like(t, dtype=float)
    idx = t >= start_t
    stat[idx] = rate * np.sqrt(t[idx] - start_t + 1.0)
    return stat


def compute_statistic_rise_then_fall(t, start_t, drop_t, rate=1.0):
    stat = np.zeros_like(t, dtype=float)
    rise_idx = (t >= start_t) & (t <= drop_t)
    fall_idx = t > drop_t
    stat[rise_idx] = rate * np.sqrt(t[rise_idx] - start_t + 1.0)
    peak = rate * np.sqrt(drop_t - start_t + 1.0)
    decay = np.exp(-0.02 * (t[fall_idx] - drop_t))
    stat[fall_idx] = peak * decay
    return stat


# -----------------------------
# Main script
# -----------------------------
def main():
    # Instance
    T = 60
    sigma = 1.0

    out_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(out_dir, exist_ok=True)

    t = np.arange(1, T + 1)

    mu_pre = 3.0
    mu_shift = 10.0
    mu_pure = 5.0

    taus_left = [1, 20, T + 1]
    mus_left = [mu_pre, mu_shift]
    X_left, mu_t_left = simulate_env(T, taus_left, mus_left, sigma=sigma, seed=1)

    taus_right = [1, 20, 40, T + 1]
    mus_right = [mu_pre, mu_shift, mu_pure]
    X_right, mu_t_right = simulate_env(T, taus_right, mus_right, sigma=sigma, seed=2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    stat_left = compute_statistic_increasing(t, start_t=20, rate=1.3)
    stat_right = compute_statistic_rise_then_fall(t, start_t=20, drop_t=40, rate=1.3)

    # Top left: single change at tau=20
    plot_environment(
        axes[0, 0],
        t,
        X_left,
        mu_t_left,
        taus_left,
        mus_left,
        stat=stat_left,
        show_legend=True,
    )
    # Top right: distribution (was bottom left)
    plot_distribution_two_gaussians(axes[0, 1], mu_pre, mu_shift, sigma)

    # Bottom left: environment with two changes (was top right)
    plot_environment(
        axes[1, 0],
        t,
        X_right,
        mu_t_right,
        taus_right,
        mus_right,
        stat=stat_right,
        show_legend=False,
    )

    # Bottom right: mixture vs pure
    plot_distribution_mixture_vs_pure(
        axes[1, 1], [mu_pre, mu_shift], [0.5, 0.5], mu_pure, sigma
    )

    fig.tight_layout()
    out_path = os.path.join(out_dir, "mixture_illustration_3x2.png")
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
