"""Experiment registry and the single unified driver for the RCDP-UCB paper.

Every experiment is declared once in ``EXPERIMENTS``; ``main.py`` is the only
user-facing entry point. The driver (``run_mapping`` / ``reproduce``) builds the
environment and learners, runs ``n_runs`` independently seeded repetitions, and
writes a regret plot + a tidy CSV to ``results/<experiment>/``.

Reproducibility: every (learner, run) rebuilds a freshly seeded environment, so
each method sees the identical environment realisation — the comparison isolates
the algorithm, not the randomness.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from contextual_dueling_bandit import (
    ContextualDuelingBanditEnv, DuelingGLMLearner,
    RCDBLearner, ColSTIMLearner, MaxInPLearner, MaxPairUCBLearner,
    RCDBPostServingLearner, ColSTIMPostServingLearner,
    MaxInPPostServingLearner, MaxPairUCBPostServingLearner,
    StrategicDelay, StochasticGaussianDelay, StrategicOutcomeCorruption,
    run_simulation,
)

OURS = "RCDP-UCB (Ours)"

# Fixed worst-case robustness provisioning (paper, App. D): the learner is
# always provisioned for the hardest setting, so the curves show pure
# environmental impact rather than per-setting tuning.
FIXED_C, FIXED_LAMBDA, FIXED_MU_TAU, KAPPA = 25.0, 10000.0, 100.0, 0.25

_PHI_ORDER = ["polynomial", "abs", "cosine", "sinusoidal"]   # seed indexing (do not reorder)
_COLORS = {"Ours": "#1f77b4", "RCDB": "#2ca02c", "ColSTIM": "#ff7f0e",
           "MaxInP": "#9467bd", "MaxPair": "#d62728"}


def post_dim(phi, d):
    """Natural post-serving dimension d_y of the mapping phi (paper Sec. 6.1):
    sinusoidal -> [cos(x), sin(x)] and polynomial -> [x^2, sqrt|x|] are 2*d_x
    dimensional; abs/cosine/piecewise are d_x; interaction is d_x(d_x+1)/2.
    Using the exact d_y avoids truncating away half of the mapping."""
    if phi in ("sinusoidal", "polynomial"):
        return 2 * d
    if phi == "interaction":
        return d * (d + 1) // 2
    return d


# --------------------------------------------------------------------------- #
# Learner builders (lazy: each call constructs a fresh learner)
# --------------------------------------------------------------------------- #
def _ours(d, e):
    return (OURS, lambda: DuelingGLMLearner(d=d, e=e, lambda_reg=1.0, alpha=0.1,
            C=FIXED_C, Lambda=FIXED_LAMBDA, mu_tau=FIXED_MU_TAU, kappa=KAPPA))


def x_only_baselines(d, e):
    """Standard dueling baselines on the pre-serving context X only (figs 0/1/6)."""
    aw = np.sqrt(d) / (np.sqrt(KAPPA) * FIXED_C)
    beta = np.sqrt(d) + aw * FIXED_C
    return [
        ("RCDB",       lambda: RCDBLearner(d=d, lambda_reg=1.0, alpha=0.5, rcdb_alpha=aw, rcdb_beta=beta, kappa=KAPPA)),
        ("ColSTIM",    lambda: ColSTIMLearner(d=d, lambda_reg=1.0, alpha=0.5)),
        ("MaxInP",     lambda: MaxInPLearner(d=d, lambda_reg=1.0, alpha=0.5)),
        ("MaxPairUCB", lambda: MaxPairUCBLearner(d=d, lambda_reg=1.0, alpha=np.sqrt(d))),
    ]


def post_serving_baselines(d, e):
    """Baselines augmented with the same neural post-serving predictor (fig 2)."""
    dim = d + e
    aw = np.sqrt(dim) / (np.sqrt(KAPPA) * FIXED_C)
    beta = np.sqrt(dim) + aw * FIXED_C
    return [
        ("RCDB+PS",       lambda: RCDBPostServingLearner(d=d, e=e, lambda_reg=1.0, alpha=0.5, rcdb_alpha=aw, rcdb_beta=beta, kappa=KAPPA)),
        ("ColSTIM+PS",    lambda: ColSTIMPostServingLearner(d=d, e=e, lambda_reg=1.0, alpha=0.5, C=FIXED_C, Lambda=FIXED_LAMBDA, mu_tau=FIXED_MU_TAU, kappa=KAPPA)),
        ("MaxInP+PS",     lambda: MaxInPPostServingLearner(d=d, e=e, lambda_reg=1.0, alpha=0.1, rcdb_alpha=aw, rcdb_beta=beta, kappa=KAPPA)),
        ("MaxPairUCB+PS", lambda: MaxPairUCBPostServingLearner(d=d, e=e, lambda_reg=1.0, alpha=0.1, rcdb_alpha=aw, rcdb_beta=beta, kappa=KAPPA)),
    ]


# --------------------------------------------------------------------------- #
# Environment factories
# --------------------------------------------------------------------------- #
def _delay(cfg):
    if cfg["delay_type"] == "strategic":
        return StrategicDelay(budget=cfg["delay_mag"], magnitude=int(np.sqrt(cfg["delay_mag"])), threshold_val=0.0)
    if cfg["delay_type"] == "stochastic":
        return StochasticGaussianDelay(mean_delay=cfg["mean"], std_delay=cfg["std"])
    return None


def _corruption(cfg):
    return None if cfg["delay_type"] == "clean" else StrategicOutcomeCorruption(budget=cfg["corruption_budget"])


def _synthetic_env(cfg, d, e, K, phi, theta, zeta):
    env = ContextualDuelingBanditEnv(d=d, e=e, k=K, delay_model=_delay(cfg),
                                     corruption_model=_corruption(cfg), phi_type=phi)
    env.theta_star, env.zeta_star = theta.copy(), zeta.copy()
    return env


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_mapping(exp, cfg, d, e, K, phi, T, n_runs):
    """Run one (experiment, config, phi) over n_runs and return {method: (n_runs, T)}."""
    spec = EXPERIMENTS[exp]
    builders = [_ours(d, e)] + spec["baselines"](d, e)
    names = [n for n, _ in builders]
    regrets = {n: np.zeros((n_runs, T)) for n in names}

    base_seed = 42 + (_PHI_ORDER.index(phi) * 1000 if phi in _PHI_ORDER else 0) + 999

    for run_idx in range(n_runs):
        run_seed = base_seed + run_idx * 100
        np.random.seed(run_seed)
        theta = np.random.randn(d) * 0.1
        zeta = np.random.randn(e) * 5.0 if e > 0 else np.array([])
        for name, build in builders:
            np.random.seed(run_seed)
            env = _synthetic_env(cfg, d, e, K, phi, theta, zeta)
            regrets[name][run_idx] = run_simulation(env, build(), T, name=None)
    return names, regrets


def _save(exp, label, names, regrets, T, n_runs):
    out = os.path.join("results", exp)
    os.makedirs(out, exist_ok=True)
    # plot
    plt.rcParams.update({"font.size": 22, "axes.labelsize": 24, "legend.fontsize": 18,
                         "lines.linewidth": 4.0, "figure.figsize": (11, 8)})
    plt.figure()
    t = np.arange(T)
    ei = np.linspace(0, T - 1, 11).astype(int)
    for name in names:
        data = regrets[name]
        mean, sd = data.mean(0), data.std(0)         # error bars: +/- 1 std over runs
        color = next((c for k, c in _COLORS.items() if k in name), "black")
        plt.plot(t, mean, label=name, color=color)
        plt.errorbar(t[ei], mean[ei], yerr=sd[ei], fmt="none", ecolor=color, capsize=4, alpha=0.6)
    plt.xlabel("Rounds (t)"); plt.ylabel("Cumulative Regret")
    plt.grid(True, alpha=0.3); plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out, f"{label}.pdf"))
    plt.close()
    # tidy CSV
    rows = []
    for name in names:
        for r in range(n_runs):
            rows.append(pd.DataFrame({"round": np.arange(1, T + 1), "regret": regrets[name][r],
                                      "method": name, "run": r}))
    pd.concat(rows, ignore_index=True).to_csv(os.path.join(out, f"{label}.csv"), index=False)
    finals = {n: regrets[n][:, -1].mean() for n in names}
    return out, finals


def _label(cfg, d, K, phi):
    tag = (f"Mag{cfg['delay_mag']}" if cfg["delay_type"] == "strategic"
           else f"Mean{cfg['mean']}_Std{cfg['std']}")
    return f"{phi}_{cfg['delay_type']}_d{d}_K{K}_C{cfg['corruption_budget']}_{tag}"


def run_one(exp, cfg, T=2000, n_runs=5, verbose=True):
    """Run a single configuration of an experiment and save plot + CSV."""
    spec = EXPERIMENTS[exp]
    d, K = cfg["d"], cfg["K"]
    phi = cfg.get("phi", spec.get("phis", ["polynomial"])[0])
    if "e" in cfg:                       # explicit override
        e = cfg["e"]
    elif spec.get("e", 0) > 0:           # post-serving experiments: use phi's natural d_y
        e = post_dim(phi, d)
    else:
        e = 0
    names, regrets = run_mapping(exp, cfg, d, e, K, phi, T, n_runs)
    out, finals = _save(exp, _label(cfg, d, K, phi), names, regrets, T, n_runs)
    if verbose:
        ranked = sorted(finals.items(), key=lambda kv: kv[1])
        print(f"  saved -> {out}/{_label(cfg, d, K, phi)}.{{pdf,csv}}")
        for rk, (n, v) in enumerate(ranked, 1):
            print(f"    {rk}. {n:16s} {v:10.1f}")
    return finals


def reproduce(exp, T=2000, n_runs=5):
    """Run the full paper sweep registered for an experiment."""
    spec = EXPERIMENTS[exp]
    if exp == "std-ablation":
        return _run_std_ablation(T, n_runs)
    sweep = spec["sweep"]
    print(f"[{exp}] {spec['paper']}  |  {len(sweep)} configs, n_runs={n_runs}, T={T}")
    for i, cfg in enumerate(sweep, 1):
        phi = cfg.get("phi", spec.get("phis", ["polynomial"])[0])
        print(f"\n[{exp}] {i}/{len(sweep)}  phi={phi} d{cfg['d']} K{cfg['K']} {cfg['delay_type']} C{cfg['corruption_budget']}")
        run_one(exp, cfg, T=T, n_runs=n_runs)


# --------------------------------------------------------------------------- #
# Sweeps (the exact paper / appendix configurations)
# --------------------------------------------------------------------------- #
def _cfg(d, K, C, mean, std, mag, delay_type, phi=None):
    c = dict(d=d, K=K, corruption_budget=C, mean=mean, std=std, delay_mag=mag, delay_type=delay_type)
    if phi:
        c["phi"] = phi
    return c


def _sweep_linear():
    settings = [(25, 100., 100., 10000), (25, 100., 100., 6400), (25, 100., 100., 3600),
                (25, 80., 80., 10000), (25, 60., 60., 10000),
                (10, 100., 100., 10000), (15, 100., 100., 10000), (20, 100., 100., 10000)]
    return [_cfg(d, K, C, mu, sd, mag, dt, phi="polynomial")
            for d in (10, 20, 30) for K in (10, 20, 30)
            for (C, mu, sd, mag) in settings for dt in ("stochastic", "strategic")]


def _sweep_representative():
    # Same configuration grid as `post-serving`; the only difference is that here
    # the baselines are restricted to pre-serving context (only RCDP-UCB uses Y).
    return _sweep_post_serving()


def _sweep_post_serving():
    out = []
    for (d, K) in [(10, 10), (20, 10), (30, 10), (10, 20), (10, 30)]:
        for phi in ("polynomial", "abs", "sinusoidal"):
            for dt in ("strategic", "stochastic"):
                out.append(_cfg(d, K, 25, 100., 100., 10000, dt, phi=phi))
    for (C, mag) in [(25, 6400), (25, 3600), (20, 10000)]:
        out.append(_cfg(10, 10, C, 100., 100., mag, "strategic", phi="abs"))
    for mu in (60., 80.):
        out.append(_cfg(10, 10, 25, mu, mu, 10000, "stochastic", phi="abs"))
    return out


def _run_std_ablation(T, n_runs, std_list=(0, 50, 100, 150), d=10, K=10):
    """RCDP-UCB only: sweep the stochastic-delay variance (paper appendix)."""
    out = os.path.join("results", "std-ablation")
    os.makedirs(out, exist_ok=True)
    results = {}
    for std in std_list:
        results[std] = np.zeros((n_runs, T))
        base_seed = 42 + int(std) * 10
        for run_idx in range(n_runs):
            run_seed = base_seed + run_idx * 100
            np.random.seed(run_seed)
            theta = np.random.randn(d) * 0.1
            np.random.seed(run_seed)
            env = ContextualDuelingBanditEnv(d=d, e=0, k=K,
                    delay_model=StochasticGaussianDelay(mean_delay=100.0, std_delay=std),
                    corruption_model=StrategicOutcomeCorruption(budget=25), phi_type="polynomial")
            env.theta_star, env.zeta_star = theta.copy(), np.array([])
            learner = DuelingGLMLearner(d=d, e=0, lambda_reg=1.0, alpha=0.1,
                        C=FIXED_C, Lambda=FIXED_LAMBDA, mu_tau=FIXED_MU_TAU, kappa=KAPPA)
            results[std][run_idx] = run_simulation(env, learner, T, name=None)
    plt.rcParams.update({"font.size": 22, "lines.linewidth": 4.0, "figure.figsize": (11, 8)})
    plt.figure(); t = np.arange(T)
    for std in std_list:
        m = results[std].mean(0)
        plt.plot(t, m, label=fr"Ours ($\sigma^2={std}$)")
    plt.xlabel("Rounds (t)"); plt.ylabel("Cumulative Regret")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out, "std_ablation.pdf")); plt.close()
    print(f"  saved -> {out}/std_ablation.pdf")


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
EXPERIMENTS = {
    "linear": dict(
        kind="synthetic", e=0, baselines=x_only_baselines, phis=["polynomial"],
        paper="Fig. 1 — linear dueling setting, no post-serving context",
        sweep=_sweep_linear()),
    "representative": dict(
        kind="synthetic", e=10, baselines=x_only_baselines, phis=["abs"],
        paper="Fig. 3 — only RCDP-UCB exploits the post-serving mapping",
        sweep=_sweep_representative()),
    "post-serving": dict(
        kind="synthetic", e=10, baselines=post_serving_baselines,
        phis=["polynomial", "abs", "sinusoidal"],
        paper="Fig. 2 — all methods learn the post-serving mapping",
        sweep=_sweep_post_serving()),
    "std-ablation": dict(
        kind="ablation", e=0, phis=["polynomial"],
        paper="Appendix — RCDP-UCB sensitivity to stochastic-delay variance",
        sweep=None),
}


# --------------------------------------------------------------------------- #
# README teaser figures  (python main.py figures)
# --------------------------------------------------------------------------- #
_FIG_COLORS = {"Ours": "#1f77b4", "RCDB": "#2ca02c", "ColSTIM": "#ff7f0e",
               "MaxInP": "#9467bd", "MaxPair": "#d62728"}
_FIG_MARK = {"Ours": "o", "RCDB": "s", "ColSTIM": "^", "MaxInP": "D", "MaxPair": "v"}


def _fig_csv(exp, phi, dt):
    tag = "Mag10000" if dt == "strategic" else "Mean100.0_Std100.0"
    return os.path.join("results", exp, f"{phi}_{dt}_d10_K10_C25_{tag}.csv")


def _fig_panel(ax, csv, title):
    df = pd.read_csv(csv)
    pick = lambda n, d: next((v for k, v in d.items() if k in n), None)
    key = lambda n: (0 if "Ours" in n else 1, n)
    for m in sorted(df["method"].unique(), key=key):
        g = df[df["method"] == m].groupby("round")["regret"]
        mu, sd, t = g.mean().values, g.std().values, g.mean().index.values
        ei = np.linspace(0, len(t) - 1, 9).astype(int)
        c, ours = pick(m, _FIG_COLORS), "Ours" in m
        ax.plot(t, mu, color=c, lw=3.3 if ours else 1.9, zorder=6 if ours else 3,
                label=m.replace("+PS", ""), solid_capstyle="round")
        ax.errorbar(t[ei], mu[ei], yerr=sd[ei], fmt=pick(m, _FIG_MARK), ms=6 if ours else 4,
                    color=c, mfc=c, mec="white", mew=0.8, capsize=3.5, elinewidth=1.4,
                    capthick=1.4, zorder=7 if ours else 4, alpha=0.95)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Round  t", fontsize=10)
    ax.margins(x=0.02)


def make_figures(n_runs=10, T=2000, rerun=True):
    """Reproduce the three README teaser images (assets/teaser_*.png), with
    error bars (±1 std). With rerun=True it re-runs the underlying configs into
    results/ first; with rerun=False it just re-plots from existing CSVs."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass
    os.makedirs("assets", exist_ok=True)
    phis, delays = ["abs", "polynomial", "sinusoidal"], ["strategic", "stochastic"]
    base = lambda phi, dt: dict(d=10, K=10, corruption_budget=25, mean=100., std=100.,
                                delay_mag=10000, delay_type=dt, phi=phi)
    if rerun:
        print(f"[figures] running configs (n_runs={n_runs}, T={T}) ...", flush=True)
        for dt in delays:
            run_one("linear", base("polynomial", dt), T=T, n_runs=n_runs, verbose=False)
        for exp in ("representative", "post-serving"):
            for phi in phis:
                for dt in delays:
                    run_one(exp, base(phi, dt), T=T, n_runs=n_runs, verbose=False)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5.3))
    _fig_panel(ax[0], _fig_csv("linear", "polynomial", "strategic"), "Strategic delay")
    _fig_panel(ax[1], _fig_csv("linear", "polynomial", "stochastic"), "Stochastic delay")
    ax[0].set_ylabel("Cumulative regret", fontsize=10)
    h, l = ax[0].get_legend_handles_labels()
    fig.suptitle("Linear setting — no post-serving context  (polynomial, d=10, K=10)",
                 y=0.995, fontsize=12.5, fontweight="bold")
    fig.legend(h, l, loc="upper center", ncol=5, fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.935))
    fig.tight_layout(rect=[0, 0, 1, 0.83])
    fig.savefig("assets/teaser_linear.png", dpi=130)
    plt.close(fig)

    for exp, fname, sup in [
        ("representative", "teaser_representative",
         "Only RCDP-UCB uses the post-serving context  (baselines: pre-serving only)"),
        ("post-serving", "teaser_postserving",
         "All methods use the post-serving context")]:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for r, dt in enumerate(delays):
            for c, phi in enumerate(phis):
                _fig_panel(axes[r, c], _fig_csv(exp, phi, dt), f"{phi}  ·  {dt}")
                if c == 0:
                    axes[r, c].set_ylabel("Cumulative regret", fontsize=10)
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.suptitle(sup, y=0.995, fontsize=13.5, fontweight="bold")
        fig.legend(h, l, loc="upper center", ncol=5, fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.955))
        fig.tight_layout(rect=[0, 0, 1, 0.9])
        fig.savefig(f"assets/{fname}.png", dpi=120)
        plt.close(fig)
    print("[figures] wrote assets/teaser_linear.png, teaser_representative.png, teaser_postserving.png", flush=True)
