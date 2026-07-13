import argparse
import os
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

from atc_scaling_scenarios_dense_adver import (
    atc_run,
    candidate_splits_all,
    candidate_splits_geom_ends,
    candidate_splits_uniform,
    gamma_constant_factory,
    gamma_paper,
    sliding_run,
    discount_run,
    resolve_window_len,
    resolve_discount,
)


def load_nab_csv(path: str) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=float, usecols=[1])
    data = np.asarray(data, dtype=float)
    if data.ndim != 1 or data.size == 0:
        raise ValueError(f"No data loaded from {path!r}.")
    return data


def build_piecewise_means(X: np.ndarray, cp_list: list[int]) -> np.ndarray:
    """
    Build piecewise-constant true means from change points (1-based indices).
    cp_list are times t where a new segment starts.
    """
    T = len(X)
    cps = [int(c) for c in cp_list if 1 <= int(c) <= T]
    cps = sorted(set(cps))
    taus = [1] + cps + [T + 1]
    mu_t = np.empty(T, dtype=float)
    for i in range(len(taus) - 1):
        start = taus[i] - 1
        end = taus[i + 1] - 1
        if end <= start:
            continue
        mu_t[start:end] = float(np.mean(X[start:end]))
    return mu_t


def estimate_sigma_from_segments(X: np.ndarray, mu_t: np.ndarray) -> float:
    resid = X - mu_t
    return float(np.std(resid, ddof=1))


def variance_bias_terms(hatmu: np.ndarray, mu_t: np.ndarray, alarms: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Decompose squared error into variance-like and bias-like terms:
      variance-like: (hatmu_t - bar_mu_t^r)^2
      bias-like:     (bar_mu_t^r - mu_t)^2
    where bar_mu_t^r is the running average of true means since last restart r.
    """
    T = len(mu_t)
    mu_cs = np.concatenate([[0.0], np.cumsum(mu_t)])
    alarms_sorted = sorted(int(a) for a in alarms)

    r = 0
    idx_alarm = 0
    var_term = np.empty(T, dtype=float)
    bias_term = np.empty(T, dtype=float)

    for t in range(1, T + 1):
        bar_mu = (mu_cs[t] - mu_cs[r]) / (t - r)
        h = hatmu[t - 1]
        m = mu_t[t - 1]
        var_term[t - 1] = (h - bar_mu) ** 2
        bias_term[t - 1] = (bar_mu - m) ** 2

        if idx_alarm < len(alarms_sorted) and t == alarms_sorted[idx_alarm]:
            r = t
            idx_alarm += 1

    return var_term, bias_term


def parse_max_splits_arg(v: str | None) -> int | str | None:
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return None
    s_low = s.lower()
    if s_low in {"logt", "log_t", "log"}:
        return "logT"
    try:
        n = int(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError("max-splits must be an int or 'logT'.") from e
    if n <= 0:
        raise argparse.ArgumentTypeError("max-splits must be positive.")
    return n


def make_split_fn(
    algo: str,
    *,
    T: int,
    max_splits: int | str | None,
    geom_base: float,
) -> Callable[[int, int], np.ndarray]:
    if algo in {"exact", "const"}:
        return candidate_splits_all
    if algo in {"geom_ends", "const_geom_ends"}:
        def _split(r: int, t: int) -> np.ndarray:
            return candidate_splits_geom_ends(r, t, base=geom_base, include_midpoint=True)
        return _split
    if algo in {"maxsplit", "const_maxsplit"}:
        if max_splits is None:
            M = 200
        elif isinstance(max_splits, str) and max_splits.lower() == "logt":
            M = max(1, int(np.ceil(np.log(max(T, 2)))))
        else:
            M = int(max_splits)

        def _split(r: int, t: int) -> np.ndarray:
            return candidate_splits_uniform(r, t, M)
        return _split
    raise ValueError(f"Unknown algo {algo!r}.")


def main() -> None:
    p = argparse.ArgumentParser(description="Run ATC on a single NAB time series.")
    p.add_argument("--csv-path", required=True, help="Path to NAB csv file.")
    p.add_argument(
        "--algo",
        choices=["exact", "geom_ends", "maxsplit", "const", "const_geom_ends", "const_maxsplit"],
        default="exact",
        help="ATC variant to use.",
    )
    p.add_argument(
        "--algos",
        nargs="+",
        choices=[
            "exact",
            "geom_ends",
            "maxsplit",
            "const",
            "const_geom_ends",
            "const_maxsplit",
            "sliding",
            "discount",
        ],
        default=None,
        help="One or more algorithms to compare in the regret plot.",
    )
    p.add_argument("--sigma", type=float, default=1.0, help="Noise std / proxy.")
    p.add_argument(
        "--sigma-list",
        nargs="+",
        type=float,
        default=None,
        help="Optional list of sigma values to compare (overrides --sigma).",
    )
    p.add_argument("--alpha", type=float, default=0.05, help="Alpha for ATC paper threshold.")
    p.add_argument("--const-gamma", type=float, default=6.0, help="Constant threshold gamma0.")
    p.add_argument("--geom-base", type=float, default=2.0, help="geom_ends base (must be > 1).")
    p.add_argument("--max-splits", type=parse_max_splits_arg, default=None, help="maxsplit count or 'logT'.")
    p.add_argument("--window", type=int, default=50, help="Sliding window length.")
    p.add_argument(
        "--window-formula",
        type=str,
        default="fixed",
        choices=["fixed", "sqrt_TlogT_over_S"],
        help="Sliding window length formula.",
    )
    p.add_argument("--discount", type=float, default=0.98, help="Discount factor in (0,1).")
    p.add_argument(
        "--discount-formula",
        type=str,
        default="fixed",
        choices=["fixed", "one_minus_sqrt_S_over_T"],
        help="Discount factor formula.",
    )
    p.add_argument("--out-dir", default="nab_figures", help="Output directory.")
    p.add_argument("--out-path", default=None, help="Optional full output path for the plot.")
    p.add_argument(
        "--cp-list",
        nargs="+",
        type=int,
        default=[377, 420, 592, 3575],
        help="Change-point indices (1-based) for piecewise-mean ground truth.",
    )
    p.add_argument("--estimate-sigma", action="store_true", help="Estimate sigma from segments.")
    p.add_argument("--plot-regret", action="store_true", help="Plot cumulative regret vs time.")
    p.add_argument("--regret-path", default=None, help="Optional full output path for regret plot.")
    p.add_argument("--no-alarms", action="store_true", help="Do not plot alarm lines.")
    p.add_argument("--no-hatmu", action="store_true", help="Do not plot the ATC estimate.")
    p.add_argument("--plot-cp", action="store_true", help="Plot change-point lines from --cp-list.")
    p.add_argument("--plot-means", action="store_true", help="Plot segment means from --cp-list.")

    args = p.parse_args()

    X = load_nab_csv(args.csv_path)
    T = len(X)
    S = len(args.cp_list)

    mu_t = build_piecewise_means(X, args.cp_list)
    sigma = float(args.sigma)
    if args.estimate_sigma:
        sigma = estimate_sigma_from_segments(X, mu_t)
        print(f"Estimated sigma: {sigma:.4f}")

    if args.sigma_list is not None:
        sigmas = [float(s) for s in args.sigma_list]
    else:
        sigmas = [sigma]

    if args.sigma_list is not None and args.estimate_sigma:
        print("Note: --estimate-sigma ignored when --sigma-list is provided.")

    algo_keys = args.algos if args.algos else [args.algo]
    if len(algo_keys) > 1 and len(sigmas) > 1:
        raise ValueError("When using --algos with multiple entries, provide a single sigma (no --sigma-list).")

    def algo_label(algo_key: str) -> str:
        labels = {
            "exact": "ATC algorithm",
            "geom_ends": "Efficient ATC",
            "maxsplit": "ATC maxsplit",
            "const": "ATC const",
            "const_geom_ends": "Efficient ATC (const)",
            "const_maxsplit": "ATC maxsplit (const)",
            "sliding": "Sliding window",
            "discount": "Discounted mean",
        }
        return labels.get(algo_key, algo_key)

    def run_algo(algo_key: str, s: float) -> dict:
        if algo_key in {"sliding", "discount"}:
            if algo_key == "sliding":
                W = resolve_window_len(
                    T=T,
                    S=S,
                    window_len=args.window,
                    window_formula=args.window_formula,
                )
                return sliding_run(X, window_len=W)
            rho = resolve_discount(
                T=T,
                S=S,
                discount=args.discount,
                discount_formula=args.discount_formula,
            )
            return discount_run(X, discount=rho)

        split_fn = make_split_fn(
            algo_key,
            T=T,
            max_splits=args.max_splits,
            geom_base=args.geom_base,
        )
        if "const" in algo_key:
            gamma_fn = gamma_constant_factory(args.const_gamma, scale_by_sigma=True)
        else:
            gamma_fn = gamma_paper
        return atc_run(
            X,
            sigma=s,
            alpha=float(args.alpha),
            split_fn=split_fn,
            gamma_fn=gamma_fn,
        )

    alarms: list[int] = []
    regret_traces: list[tuple[str, np.ndarray]] = []
    alarms_for_regret: list[int] = []

    if len(algo_keys) == 1 and len(sigmas) > 1:
        algo_key = algo_keys[0]
        last_idx = len(sigmas) - 1
        for idx, s in enumerate(sigmas):
            res = run_algo(algo_key, s)
            if idx == 0:
                alarms = res["alarms"]
            if idx == last_idx:
                alarms_for_regret = list(res["alarms"])
            if args.plot_regret:
                err = res["hatmu"] - mu_t
                cum_regret = np.cumsum(err * err)
                regret_traces.append((f"$\\sigma$={s:g}", cum_regret))
    else:
        s = sigmas[0]
        for idx, algo_key in enumerate(algo_keys):
            res = run_algo(algo_key, s)
            if idx == 0:
                alarms = res["alarms"]
                alarms_for_regret = list(res["alarms"])
            if args.plot_regret:
                err = res["hatmu"] - mu_t
                cum_regret = np.cumsum(err * err)
                regret_traces.append((algo_label(algo_key), cum_regret))

    os.makedirs(args.out_dir, exist_ok=True)
    if args.out_path:
        out_path = args.out_path
    else:
        base = os.path.splitext(os.path.basename(args.csv_path))[0]
        out_path = os.path.join(args.out_dir, f"atc_alarms_{base}_{args.algo}.png")

    t = np.arange(1, T + 1)

    def plot_env(ax) -> None:
        ax.plot(t, X, color="tab:blue", alpha=0.9, linewidth=1.8, label="raw data")
        if args.plot_means:
            ax.plot(t, mu_t, color="tab:red", linewidth=1.6, label="segment means")
        if args.plot_cp:
            for i, cp in enumerate(args.cp_list):
                ax.axvline(
                    cp,
                    color="tab:green",
                    linestyle=":",
                    linewidth=1.6,
                    alpha=0.8,
                    label="change points" if i == 0 else None,
                )
        if not args.no_alarms:
            for i, a in enumerate(alarms):
                ax.axvline(
                    a,
                    color="tab:red",
                    linewidth=1.2,
                    alpha=0.9,
                    label="alarm" if i == 0 else None,
                )
        ax.set_ylabel("value", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=15, frameon=True)

    def plot_regret_ax(ax) -> None:
        last_sigma_color = None
        for label, cum_regret in regret_traces:
            line = ax.plot(t, cum_regret, linewidth=1.6, label=label)[0]
            last_sigma_color = line.get_color()
        ax.set_ylabel("cumulative regret", fontsize=14)
        ax.grid(True, alpha=0.3)
        if not args.no_alarms:
            for i, a in enumerate(alarms_for_regret):
                alarm_color = last_sigma_color or "tab:red"
                ax.axvline(
                    a,
                    color=alarm_color,
                    linewidth=1.0,
                    alpha=0.7,
                    label="alarm" if i == 0 else None,
                )
        if len(regret_traces) > 1:
            handles, labels = ax.get_legend_handles_labels()
            order = ["Discounted mean", "Sliding window", "ATC algorithm"]
            ordered = []
            for name in order:
                for h, l in zip(handles, labels):
                    if l == name:
                        ordered.append((h, l))
                        break
            for h, l in zip(handles, labels):
                if l not in order:
                    ordered.append((h, l))
            if ordered:
                ax.legend(
                    [h for h, _ in ordered],
                    [l for _, l in ordered],
                    fontsize=15,
                    frameon=True,
                )
            else:
                ax.legend(fontsize=15, frameon=True)

    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    plot_env(ax)
    ax.set_xlabel("time index", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Wrote: {out_path}")
    print(f"Alarms ({len(alarms)}): {alarms}")

    if args.plot_regret:
        if args.regret_path:
            regret_path = args.regret_path
        else:
            base = os.path.splitext(os.path.basename(args.csv_path))[0]
            regret_path = os.path.join(args.out_dir, f"regret_{base}_{args.algo}.png")

        plt.figure(figsize=(10, 4))
        ax = plt.gca()
        plot_regret_ax(ax)
        ax.set_xlabel("time index", fontsize=14)
        plt.tight_layout()
        plt.savefig(regret_path, dpi=200)
        plt.close()
        print(f"Wrote: {regret_path}")

        stacked_path = os.path.join(args.out_dir, f"stacked_{base}_{args.algo}.png")
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        plot_env(axes[0])
        plot_regret_ax(axes[1])
        axes[0].set_xlabel("")
        axes[1].set_xlabel("time index", fontsize=14)
        fig.tight_layout()
        fig.savefig(stacked_path, dpi=200)
        plt.close(fig)
        print(f"Wrote: {stacked_path}")


if __name__ == "__main__":
    main()
