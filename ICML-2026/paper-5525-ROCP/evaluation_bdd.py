"""
Evaluate ROCP/RAC baselines on BDD using a single NPZ and repeated random splits.

This mirrors evaluation.py but uses one dataset file and creates N random
splits (default 20). Each split is partitioned into:
  - D_phi: fit isotonic calibrators for 3 hazard bits
  - D_cal: conformal calibration
  - D_test: evaluation
"""

import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression

from evaluation import evaluate_seed, mean_and_se, format_mean_se


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BDD with repeated random splits.")
    parser.add_argument(
        "--npz",
        type=Path,
        default=Path("BDD/outputs/prob_true.npz"),
        help="Path to NPZ with keys scores (N,3) and true_bits (N,3).",
    )
    parser.add_argument("--num-splits", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=23)
    parser.add_argument(
        "--phi-frac",
        type=float,
        default=0.2,
        help="Fraction used for isotonic regression (D_phi).",
    )
    parser.add_argument(
        "--cal-frac",
        type=float,
        default=0.4,
        help="Fraction used for conformal calibration (D_cal). Remaining is test (D_test).",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-3,
        help="Clip isotonic outputs to [eps, 1-eps] to avoid 0/1 probabilities.",
    )
    parser.add_argument(
        "--alpha-list",
        type=float,
        nargs="*",
        default=[0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1],
    )
    return parser.parse_args()


def encode_labels(y_bits: np.ndarray) -> np.ndarray:
    if y_bits.ndim != 2 or y_bits.shape[1] != 3:
        raise ValueError("true_bits must have shape (N,3).")
    return (y_bits[:, 0] * 4 + y_bits[:, 1] * 2 + y_bits[:, 2]).astype(np.int64)


def top1_accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    if probs.ndim != 2:
        raise ValueError("probs must be a 2D array.")
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array.")
    if len(probs) != len(labels):
        raise ValueError("probs and labels must have the same length.")
    pred = np.argmax(probs, axis=1).astype(np.int64)
    return float(np.mean(pred == labels.astype(np.int64)))


def build_fx(p: np.ndarray) -> np.ndarray:
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("p must have shape (N,3).")
    p1 = np.clip(p[:, 0], 0.0, 1.0)
    p2 = np.clip(p[:, 1], 0.0, 1.0)
    p3 = np.clip(p[:, 2], 0.0, 1.0)
    f000 = (1 - p1) * (1 - p2) * (1 - p3)
    f001 = (1 - p1) * (1 - p2) * p3
    f010 = (1 - p1) * p2 * (1 - p3)
    f011 = (1 - p1) * p2 * p3
    f100 = p1 * (1 - p2) * (1 - p3)
    f101 = p1 * (1 - p2) * p3
    f110 = p1 * p2 * (1 - p3)
    f111 = p1 * p2 * p3
    return np.stack([f000, f001, f010, f011, f100, f101, f110, f111], axis=1)


def build_loss_matrix(
    M=60.0,
    c_turn=3.0,
    c_unnec=2.0,
    c_stop_free=6.0,
    c_stop_block=2.0,
):
    # action order: KEEP, LEFT, RIGHT, STOP (columns)
    loss = np.zeros((8, 4), dtype=float)
    for idx in range(8):
        y_a = (idx >> 2) & 1
        y_l = (idx >> 1) & 1
        y_r = idx & 1
        # KEEP
        loss[idx, 0] = M if y_a == 1 else 0.0
        # LEFT
        loss[idx, 1] = (M if y_l == 1 else 0.0) + c_turn + (c_unnec if y_a == 0 else 0.0)
        # RIGHT
        loss[idx, 2] = (M if y_r == 1 else 0.0) + c_turn + (c_unnec if y_a == 0 else 0.0)
        # STOP
        loss[idx, 3] = c_stop_block if y_a == 1 else c_stop_free
    return loss


def plot_with_band(ax, x, mean, se, label, color, linestyle="-", marker="o"):
    line_kwargs = {
        "label": label,
        "color": color,
        "linestyle": linestyle,
    }
    if marker is not None:
        line_kwargs["marker"] = marker
    ax.plot(x, mean, **line_kwargs)
    if se is not None:
        ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.15, linewidth=0)


def fit_isotonic_models(scores_phi: np.ndarray, y_bits_phi: np.ndarray, eps: float):
    if scores_phi.ndim != 2 or scores_phi.shape[1] != 3:
        raise ValueError("scores must have shape (N,3).")
    if y_bits_phi.ndim != 2 or y_bits_phi.shape[1] != 3:
        raise ValueError("true_bits must have shape (N,3).")
    models = []
    for k in range(3):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=eps, y_max=1.0 - eps)
        ir.fit(scores_phi[:, k].astype(float), y_bits_phi[:, k].astype(float))
        models.append(ir)
    return models


def apply_isotonic(models, scores: np.ndarray, eps: float) -> np.ndarray:
    p = np.empty_like(scores, dtype=float)
    for k, ir in enumerate(models):
        p[:, k] = ir.transform(scores[:, k].astype(float))
    return np.clip(p, eps, 1.0 - eps)


def make_splits(scores, y_bits, phi_frac, cal_frac, seeds):
    n = len(scores)
    phi_n = int(math.floor(phi_frac * n))
    cal_n = int(math.floor(cal_frac * n))
    test_n = n - phi_n - cal_n
    if phi_n <= 0 or cal_n <= 0 or test_n <= 0:
        raise ValueError("phi-frac/cal-frac lead to an empty split. Need phi, cal, test all non-empty.")
    for seed in seeds:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        phi_idx = idx[:phi_n]
        cal_idx = idx[phi_n : phi_n + cal_n]
        test_idx = idx[phi_n + cal_n :]
        yield (
            seed,
            scores[phi_idx],
            y_bits[phi_idx],
            scores[cal_idx],
            y_bits[cal_idx],
            scores[test_idx],
            y_bits[test_idx],
        )


def main():
    args = parse_args()

    if not args.npz.exists():
        raise FileNotFoundError(f"NPZ not found: {args.npz}")

    with np.load(args.npz) as data:
        if "scores" not in data or "true_bits" not in data:
            raise KeyError(
                "NPZ must contain keys: scores (N,3), true_bits (N,3). "
                "Please regenerate with BDD/build_npz.py."
            )
        scores = data["scores"]
        y_bits = data["true_bits"]

    if scores.ndim != 2 or scores.shape[1] != 3:
        raise ValueError("scores must have shape (N,3).")
    if y_bits.ndim != 2 or y_bits.shape[1] != 3:
        raise ValueError("true_bits must have shape (N,3).")
    if len(scores) != len(y_bits):
        raise ValueError("scores and true_bits must have the same length.")

    alpha_list = [float(a) for a in args.alpha_list]
    if not any(abs(a - 0.05) < 1e-12 for a in alpha_list):
        raise ValueError("alpha-list must include 0.05 because evaluation.py computes critical mistakes at alpha=0.05.")

    loss_matrix = build_loss_matrix()
    critical_labels = [1, 2, 3, 4, 5, 6, 7]  # all labels except 000
    bad_action_threshold = 60.0  

    seeds = [args.seed_start + i for i in range(args.num_splits)]
    split_results = []
    for (
        seed,
        scores_phi,
        y_bits_phi,
        scores_cal,
        y_bits_cal,
        scores_test,
        y_bits_test,
    ) in make_splits(
        scores, y_bits, args.phi_frac, args.cal_frac, seeds
    ):
        models = fit_isotonic_models(scores_phi, y_bits_phi, eps=args.eps)
        p_cal = apply_isotonic(models, scores_cal, eps=args.eps)
        p_test = apply_isotonic(models, scores_test, eps=args.eps)

        cal_probs = build_fx(p_cal).astype(np.float32)
        test_probs = build_fx(p_test).astype(np.float32)
        cal_labels = encode_labels(y_bits_cal)
        test_labels = encode_labels(y_bits_test)

        cal_top1 = top1_accuracy(cal_probs, cal_labels)
        test_top1 = top1_accuracy(test_probs, test_labels)
        print(f"Split seed {seed}: top1(cal)={cal_top1:.4f}, top1(test)={test_top1:.4f}")

        split_results.append(
            evaluate_seed(
                cal_probs,
                cal_labels,
                test_probs,
                test_labels,
                loss_matrix,
                alpha_list,
                critical_labels,
                bad_action_threshold=bad_action_threshold,
            )
        )
        print(f"Processed split seed {seed}")

    score_names = ["LAS", "APS", "SOCOP"]

    wcr_mean = {}
    wcr_se = {}
    for m in ["ROCP", "RAC"]:
        values = [res["worst_case_risk"][m] for res in split_results]
        wcr_mean[m], wcr_se[m] = mean_and_se(values)

    wcr_scores_mean = {s: {} for s in score_names}
    wcr_scores_se = {s: {} for s in score_names}
    for s in score_names:
        for rule in ["a_ROCP", "a_RAC"]:
            values = [res["worst_case_risk_scores"][s][rule] for res in split_results]
            wcr_scores_mean[s][rule], wcr_scores_se[s][rule] = mean_and_se(values)

    rl_mean = {}
    rl_se = {}
    for m in ["ROCP", "RAC"]:
        values = [res["realized_loss"][m] for res in split_results]
        rl_mean[m], rl_se[m] = mean_and_se(values)

    rl_scores_mean = {s: {} for s in score_names}
    rl_scores_se = {s: {} for s in score_names}
    for s in score_names:
        for rule in ["a_ROCP", "a_RAC"]:
            values = [res["realized_loss_scores"][s][rule] for res in split_results]
            rl_scores_mean[s][rule], rl_scores_se[s][rule] = mean_and_se(values)

    mis_mean = {}
    mis_se = {}
    for m in ["ROCP", "RAC"] + score_names:
        values = [res["miscoverage"][m] for res in split_results]
        mis_mean[m], mis_se[m] = mean_and_se(values)

    best_realized_losses = [res["best_realized_loss"] for res in split_results]
    best_mean, best_se = mean_and_se(best_realized_losses)

    bad_mean = {"best": {}, "rocp": {}, "rac": {}}
    bad_se = {"best": {}, "rocp": {}, "rac": {}}
    for method in ["best", "rocp", "rac"]:
        for lbl in critical_labels:
            vals = [res["critical_bad_action"][method][lbl] for res in split_results]
            bad_mean[method][lbl], bad_se[method][lbl] = mean_and_se(vals)

    print("\nAlpha list:", alpha_list)
    print("\nWorst-case risk (mean +- SE):")
    print(f"method ROCP: {format_mean_se(wcr_mean['ROCP'], wcr_se['ROCP'])}")
    print(f"method RAC: {format_mean_se(wcr_mean['RAC'], wcr_se['RAC'])}")
    for s in score_names:
        print(f"method {s} (a_ROCP): {format_mean_se(wcr_scores_mean[s]['a_ROCP'], wcr_scores_se[s]['a_ROCP'])}")
        print(f"method {s} (a_RAC): {format_mean_se(wcr_scores_mean[s]['a_RAC'], wcr_scores_se[s]['a_RAC'])}")

    print("\nRealized loss (mean +- SE):")
    print(f"method ROCP: {format_mean_se(rl_mean['ROCP'], rl_se['ROCP'])}")
    print(f"method RAC: {format_mean_se(rl_mean['RAC'], rl_se['RAC'])}")
    for s in score_names:
        print(f"method {s} (a_ROCP): {format_mean_se(rl_scores_mean[s]['a_ROCP'], rl_scores_se[s]['a_ROCP'])}")
        print(f"method {s} (a_RAC): {format_mean_se(rl_scores_mean[s]['a_RAC'], rl_scores_se[s]['a_RAC'])}")
    best_line_mean = np.full(len(alpha_list), best_mean, dtype=float)
    best_line_se = np.full(len(alpha_list), best_se, dtype=float)
    print(f"method best-resp: {format_mean_se(best_line_mean, best_line_se)}")

    print("\nMiscoverage (mean +- SE):")
    for m in ["ROCP", "RAC"] + score_names:
        print(f"method {m}: {format_mean_se(mis_mean[m], mis_se[m])}")

    print(f"\nCritical mistake rates at alpha=0.05 (loss >= {bad_action_threshold:g}):")
    for method in ["best", "rocp", "rac"]:
        vals = [f"{lbl}:{bad_mean[method][lbl]:.4f}±{bad_se[method][lbl]:.4f}" for lbl in critical_labels]
        print(f"{method}: " + ", ".join(vals))

    # ----------------------------
    # Plot (covid-style)
    # ----------------------------
    # Plot against the actual alpha values, but show clean ticks 0.00, 0.02, ..., max(alpha).
    x_alpha = np.asarray(alpha_list, dtype=float)
    max_alpha = float(np.max(x_alpha)) if x_alpha.size else 0.0
    tick_step = 0.02
    tick_end = round(math.ceil(max_alpha / tick_step) * tick_step, 10) if max_alpha > 0 else tick_step
    x_ticks = np.arange(0.0, tick_end + 1e-12, tick_step)

    score_colors = {"LAS": "#ff7f0e", "APS": "#2ca02c", "SOCOP": "#d62728"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    ax_wcr, ax_rl = axes[0, 0], axes[0, 1]
    ax_mis, ax_cm = axes[1, 0], axes[1, 1]

    # (1) Worst-case risk
    plot_with_band(ax_wcr, x_alpha, wcr_mean["ROCP"], wcr_se["ROCP"], "ROCP", "black")
    plot_with_band(ax_wcr, x_alpha, wcr_mean["RAC"], wcr_se["RAC"], "RAC", "gray")
    for s in score_names:
        c = score_colors[s]
        plot_with_band(
            ax_wcr,
            x_alpha,
            wcr_scores_mean[s]["a_ROCP"],
            wcr_scores_se[s]["a_ROCP"],
            f"{s} (a_ROCP)",
            c,
            linestyle="-",
        )
        plot_with_band(
            ax_wcr,
            x_alpha,
            wcr_scores_mean[s]["a_RAC"],
            wcr_scores_se[s]["a_RAC"],
            f"{s} (a_RAC)",
            c,
            linestyle="--",
        )
    ax_wcr.set_title("(a) Averaged realized worst-case risk")
    ax_wcr.set_xlim([-tick_step / 4.0, tick_end + tick_step / 4.0])
    ax_wcr.set_xticks(x_ticks)
    ax_wcr.set_xlabel("alpha")
    ax_wcr.set_ylabel("Averaged set risk")
    ax_wcr.grid(True, linestyle="--", alpha=0.4)
    ax_wcr.legend(fontsize=8)

    # (2) Realized loss
    plot_with_band(ax_rl, x_alpha, rl_mean["ROCP"], rl_se["ROCP"], "ROCP", "black")
    plot_with_band(ax_rl, x_alpha, rl_mean["RAC"], rl_se["RAC"], "RAC", "gray")
    for s in score_names:
        c = score_colors[s]
        plot_with_band(
            ax_rl,
            x_alpha,
            rl_scores_mean[s]["a_ROCP"],
            rl_scores_se[s]["a_ROCP"],
            f"{s} (a_ROCP)",
            c,
            linestyle="-",
        )
        plot_with_band(
            ax_rl,
            x_alpha,
            rl_scores_mean[s]["a_RAC"],
            rl_scores_se[s]["a_RAC"],
            f"{s} (a_RAC)",
            c,
            linestyle="--",
        )
    plot_with_band(
        ax_rl,
        x_alpha,
        np.full(len(x_alpha), best_mean, dtype=float),
        np.full(len(x_alpha), best_se, dtype=float),
        "best-resp",
        "#1f77b4",
        linestyle=":",
        marker=None,
    )
    ax_rl.set_title("(b) Averaged realized loss")
    ax_rl.set_xlim([-tick_step / 4.0, tick_end + tick_step / 4.0])
    ax_rl.set_xticks(x_ticks)
    ax_rl.set_xlabel("alpha")
    ax_rl.set_ylabel("Averaged realized loss")
    ax_rl.grid(True, linestyle="--", alpha=0.4)
    ax_rl.legend(fontsize=8)

    # (3) Miscoverage
    plot_with_band(ax_mis, x_alpha, mis_mean["ROCP"], mis_se["ROCP"], "ROCP", "black")
    plot_with_band(ax_mis, x_alpha, mis_mean["RAC"], mis_se["RAC"], "RAC", "gray")
    for s in score_names:
        plot_with_band(ax_mis, x_alpha, mis_mean[s], mis_se[s], s, score_colors[s])
    # y=x reference line (true diagonal)
    ax_mis.plot([0, tick_end], [0, tick_end], linestyle="--", color="gray", linewidth=1.5, label="y=x", alpha=0.7)
    ax_mis.set_title("(c) Averaged miscoverage")
    ax_mis.set_xlabel("alpha")
    ax_mis.set_ylabel("Averaged miscoverage")
    ax_mis.set_xlim([-tick_step / 4.0, tick_end + tick_step / 4.0])
    ax_mis.set_ylim([-tick_step / 4.0, tick_end + tick_step / 4.0])
    ax_mis.set_xticks(x_ticks)
    ax_mis.grid(True, linestyle="--", alpha=0.4)
    ax_mis.legend(fontsize=8)

    # (4) Critical mistake rates at alpha=0.05 (loss >= threshold)
    label_names = {i: format(i, "03b") for i in range(8)}
    vals_best = [100.0 * bad_mean["best"][l] for l in critical_labels]
    vals_rocp = [100.0 * bad_mean["rocp"][l] for l in critical_labels]
    vals_rac = [100.0 * bad_mean["rac"][l] for l in critical_labels]
    err_best = [100.0 * bad_se["best"][l] for l in critical_labels]
    err_rocp = [100.0 * bad_se["rocp"][l] for l in critical_labels]
    err_rac = [100.0 * bad_se["rac"][l] for l in critical_labels]

    gx = np.arange(len(critical_labels))
    bar_w = 0.25
    ax_cm.bar(
        gx - bar_w,
        vals_best,
        width=bar_w,
        label="Best response",
        color="#1f77b4",
        yerr=err_best,
        capsize=3,
    )
    ax_cm.bar(
        gx,
        vals_rocp,
        width=bar_w,
        label="ROCP (alpha=0.05)",
        color="#ff7f0e",
        yerr=err_rocp,
        capsize=3,
    )
    ax_cm.bar(
        gx + bar_w,
        vals_rac,
        width=bar_w,
        label="RAC (alpha=0.05)",
        color="#2ca02c",
        yerr=err_rac,
        capsize=3,
    )
    ax_cm.set_title(f"(d) Critical mistake rates (alpha=0.05, loss >= {bad_action_threshold:g})")
    ax_cm.set_xticks(gx)
    ax_cm.set_xticklabels([label_names[l] for l in critical_labels], rotation=15, ha="right")
    ax_cm.set_ylabel("Percentage of critical decisions")
    ax_cm.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax_cm.legend(fontsize=8)

    fig.tight_layout()
    figures_dir = Path("Figures")
    figures_dir.mkdir(exist_ok=True)
    out_path = figures_dir / "evaluation_bdd_1.png"
    fig.savefig(out_path)
    print(f"\nSaved combined figure to: {out_path}")


if __name__ == "__main__":
    main()
