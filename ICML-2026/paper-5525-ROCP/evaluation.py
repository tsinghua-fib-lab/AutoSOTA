"""
Evaluate ROCP/RAC baselines across multiple random seeds and plot mean ± SE bands.

Loads per-seed calibration/test probabilities from results/*/probabilities/seed_*.npz.
"""

import argparse
import math
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from rocp import RiskOptimalConformalPredictor
from baselines import (
    RACBaseline,
    LASConformal,
    APSConformal,
    SOCOPConformal,
    BestResponseBaseline,
)
from eval_utility import (
    averaged_realized_worst_case_risk,
    averaged_realized_loss,
    critical_mistake_rates,
    critical_bad_action_rates,
    averaged_miscoverage,
)




def get_dataset_config(dataset: str):
    if dataset == "covid":
        # action 0: No action
        # action 1: Antibiotics
        # action 2: Quarantine
        # action 3: Additional Testing
        loss_matrix = [
            [0,    8,    8,   6], # Label 0: Normal
            [10,   0,   7,  3], # Label 1: Pneumonia
            [10,  7,   0,   2], # Label 2: COVID19
            [9,  6,   6,  0], # Label 3: LungOpacity
        ]
        critical_labels = [1, 2, 3]
        label_names = {1: "Pneumonia", 2: "COVID19", 3: "LungOpacity"}
        critical_mistake_suffix = "-> No action"
        alpha_list = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
    elif dataset == "movielens":
        # action 0: Not recommend
        # action 1: Recommend
        loss_matrix = [
            [2,  4],  # Rating 1:
            [2,  3],  # Rating 2: 
            [2,  2],  # Rating 3: 
            [2,  1],  # Rating 4: 
            [2,  0],  # Rating 5: 
        ]
        critical_labels = [0, 1]
        label_names = {0: "Rating 1", 1: "Rating 2"}
        critical_mistake_suffix = "-> Recommend"
        alpha_list = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return {
        "loss_matrix": loss_matrix,
        "critical_labels": critical_labels,
        "label_names": label_names,
        "critical_mistake_suffix": critical_mistake_suffix,
        "alpha_list": alpha_list,
    }


def find_seed_files(prob_dir: Path):
    seed_files = []
    for path in prob_dir.glob("seed_*.npz"):
        m = re.match(r"seed_(\d+)\.npz", path.name)
        if m:
            seed_files.append((int(m.group(1)), path))
    seed_files.sort(key=lambda x: x[0])
    return seed_files


def select_seed_files(seed_files, seeds=None, start_seed=None, end_seed=None):
    seed_map = {s: p for s, p in seed_files}
    if seeds:
        missing = [s for s in seeds if s not in seed_map]
        if missing:
            raise ValueError(f"Missing seed files for: {missing}")
        return [(s, seed_map[s]) for s in seeds]
    if start_seed is not None or end_seed is not None:
        if not seed_files:
            return []
        min_seed = seed_files[0][0]
        max_seed = seed_files[-1][0]
        start = min_seed if start_seed is None else start_seed
        end = max_seed if end_seed is None else end_seed
        return [(s, p) for s, p in seed_files if start <= s <= end]
    return seed_files


def mean_and_se(values):
    arr = np.asarray(values, dtype=float)
    mean = np.mean(arr, axis=0)
    if arr.shape[0] > 1:
        se = np.std(arr, axis=0, ddof=1) / math.sqrt(arr.shape[0])
    else:
        se = np.zeros_like(mean)
    return mean, se


def evaluate_seed(
    cal_probs,
    cal_labels,
    test_probs,
    test_labels,
    loss_matrix,
    alpha_list,
    critical_labels,
    bad_action_threshold=None,
):
    loss_arr = np.asarray(loss_matrix, dtype=float)
    max_loss = float(np.max(loss_arr))
    utility_matrix = max_loss - loss_arr
    actions = list(range(loss_arr.shape[1]))

    predictor = RiskOptimalConformalPredictor(actions=actions, loss_matrix=loss_matrix)
    rac = RACBaseline(actions=actions, utility_matrix=utility_matrix)
    best_resp = BestResponseBaseline(actions=actions, loss_matrix=loss_matrix)
    best_resp.fit(cal_probs, cal_labels)

    best_actions = [best_resp.predict_action(x_probs) for x_probs in test_probs]
    best_realized_loss = averaged_realized_loss(best_actions, test_labels, actions, loss_matrix)

    score_names = ["LAS", "APS", "SOCOP"]
    methods_with_sets = ["ROCP", "RAC"] + score_names

    worst_case_risk = {"ROCP": [], "RAC": []}
    realized_loss = {"ROCP": [], "RAC": []}
    worst_case_risk_scores = {s: {"a_ROCP": [], "a_RAC": []} for s in score_names}
    realized_loss_scores = {s: {"a_ROCP": [], "a_RAC": []} for s in score_names}
    miscoverage = {m: [] for m in methods_with_sets}

    rocp_actions_alpha05 = None
    rac_actions_alpha05 = None

    for alpha in alpha_list:
        beta_dict_rocp, global_beta_rocp = predictor.calibrate_beta_class_conditional(
            cal_probs, cal_labels, alpha=alpha)
        beta_rac = rac.calibrate_beta(cal_probs, cal_labels, alpha=alpha)

        las = LASConformal(actions=actions)
        las.calibrate(cal_probs, cal_labels, alpha)
        aps = APSConformal(actions=actions)
        aps.calibrate(cal_probs, cal_labels, alpha)
        socop = SOCOPConformal(actions=actions, lambda_param=0.25)
        socop.calibrate(cal_probs, cal_labels, alpha)

        rocp_sets, rocp_actions = [], []
        rac_sets, rac_actions = [], []
        score_sets = {s: [] for s in score_names}
        score_actions_rocp = {s: [] for s in score_names}
        score_actions_rac = {s: [] for s in score_names}

        full_set = set(range(len(actions)))

        for x_probs in test_probs:
            s_rocp = predictor.predict_set_cc(x_probs, beta_dict_rocp, global_beta_rocp)
            rocp_sets.append(s_rocp)
            a_rocp, _ = predictor.action_and_certificate(
                s_rocp if s_rocp else full_set,
                alpha=alpha,
                num_labels=len(x_probs),
            )
            rocp_actions.append(a_rocp)

            s_rac = rac.predict_set(x_probs, beta_rac)
            rac_sets.append(s_rac)
            a_rac, _ = rac.a_RAC(s_rac if s_rac else full_set)
            rac_actions.append(a_rac)

            for name, scorer in [("LAS", las), ("APS", aps), ("SOCOP", socop)]:
                s_sc = scorer.predict_set(x_probs)
                score_sets[name].append(s_sc)

                a_sc_rocp, _ = predictor.action_and_certificate(
                    s_sc if s_sc else full_set,
                    alpha=alpha,
                    num_labels=len(x_probs),
                )
                score_actions_rocp[name].append(a_sc_rocp)

                a_sc_rac, _ = rac.a_RAC(s_sc if s_sc else full_set)
                score_actions_rac[name].append(a_sc_rac)

        worst_case_risk["ROCP"].append(
            averaged_realized_worst_case_risk(rocp_sets, rocp_actions, actions, loss_matrix, alpha)
        )
        worst_case_risk["RAC"].append(
            averaged_realized_worst_case_risk(rac_sets, rac_actions, actions, loss_matrix, alpha)
        )
        for s in score_names:
            worst_case_risk_scores[s]["a_ROCP"].append(
                averaged_realized_worst_case_risk(
                    score_sets[s], score_actions_rocp[s], actions, loss_matrix, alpha
                )
            )
            worst_case_risk_scores[s]["a_RAC"].append(
                averaged_realized_worst_case_risk(
                    score_sets[s], score_actions_rac[s], actions, loss_matrix, alpha
                )
            )

        realized_loss["ROCP"].append(
            averaged_realized_loss(rocp_actions, test_labels, actions, loss_matrix)
        )
        realized_loss["RAC"].append(
            averaged_realized_loss(rac_actions, test_labels, actions, loss_matrix)
        )
        for s in score_names:
            realized_loss_scores[s]["a_ROCP"].append(
                averaged_realized_loss(score_actions_rocp[s], test_labels, actions, loss_matrix)
            )
            realized_loss_scores[s]["a_RAC"].append(
                averaged_realized_loss(score_actions_rac[s], test_labels, actions, loss_matrix)
            )

        miscoverage["ROCP"].append(averaged_miscoverage(rocp_sets, test_labels))
        miscoverage["RAC"].append(averaged_miscoverage(rac_sets, test_labels))
        for s in score_names:
            miscoverage[s].append(averaged_miscoverage(score_sets[s], test_labels))

        if abs(alpha - 0.05) < 1e-12:
            rocp_actions_alpha05 = rocp_actions
            rac_actions_alpha05 = rac_actions

    if rocp_actions_alpha05 is None or rac_actions_alpha05 is None:
        raise RuntimeError("Failed to capture actions for alpha=0.05.")

    cm_rocp = critical_mistake_rates(
        rocp_actions_alpha05, test_labels, actions, critical_labels, loss_matrix
    )
    cm_rac = critical_mistake_rates(
        rac_actions_alpha05, test_labels, actions, critical_labels, loss_matrix
    )
    cm_best = critical_mistake_rates(
        best_actions, test_labels, actions, critical_labels, loss_matrix
    )

    bad_action = None
    if bad_action_threshold is not None:
        bad_rocp = critical_bad_action_rates(
            rocp_actions_alpha05,
            test_labels,
            actions,
            critical_labels,
            loss_matrix,
            loss_threshold=float(bad_action_threshold),
        )
        bad_rac = critical_bad_action_rates(
            rac_actions_alpha05,
            test_labels,
            actions,
            critical_labels,
            loss_matrix,
            loss_threshold=float(bad_action_threshold),
        )
        bad_best = critical_bad_action_rates(
            best_actions,
            test_labels,
            actions,
            critical_labels,
            loss_matrix,
            loss_threshold=float(bad_action_threshold),
        )
        bad_action = {"best": bad_best, "rocp": bad_rocp, "rac": bad_rac}

    return {
        "worst_case_risk": worst_case_risk,
        "worst_case_risk_scores": worst_case_risk_scores,
        "realized_loss": realized_loss,
        "realized_loss_scores": realized_loss_scores,
        "miscoverage": miscoverage,
        "best_realized_loss": best_realized_loss,
        "critical_mistake": {
            "best": cm_best,
            "rocp": cm_rocp,
            "rac": cm_rac,
        },
        "critical_bad_action": bad_action,
    }


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


def format_mean_se(mean, se, precision=4):
    mean = np.asarray(mean, dtype=float)
    se = np.asarray(se, dtype=float)
    return "[" + ", ".join(
        f"{m:.{precision}f} \\pm {s:.{precision}f}" for m, s in zip(mean, se)
    ) + "]"


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and plot averaged metrics across multiple seeds."
    )
    parser.add_argument(
        "--dataset",
        choices=["covid", "movielens"],
        required=True,
        help="Dataset to use: 'covid' or 'movielens'.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Root results directory (default: results/Covid_data or results/MovieLens_data).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Explicit list of seeds to include (overrides start/end).",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=None,
        help="Start seed (inclusive) if using a range.",
    )
    parser.add_argument(
        "--end-seed",
        type=int,
        default=None,
        help="End seed (inclusive) if using a range.",
    )
    args = parser.parse_args()

    dataset = args.dataset
    cfg = get_dataset_config(dataset)
    loss_matrix = cfg["loss_matrix"]
    alpha_list = cfg["alpha_list"]
    critical_labels = cfg["critical_labels"]
    label_names = cfg["label_names"]
    critical_mistake_suffix = cfg["critical_mistake_suffix"]

    if args.results_dir is None:
        results_root = Path("results/Covid_data" if dataset == "covid" else "results/MovieLens_data")
    else:
        results_root = Path(args.results_dir)

    prob_dir = results_root / "probabilities"
    seed_files_all = find_seed_files(prob_dir)
    seed_files = select_seed_files(seed_files_all, args.seeds, args.start_seed, args.end_seed)

    if not seed_files:
        raise RuntimeError(f"No seed files found in {prob_dir}")

    seed_results = []
    for seed, path in seed_files:
        with np.load(path) as data:
            cal_probs = data["cal_probs"]
            cal_labels = data["cal_labels"]
            test_probs = data["test_probs"]
            test_labels = data["test_labels"]
        seed_results.append(
            evaluate_seed(
                cal_probs,
                cal_labels,
                test_probs,
                test_labels,
                loss_matrix,
                alpha_list,
                critical_labels,
            )
        )
        print(f"Processed seed {seed} from {path}")

    score_names = ["LAS", "APS", "SOCOP"]

    wcr_mean = {}
    wcr_se = {}
    for m in ["ROCP", "RAC"]:
        values = [res["worst_case_risk"][m] for res in seed_results]
        wcr_mean[m], wcr_se[m] = mean_and_se(values)

    wcr_scores_mean = {s: {} for s in score_names}
    wcr_scores_se = {s: {} for s in score_names}
    for s in score_names:
        for rule in ["a_ROCP", "a_RAC"]:
            values = [res["worst_case_risk_scores"][s][rule] for res in seed_results]
            wcr_scores_mean[s][rule], wcr_scores_se[s][rule] = mean_and_se(values)

    rl_mean = {}
    rl_se = {}
    for m in ["ROCP", "RAC"]:
        values = [res["realized_loss"][m] for res in seed_results]
        rl_mean[m], rl_se[m] = mean_and_se(values)

    rl_scores_mean = {s: {} for s in score_names}
    rl_scores_se = {s: {} for s in score_names}
    for s in score_names:
        for rule in ["a_ROCP", "a_RAC"]:
            values = [res["realized_loss_scores"][s][rule] for res in seed_results]
            rl_scores_mean[s][rule], rl_scores_se[s][rule] = mean_and_se(values)

    mis_mean = {}
    mis_se = {}
    for m in ["ROCP", "RAC"] + score_names:
        values = [res["miscoverage"][m] for res in seed_results]
        mis_mean[m], mis_se[m] = mean_and_se(values)

    best_realized_losses = [res["best_realized_loss"] for res in seed_results]
    best_mean, best_se = mean_and_se(best_realized_losses)

    cm_mean = {"best": {}, "rocp": {}, "rac": {}}
    cm_se = {"best": {}, "rocp": {}, "rac": {}}
    for method in ["best", "rocp", "rac"]:
        for lbl in critical_labels:
            vals = [res["critical_mistake"][method][lbl] for res in seed_results]
            cm_mean[method][lbl], cm_se[method][lbl] = mean_and_se(vals)

    # Print mean ± SE per method (per alpha)
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
    print(f"method ROCP: {format_mean_se(mis_mean['ROCP'], mis_se['ROCP'])}")
    print(f"method RAC: {format_mean_se(mis_mean['RAC'], mis_se['RAC'])}")
    for s in score_names:
        print(f"method {s}: {format_mean_se(mis_mean[s], mis_se[s])}")

    cm_labels = [f"{label_names[l]} {critical_mistake_suffix}" for l in critical_labels]
    print("\nCritical mistakes at alpha=0.05 (mean +- SE, %), label order:")
    print(cm_labels)
    for method in ["best", "rocp", "rac"]:
        cm_mean_list = [100.0 * cm_mean[method][l] for l in critical_labels]
        cm_se_list = [100.0 * cm_se[method][l] for l in critical_labels]
        print(f"method {method}: {format_mean_se(cm_mean_list, cm_se_list)}")

    # Plot against the actual alpha values, but show clean ticks 0.00, 0.02, ..., max(alpha).
    x_alpha = np.asarray(alpha_list, dtype=float)
    max_alpha = float(np.max(x_alpha)) if x_alpha.size else 0.0
    tick_step = 0.02
    tick_end = round(math.ceil(max_alpha / tick_step) * tick_step, 10) if max_alpha > 0 else tick_step
    x_ticks = np.arange(0.0, tick_end + 1e-12, tick_step)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    ax_wcr, ax_rl = axes[0, 0], axes[0, 1]
    ax_mis, ax_cm = axes[1, 0], axes[1, 1]

    score_colors = {"LAS": "#ff7f0e", "APS": "#2ca02c", "SOCOP": "#d62728"}

    # (1) Averaged worst-case risk
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

    # (2) Averaged realized loss
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
        np.full_like(x_alpha, best_mean, dtype=float),
        np.full_like(x_alpha, best_se, dtype=float),
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

    # (3) Averaged miscoverage
    plot_with_band(
        ax_mis,
        x_alpha,
        mis_mean["ROCP"],
        mis_se["ROCP"],
        "ROCP",
        "black",
    )
    plot_with_band(
        ax_mis,
        x_alpha,
        mis_mean["RAC"],
        mis_se["RAC"],
        "RAC",
        "gray",
    )
    for s in score_names:
        plot_with_band(
            ax_mis,
            x_alpha,
            mis_mean[s],
            mis_se[s],
            s,
            score_colors[s],
        )
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

    # (4) Critical mistake rates at alpha=0.05
    vals_best = [100.0 * cm_mean["best"][l] for l in critical_labels]
    vals_rocp = [100.0 * cm_mean["rocp"][l] for l in critical_labels]
    vals_rac = [100.0 * cm_mean["rac"][l] for l in critical_labels]
    err_best = [100.0 * cm_se["best"][l] for l in critical_labels]
    err_rocp = [100.0 * cm_se["rocp"][l] for l in critical_labels]
    err_rac = [100.0 * cm_se["rac"][l] for l in critical_labels]

    gx = np.arange(len(critical_labels))
    bar_w = 0.25
    ax_cm.bar(gx - bar_w, vals_best, width=bar_w, label="Best response", color="#1f77b4", yerr=err_best, capsize=3)
    ax_cm.bar(gx, vals_rocp, width=bar_w, label="ROCP (alpha=0.05)", color="#ff7f0e", yerr=err_rocp, capsize=3)
    ax_cm.bar(gx + bar_w, vals_rac, width=bar_w, label="RAC (alpha=0.05)", color="#2ca02c", yerr=err_rac, capsize=3)
    ax_cm.set_title("(d) Critical mistake rates (alpha=0.05)")
    ax_cm.set_xticks(gx)
    ax_cm.set_xticklabels([f"{label_names[l]} {critical_mistake_suffix}" for l in critical_labels], rotation=15, ha="right")
    ax_cm.set_ylabel("Percentage of critical decisions")
    ax_cm.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax_cm.legend(fontsize=8)

    fig.tight_layout()
    figures_dir = Path("Figures")
    figures_dir.mkdir(exist_ok=True)
    out_path = figures_dir / f"evaluation_{dataset}_x5_3.png"
    fig.savefig(out_path)
    print(f"Saved combined figure to: {out_path}")


if __name__ == "__main__":
    main()
