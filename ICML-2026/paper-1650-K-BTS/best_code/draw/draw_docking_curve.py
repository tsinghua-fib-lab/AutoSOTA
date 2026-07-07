import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
def aggregate_docking_curve(
    all_scores,
    step=10,
    max_mols=100,
):

    bucket_scores = {k: [] for k in range(step, max_mols + 1, step)}

    for scores in all_scores:
        if scores is None or len(scores) == 0:
            continue

        scores = np.asarray(scores)
        max_k = min(len(scores), max_mols)

        for k in range(step, max_k + 1, step):
            bucket = scores[k-step:k]
            bucket = bucket[bucket <= 0]   # 跳过 >0

            if len(bucket) == 0:
                continue

            bucket_scores[k].append(np.mean(bucket))

    x, y = [], []
    for k in sorted(bucket_scores.keys()):
        if len(bucket_scores[k]) == 0:
            continue
        x.append(k)
        y.append(np.mean(bucket_scores[k]))

    return x, y

def load_KBTS_rand(base_dir, num_targets=100):

    all_scores = []

    for i in range(num_targets):
        csv_path = os.path.join(base_dir, str(i), "1_result.csv")

        if not os.path.exists(csv_path):
            all_scores.append(None)
            continue

        df = pd.read_csv(csv_path)
        if "docking_score" not in df.columns:
            all_scores.append(None)
            continue

        all_scores.append(df["docking_score"].to_numpy())

    return all_scores

def load_LMLF_rand(base_dir, num_targets=100, seed_range=(1, 2)):

    all_scores = []

    for i in range(num_targets):
        target_dir = os.path.join(base_dir, str(i))
        lmlf_dockings = []

        if not os.path.exists(target_dir):
            all_scores.append(None)
            continue

        for seed in range(*seed_range):
            result_path = os.path.join(target_dir, f"{seed}_result_ori.csv")
            if not os.path.exists(result_path):
                continue

            df = pd.read_csv(result_path)
            if "Docking Score" not in df.columns:
                continue

            lmlf_dockings.extend(df["Docking Score"].tolist())

        if len(lmlf_dockings) == 0:
            all_scores.append(None)
        else:
            all_scores.append(lmlf_dockings)

    return all_scores



def load_ELILLM_rand(base_dir, num_targets=100, seed_range=(1, 2)):
    all_scores = []
    for i in range(num_targets):
        target_dir = os.path.join(base_dir, str(i))

        if not os.path.exists(target_dir):
            all_scores.append(None)
            continue

        bo_dockings = []

        for seed in range(*seed_range):
            result_path = os.path.join(target_dir, f"{seed}_result.csv")
            if not os.path.exists(result_path):
                continue

            df = pd.read_csv(result_path)

            if "Docking Score" not in df.columns:
                continue

            bo_dockings.extend(df["Docking Score"].tolist())

        if len(bo_dockings) == 0:
            all_scores.append(None)
        else:
            all_scores.append(bo_dockings)

    return all_scores

def load_KBTS_diff(base_dir, num_targets=100):


    all_scores = []

    for i in range(num_targets):
        csv_path = os.path.join(base_dir, str(i), "1_result.csv")

        if not os.path.exists(csv_path):
            all_scores.append(None)
            continue

        df = pd.read_csv(csv_path)
        if "docking_score" not in df.columns:
            all_scores.append(None)
            continue

        all_scores.append(df["docking_score"].to_numpy())

    return all_scores

def load_LMLF_diff(base_dir, num_targets=100, seed_range=(1, 2)):


    all_scores = []

    for i in range(num_targets):
        target_dir = os.path.join(base_dir, str(i))
        lmlf_dockings = []

        if not os.path.exists(target_dir):
            all_scores.append(None)
            continue

        for seed in range(*seed_range):
            result_path = os.path.join(target_dir, f"{seed}_result_ori.csv")
            if not os.path.exists(result_path):
                continue

            df = pd.read_csv(result_path)
            if "Docking Score" not in df.columns:
                continue
            lmlf_dockings.extend(df["Docking Score"].tolist())

        if len(lmlf_dockings) == 0:
            all_scores.append(None)
        else:
            all_scores.append(lmlf_dockings)

    return all_scores

def load_ELILLM_diff(base_dir, num_targets=100, seed_range=(1, 2)):
    all_scores = []
    for i in range(num_targets):
        target_dir = os.path.join(base_dir, str(i))

        if not os.path.exists(target_dir):
            all_scores.append(None)
            continue

        bo_dockings = []

        for seed in range(*seed_range):
            result_path = os.path.join(target_dir, f"{seed}_result.csv")
            if not os.path.exists(result_path):
                continue

            df = pd.read_csv(result_path)

            if "Docking Score" not in df.columns:
                continue

            bo_dockings.extend(df["Docking Score"].tolist())

        if len(bo_dockings) == 0:
            all_scores.append(None)
        else:
            all_scores.append(bo_dockings)

    return all_scores

def plot_multiple_curves(curves, save_path=None):
    """
    curves: dict[str, tuple[list, list]]
        method_name -> (x, y)
    """
    plt.figure(figsize=(8, 6))
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:6.2f}"))
    plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.15)

    markers = ['o', 's', '^', 'D', 'v', '*', 'p', 'H', 'X', '+']

    for idx, (name, (x, y)) in enumerate(curves.items()):
        plt.plot(
            x, y,
            marker=markers[idx % len(markers)],
            linewidth=2.5,
            markersize=8,
            label=name
        )

    plt.xlabel("Iteration (number of generated molecules)", fontsize=18, fontweight='bold')
    plt.ylabel("Average docking score", fontsize=18, fontweight='bold')
    plt.legend(
        fontsize=16,
        frameon=True,
        framealpha=0.9,
        edgecolor="black",
        handlelength=2.5,
        handletextpad=0.8,
        loc='best'
    )

    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path,
                    format="pdf",
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=0
                    )

    plt.show()





if __name__ == "__main__":

    curves = {}

    # scores_kbts = load_KBTS_rand("../results/rand", num_targets=10)
    # curves["K-BTS-rand"] = aggregate_docking_curve(scores_kbts, step=8, max_mols=40)
    #
    # scores_wo_warmstart = load_KBTS_rand("../results/rand_wo_warmstart", num_targets=10)
    # curves["wo_warmstart"] = aggregate_docking_curve(scores_wo_warmstart, step=8, max_mols=40)
    #
    # scores_wo_knowledge = load_KBTS_rand("../results/rand_wo_knowledge", num_targets=10)
    # curves["wo_knowledge"] = aggregate_docking_curve(scores_wo_knowledge, step=8, max_mols=40)
    #
    # scores_wo_lower = load_KBTS_rand("../results/rand_wo_lower", num_targets=10)
    # curves["wo_lower"] = aggregate_docking_curve(scores_wo_lower, step=8, max_mols=40)
    #
    # scores_wo_upper = load_KBTS_rand("../results/rand_wo_upper", num_targets=10)
    # curves["wo_upper"] = aggregate_docking_curve(scores_wo_upper, step=8, max_mols=40)

    scores_kbts = load_KBTS_rand("../results/rand", num_targets=100)
    curves["K-BTS-rand"] = aggregate_docking_curve(scores_kbts)

    scores_elillm = load_ELILLM_rand("../baselines/ELILLM-rand", num_targets=100)
    curves["ELILLM-rand"] = aggregate_docking_curve(scores_elillm)

    scores_lmlf = load_LMLF_rand("../baselines/LMLF-rand", num_targets=100)
    curves["LMLF-rand"] = aggregate_docking_curve(scores_lmlf)

    # scores_kbts = load_KBTS_diff("../results/diff", num_targets=100)
    # curves["K-BTS-diff"] = aggregate_docking_curve(scores_kbts)
    #
    # scores_elillm = load_ELILLM_diff("../baselines/ELILLM-diff", num_targets=100)
    # curves["ELILLM-diff"] = aggregate_docking_curve(scores_elillm)
    #
    # scores_lmlf = load_LMLF_diff("../baselines/LMLF-diff", num_targets=100)
    # curves["LMLF-diff"] = aggregate_docking_curve(scores_lmlf)


    plot_multiple_curves(
        curves,
        # save_path="docking_curve_comparison_ablation.pdf"
        save_path="docking_curve_comparison.pdf"
    )
