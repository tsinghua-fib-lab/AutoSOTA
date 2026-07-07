import os
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

def get_topk_means(scores, top_ks=[1, 5, 10, 20]):
    sorted_scores = np.sort(scores)
    return {f"top{k}": np.mean(sorted_scores[:k]) for k in top_ks}

def load_alidiff(dir_path="results/diff", num_samples=100):
    results = {f"top{k}": [] for k in [1, 5, 10, 20]}
    for i in range(num_samples):
        path = os.path.join("..", dir_path, str(i), "init_scores.csv")
        df = pd.read_csv(path)
        res = get_topk_means(df["rdkit3d_docking_score"].values)
        for k in res: results[k].append(res[k])
    return results

def load_kbts(dir_path="results/diff", num_samples=100):
    results = {f"top{k}": [] for k in [1, 5, 10, 20]}
    for i in range(num_samples):
        path = os.path.join("..", dir_path, str(i), "1_result.csv")
        df = pd.read_csv(path)
        res = get_topk_means(df["docking_score"].values)
        for k in res: results[k].append(res[k])
    return results

def load_elillm(dir_path="baselines/ELILLM-diff", num_samples=100):
    results = {f"top{k}": [] for k in [1, 5, 10, 20]}
    for i in range(num_samples):
        path = os.path.join("..", dir_path, str(i), "1_result.csv")
        df = pd.read_csv(path)
        res = get_topk_means(df["Docking Score"].values)
        for k in res: results[k].append(res[k])
    return results

def load_lmlf(dir_path="baselines/LMLF-diff", num_samples=100):
    results = {f"top{k}": [] for k in [1, 5, 10, 20]}
    for i in range(num_samples):
        path = os.path.join("..", dir_path, str(i), "1_result_ori.csv")
        df = pd.read_csv(path)
        res = get_topk_means(df["Docking Score"].values)
        for k in res: results[k].append(res[k])
    return results

def load_targetdiff(dir_path="baselines/targetdiff_result", num_samples=100):
    results = {f"top{k}": [] for k in [1, 5, 10, 20]}
    for i in range(num_samples):
        path = os.path.join("..", dir_path, str(i), "score.csv")
        df = pd.read_csv(path)
        res = get_topk_means(df["docking score"].values)
        for k in res: results[k].append(res[k])
    return results



def run_superiority_test_with_holm(kbts_data, other_methods_dict, top_ks=[1, 5, 10, 20]):
    """
    For each top-K setting, test whether K-BTS significantly outperforms other methods
    using one-sided paired Wilcoxon tests (lower is better),
    with Holm–Bonferroni correction applied across methods.
    """
    all_results = []

    for k in top_ks:
        key = f"top{k}"
        per_k_results = []

        for m_name, m_data in other_methods_dict.items():
            kbts_vec = np.array(kbts_data[key])
            other_vec = np.array(m_data[key])

            mask = ~np.isnan(kbts_vec) & ~np.isnan(other_vec)

            stat, p_val = stats.wilcoxon(
                kbts_vec[mask],
                other_vec[mask],
                alternative='less'  # lower docking score is better
            )

            per_k_results.append({
                "Comparison": f"K-BTS vs {m_name}",
                "Metric": key,
                "KBTS_Mean": np.mean(kbts_vec[mask]),
                "Other_Mean": np.mean(other_vec[mask]),
                "P_raw": p_val
            })

        # Holm
        df_k = pd.DataFrame(per_k_results)

        reject, p_corrected, _, _ = multipletests(
            df_k["P_raw"],
            alpha=0.05,
            method="holm"
        )

        df_k["P_corrected"] = p_corrected
        df_k["Significant_Holm"] = reject

        all_results.append(df_k)

    return pd.concat(all_results, ignore_index=True)



def plot_paired_difference_boxplot(
    kbts_data,
    other_methods_dict,
    top_k=1,
    save_path=None
):
    """
    Paired difference boxplot:
    Δ = Other - K-BTS (positive means K-BTS is better)
    """
    key = f"top{top_k}"
    diff_data = []
    labels = []

    for m_name, m_data in other_methods_dict.items():
        kbts_vec = np.array(kbts_data[key])
        other_vec = np.array(m_data[key])

        mask = ~np.isnan(kbts_vec) & ~np.isnan(other_vec)
        diffs = other_vec[mask] - kbts_vec[mask]

        diff_data.append(diffs)
        labels.append(m_name)

    plt.figure(figsize=(8, 4))
    plt.boxplot(
        diff_data,
        vert=True,
        showfliers=True
    )
    plt.axhline(0, linestyle="--")
    plt.xticks(range(1, len(labels) + 1), labels, rotation=15)
    plt.ylabel("Docking Score Difference (Baseline − K-BTS)")
    plt.title(f"Paired Difference Distribution (Top-{top_k})")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_paired_difference_boxplot_multitop(
    kbts_data,
    other_methods_dict,
    top_ks=[1, 5, 10, 20],
    save_path=None
):
    n_top = len(top_ks)
    labels = list(other_methods_dict.keys())

    fig, axes = plt.subplots(n_top, 1, figsize=(8, 3 * n_top), sharex=True)

    if n_top == 1:
        axes = [axes]

    for ax, k in zip(axes, top_ks):
        key = f"top{k}"
        diff_data = []
        for m_name in labels:
            kbts_vec = np.array(kbts_data[key])
            other_vec = np.array(other_methods_dict[m_name][key])
            mask = ~np.isnan(kbts_vec) & ~np.isnan(other_vec)
            diffs = other_vec[mask] - kbts_vec[mask]
            diff_data.append(diffs)

        ax.boxplot(diff_data, vert=True, showfliers=True)
        ax.axhline(0, linestyle="--", color="gray")
        ax.set_ylabel(f"Top-{k} Δ")
        ax.set_title(f"Top-{k} Docking Score Difference (Baseline − K-BTS)")

    axes[-1].set_xticks(range(1, len(labels) + 1))
    axes[-1].set_xticklabels(labels, rotation=15)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


if __name__ == "__main__":
    alidiff = load_alidiff()
    kbts = load_kbts()
    elillm = load_elillm()
    lmlf = load_lmlf()
    targetdiff = load_targetdiff()
    comparison_map = {
        "AlIDIFF": alidiff,
        "TargetDiff": targetdiff,
        "ELILLM-diff": elillm,
        "LMLF-diff": lmlf
    }
    plot_paired_difference_boxplot_multitop(
        kbts_data=kbts,
        other_methods_dict=comparison_map,
        top_ks=[1, 5, 10, 20],
        save_path="boxplot_top1-20.pdf"
    )



