"""
Spearman rank correlation stability across cross-validation folds.

Measures pairwise Spearman correlation between importance rankings
on different held-out folds for ensemble vs. sub-models strategies.
"""

# %%
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib.ticker import ScalarFormatter
from scipy.stats import spearmanr
from tqdm import tqdm

# %%

results_dir = Path("../results")


def read_one(dataset, model, n_samples, p, ensembling, snr, seed, method):
    """
        Read and process, by computing Spearman rank correlation across folds, the
        results of one experiment. An experiment is defined by a dataset, a model,
        a number of samples, a number of features, an ensembling strategy, a
    signal-to-noise ratio, a random seed and a method (LOCO, SAGE...).
    """
    file_path = (
        results_dir
        / f"{dataset}_{model}_n{n_samples}_p{p}_{ensembling}10_snr{snr}"
        / f"{method}_{dataset}_{seed}.csv"
    )
    if not file_path.exists():
        print(f"File {file_path} does not exist.")
        return pd.DataFrame()

    df = pd.read_csv(file_path)
    df["model"] = df["model"].apply(
        lambda x: "ensemble" if x == "ensemble" else "sub-models"
    )
    df = df.groupby(["model", "feature", "fold"])["importance"].mean().reset_index()

    corrs_ens = []
    corrs_sub = []
    df_ens = df[df["model"] == "ensemble"]
    df_sub = df[df["model"] == "sub-models"]
    n_folds = df["fold"].nunique()
    for i in range(n_folds):
        for j in range(i + 1, n_folds):
            df_ens_i = df_ens[df_ens["fold"] == i]
            df_ens_k = df_ens[df_ens["fold"] == j]
            corr_ens, _ = spearmanr(df_ens_i["importance"], df_ens_k["importance"])
            corrs_ens.append(corr_ens)

            df_sub_i = df_sub[df_sub["fold"] == i]
            df_sub_k = df_sub[df_sub["fold"] == j]
            corr_sub, _ = spearmanr(df_sub_i["importance"], df_sub_k["importance"])
            corrs_sub.append(corr_sub)

    return pd.DataFrame(
        {
            "spearmanr": [np.mean(corrs_ens), np.mean(corrs_sub)],
            "method": ["ensemble", "sub-models"],
            "dataset": dataset,
            "model": model,
            "seed": seed,
            "n_samples": n_samples,
        }
    )


# %%

n_jobs = 10

dataset_list = ["friedman1", "ishigami", "g_function"]
n_samples_list = [128, 256, 512, 1024, 2048]
p = 20
ensembling_list = ["voting", "bagging"]
snr = 1
seeds = range(1, 101)
method_list = ["loco", "sage"]
model_list = ["mlp", "rf"]

results = Parallel(n_jobs=n_jobs)(
    delayed(read_one)(d, m, n, p, ensembling, snr, s, method)
    for d in dataset_list
    for m in model_list
    for n in n_samples_list
    for ensembling in ensembling_list
    for method in method_list
    for s in tqdm(seeds, desc="Processing seeds")
)

# %%
mpl.rcParams["figure.dpi"] = 300

df_plot = pd.concat(results, ignore_index=True)

palette = {
    "ensemble": "#648fff",
    "sub-models": "#fe6100",
}
_, axes = plt.subplots(2, 3, figsize=(9, 6), sharex=True)

for j, method in enumerate(method_list):
    for i, dataset in enumerate(dataset_list):
        ax = axes[j, i]
        df_plot_subset = df_plot[df_plot["dataset"] == dataset]
        sns.lineplot(
            data=df_plot_subset,
            x="n_samples",
            y="spearmanr",
            hue="method",
            style="model",
            markers=["o", "^"],
            palette=palette,
            ax=ax,
            legend=i == 0,
        )
        ax.set_title(dataset, y=0.9)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel("Number of samples")

    axes[j, 1].annotate(
        method.upper(),
        xy=(0.5, 1.1),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1),
    )
sns.despine()
plt.tight_layout()

plt.savefig("./figures_pdf/spearman_rank.pdf", bbox_inches="tight")
plt.show()
# %%
