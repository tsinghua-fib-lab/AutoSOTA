import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import snapshot_download
from collections import defaultdict
import sys
sys.path.append("..")
from utils import calculate_flop
from tueplots import bundles
bundles.icml2024()
    
DATASETS = ["mmlu_anatomy", "arc_challenge", "hellaswag", "mmlu_abstract_algebra", "mmlu_astronomy"]
METRICS = ["prob_vocab_correct", "prob_choices_correct", "acc"]
DISPLAY_METRICS = {
    "prob_vocab_correct": "p_vocab",
    "prob_choices_correct": "p_choices", 
    "acc": "acc"
}
SEEDS = range(10) 

if __name__ == "__main__":
    subset_size = 50
    eps = 1e-15
    results_dict = defaultdict(lambda: defaultdict(dict))
    
    # organize data
    for dataset in tqdm(DATASETS):
        for metric in METRICS:
            ds = load_dataset(f"RylanSchaeffer/per_sample_scores.{dataset}.{metric}.2024-05-31")
            df = ds["train"].to_pandas()
            model_nicknames = df["Model Nickname"].unique()
            model_families = pd.Series(model_nicknames).str.split("_").str[0].unique()
            
            for model_family in model_families:
                df_f = df[df["Model Nickname"].astype(str).str.startswith(model_family)].copy()
                resmat = df_f.pivot_table(
                    index="Model Nickname",
                    columns="sample_idx",
                    values="score",
                )
                resmat = resmat.sort_index(axis=1)
                resmat = resmat.loc[sorted(resmat.index, key=lambda x: calculate_flop(x))]
                results_dict[dataset][model_family][f"resmat_{metric}"] = resmat
            
                # y_full
                y_full = resmat.mean(axis=1, skipna=True).to_numpy()
                y_full_by_seed = {}
                for s in SEEDS:
                    rng = np.random.default_rng(s)
                    y_full_seed = []
                    for _, row in resmat.iterrows():
                        vals = row.to_numpy()
                        k = int(0.8 * len(vals))
                        chosen = rng.choice(len(vals), size=k, replace=False)
                        y_full_seed.append(float(np.nanmean(vals[chosen])))
                    y_full_by_seed[s] = y_full_seed
                y_full_3std = (3.0 * np.nanstd(np.array([y_full_by_seed[s] for s in SEEDS], dtype=float), axis=0))
                
                # y_sub
                y_sub_by_seed = {}
                for s in SEEDS:
                    rng = np.random.default_rng(s)
                    y_sub_seed = []
                    for _, row in resmat.iterrows():
                        vals = row.to_numpy()
                        k = subset_size
                        chosen = rng.choice(len(vals), size=k, replace=False)
                        y_sub_seed.append(float(np.nanmean(vals[chosen])))
                    y_sub_by_seed[s] = y_sub_seed
                y_sub_matrix = np.array([y_sub_by_seed[s] for s in SEEDS], dtype=float)  # (n_seeds, n_models)
                y_sub_3std = (3.0 * np.nanstd(y_sub_matrix, axis=0))
                y_sub_avg = np.nanmean(y_sub_matrix, axis=0)
                
                assert y_full.shape[0] == y_full_3std.shape[0] == y_sub_avg.shape[0] == y_sub_3std.shape[0] == resmat.shape[0]
                results_dict[dataset][model_family][f"y_full_{metric}"] = y_full
                results_dict[dataset][model_family][f"y_full_3std_{metric}"] = y_full_3std
                results_dict[dataset][model_family][f"y_sub_{metric}"] = y_sub_avg
                results_dict[dataset][model_family][f"y_sub_3std_{metric}"] = y_sub_3std
    
    for dataset, fam_dict in results_dict.items():
        dataset_temp = "mmlu" if dataset.startswith("mmlu") else dataset
        for model_family, value_dict in fam_dict.items():
            resmats = {m: value_dict[f"resmat_{m}"] for m in METRICS}
            base = resmats[METRICS[0]]
            for m in METRICS[1:]:
                assert resmats[m].index.equals(base.index)
                assert resmats[m].columns.equals(base.columns)

            # store item parameters and flops
            flops = np.array([calculate_flop(model_name) for model_name in base.index], dtype=float)
            cache_dir = snapshot_download(repo_id="allenai/fluid-benchmarking", repo_type="dataset")
            irt_df = pd.read_csv(f"{cache_dir}/data/irt_models/{dataset_temp}.csv", names=["raw_idx", "a", "b"], header=0).set_index("raw_idx")
            col_with_dataset = [f"{dataset}_{c}" for c in base.columns]
            irt_df = irt_df.loc[col_with_dataset]
            discris, zs = irt_df["a"].to_numpy(), irt_df["b"].to_numpy()
            assert discris.shape[0] == zs.shape[0] == base.shape[1]
            value_dict["flops"] = flops
            value_dict["discris"] = discris
            value_dict["zs"] = zs
            
    final_results_dict = {k: dict(v) for k, v in results_dict.items()}
    with open(f"preprocessed_resmat2.pkl", "wb") as f:
        pickle.dump(final_results_dict, f)
    
    # with open(f"preprocessed_resmat2.pkl", "rb") as f:
    #     final_results_dict = pickle.load(f)
    
    # law curve
    output_dir = "../result/pretrain_naive_laws_resmat2"
    os.makedirs(output_dir, exist_ok=True)
    for dataset, fam_dict in final_results_dict.items():
        for model_family, value_dict in fam_dict.items():
            flops = value_dict["flops"]
            with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
                fig, ax = plt.subplots(figsize=(6, 4))
                ax2 = ax.twinx()
                for metric in METRICS:
                    y_full      = value_dict[f"y_full_{metric}"]
                    y_full_3std = value_dict[f"y_full_3std_{metric}"]
                    y_sub       = value_dict[f"y_sub_{metric}"]
                    y_sub_3std  = value_dict[f"y_sub_3std_{metric}"]
                    line_full, = ax.plot(flops, y_full, linewidth=1.5, linestyle="-", label=f"fullset {DISPLAY_METRICS[metric]}")
                    ax.fill_between(flops, y_full - y_full_3std, y_full + y_full_3std, alpha=0.15)
                    color = line_full.get_color()
                    ax.plot(flops, y_sub, linewidth=1.5, linestyle="--", label=f"subset {DISPLAY_METRICS[metric]}", color=color)
                    ax.fill_between(flops, y_sub - y_sub_3std, y_sub + y_sub_3std, alpha=0.15, color=color)

                    if metric == "prob_vocab_correct":
                        y_full_c = np.clip(y_full, eps, 1.0 - eps)
                        y_sub_c  = np.clip(y_sub,  eps, 1.0 - eps)
                        neglogy_full = -np.log(y_full_c)
                        neglogy_sub  = -np.log(y_sub_c)
                        ax2.plot(flops, neglogy_full, linewidth=1.5, linestyle=":", label="fullset -log(p_vocab)", color=color)
                        ax2.plot(flops, neglogy_sub, linewidth=1.5, linestyle="-.", label="fullset -log(p_vocab)", color=color)

                ax.set_xscale("log")
                ax.set_xlabel("FLOP", fontsize=12)
                ax.set_ylabel("Score", fontsize=12)
                ax2.set_ylabel("-log(p_vocab)", fontsize=12)
                ax.tick_params(axis="x", labelsize=10)
                ax.tick_params(axis="y", labelsize=10)
                ax2.tick_params(axis="y", labelsize=10)
                ax.set_title(f"{dataset}, {model_family}", fontsize=14)
                lines, labels = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines + lines2, labels + labels2, fontsize=9, ncol=2)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/{dataset}_{model_family}_law_curves.png", dpi=300, bbox_inches="tight")
                plt.close()
    
    
    
    
    
    
    
    
    