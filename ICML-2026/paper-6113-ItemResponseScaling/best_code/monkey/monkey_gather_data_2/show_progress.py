import pandas as pd
from pathlib import Path
from huggingface_hub import snapshot_download
import matplotlib.pyplot as plt
import seaborn as sns

ordered_scenarios = [
    "aime2025",
    "aime2024",
    "mmlu_pro",
    "global_mmlu_lite",
]

ordered_models = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "meta-llama/Llama-3.3-70B-Instruct",
    "microsoft/Phi-4-mini-reasoning",
    "microsoft/Phi-4-reasoning-plus",
    "HuggingFaceTB/SmolLM3-3B",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/QwQ-32B",
    "allenai/OLMo-2-1124-7B-Instruct",
    "allenai/OLMo-2-1124-13B-Instruct",
    "allenai/OLMo-2-0325-32B-Instruct",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
]

ordered_models = [om.split("/")[-1] for om in ordered_models]

if __name__ == "__main__":
    base_eval_dir = Path(snapshot_download(
        repo_id="stair-lab/denoise_eval_query",
        repo_type="dataset"
    ))
    counts = {
        scen: {model: 0 for model in ordered_models}
        for scen in ordered_scenarios
    }

    for scen in ordered_scenarios:
        scen_dir = base_eval_dir / scen
        if not scen_dir.is_dir():
            continue

        for model in ordered_models:
            model_dir = scen_dir / model
            if not model_dir.is_dir():
                continue

            prompt_dirs = list(model_dir.glob("prompt=*"))
            valid = 0
            for pdir in prompt_dirs:
                for bf in pdir.glob("batch_*.parquet"):
                    if bf.is_file() and bf.stat().st_size > 0:
                        valid += 1
                        break
            counts[scen][model] = valid

    counts_df = pd.DataFrame.from_dict(counts, orient="index")
    counts_df_T = counts_df.T
    print(counts_df_T.shape)
    plt.figure(
        figsize=(len(ordered_scenarios) * 0.6 + 2,
                len(ordered_models) * 0.5)
    )

    sns.set_style("white")
    ax = sns.heatmap(
        counts_df_T,
        cmap="YlGnBu",
        linewidths=0.5,
        linecolor="lightgray",
        annot=True,
        fmt="d",
        cbar_kws={"label": "Processed Prompts"},
        annot_kws={"fontsize": 8}
    )

    # swap axis labels
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Model")

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("show_progress.png", dpi=300, bbox_inches="tight")