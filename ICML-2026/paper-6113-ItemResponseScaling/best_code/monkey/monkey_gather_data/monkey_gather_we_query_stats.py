import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import snapshot_download
import glob

# ---- 0) map raw names to aligned names ----
model_name_map = {
    'meta-llama-Meta-Llama-3-8B-Instruct': 'Meta-Llama-3-8B-Instruct',
    'meta-llama-Meta-Llama-3-70B':      'Meta-Llama-3-70B-Instruct',
    'pythia-12b':                       'Pythia_12B',
    'pythia-6.9b':                      'Pythia_6.9B',
}

# 1) Define your ordered scenarios
ordered_scenarios = [
    "gsm", "mmlu", "lsat_qa", "legalbench",
    "bbq", "commonsense", "math", "med_qa", "legal_support"
]

benchmark2scenario = {
    "lite":    ["legalbench", "math", "commonsense", "med_qa", "gsm"],
    "mmlu":    ["mmlu"],
    "classic": ["bbq", "lsat_qa", "legal_support"]
}
scenario2benchmark = {
    **{
        scen: bench
        for bench, scenarios in benchmark2scenario.items()
        for scen in scenarios
    },
    'harm_bench': 'safety',
    'gsm':        'lite',
}

# download all files
local_dir = snapshot_download(
    repo_id="stair-lab/monkey_queries",
    repo_type="dataset"
)
paths = []
for scen in scenario2benchmark:
    paths.extend(glob.glob(f"{local_dir}/*{scen}.json"))

# 2) Count questions per (aligned) model × scenario
monkey_counts_by_scenario = {}
for path in tqdm(paths):
    stem = Path(path).stem
    scenario_name = next((s for s in scenario2benchmark if stem.endswith(f"_{s}")), None)

    # skip harm_bench entirely
    if scenario_name == "harm_bench":
        continue

    # extract raw model name
    raw_model = stem[: -(len(scenario_name) + 1)]

    # skip Pythia variants other than 6.9B or 12B
    if raw_model.startswith("Pythia") and not (
        raw_model.endswith("6.9B") or raw_model.endswith("12B")
    ):
        continue

    # map raw_model to aligned name
    model_name = model_name_map.get(raw_model, raw_model)

    df = pd.read_json(path)
    count = len(df)
    monkey_counts_by_scenario.setdefault(scenario_name, []).append((model_name, count))

# 3) Build sorted list of all aligned models
all_models = sorted({
    m for entries in monkey_counts_by_scenario.values() for m, _ in entries
})

# 4) Assemble DataFrame of counts
count_matrix = []
for scenario in ordered_scenarios:
    sc_map = {m: np.nan for m in all_models}
    for m, cnt in monkey_counts_by_scenario.get(scenario, []):
        sc_map[m] = cnt
    count_matrix.append([sc_map[m] for m in all_models])

counts_df = pd.DataFrame(
    count_matrix,
    index=ordered_scenarios,
    columns=all_models
)

# 5) Plot heatmap
plt.figure(figsize=(len(all_models)*0.5 + 3,
                    len(ordered_scenarios)*0.5 + 3))
sns.set_style("white")
ax = sns.heatmap(
    counts_df,
    cmap="YlGnBu",
    linewidths=0.5,
    linecolor="lightgray",
    annot=True,
    fmt=".0f",
    cbar_kws={"label": "Number of Questions"},
    annot_kws={"fontsize": 8}
)
ax.set_xlabel("Model")
ax.set_ylabel("Scenario")
ax.set_title("Number of Questions per Model and Scenario")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("monkey_question_counts_heatmap.png", dpi=300,
            bbox_inches="tight")