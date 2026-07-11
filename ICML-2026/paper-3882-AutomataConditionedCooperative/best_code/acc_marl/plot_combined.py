import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

plt.rcParams.update({
    "font.size": 24,
    "axes.titlesize": 28,
    "axes.labelsize": 28,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 28,
    "axes.titleweight": "bold",
})

# Take multiple experiments from command line
exp_names = sys.argv[1:]  # e.g. python script.py 2buttons_2agents 2rooms_2agents

tasks = {
    "RAD Embd; PBRS": "log_rad_pbrs_?.csv",
    "RAD Embd; no PBRS": "log_rad_no_pbrs_?.csv",
    "no RAD Embd; PBRS": "log_no_rad_pbrs_?.csv",
    "no RAD Embd; no PBRS": "log_no_rad_no_pbrs_?.csv",
}

exp_name_map = {
    "2buttons_2agents": "Buttons-2",
    "2rooms_2agents": "Rooms-2",
    "4buttons_4agents": "Buttons-4",
    "4rooms_4agents": "Rooms-4",
}

col_map = {
    "disc_return_mean": "Discounted Return",
    "prob_success": "Success Probability",
    "prob_fail": "Fail Probability",
    "return_mean": "Return Mean",
}

# --- Function to load results for one experiment ---
def load_results(exp_name):
    results = {}
    for task, pattern in tasks.items():
        log_files = glob.glob(f"storage/{exp_name}/{pattern}")
        if not log_files:
            print(f"⚠️ Warning: No files for {task} in {exp_name}")
            continue

        dfs = [pd.read_csv(f) for f in log_files]

        # deduplicate timesteps
        for i in range(len(dfs)):
            if dfs[i]["timestep"].duplicated().any():
                dfs[i] = dfs[i].groupby("timestep", as_index=False).mean()

        all_timesteps = sorted(set().union(*[df["timestep"].values for df in dfs]))

        reindexed_dfs = []
        for df in dfs:
            df = df.set_index("timestep").reindex(all_timesteps)
            reindexed_dfs.append(df.reset_index().rename(columns={"index": "timestep"}))

        timesteps = np.array(all_timesteps)
        base_columns = [c for c in dfs[0].columns if c != "timestep"]

        data_mean, data_std = {}, {}
        for col in base_columns:
            values = np.stack([df[col].values for df in reindexed_dfs], axis=1)
            data_mean[col] = np.nanmean(values, axis=1)
            data_std[col] = np.nanstd(values, axis=1)

        results[task] = {
            "timesteps": timesteps,
            "mean": data_mean,
            "std": data_std,
            "columns": base_columns,
        }
    return results


# --- Plot side by side for each column ---
for col in list(col_map.keys()):
    fig, axes = plt.subplots(1, len(exp_names), figsize=(7 * len(exp_names), 6), sharey=True)

    if len(exp_names) == 1:
        axes = [axes]  # make iterable

    for ax, exp_name in zip(axes, exp_names):
        results = load_results(exp_name)
        for task in tasks.keys():
            if task not in results:
                continue
            timesteps = results[task]["timesteps"]
            mean = results[task]["mean"][col]
            std = results[task]["std"][col]

            ax.plot(timesteps, mean, label=task, linewidth=2.5)
            ax.fill_between(timesteps, mean - std, mean + std, alpha=0.3)

        ax.set_xlabel("Timestep")
        ax.set_ylabel(col_map[col])
        ax.set_title(exp_name_map.get(exp_name, exp_name))
        ax.grid(True, alpha=0.3)

        if "prob_success" in col.lower():
            ax.set_ylim(0.0, 1)
        if "disc_return_mean" in col.lower():
            ax.set_ylim(0, 1.2)

    # shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.1, 1, 1])

    os.makedirs(f"storage/plots/combined", exist_ok=True)
    pdf_path = f"storage/plots/combined/{col}.pdf"
    plt.savefig(pdf_path)
    plt.close()

print(f"✅ Combined plots saved in storage/plots/combined")
