import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

exp_name = sys.argv[1]

# Define tasks and corresponding directories
tasks = {
    "RAD; PBRS": f"storage/{exp_name}/log_rad_pbrs_?.csv",
    "RAD; no PBRS": f"storage/{exp_name}/log_rad_no_pbrs_?.csv",
    "no RAD; PBRS": f"storage/{exp_name}/log_no_rad_pbrs_?.csv",
    "no RAD; no PBRS": f"storage/{exp_name}/log_no_rad_no_pbrs_?.csv",
    "RAD; PBRS (LSTM)": f"storage/{exp_name}/log_rad_pbrs_lstm_?.csv",
    "RAD; no PBRS (LSTM)": f"storage/{exp_name}/log_rad_no_pbrs_lstm_?.csv",
    "no RAD; PBRS (LSTM)": f"storage/{exp_name}/log_no_rad_pbrs_lstm_?.csv",
    "no RAD; no PBRS (LSTM)": f"storage/{exp_name}/log_no_rad_no_pbrs_lstm_?.csv",
    # "rad_pbrs_gru": f"storage/{exp_name}/log_rad_pbrs_gru_?.csv",
    # "rad_no_pbrs_gru": f"storage/{exp_name}/log_rad_no_pbrs_gru_?.csv",
    # "no_rad_pbrs_gru": f"storage/{exp_name}/log_no_rad_pbrs_gru_?.csv",
    # "no_rad_no_pbrs_gru": f"storage/{exp_name}/log_no_rad_no_pbrs_gru_?.csv",
}

# colors = {"RAD": "tab:blue", "NO RAD": "tab:orange"}
# colors = {
#     "rad_pbrs": f"storage/{exp_name}/log_rad_pbrs_*.csv",
#     "rad_no_pbrs": f"storage/{exp_name}/log_rad_no_pbrs_*.csv",
#     "no_rad_pbrs": f"storage/{exp_name}/log_no_rad_pbrs_*.csv",
#     "no_rad_no_pbrs": f"storage/{exp_name}/log_no_rad_no_pbrs_*.csv",
# }

# Dictionary to hold mean/std for each task
results = {}

for task, pattern in tasks.items():
    log_files = glob.glob(pattern)
    if not log_files:
        print(f"Warning: No files found for task={task} with pattern {pattern}")
        continue

    dfs = [pd.read_csv(f) for f in log_files]

    # ensure unique timesteps per df (take mean if duplicates exist)
    for i in range(len(dfs)):
        if dfs[i]["timestep"].duplicated().any():
            dfs[i] = dfs[i].groupby("timestep", as_index=False).mean()

    # union of all timesteps
    all_timesteps = sorted(set().union(*[df["timestep"].values for df in dfs]))

    # reindex each df to have the full timestep range
    reindexed_dfs = []
    for df in dfs:
        df = df.set_index("timestep").reindex(all_timesteps)
        reindexed_dfs.append(df.reset_index().rename(columns={"index": "timestep"}))

    timesteps = np.array(all_timesteps)
    base_columns = [c for c in dfs[0].columns if c != "timestep"]

    data_mean, data_std = {}, {}
    for col in base_columns:
        values = np.stack([df[col].values for df in reindexed_dfs], axis=1)  # (T, num_seeds)
        data_mean[col] = np.nanmean(values, axis=1)
        data_std[col] = np.nanstd(values, axis=1)

    results[task] = {
        "timesteps": timesteps,
        "mean": data_mean,
        "std": data_std,
        "columns": base_columns,
    }

# Create folder to save plots
os.makedirs(f"storage/plots/{exp_name}", exist_ok=True)

# Plot
for col in results[list(tasks.keys())[0]]["columns"]:
    plt.figure(figsize=(8, 4))
    for task in tasks.keys():
        if task not in results:
            continue
        timesteps = results[task]["timesteps"]
        mean = results[task]["mean"][col]
        std = results[task]["std"][col]

        # plt.plot(timesteps, mean, label=task, color=colors[task])
        # plt.fill_between(timesteps, mean - std, mean + std, alpha=0.2, color=colors[task])
        plt.plot(timesteps, mean, label=task)
        plt.fill_between(timesteps, mean - std, mean + std, alpha=0.2)

    plt.xlabel("timestep")
    plt.ylabel(col)
    plt.title(f"{exp_name} -- {col}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if "prob_success" in col.lower():
        plt.ylim(0.15, 1)
    if "disc_return_mean" in col.lower():
        plt.ylim(0, 1.2)
    plt.tight_layout()

    pdf_path = os.path.join("storage", "plots", exp_name, f"{col}.pdf")
    plt.savefig(pdf_path)
    plt.close()

print(f"✅ Plots saved in storage/plots/{exp_name}")
