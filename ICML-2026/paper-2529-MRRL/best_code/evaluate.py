import argparse
import glob
import json
import os
import re
import warnings

import lightning as L
import numpy as np
import pandas as pd
import torch
from competitors import MREgger
from data_setup import setup_dgp_args
from ivmodels import KClass
from mdcrl import SimDataset
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader
from utils_eval import (
    compute_estimates,
    compute_gcm,
    compute_recon_err,
    load_model,
)

warnings.filter_report = "ignore"
warnings.filterwarnings(
    "ignore",
    message=".*is an instance of `nn.Module` and is already saved during checkpointing.*",
)

# %%


class ModelSelector:
    def __init__(
        self,
        base_output_dir="outputs",
        ckpt_strategy="best",
        metric_key="val/tot_loss",  # Keep original key with slash
        mode="min",
        last_k=20,
        exclude_sim_ids=None,
    ):
        self.base_output_dir = base_output_dir
        self.ckpt_strategy = ckpt_strategy
        self.metric_key = metric_key
        self.mode = mode
        self.last_k = last_k
        self.exclude_sim_ids = exclude_sim_ids or []

    def find_best_sim(self, exp_prefix):
        exp_dir = os.path.join(self.base_output_dir, exp_prefix)
        sim_results = []

        if not os.path.exists(exp_dir):
            print(f"Experiment dir not found: {exp_dir}")
            return None, None, None

        for sim_dir in os.listdir(exp_dir):
            if not sim_dir.isdigit() or int(sim_dir) in self.exclude_sim_ids:
                continue

            sim_id = int(sim_dir)

            if self.ckpt_strategy == "best":
                # --- STRATEGY: BEST ---
                # 1. Locate the Best Checkpoint File
                ckpt_path = os.path.join(exp_dir, sim_dir, "checkpoints")
                best_files = glob.glob(os.path.join(ckpt_path, "best-*.ckpt"))

                if not best_files:
                    continue

                # Get the most recent one if multiple exist
                best_files.sort(key=os.path.getmtime, reverse=True)
                latest_best = os.path.basename(best_files[0])

                # 2. Extract the Epoch Number
                # Matches "best-epoch=388" or "best-epoch=388.ckpt"
                match_epoch = re.search(r"epoch=(\d+)", latest_best)

                if not match_epoch:
                    print(
                        f"Could not parse epoch from {latest_best} in sim {sim_id}"
                    )
                    continue

                best_epoch = int(match_epoch.group(1))

                # 3. Retrieve the Score from CSV (Lookup by Epoch)
                csv_path = os.path.join(
                    exp_dir, sim_dir, "metrics", "val_metrics.csv"
                )

                if not os.path.exists(csv_path):
                    # Fallback: If no CSV, we can't know the score. Skip.
                    print(
                        f"Warning: Checkpoint exists but CSV missing for sim {sim_id}"
                    )
                    continue

                try:
                    df = pd.read_csv(csv_path)

                    # Filter for the specific epoch and metric
                    # Use the original key with slash ("val/tot_loss") because that matches CSV content
                    row = df[
                        (df["epoch"] == best_epoch)
                        & (df["metric"] == self.metric_key)
                    ]

                    if not row.empty:
                        # Take the last logged value for that epoch
                        score = float(row.iloc[-1]["value"])
                        sim_results.append({"sim_id": sim_id, "score": score})
                    else:
                        print(
                            f"Metric {self.metric_key} not found for epoch {best_epoch} in sim {sim_id}"
                        )

                except Exception as e:
                    print(f"Error reading CSV for sim {sim_id}: {e}")

            else:
                # --- STRATEGY: LAST ---
                csv_path = os.path.join(
                    exp_dir, sim_dir, "metrics", "val_metrics.csv"
                )
                if not os.path.exists(csv_path):
                    continue
                try:
                    df = pd.read_csv(csv_path)

                    # 1. FILTER FIRST: Isolate the metric we care about
                    df = df[df["metric"] == self.metric_key].copy()

                    if df.empty:
                        continue

                    # 2. SORT & DEDUPLICATE: Now safe to drop epoch duplicates for THIS metric
                    df = df.sort_values(by=["epoch", "timestamp"])
                    df = df.drop_duplicates(subset=["epoch"], keep="last")

                    # 3. Calculate Score
                    df["value"] = pd.to_numeric(df["value"], errors="coerce")
                    score = df["value"].tail(self.last_k).mean()
                    sim_results.append({"sim_id": sim_id, "score": score})

                except Exception as e:
                    print(f"Error reading {csv_path}: {e}")

        if not sim_results:
            print(f"No valid results found in {exp_dir}")
            return None, None, None

        summary = pd.DataFrame(sim_results)
        if self.mode == "min":
            best_row = summary.loc[summary["score"].idxmin()]
        else:
            best_row = summary.loc[summary["score"].idxmax()]

        return int(best_row["sim_id"]), best_row["score"], summary


def load_hydra_config(exp_prefix, sim_id):
    config_path = os.path.join(
        "outputs", exp_prefix, str(sim_id), ".hydra", "config.yaml"
    )
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Could not find Hydra config at: {config_path}"
        )
    return OmegaConf.load(config_path)


def find_and_load_best_context(args, exp_prefix):
    """
    1. Finds the best simulation ID using ModelSelector.
    2. Loads the Hydra Configuration for that sim.
    3. Loads the PyTorch Model Checkpoint.
    Returns: (model, cfg, sim_id)
    """
    # 1. Find Best Sim
    print(f"\nScanning for best simulation in {exp_prefix}...")
    selector = ModelSelector(
        base_output_dir="outputs",
        ckpt_strategy=args.ckpt_strategy,
        metric_key=args.metric_key,
        mode=args.selection_mode,
        last_k=args.last_k,
        exclude_sim_ids=args.exclude_sim_ids,
    )
    best_sim, best_score, _ = selector.find_best_sim(exp_prefix)

    if best_sim is None:
        raise ValueError(f"No valid simulation found for {exp_prefix}")

    print(
        f" -> Selected Sim ID: {best_sim} (Score: {best_score:.4f} [{args.metric_key}])"
    )

    # 2. Load Config
    cfg = load_hydra_config(exp_prefix, best_sim)

    # 3. Load Model
    # Uses the same args.ckpt_strategy ('best' or 'last') to decide which file to open
    print(f" -> Loading model checkpoint ({args.ckpt_strategy})...")
    model = load_model(
        exp_name=exp_prefix,
        sim_id=best_sim,
        selection_strategy=args.ckpt_strategy,
    )

    return model, cfg, best_sim


def run_evaluation(model, cfg, sim_id, exp_prefix, args):
    """
    Runs evaluation on either In-Sample (Train+Val) or Out-of-Sample (Test) data.
    Now accepts the loaded 'model' object directly.
    """
    # 1. Determine Seeds & Data Mode
    train_seed = cfg.data_seed
    val_seed = cfg.data_seed + 10000
    test_seed = cfg.data_seed + 20000

    mode_str = (
        "Out-of-Sample (Test)"
        if args.use_test_data
        else "In-Sample (Train + Val)"
    )
    print(f"\n--- Evaluating {exp_prefix} | Sim ID: {sim_id} ---")
    print(f"Mode: {mode_str}")

    # 2. Setup Data (Recover DGP args)
    dataset_args, _, _ = setup_dgp_args(cfg)

    # 3. Construct DataLoader
    if not args.use_test_data:
        print(f"Loading Train (seed {train_seed}) and Val (seed {val_seed})...")
        d_train = SimDataset(
            num_draws=cfg.data.n_train,
            num_obs=[cfg.data.n_train for _ in range(cfg.data.n_pop)],
            **dataset_args,
            seed=train_seed,
        )
        d_val = SimDataset(
            num_draws=cfg.data.n_val,
            num_obs=[cfg.data.n_val for _ in range(cfg.data.n_pop)],
            **dataset_args,
            seed=val_seed,
        )
        eval_dataset = ConcatDataset([d_train, d_val])
        # We also need d_train/d_val specifically for DVAECIV training later if selected
        d_train_for_competitor = d_train
        d_val_for_competitor = d_val
    else:
        print(f"Loading Test Data (seed {test_seed})...")
        eval_dataset = SimDataset(
            num_draws=args.n_test,
            num_obs=[args.n_test for _ in range(cfg.data.n_pop)],
            **dataset_args,
            seed=test_seed,
        )
        # For competitors in Test mode, we technically still need a training set to fit them
        # if they aren't pre-trained. Here we reload train just for fitting competitors.
        d_train_for_competitor = SimDataset(
            num_draws=cfg.data.n_train,
            num_obs=[cfg.data.n_train for _ in range(cfg.data.n_pop)],
            **dataset_args,
            seed=train_seed,
        )
        d_val_for_competitor = SimDataset(
            num_draws=cfg.data.n_val,
            num_obs=[cfg.data.n_val for _ in range(cfg.data.n_pop)],
            **dataset_args,
            seed=val_seed,
        )

    eval_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # 4. Run Inference (Using the passed 'model')
    print("Running model inference...")
    df_lst = model.encode_dataset(eval_loader)

    # Add population index for tracking
    for i, df in enumerate(df_lst):
        df["pop_num"] = i
        df["batch_num"] = 0

    # Create Combined DataFrame
    dfs_combined = pd.concat(df_lst, ignore_index=True)

    # 5. Compute Estimates
    iv_method = getattr(args, "method", "tsls")
    est_res = compute_estimates(
        df_lst,
        plot=False,
        theta=cfg.data.theta if hasattr(cfg.data, "theta") else 1.0,
        method=iv_method,
    )

    est_combined = compute_estimates([dfs_combined], plot=False, method=iv_method)
    est_combined["pop_num"] = -1

    # 6. Run Competitors
    Z_data = dfs_combined.filter(regex="^Z_").to_numpy()
    D_data = dfs_combined.filter(regex="^D").to_numpy()
    Y_data = dfs_combined.filter(regex="^Y").to_numpy()
    C_data = dfs_combined["pop_num"].to_numpy().reshape(-1, 1)
    print("Running KClass (TSLS/LIML)...")
    for method in ["tsls", "liml"]:
        try:
            mod_kclass = KClass(kappa=method, fit_intercept=True).fit(
                Z=Z_data, X=D_data, y=Y_data, C=C_data
            )
            est_val = mod_kclass.coef_[0] if len(mod_kclass.coef_) > 0 else 0.0
            inst_name = f"{method}CondPop"
            est_combined = pd.concat(
                [
                    est_combined,
                    pd.DataFrame(
                        {
                            "instrument": [inst_name],
                            "pop_num": [-1],
                            "estimate": [est_val],
                        }
                    ),
                ],
                ignore_index=True,
            )
        except Exception as e:
            print(f"Skipping {method}CondPop: {e}")

    # MR Egger
    print("Running MR Egger...")
    try:
        mod_egger = MREgger().fit(Z=Z_data, X=D_data, y=Y_data, C=None)
        est_val = mod_egger.coef_[0]
        inst_name = "MREgger"
        est_combined = pd.concat(
            [
                est_combined,
                pd.DataFrame(
                    {
                        "instrument": [inst_name],
                        "pop_num": [-1],
                        "estimate": [est_val],
                    }
                ),
            ],
            ignore_index=True,
        )
    except Exception as e:
        print(f"Skipping MREggerCondPop: {e}")
    try:
        mod_egger = MREgger().fit(Z=Z_data, X=D_data, y=Y_data, C=C_data)
        est_val = mod_egger.coef_[0]
        inst_name = "MREggerCondPop"
        est_combined = pd.concat(
            [
                est_combined,
                pd.DataFrame(
                    {
                        "instrument": [inst_name],
                        "pop_num": [-1],
                        "estimate": [est_val],
                    }
                ),
            ],
            ignore_index=True,
        )
    except Exception as e:
        print(f"Skipping MREggerCondPop: {e}")

    # Merge results
    final_est = pd.concat([est_res, est_combined], ignore_index=True)
    final_est["sim_id"] = sim_id
    final_est["use_test_data"] = args.use_test_data

    # 7. Compute Extra Metrics
    metrics_dict = {}
    if args.compute_extras:
        print("Computing GCM and reconstruction error...")
        metrics_dict["gcm"] = compute_gcm(df_lst)
        metrics_dict["recon"] = compute_recon_err(df_lst)

    # 8. Training Stats
    stats_path = os.path.join(
        "outputs", exp_prefix, str(sim_id), "training_stats.json"
    )
    train_time = 0
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            for line in f:
                try:
                    stats = json.loads(line)
                    train_time += stats.get("train_time_sec", 0)
                except json.JSONDecodeError:
                    continue
    final_est["train_time"] = train_time

    return final_est, metrics_dict


# %%

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", type=str, required=True)
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--use_test_data", action="store_true")
    parser.add_argument("--n_test", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=3000)
    parser.add_argument("--compute_extras", action="store_true")
    parser.add_argument(
        "--ckpt_strategy", type=str, default="best", choices=["best", "last"]
    )
    parser.add_argument("--metric_key", type=str, default="val_tot_loss")
    parser.add_argument("--selection_mode", type=str, default="min")
    parser.add_argument("--last_k", type=int, default=20)
    parser.add_argument("--exclude_sim_ids", type=int, nargs="*", default=[])

    args = parser.parse_args()
    exp_prefix_full = f"{args.exp_id}-ds{args.data_seed}"

    # 1. UNIFIED LOADING: Get everything we need (Model + Config + ID)
    # This guarantees the 'model' object matches the 'best_sim' logic
    try:
        model, cfg, best_sim = find_and_load_best_context(args, exp_prefix_full)
    except Exception as e:
        print(f"Critical Error during setup: {e}")
        exit(1)

    # 2. RUN EVALUATION: Pass the loaded objects
    estimates_df, extra_metrics = run_evaluation(
        model=model,
        cfg=cfg,
        sim_id=best_sim,
        exp_prefix=exp_prefix_full,
        args=args,
    )

    # 3. SAVE RESULTS
    suffix = "outsample" if args.use_test_data else "insample"
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    base_filename = (
        f"{out_dir}/{args.exp_id}-ds{args.data_seed}_bestsim{best_sim}_{suffix}"
    )

    estimates_df.to_csv(f"{base_filename}_estimates.csv", index=False)
    for name, df in extra_metrics.items():
        df.to_csv(f"{base_filename}_{name}.csv", index=False)

    print(f"\nSaved results to {base_filename}*")

    # 4. SUMMARY
    print("\n--- Summary ---")
    summary_cols = ["instrument", "estimate"]
    summary_view = estimates_df[
        estimates_df["instrument"].isin(
            ["hW", "hWchV", "tslsCondPop", "limlCondPop"]
        )
    ]
    if "pop_num" in summary_view.columns:
        summary_cols.insert(0, "pop_num")
    print(summary_view[summary_cols].to_string(index=False))
