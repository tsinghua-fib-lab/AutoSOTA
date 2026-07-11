import traceback
from itertools import product

import pandas as pd
import numpy as np

from tcga_tools import run_tcga_experiment

# Validation sweep config
seeds_list = [0]
max_depth_s = [5]#[2, 3, 4, 5]
eta_s = [0.03]#[0.03, 0.05, 0.1, 0.2]
num_boost_round_s = [200]#[50, 100, 200]

# Optional extra knobs to sweep later if needed.
subsample_s = [1.0]
colsample_bytree_s = [1.0]

val_or_test = "test"
if val_or_test == "val":
    folder_for_pickle = f"ICML_experiments/tcga/{val_or_test}/xgboost"
else:
    folder_for_pickle = f"ICML_experiments/tcga/{val_or_test}"
num_important_features = 20
stop_on_failure = False


def iter_xgb_params():
    for seed, max_depth, eta, num_boost_round, subsample, colsample_bytree in product(
        seeds_list,
        max_depth_s,
        eta_s,
        num_boost_round_s,
        subsample_s,
        colsample_bytree_s,
    ):
        xgb_params = {
            # Let tools.py infer objective defaults from num_classes when desired.
            # Keeping explicit multiclass defaults here for 4-class TCGA.
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "num_class": 4,
            "max_depth": max_depth,
            "eta": eta,
            "num_boost_round": num_boost_round,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
        }
        yield seed, xgb_params


def main():
    runs = list(iter_xgb_params())
    records = []

    print(f"Running {len(runs)} xgboost TCGA validation experiment(s)...")

    for idx, (seed, xgb_params) in enumerate(runs, start=1):
        print(f"[{idx}/{len(runs)}] RUNNING seed={seed}, xgb_params={xgb_params}")
        used_num_boost_round = int(xgb_params.get("num_boost_round", 100))
        try:
            results = run_tcga_experiment(
                model_type="shap_xgboost",
                lmbda=0.0,
                seed=seed,
                val_or_test=val_or_test,
                folder_for_pickle=folder_for_pickle,
                epochs=None,
                num_important_features=num_important_features,
                xgb_params=xgb_params,
            )
            records.append(
                {
                    "status": "ok",
                    "seed": seed,
                    "max_depth": xgb_params["max_depth"],
                    "eta": xgb_params["eta"],
                    "num_boost_round": used_num_boost_round,
                    "subsample": xgb_params["subsample"],
                    "colsample_bytree": xgb_params["colsample_bytree"],
                    "accuracy": results.get("accuracy"),
                    "roc_auc": results.get("roc_auc"),
                    "pr_auc": results.get("pr_auc"),
                    "run_type": results.get("run_type"),
                    "time_run": results.get("time_run"),
                    "time_end": results.get("time_end"),
                }
            )
        except Exception as exc:
            records.append(
                {
                    "status": "failed",
                    "seed": seed,
                    "max_depth": xgb_params["max_depth"],
                    "eta": xgb_params["eta"],
                    "num_boost_round": used_num_boost_round,
                    "subsample": xgb_params["subsample"],
                    "colsample_bytree": xgb_params["colsample_bytree"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print("FAILED")
            traceback.print_exc()
            if stop_on_failure:
                break

    summary = pd.DataFrame(records)
    ok_rows = summary[summary["status"] == "ok"].copy()
    if not ok_rows.empty and "roc_auc" in ok_rows.columns:
        roc_auc_num = pd.to_numeric(ok_rows["roc_auc"], errors="coerce").to_numpy()
        order = np.argsort(roc_auc_num)[::-1]
        ok_rows = ok_rows.iloc[order]

    print("\nValidation sweep finished.")
    if ok_rows.empty:
        print("No successful runs.")
    else:
        print("Top runs by roc_auc:")
        print(ok_rows.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
