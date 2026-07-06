"""Hyperparameter-grid builders for the experiment entry points."""

import pandas as pd
from typing import Dict, Any, Optional


def get_best_hparams(dataset, penalty, series, weight_decay=1e-5):
    """Read tuned real-data hyperparameters from server_results."""

    df = pd.read_csv(f'./server_results/real_data_{penalty}.csv')
    row = df[(df["Dataset"] == dataset) &
             (df["series"] == series)]

    if row.empty:
        raise ValueError(
            f"No tuned hyperparameters for (dataset={dataset}, series={series})"
        )

    row = row.iloc[0]

    return {
        "lag": int(row["lag"]),
        "lr": float(row["lr"]),
        "hidden_dim": int(row["hidden_dim"]),
        "dropout": float(row["dropout"]),
        "layers": int(row["layers"]),
        "ind_lambda": float(row["ind_lambda"]),
        "int_lambda": float(row["int_lambda"]),
        "weight_decay": float(weight_decay),
    }


def build_training_config(
    dataset: str,
    penalty_type: str,
    data_dim: int,
    use_best: bool = False,  
    best_hparams: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the default grid and evaluation settings for one experiment."""

    penalty_to_importance = {
        "Fast_Shap": "Shapley",
        "Fast_Shap_1": "Shapley",
        "Fast_Shap_3": "Shapley",
        "Fast_Shap_5": "Shapley",
        "Shapley": "Shapley",
        "Jacob_F": "Jacobian",
        "Jacob_L1": "Jacobian",
        "Layer_Weight": "Layer_Weight",
    }
    if penalty_type not in penalty_to_importance:
        raise ValueError(f"Unknown penalty_type: {penalty_type}")
    importance_type = penalty_to_importance[penalty_type]

    ignore_diagonal = dataset in ("DREAM3", "DREAM4")

    batch_size = 512 if dataset == "CausalTime" else -1

    if dataset in ("DREAM3", "DREAM4"):
        hidden_dim_grid = [
            int(0.5 * data_dim),
            data_dim,
            2 * data_dim,
            3 * data_dim,
            4 * data_dim,
        ]
    else:
        hidden_dim_grid = [
            data_dim,
            2 * data_dim,
            3 * data_dim,
            4 * data_dim,
            5 * data_dim
        ]

    if penalty_type.startswith("Fast_Shap"):
        int_lambda_grid = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    else:
        int_lambda_grid = [0.0]

    # Reproduction mode uses a one-point grid from the tuning table.
    if use_best and best_hparams is not None:
        param_grid = {
            "lr": [best_hparams["lr"]],
            "hidden_dim": [best_hparams["hidden_dim"]],
            "layers": [best_hparams["layers"]],
            "dropout": [best_hparams["dropout"]],
            "ind_lambda": [best_hparams["ind_lambda"]],
            "int_lambda": [best_hparams["int_lambda"]],
            "weight_decay": [best_hparams.get("weight_decay", 1e-5)],
        }
    else:
        param_grid = {
            "lr": [5e-4, 1e-4, 1e-3, 5e-3],
            "hidden_dim": hidden_dim_grid,
            "layers": [0, 1, 2, 3, 4, 5],
            "dropout": [0, 0.1, 0.2],
            "ind_lambda": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
            "int_lambda": int_lambda_grid,
            "weight_decay": [1e-5],
        }

    return {
        "importance_type": importance_type,
        "ignore_diagonal": ignore_diagonal,
        "batch_size": batch_size,
        "param_grid": param_grid,
    }
