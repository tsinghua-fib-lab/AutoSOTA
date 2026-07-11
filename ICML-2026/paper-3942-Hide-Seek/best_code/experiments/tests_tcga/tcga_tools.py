import pandas as pd
import numpy as np
import argparse
import ast
import json
from sklearn.model_selection import train_test_split
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import repo_paths  # noqa: F401

from tools import run_feature_selection_model

RANDOM_STATE = 42

from params import CONSTANT_PARAMS as PARAMS


def int_or_none(x):
    return None if str(x).lower() == "none" else int(x)


def parse_xgb_params(xgb_params_str):
    if xgb_params_str is None:
        return None

    # Accept both strict JSON and Python-style dict strings.
    try:
        parsed = json.loads(xgb_params_str)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(xgb_params_str)

    if not isinstance(parsed, dict):
        raise ValueError("xgb_params must parse to a dict")

    if "num_boost_round" in parsed:
        parsed["num_boost_round"] = int(parsed["num_boost_round"])

    return parsed

def one_hot_encoder(y):
    from sklearn.preprocessing import OneHotEncoder
    try:
        encoder = OneHotEncoder(sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(sparse=False) #for l2x environment
    y_reshaped = y.reshape(-1, 1)
    y_one_hot = encoder.fit_transform(y_reshaped)
    return y_one_hot

def run_tcga_experiment(model_type="hide_and_seek",
                        lmbda=0.3,
                        seed=0,
                        val_or_test='val',
                        folder_for_pickle='ICML_experiments/tcga',
                        epochs=None,
                        num_important_features=20,
                        xgb_params=None): #20 is arbitrary for hide_and_seek, invase, realx

    # Load data
    HERE = Path(__file__).resolve().parent
    data = pd.read_csv(HERE / "brca_small.csv", index_col=0)
    X = data.values[:, :-1]
    Y = data.values[:, -1]
    Y = one_hot_encoder(Y)

    assert ~(np.isnan(X).any())

    genes = data.columns.tolist()[:-1]

    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=100, random_state=0)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=100, random_state=1)

    if val_or_test == 'val':
        full_data_dict = {'x_train':X_train,
                        'y_train':Y_train,
                        'x_test':X_val,
                        'y_test':Y_val,
                        'g_test':None
                        }
    elif val_or_test == 'test':
        full_data_dict = {'x_train':X_train,
                        'y_train':Y_train,
                        'x_test':X_test,
                        'y_test':Y_test,
                        'g_test':None
                        }
    else:
        raise ValueError("val_or_test must be either 'val' or 'test'.")

    print(f"Using {val_or_test} set for evaluation")

    if model_type in PARAMS:
        if epochs is None:
            epochs = PARAMS[model_type]['epochs']
        batch_size = PARAMS[model_type]['batch_size']
    else:
        raise ValueError(f"Unsupported model_type for tcga_tools: {model_type}")

    return_results = True
    task = 'classification'
    save_experiment_data = True
    data_type = 'tcga'
    batchnorm_hs = False
    num_syn_features = None
    train_N = None
    test_N = None
    data_mode = 'none'

    hide_hidden_dim = 32
    seek_hidden_dim = 32
    hide_num_hidden_layers = 2
    seek_num_hidden_layers = 2
    lmbda_exponent = 2

    print(f"Running with seed={seed}, lmbda={lmbda}, epochs={epochs}")

    results = run_feature_selection_model(data_type=data_type, 
                        folder_for_pickle=folder_for_pickle,
                        num_important_features=num_important_features,
                        full_data_dict=full_data_dict,
                        model_type=model_type,
                        batch_size=batch_size,
                        epochs=epochs,
                        lmbda=lmbda,
                        task=task,
                        hide_hidden_dim=hide_hidden_dim,
                        seek_hidden_dim=seek_hidden_dim, 
                        hide_num_hidden_layers=hide_num_hidden_layers, 
                            seek_num_hidden_layers=seek_num_hidden_layers,
                            train_N = train_N,
                            test_N = test_N,
                            seed=seed,
                            num_syn_features=num_syn_features,
                            return_results=return_results,
                            batchnorm_hs=batchnorm_hs,
                            save_experiment_data=save_experiment_data,
                            lmbda_exponent=lmbda_exponent,
                            data_mode=data_mode,
                            column_names=genes,
                            xgb_params=xgb_params)

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run a single TCGA feature-selection experiment.")
    parser.add_argument("--model_type", type=str, default="hide_and_seek")
    parser.add_argument("--lmbda", type=float, default=1.5)
    parser.add_argument("--epochs", type=int_or_none, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val_or_test", type=str, choices=["val", "test"], default="val")
    parser.add_argument("--folder-for-pickle", type=str, default="ICML_experiments/tcga")
    parser.add_argument("--num_important_features", type=int, default=20)
    parser.add_argument(
        "--xgb-params",
        type=str,
        default=None,
        help=(
            "XGBoost parameter dict as JSON or Python literal string. "
            "Example: '{\"objective\":\"binary:logistic\",\"eval_metric\":\"logloss\","
            "\"max_depth\":3,\"eta\":0.1,\"num_boost_round\":100}'"
        ),
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    xgb_params = parse_xgb_params(args.xgb_params)

    run_tcga_experiment(
        model_type=args.model_type,
        lmbda=args.lmbda,
        epochs=args.epochs,
        seed=args.seed,
        val_or_test=args.val_or_test,
        folder_for_pickle=args.folder_for_pickle,
        num_important_features=args.num_important_features,
        xgb_params=xgb_params,
    )

