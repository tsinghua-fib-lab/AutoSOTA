import json
import os
import sys
import pandas as pd
from datetime import datetime
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import repo_paths  # noqa: F401
from tools import run_feature_selection_model, parse_args
from params import CONSTANT_PARAMS as PARAMS

if __name__ == '__main__':
    args = parse_args(lmbda_default=0.3,
                      epochs_default=None,
                      batch_size_default=None,
                     seed_default=0,
                     syn_idx_default=3,
                     num_syn_features_default=11,
                     train_N_default=10_000,#10_000,
                     test_N_default=10_000,
                     hide_hidden_dim_default=32,
                     seek_hidden_dim_default=32,
                     hide_num_hidden_layers_default=2,
                     seek_num_hidden_layers_default=2,
                     lmbda_exponent_default=2,
                     data_mode_default='synthetic', #'credit_data_val', #'synthetic'
                     model_type_default='hide_and_seek',
                     folder_for_pickle_default='ICML_experiments/unallocated')
                     

    model_type = args.model_type

    if model_type not in PARAMS:
        raise ValueError(f"Unsupported model_type for PARAMS lookup: {model_type}")

    lmbda = args.lmbda
    batch_size = PARAMS[model_type]['batch_size'] if args.batch_size is None else args.batch_size
    epochs = PARAMS[model_type]['epochs'] if args.epochs is None else args.epochs
    folder_for_pickle = args.folder_for_pickle

    seed = args.seed
    num_syn_features = args.num_syn_features
    train_N = args.train_N
    test_N = args.test_N
    rho = args.rho

    data_mode = args.data_mode
    perturbation_method = args.perturbation_method
    n_ensemble = args.n_ensemble
    colsample = args.colsample
    ensemble_parallel = args.ensemble_parallel
    ensemble_n_jobs = args.ensemble_n_jobs
    ensemble_backend = args.ensemble_backend
    xgb_params = args.xgb_params

    hide_hidden_dim = args.hide_hidden_dim
    seek_hidden_dim = args.seek_hidden_dim
    hide_num_hidden_layers = args.hide_num_hidden_layers
    seek_num_hidden_layers = args.seek_num_hidden_layers
    lmbda_exponent = args.lmbda_exponent

    batchnorm_hs = args.batchnorm_hs

    return_losses_on_val = args.return_losses_on_val

    syn_switch_quantile = args.syn_switch_quantile

    print(f"Running with seed={seed}, lmbda={lmbda}")

    # ---- Resolve the single data set to run ----
    # This script now processes EXACTLY ONE data set per invocation. Cross-dataset
    # parallelism lives in run_multiple_tests.py.
    if args.data_set is None:
        raise ValueError(
            "run_synthetic_tests.py is single-dataset-only. Pass --data-set <name>."
        )
    data_set = args.data_set

    task = args.task
    save_experiment_data = True

    num_important_features = args.num_important_features
    # num_important_features = 4 #[3,4]

    if model_type == 'lime' and (epochs != 500 or batch_size != None): raise ValueError("lime baseline usually requires uses epochs=500, batch_size=None")
    if num_syn_features != 11:
        print("WARNING NOT TESTING ON STANDARD NUMBER OF SYN FEATURES")

    #for 100 features, invase lmbda 0.5, realx 1, hide&seek 1.8

    if model_type != 'l2x':
        from tabulate import tabulate

    results = run_feature_selection_model(data_type=data_set,
                        folder_for_pickle=folder_for_pickle,
                        num_important_features=num_important_features,
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
                            return_results=True,
                            batchnorm_hs=batchnorm_hs,
                            save_experiment_data=save_experiment_data,
                            lmbda_exponent=lmbda_exponent,
                            return_losses_on_val=return_losses_on_val,
                            data_mode=data_mode,
                            perturbation_method=perturbation_method,
                            n_ensemble=n_ensemble,
                            colsample=colsample,
                            ensemble_parallel=ensemble_parallel,
                            ensemble_n_jobs=ensemble_n_jobs,
                            ensemble_backend=ensemble_backend,
                            xgb_params=xgb_params,
                            rho=rho,
                            syn_switch_quantile=syn_switch_quantile
                            )
    row = {
            "syn": data_set,
            "TPR": round(results["TPR_mean"], 4),
            "FDR": round(results["FDR_mean"], 4),
            "F1": round(results["f1"], 4)
        }

    df = pd.DataFrame([row])

    if args.metrics_out is not None:
        with open(args.metrics_out, 'w') as f:
            json.dump({
                "data_set": data_set,
                "TPR_mean": float(results["TPR_mean"]),
                "FDR_mean": float(results["FDR_mean"]),
                "f1": float(results["f1"]),
            }, f)

    print("\n===== RUN CONFIG =====")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print(f"resolved_epochs: {epochs}")
    print(f"resolved_batch_size: {batch_size}")
    print("======================\n")
    if model_type != 'l2x':
        print(tabulate(df, headers="keys", tablefmt="fancy_grid"))
    else:
        print(df)

    gc.collect()
    if model_type != 'l2x' and model_type != 'realx':
        import torch
        torch.cuda.empty_cache()
