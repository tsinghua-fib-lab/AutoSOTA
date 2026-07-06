import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import grid_search
from data.Dataset import TimeSeriesDataset, Data
from utils.configs import build_training_config
from utils.experiment import append_results_csv, select_device, suppress_cuda_context_warning

suppress_cuda_context_warning()

def build_rebuttal_config(dataset, penalty_type, data_dim, model_type):
    # Call original config safely, strictly disabling pre-tuned parameters
    cfg = build_training_config(dataset, penalty_type, data_dim, use_best=False, best_hparams=None)

    # Override the param_grid ONLY for the new architectures
    if model_type in ["SRU"]:
        ind_lambda_grid = [1e-4, 1e-3, 1e-2, 1e-1, 1]
        int_lambda_grid = [1e-4, 1e-3, 1e-2, 1e-1, 1] if penalty_type == "Fast_Shap" else [0.0]

        inf_hidden_grid = [data_dim, 2 * data_dim, 3 * data_dim, 4 * data_dim, 5 * data_dim]
        cfg["param_grid"] = {
            "lr": [1e-2, 5e-4, 1e-4, 1e-3, 5e-3], 
            "hidden_dim": inf_hidden_grid, 
            "layers": [1, 2, 3],
            "dropout": [0],
            "ind_lambda": ind_lambda_grid,
            "int_lambda": int_lambda_grid,
            "weight_decay": [1e-5, 1e-4],
        }
    return cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CausalTime")
    parser.add_argument("--series", type=int, default=1)
    parser.add_argument("--subject", type=int, default=1) 
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--exec_idx", type=int, default=1)
    parser.add_argument("--penalty_type", type=str, required=True,
                        choices=["Fast_Shap", "Shapley", "Jacob_F", "Jacob_L1"])
    parser.add_argument("--model_type", type=str, default="MLP", 
                        choices=["DLinear", "Informer", "TSMixer", "SimpleMamba", "MLP", "SRU"])
    parser.add_argument("--lag", type=int, default=3,
                        help="Lag length for the time series dataset")

    args = parser.parse_args()

    device = select_device(args.exec_idx)

    save_path = (
        f"./results/rebuttal_{args.dataset}_{args.series}_{args.lag}_"
        f"{args.model_type}_{args.penalty_type}.csv"
    )

    data_extractor = Data("./data", args.dataset, args.series, args.subject)
    X, network, gene_names = data_extractor.load_data()
    
    # Pass the command line argument directly to the dataset
    dataset = TimeSeriesDataset(X, lag=args.lag, Norm=True, device=device)
    data_dim = dataset.output_dim

    # Call the simplified config wrapper
    cfg = build_rebuttal_config(
        dataset=args.dataset,
        penalty_type=args.penalty_type,
        data_dim=data_dim,
        model_type=args.model_type
    )

    results = grid_search(
        dataset=dataset,
        network=network,
        model_type=args.model_type,
        penalty_type=args.penalty_type,
        importance_type=cfg["importance_type"],
        ignore_diagonal=cfg["ignore_diagonal"],
        param_grid=cfg["param_grid"],
        batch_size=cfg["batch_size"],
        save_path=save_path,
        num_workers=args.num_workers,
        exec_idx=args.exec_idx,
        device=device,
        seed=args.seed,
    )

    append_results_csv(results, save_path)
