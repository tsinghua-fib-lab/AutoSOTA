import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import grid_search
from data.Dataset import TimeSeriesDataset, Data
from utils.configs import build_training_config, get_best_hparams
from utils.experiment import append_results_csv, select_device, suppress_cuda_context_warning

suppress_cuda_context_warning()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="DREAM3")
    parser.add_argument("--series", type=int, default=1)
    parser.add_argument("--subject", type=int, default=1) 
    parser.add_argument("--lag", type=int, default=3,
                        help="Lag used for full grid search. Ignored when --use_best is set.")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--exec_idx", type=int, default=1)
    parser.add_argument("--penalty_type", type=str, required=True,
                        choices=["Fast_Shap", "Shapley", "Jacob_F", "Jacob_L1"])
    parser.add_argument("--use_best", action="store_true",
                        help="Use tuned hyperparameters instead of full grid search")
    parser.add_argument("--lag_override", type=int, default=None,
                        help="Override the lag value (useful when --use_best is set)")

    args = parser.parse_args()
    model_type = "ResidualMLP"

    device = select_device(args.exec_idx)

    save_path = (
        f"./results/real_data_{args.dataset}_{args.series}_{args.subject}_"
        f"{model_type}_{args.penalty_type}.csv"
    )

    # ----------------- optional best hparams -----------------
    best_hparams = None
    if args.use_best:
        best_hparams = get_best_hparams(
            args.dataset,
            args.penalty_type,
            args.series,
            weight_decay=1e-5,
        )

    # ----------------- data -----------------
    data_extractor = Data("./data", args.dataset, args.series, args.subject)
    X, network, gene_names = data_extractor.load_data()
    lag = best_hparams["lag"] if best_hparams else args.lag
    if args.lag_override is not None:
        lag = args.lag_override
    dataset = TimeSeriesDataset(X, lag, Norm=True, device=device)
    data_dim = dataset.output_dim

    # ----------------- single elegant config call -----------------
    cfg = build_training_config(
        dataset=args.dataset,
        penalty_type=args.penalty_type,
        data_dim=data_dim,
        use_best=args.use_best,
        best_hparams=best_hparams,
    )
    
    importance_type = cfg["importance_type"]
    ignore_diagonal = cfg["ignore_diagonal"]
    batch_size = cfg["batch_size"]
    param_grid = cfg["param_grid"]

    # ----------------- run grid_search -----------------
    results = grid_search(
        dataset=dataset,
        network=network,
        model_type=model_type,
        penalty_type=args.penalty_type,
        importance_type=importance_type,
        ignore_diagonal=ignore_diagonal,
        param_grid=param_grid,
        batch_size=batch_size,
        save_path=save_path,
        num_workers=args.num_workers,
        exec_idx=args.exec_idx,
        device=device,
        seed=args.seed,
    )

    append_results_csv(results, save_path)
