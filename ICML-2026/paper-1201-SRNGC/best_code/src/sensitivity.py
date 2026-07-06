import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import grid_search
from data.Dataset import TimeSeriesDataset, Data
from utils.experiment import append_results_csv, select_device, suppress_cuda_context_warning

suppress_cuda_context_warning()

# ==========================================================
# Main
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="fMRI")
    parser.add_argument("--series", type=int, default=1)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--lag", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel workers for grid search")
    parser.add_argument("--exec_idx", type=int, default=1, help="Execution index for this worker (1-based)")
    parser.add_argument("--penalty_type", type=str, required=True,
                        choices=["Fast_Shap_1", "Fast_Shap_3", "Fast_Shap_5", "Shapley"],
                        help="Penalty used during training")

    args = parser.parse_args()
    model_type = 'ResidualMLP'
    
    importance_type = 'Shapley'
    ignore_diagonal =  False
    batch_size = -1

    device = select_device(args.exec_idx)

    save_path = f"./results/sensitivity_{args.dataset}_{args.series}_{args.subject}_{model_type}_{args.penalty_type}.csv"

    # Load dataset
    data_extractor = Data('./data', args.dataset, args.series, args.subject)
    X, network, gene_names = data_extractor.load_data()
    dataset = TimeSeriesDataset(X, args.lag, Norm=True, device=device)

    int_lambda = [0.0] if args.penalty_type in ['Shapley'] else [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    data_dim = dataset.output_dim
    hidden_dim = [100]

    # Hyperparameter grid
    param_grid = {
        'lr': [5e-5, 1e-4, 5e-4, 1e-3],
        'hidden_dim': hidden_dim,
        'layers': [1, 2, 3, 4, 5],
        'dropout': [0],
        'ind_lambda': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
        'int_lambda': int_lambda,
        'weight_decay': [0]
    }

    results = grid_search(dataset=dataset, network=network, model_type=model_type, penalty_type=args.penalty_type, importance_type=importance_type, ignore_diagonal=ignore_diagonal, param_grid=param_grid, batch_size=batch_size, save_path=save_path, num_workers=args.num_workers, exec_idx=args.exec_idx, device=device, seed=args.seed)

    append_results_csv(results, save_path)
