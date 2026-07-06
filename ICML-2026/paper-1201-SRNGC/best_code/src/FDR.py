import argparse
import torch
import os
import sys
import itertools
import traceback
import numpy as np
from torch.utils.data import DataLoader
import warnings
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch

# Silence CUDA context warnings
warnings.filterwarnings(
    "ignore",
    message=".*Attempting to run cuBLAS, but there was no current CUDA context.*"
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.cls_performance import cls_metrics
from train import train_model, setup_penalties, setup_importance, set_seed
from data.Dataset import TimeSeriesDataset, Data
from model.models import ResidualMLP

# -----------------------------
# Hyperparameter tuning and model fitting
# -----------------------------
def tune_and_fit(dataset, model_type, penalty_type, param_grid, batch_size, device, seed, input_dim, output_dim):
    train_dataset, val_dataset = dataset.split_series(split_ratio=0.7)
    # set for knockoff training
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    X_train, Y_train = next(iter(train_loader))
    X_val, Y_val = next(iter(val_loader))
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train[:, :output_dim])
    val_dataset = torch.utils.data.TensorDataset(X_val, Y_val[:, :output_dim])

    penalty_obj = setup_penalties(model_type, penalty_type, device)

    keys, values = zip(*param_grid.items())
    all_combinations = list(itertools.product(*values))

    best_val_loss = float("inf")
    best_state_dict = None
    best_params = None

    for idx, combo in enumerate(all_combinations):
        params = dict(zip(keys, combo))
        set_seed(seed)
        try:
            print(f'Config {idx}/{len(all_combinations)} | params={params}')
            model = ResidualMLP(input_dim=input_dim, output_dim=output_dim, layers=params['layers'], hidden_dim=params['hidden_dim'], dropout=params['dropout']).to(device)

            trained_model, val_loss = train_model(model=model, train_dataset=train_dataset, val_dataset=val_dataset, penalty=penalty_obj, batch_size=batch_size, ind_lambda=params['ind_lambda'], int_lambda=params['int_lambda'], weight_decay=params['weight_decay'], lr=params['lr'], penalty_type=penalty_type)
            
            print(f'Val loss | {val_loss:.6f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = {k: v.detach().clone() for k, v in trained_model.state_dict().items()}
                best_params = params
                print('Best model update!')

            

        except Exception as e:
            print(f"[WARNING] Error with params {params}: {e}")
            traceback.print_exc()

    if best_state_dict is None:
        raise RuntimeError("No valid model found during hyperparameter tuning.")

    best_model = ResidualMLP(input_dim=input_dim,output_dim=output_dim, layers=best_params['layers'], hidden_dim=best_params['hidden_dim'], dropout=best_params['dropout']).to(device)
    best_model.load_state_dict(best_state_dict)
    best_model.eval()
    return best_model, best_params

# -----------------------------
# Construct residual knockoffs
# -----------------------------
@torch.no_grad()
def construct_residual_knockoffs(dataset, trained_model, seed):
    set_seed(seed)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    X, Y = next(iter(loader))

    Y_hat = trained_model(X)
    residuals = Y - Y_hat

    T, output_dim = residuals.shape
    bootstrap_idx = torch.randint(low=0, high=T, size=(T,), device=residuals.device)
    residuals_boot = residuals[bootstrap_idx]

    X_knockoff = Y.clone()
    X_knockoff[..., -output_dim:] = Y_hat + residuals_boot

    return X, X_knockoff, Y

# -----------------------------
# Split Shapley matrix by lag
# -----------------------------
def split_matrix_by_lags(matrix: torch.Tensor, output_dim: int, lag: int):
    Z_list, Z_tilde_list = [], []
    for l in range(lag):
        start = l * 2 * output_dim
        Z_list.append(matrix[:, start:(start+output_dim)])
        Z_tilde_list.append(matrix[:, (start+output_dim):(start+2*output_dim)])
    Z = torch.cat(Z_list, dim=-1)
    Z_tilde = torch.cat(Z_tilde_list, dim=-1)
    return Z, Z_tilde

# -----------------------------
# Compute knockoff statistics
# -----------------------------
def compute_knockoff_statistics(model, dataset, importance_type, output_dim, lag, device, save_path=None):
    importance = setup_importance("ResidualMLP", importance_type, device)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    inputs, _ = next(iter(loader))
    inputs = inputs.clone().to(device)

    Y_pred = model(inputs)
    matrix = importance.cal_shapley_value(Y_pred, inputs)

    if save_path is not None:
        torch.save(matrix.cpu(), save_path)
        print(f"Shapley matrix saved to {save_path}")

    Z, Z_tilde = split_matrix_by_lags(matrix, output_dim, lag)
    W = Z - Z_tilde
    return W, matrix, Z, Z_tilde

# -----------------------------
# FDR thresholding
# -----------------------------
def knockoff_fdr_threshold(W: torch.Tensor, q: float = 0.1) -> float:
    W_flat = W.flatten().cpu().numpy()
    sorted_abs_W = np.sort(np.abs(W_flat))
    for t in sorted_abs_W:
        selected = (W_flat >= t).sum()
        false_pos = (W_flat <= -t).sum()
        if selected == 0:
            continue
        fdr = false_pos / max(1, selected)
        if fdr <= q:
            return t
    return np.inf

# -----------------------------
# Estimate graphs
# -----------------------------
def estimate_graphs(W, output_dim, lag, threshold):
    graphs_per_lag = []
    for l in range(lag):
        W_l = W[:, l*output_dim:(l+1)*output_dim]
        G_l = (W_l >= threshold).cpu().numpy().astype(int)
        graphs_per_lag.append(G_l)
    summary_graph = np.maximum.reduce(graphs_per_lag)
    return graphs_per_lag, summary_graph

# -----------------------------
# Parse true graphs
# -----------------------------
def parse_lag_graphs(lags):
    true_graphs = []
    
    for lag in range(lags, 0, -1):
        file_path = f'./data/VAR3/label/D10_T1000_VAR3_lag_{lag}.npy'
        try:
            lag_graph = np.load(file_path)
            true_graphs.append(lag_graph)
        except FileNotFoundError:
            print(f"Warning: file {file_path} does not exist!")
            raise
    
    return true_graphs

# -----------------------------
# ICML-style plotting
# -----------------------------
def error_map(true_graph, est_graph):
    """
    Encode graph comparison into 4 states:
      0: TN
      1: TP
      2: FP
      3: FN
    """
    true = np.int32(true_graph)
    est  = np.int32(est_graph)

    out = np.zeros_like(true)
    out[(true == 1) & (est == 1)] = 1  # TP
    out[(true == 0) & (est == 1)] = 2  # FP
    out[(true == 1) & (est == 0)] = 3  # FN
    return out

def plot_shapley_matrix(
    Z,
    Z_tilde,
    output_dim,
    lag,
    save_path,
):
    # ---------- Layout Setup ----------
    # Row 1: Original Matrices
    # Row 2: Knockoff Matrices
    ncols = lag
    nrows = 2
    
    # Calculate figure size to maintain aspect ratio
    # Each panel is roughly square (d x d)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.0 * ncols, 6.0),
        constrained_layout=False,
        gridspec_kw={
            'height_ratios': [1, 1],
            'hspace': 0.15,  # This creates the clean blank gap
            'wspace': 0.1,
            'left': 0.12,    # Margin for row labels
            'right': 0.88    # Margin for colorbar
        }
    )

    # Ensure axes is always a 2D array for consistent indexing
    if lag == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    elif lag > 1 and axes.ndim == 1:
        # Fallback if dimensions are collapsed (unlikely with nrows=2)
        axes = axes.reshape(nrows, ncols)

    # ---------- Global Styling ----------
    vmax = np.max(np.abs(np.concatenate([Z, Z_tilde], axis=1)))
    norm = Normalize(vmin=0.0, vmax=vmax)

    # ICML-style Soft Red Colormap
    soft_red_cmap = LinearSegmentedColormap.from_list(
        "soft_red_icml_v2",
        [
            (1.00, 1.00, 1.00),   # white
            (0.97, 0.8, 0.8),     # very light red
            (0.92, 0.6, 0.6),     # soft red
            (0.80, 0.1, 0.1),     # medium red
            (0.65, 0.0, 0.0),     # darker red
        ],
    )

    # ---------------- Plotting Loop ----------------
    for l in range(lag):
        # Logic: Leftmost is Lag L, Rightmost is Lag 1
        # Z is shaped [d, lag*d]. Slice the correct block.
        start_idx = l * output_dim
        end_idx   = (l + 1) * output_dim
        
        Z_block  = Z[:, start_idx:end_idx]
        Zt_block = Z_tilde[:, start_idx:end_idx]
        # --- Row 1: Original ---
        ax_orig = axes[0, lag - 1 - l]
        im = ax_orig.imshow(
            Z_block, 
            cmap=soft_red_cmap, 
            norm=norm, 
            aspect='equal', 
            interpolation='nearest'
        )
        ax_orig.set_title(f"Lag {lag - l}", fontsize=16, pad=10)
        
        # --- Row 2: Knockoff ---
        ax_knock = axes[1, lag - 1 - l]
        ax_knock.imshow(
            Zt_block, 
            cmap=soft_red_cmap, 
            norm=norm, 
            aspect='equal', 
            interpolation='nearest'
        )

        # --- Common Formatting (Borders & Ticks) ---
        for ax in [ax_orig, ax_knock]:
            ax.set_xticks([])
            ax.set_yticks([])
            # Ensure borders (spines) are visible and thick
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(0.6)

    # ---------------- Labels ----------------
    # Set Row Labels on the first column only
    axes[0, 0].set_ylabel("Original", fontsize=16, labelpad=12, color='black')
    axes[1, 0].set_ylabel("Knockoff", fontsize=16, labelpad=12, color='black')

    # ---------------- Colorbar ----------------
    # Add a dedicated axis for the colorbar on the right
    # [left, bottom, width, height] in figure fraction
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7]) 
    
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=14, length=4, width=0.8)
    cbar.outline.set_linewidth(1.0)
    
    # Optional: Symmetric ticks formatting
    ticks = np.linspace(0, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.2f}" for t in ticks])

    plt.savefig(save_path, format='pdf', bbox_inches="tight", dpi=300)
    plt.close()

def plot_graph_comparison(
    true_graphs,
    est_graphs,
    true_summary,
    est_summary,
    lag,
    save_path,
):
    ncols = lag + 1
    fig, axes = plt.subplots(
        2, ncols,
        figsize=(3 * ncols, 5.0),
        constrained_layout=False,
        gridspec_kw={
            'height_ratios': [1, 1],
            'hspace': 0.16,
            'wspace': 0.22
        }
    )

    # ---------------- Soft ICML-style Colors ----------------
    color_no_edge = [1, 1, 1]
    color_edge    = [0, 0, 0]
    color_tp      = [0.9, 0.5, 0.5]
    color_fp      = [0.5, 0.7, 0.9]
    color_fn      = [0.65, 0.15, 0.15]
    color_tn      = [0.95, 0.95, 0.95]

    cmap_true = LinearSegmentedColormap.from_list(
        "soft_true_icml", [color_no_edge, color_edge], N=2
    )
    cmap_error = LinearSegmentedColormap.from_list(
        "soft_error_icml", [color_tn, color_tp, color_fp, color_fn], N=4
    )

    # ---------------- Row 1: True Graphs ----------------
    for l in range(lag):
        axes[0, l].imshow(true_graphs[lag-1-l], cmap=cmap_true, vmin=0, vmax=1)
        axes[0, l].set_title(f"Lag {l+1}", fontsize=16)
        axes[0, l].set_xticks([])
        axes[0, l].set_yticks([])

    axes[0, lag].imshow(true_summary, cmap=cmap_true, vmin=0, vmax=1)
    axes[0, lag].set_title("Summary", fontsize=16)
    axes[0, lag].set_xticks([])
    axes[0, lag].set_yticks([])
    axes[0, 0].set_ylabel("True Graphs", fontsize=16, labelpad=16)

    # ---------------- Row 2: Estimated/Error Maps ----------------
    for l in range(lag + 1):
        if l < lag:
            err = error_map(true_graphs[lag-1-l], est_graphs[lag-1-l])  # 0=TN,1=TP,2=FP,3=FN
        else:
            err = error_map(true_summary, est_summary)

        # ---------------- Mask FP and FN for background ----------------
        err_plot = err.copy()
        err_plot[err_plot == 2] = 0  # FP → TN color
        err_plot[err_plot == 3] = 0  # FN → TN color

        axes[1, l].imshow(err_plot, cmap=cmap_error, vmin=0, vmax=3)
        axes[1, l].set_xticks([])
        axes[1, l].set_yticks([])

        # ---------------- Overlay FP/FN markers ----------------
        fp_coords = np.argwhere(err == 2)
        fn_coords = np.argwhere(err == 3)

        if fp_coords.size > 0:
            axes[1, l].scatter(
                fp_coords[:, 1], fp_coords[:, 0],
                marker='x', s=80,
                color=color_fp, linewidths=3.6, zorder=5
            )

        if fn_coords.size > 0:
            axes[1, l].scatter(
                fn_coords[:, 1], fn_coords[:, 0],
                marker='o', s=80,
                facecolor='none', edgecolor=color_fn, linewidths=2.2, zorder=5
            )


    axes[1, 0].set_ylabel("Inferred Graphs", fontsize=16, labelpad=16)
    
    # ---------------- THE LABELS (Key Change) ----------------
    
    # # 1. Row Titles (Far Left) - Identifying the Experiment
    # fig.text(0.01, 0.7, "True Graph", fontsize=14, fontweight='bold', 
    #          rotation='vertical', va='center', ha='left')
    # fig.text(0.01, 0.28, "Inferred Graph", fontsize=14, fontweight='bold', 
    #          rotation='vertical', va='center', ha='left')

    # # 2. Axis Definitions (The Frame) - Identifying the Math
    # # "Effect" goes slightly to the right of the Row Titles
    # fig.text(0.08, 0.5, "Effect (Series Index)", fontsize=14, 
    #          rotation='vertical', va='center', ha='center', color='gray')
             
    # # "Cause" goes at the bottom, centered
    # fig.text(0.55, 0.09, "Cause (Series Index)", fontsize=14, 
    #          va='center', ha='center', color='gray')
    
    # # ---------------- Add Cause/Effect Labels ----------------
    # # Add "Effect" label to the left-most plots only
    # # (We typically treat Rows as Effect/Target and Cols as Cause/Source in GC matrices)
    # for row in [0, 1]:
    #     # Preserve the existing "True"/"Estimated" labels by adding a second label or combining
    #     # Ideally, "True" is the Row label, and "Effect" is the Y-axis label.
        
    #     # This adds "Effect" to the Y-axis of the first column
    #     axes[row, 0].set_ylabel(f"{axes[row, 0].get_ylabel()}\n(Effect)", fontsize=14)

    # # Add "Cause" label to the bottom-most plots only (Row 1)
    # for col in range(ncols):
    #     axes[1, col].set_xlabel("Cause", fontsize=14)

    # ---------------- Vertical Separator ----------------
    pos_left = axes[0, lag-1].get_position()
    pos_right = axes[0, lag].get_position()
    x_sep = (pos_left.x1 + pos_right.x0) / 2
    from matplotlib.lines import Line2D
    line = Line2D([x_sep, x_sep], [0.05, 0.95], transform=fig.transFigure,
                  color='darkgray', linestyle='--', lw=1.6)
    fig.add_artist(line)

    # ---------------- Legends ----------------
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_true = [
        Patch(facecolor=color_edge, edgecolor='k', label='True Edge'),
        Patch(facecolor=color_no_edge, edgecolor='k', label='No Edge')
    ]

    legend_error = [
        Patch(facecolor=color_tp, edgecolor='k', label='TP'),
        Patch(facecolor=color_tn, edgecolor='k', label='TN'),
        Line2D([0], [0], marker='x', color='w', label='FP',
               markerfacecolor=color_fp, markeredgecolor=color_fp, markersize=10, linewidth=0, markeredgewidth=4),
        Line2D([0], [0], marker='o', color='w', label='FN',
               markerfacecolor='none', markeredgecolor=color_fn, markersize=10, linewidth=0, markeredgewidth=2),
    ]

    fig.legend(handles=legend_true + legend_error,
               loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.06),
               frameon=False, fontsize=16)

    plt.savefig(save_path, format='pdf', bbox_inches="tight", dpi=300)
    plt.close()

def post_hoc_analysis(shapley_matrix_path, output_dim, lag, fdr, thresholding="overall"):
    try: # load from path
        shapley_matrix = torch.load(shapley_matrix_path, weights_only=True)
    except: # use directly
        shapley_matrix = shapley_matrix_path
    Z, Z_tilde = split_matrix_by_lags(shapley_matrix, output_dim, lag)
    W = Z - Z_tilde  # shape (T, d*lag)

    true_summary = network
    true_lag_graphs = parse_lag_graphs(lag)  # list of (d x d) per lag

    # ------------------------------
    # Threshold determination
    # ------------------------------
    if thresholding == "overall":
        threshold = knockoff_fdr_threshold(W, q=fdr)
        thresholds_per_lag = [threshold] * lag
    elif thresholding == "per_lag":
        thresholds_per_lag = []
        for l in range(lag):
            W_l = W[:, l*output_dim:(l+1)*output_dim]
            t_l = knockoff_fdr_threshold(W_l, q=fdr)
            thresholds_per_lag.append(t_l)
    else:
        raise ValueError("thresholding must be 'overall' or 'per_lag'")

    # ------------------------------
    # Compute per-lag cls_metrics
    # ------------------------------
    per_lag_metrics = []
    estimate_sub_graphs = []
    for l in range(lag):
        W_l = W[:, l*output_dim:(l+1)*output_dim]
        metric = cls_metrics(
            true_lag_graphs[l],
            W_l,
            ignore_diagonal=False,
            threshold=thresholds_per_lag[l]
        )
        per_lag_metrics.append(metric)
        print(f"Lag {l+1} metrics (threshold={thresholds_per_lag[l]:.4f}):")
        print(metric)
        G_l = (W_l >= thresholds_per_lag[l]).cpu().numpy().astype(int)
        estimate_sub_graphs.append(G_l)

    # ------------------------------
    # Compute summary graph
    # ------------------------------
    summary_graph = np.maximum.reduce([
        (W[:, l*output_dim:(l+1)*output_dim] >= thresholds_per_lag[l]).cpu().numpy().astype(int)
        for l in range(lag)
    ])
    summary_metric = cls_metrics(true_summary, summary_graph, ignore_diagonal=False, threshold=1e-5)
    print("Summary graph metrics:")
    print(summary_metric)
    
    plot(Z, Z_tilde, true_lag_graphs, estimate_sub_graphs, true_summary, summary_graph, output_dim, args)

    return estimate_sub_graphs, summary_graph

def plot(Z, Z_tilde, true_lag_graphs, estimate_sub_graphs, true_summary, est_summary, output_dim, args):
    plot_shapley_matrix(Z.cpu(), Z_tilde.cpu(), output_dim, args.lag, save_path=f"./assets/shapley_matrix_{args.dataset}_{args.subject}_{args.lag}.pdf")
    plot_graph_comparison(
        true_graphs=true_lag_graphs,
        est_graphs=estimate_sub_graphs,
        true_summary=true_summary,
        est_summary=est_summary,
        lag=args.lag,
        save_path=f"./assets/graph_comparison_{args.dataset}_{args.subject}_{args.lag}.pdf",
    )

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="VAR3")
    parser.add_argument("--series", type=int, default=2)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--lag", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--fdr", type=float, default=0.01)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    data_extractor = Data("./data", args.dataset, args.series, args.subject)
    X, network, gene_names = data_extractor.load_data()
    dataset = TimeSeriesDataset(X, args.lag, Norm=True, device=device)
    output_dim = dataset.output_dim
    
    true_summary = network
    true_lag_graphs = parse_lag_graphs(args.lag)

    if not np.allclose(true_summary, np.maximum.reduce(true_lag_graphs)):
        print("[WARNING] True summary graph does not match the maximum over true lag graphs.")
    
    post_hoc_analysis(shapley_matrix_path=f"./assets/shapley_matrix_VAR3_1_4.pt", output_dim=output_dim, lag=args.lag, fdr=args.fdr)
    quit()
    
    # ---------------------
    # Base / knockoff grids
    # ---------------------
    base_param_grid = {
        "lr":[5e-4, 1e-3, 5e-3],
        "hidden_dim":[25, 50, 100],
        "layers":[1, 2, 3],
        "dropout":[0.1],
        "ind_lambda":[1e-5, 1e-4, 1e-3, 1e-2],
        "int_lambda":[0.0],
        "weight_decay":[1e-5]
    }
    knockoff_param_grid = {
        "lr":[5e-4, 1e-3, 5e-3],
        "hidden_dim":[25, 50, 100],
        "layers":[1, 2, 3],
        "dropout":[0.1],
        "ind_lambda":[1e-5, 1e-4, 1e-3, 1e-2],
        "int_lambda":[0.0],
        "weight_decay":[1e-5]
    }
    # the best params for VAR3_2_1_4 (series 2, subject 1, lag 4)
    # base_param_grid = {
    #     "lr":[5e-3],
    #     "hidden_dim":[50],
    #     "layers":[1],
    #     "dropout":[0.1],
    #     "ind_lambda":[0.01],
    #     "int_lambda":[0.0],
    #     "weight_decay":[1e-5]
    # }
    # knockoff_param_grid = {
    #     "lr":[5e-3],
    #     "hidden_dim":[25],
    #     "layers":[1],
    #     "dropout":[0.1],
    #     "ind_lambda":[0.01],
    #     "int_lambda":[0.0],
    #     "weight_decay":[1e-5]
    # }

    # ---------------------
    # Step 1: Fit base model
    # ---------------------
    base_model, best_base_params = tune_and_fit(dataset, "ResidualMLP", "Shapley", base_param_grid, batch_size=-1, device=device, seed=args.seed, input_dim=args.lag * output_dim, output_dim=output_dim)
    print(f"Base model params: {best_base_params}")

    # ---------------------
    # Step 2: Construct knockoffs
    # ---------------------
    X_all, X_knockoff, Y_all = construct_residual_knockoffs(dataset, base_model, args.seed)
    X_augmented = torch.cat([Y_all, X_knockoff], dim=-1)
    dataset_aug = TimeSeriesDataset(X_augmented, dataset.lag, Norm=False, device=device)

    # ---------------------
    # Step 3: Fit knockoff model
    # ---------------------
    knock_model, best_knockoff_params = tune_and_fit(dataset_aug, "ResidualMLP", "Shapley", knockoff_param_grid, batch_size=-1, device=device, seed=args.seed, input_dim=2 * args.lag * output_dim, output_dim=output_dim)
    print(f"Knockoff model params: {best_knockoff_params}")

    # ---------------------
    # Step 4: Compute knockoff statistics
    # ---------------------
    W, matrix, Z, Z_tilde = compute_knockoff_statistics(knock_model, dataset_aug, "Shapley", output_dim, args.lag, device, save_path=f"./assets/shapley_matrix_{args.dataset}_{args.subject}_{args.lag}.pt")

    # ---------------------
    # Step 5: Post-hoc analysis
    # ---------------------
    estimate_sub_graphs, est_summary = post_hoc_analysis(matrix, output_dim, args.lag, args.fdr, thresholding="overall")
