import pandas as pd
import matplotlib.pyplot as plt

def main():
    # === 1. Read the two result files ===
    file_regular = "convergence_results_combined.csv"
    file_stretched = "stretched_convergence_results_combined.csv"
    
    try:
        df_reg = pd.read_csv(file_regular, header=[0, 1], index_col=0)
        df_str = pd.read_csv(file_stretched, header=[0, 1], index_col=0)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure both CSV files are in the current directory.")
        return

    # === 2. SCI Publication Style Configuration ===
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        # Enforce Times New Roman globally
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        # Use 'stix' to make math text align with Times New Roman
        "mathtext.fontset": "stix",
        "font.size": 16,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "legend.fontsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "lines.linewidth": 2,
        "lines.markersize": 6,
    })

    # Original CSV keys (now matched exactly to the 4 sparsity schemes)
    csv_alphas = ['n^(-1/4)', 'n^(-1/2)', 'log(n)/n', '1/n']
    # Standard LaTeX math formulas for plotting
    plot_alphas = ['n^{-1/4}', 'n^{-1/2}', '\\log(n)/n', '1/n']
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    markers = ["o", "s", "^", "D"]
    linestyles = ["-", "--", "-.", ":"]

    # === 3. Helper function for single plots ===
    def create_and_save_plot(df, metric_mean, metric_std, ylabel, save_name, show_legend=False):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ns = df.index
        
        for i, alpha_key in enumerate(csv_alphas):
            mean_vals = df[metric_mean][alpha_key]
            std_vals = df[metric_std][alpha_key]
            
            # Render standard LaTeX with Times font style
            label_str = f'$\\alpha(n) = {plot_alphas[i]}$'
            
            ax.errorbar(
                x=ns,
                y=mean_vals,
                yerr=std_vals,
                label=label_str,
                marker=markers[i],
                linestyle=linestyles[i],
                color=colors[i],
                capsize=4,        
                elinewidth=1,     
                markeredgewidth=1 
            )
            
        ax.set_xlabel(r"Graph Size ($n$)")
        ax.set_ylabel(ylabel)
        
        # Remove top and right spines, disable default grid
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)
        
        if show_legend:
            ax.legend(frameon=False)
            
        plt.tight_layout()
        plt.savefig(save_name, bbox_inches="tight")
        print(f"Successfully saved: {save_name}")

    # === 4. Generate and save the 4 subplots ===
    
    # 1. c-GCN: Edge Density (With legend)
    create_and_save_plot(df_reg, 'Density_Mean', 'Density_Std', 
                         'Edge Density', 'fig_cGCN_Density.pdf', show_legend=True)
    
    # 2. c-GCN: Convergence Error (With legend)
    create_and_save_plot(df_reg, 'Error_Mean', 'Error_Std', 
                         'Convergence Error', 'fig_cGCN_Error.pdf', show_legend=True)
    
    # 3. GWCN: Edge Density (With legend)
    create_and_save_plot(df_str, 'Density_Mean', 'Density_Std', 
                         'Edge Density', 'fig_GWCN_Density.pdf', show_legend=True)
    
    # 4. GWCN: Convergence Error (With legend)
    create_and_save_plot(df_str, 'Error_Mean', 'Error_Std', 
                         'Convergence Error', 'fig_GWCN_Error.pdf', show_legend=True)

    # === 5. Display plots ===
    print("Displaying plots. Close all windows to exit...")
    plt.show()

if __name__ == "__main__":
    main()