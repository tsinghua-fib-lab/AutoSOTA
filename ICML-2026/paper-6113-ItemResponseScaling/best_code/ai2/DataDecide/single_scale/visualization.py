import os
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import format_tokens, model_order

# TODO: Visualize other evaluator metrics

class Visualization:
    def __init__(self, plot_dir="figures"):
        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)

    def _process_data_for_metric_max(self, data, value_vars):
        """Preprocess the data to get max or min values per group."""
        # Melt
        melted_df = data.melt(
            id_vars=["compute", "tokens", "metric", "model"],
            value_vars=value_vars,
            var_name="result_metric",
            value_name="value"
        ).dropna(subset=["compute", "model", "result_metric", "value"])

        # Identify agg
        agg_dfs = {}
        for result_metric in value_vars:
            group = melted_df[melted_df["result_metric"] == result_metric]
            if result_metric == "magnitude_ece":
                max_idx = group.groupby(["compute", "model", "result_metric"], observed=True)["value"].idxmin()
            else:
                max_idx = group.groupby(["compute", "model", "result_metric"], observed=True)["value"].idxmax()

            agg_max_df = melted_df.loc[max_idx].reset_index(drop=True)
            dropped_points = group.drop(index=max_idx)
            agg_dfs[result_metric] = (agg_max_df, dropped_points)

        return agg_dfs

    def plot_metric_max(self, value_vars, max_tokens):
        """Plot the max/min values per metric along with dropped points."""
        agg_dfs = self._process_data_for_metric_max(value_vars)

        for result_metric, (agg_max_df, dropped_points) in agg_dfs.items():
            df_result = agg_max_df[agg_max_df['result_metric'] == result_metric]
            df_dropped = dropped_points[dropped_points['result_metric'] == result_metric]

            # Plot scatter points
            g = sns.relplot(
                data=df_dropped,
                x='compute',
                y='value',
                hue='model',
                kind='scatter',
                style='metric',
                s=50,
                palette='viridis',
                alpha=0.6,
                # height=8,
                # aspect=1.5,
                legend="auto",
            )

            # Overlay max points and lines
            sns.scatterplot(
                data=df_result,
                x='compute',
                y='value',
                hue='model',
                style='metric',
                s=120,
                palette='viridis',
                legend=False,
                ax=g.ax
            )
            sns.lineplot(
                data=df_result,
                x='compute',
                y='value',
                hue='model',
                palette='viridis',
                legend=False,
                ax=g.ax
            )

            # Set axis labels and title
            ax = g.ax
            ax.set_xlabel('Compute (6ND)')
            ax.set_ylabel('Value')
            ax.set_title(f'{result_metric}')

            if result_metric == "magnitude_ece":
                ax.invert_yaxis()

            # Add secondary x-axis for Tokens
            token_vals = np.linspace(0, max_tokens, 5)
            newpos = [self.calc_compute("1B", t) for t in token_vals]
            ax2 = ax.twiny()
            ax2.set_xticks(newpos)
            ax2.set_xticklabels([format_tokens(int(t)) for t in token_vals])
            ax2.xaxis.set_ticks_position('bottom')
            ax2.xaxis.set_label_position('bottom')
            ax2.set_xlabel("Tokens (B)")
            ax2.spines['bottom'].set_position(('outward', 36))
            ax2.grid(False)
            ax2.yaxis.grid(False)

            # Manually create the legend
            handles, labels = ax.get_legend_handles_labels()
            g.fig.legend(
                handles=handles,
                labels=labels,
                loc="center right",
                bbox_to_anchor=(1.18, 0.5),
                frameon=False,
                title="",
                ncol=2  # Split into 2 columns
            )

            # remove original g's legend
            g._legend.remove()

            # Adjust space to avoid overlap
            g.fig.tight_layout(rect=[0, 0, 0.55, 1])

            # Save
            plt.savefig(f"{self.plot_dir}/max_values_{result_metric}.png", dpi=500)
            plt.savefig(f"{self.plot_dir}/max_values_{result_metric}.pdf", dpi=500)
            logging.info(f"Saved: '{self.plot_dir}/max_values_{result_metric}.png/.pdf'")
            plt.close()

    def plot_heatmap_model_scale(self, data, metrics, save_name="pivot_heatmap", value_col="binary_accuracy", seed = 'aggregate_seeds'):
        """
        Plot a single figure containing heatmaps for all metrics, arranged in an nxn grid.
        Apply hatched pattern to the best-performing metric for each compute.
        """
        import math
        import matplotlib.patches as patches

        # Calculate grid dimensions
        n_metrics = len(metrics)
        grid_size = math.ceil(math.sqrt(n_metrics))  # Determine nxn grid dimensions

        if seed == 'aggregate_seeds':
            df = data.groupby(["model", "proportion", "metric"])[value_col].agg(['mean', 'std']).reset_index()
        else:
            assert 'seed' in data.columns, "Seed column not found in data"
            assert seed in data.seed.unique(), f"Seed value '{seed}' not found in data"
            df = data[data['seed'] == seed]
            df = df[['model', 'proportion', 'metric', value_col]].rename(columns={value_col: 'mean'})
            df['std'] = None
            if  seed == 'all seeds':
                df['proportion'] = df['proportion'] * 3

        # Create a shared color scale)
        vmin, vmax = df["mean"].min(), df["mean"].max()

        # Set up the grid for plotting
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 6, grid_size * 5))
        axes = axes.flatten()

        # Precompute the best-performing cells with ties allowed
        best_cells = {}
        for (model, proportion_target), group in df.groupby(["model", "proportion"]):
            if group.empty:
                continue
            # Identify the maximum value(s) for this (model, proportion_target)
            max_value = group["mean"].max()
            best_metrics = group[group["mean"] == max_value]["metric"].tolist()
            best_cells[(model, proportion_target)] = best_metrics

        for i, metric in enumerate(metrics):
            ax = axes[i]
            metric_data = df[df["metric"] == metric]

            # Pivot data for heatmap plotting
            heatmap_data = metric_data.pivot(index="model", columns="proportion", values="mean")
            std_data = metric_data.pivot(index="model", columns="proportion", values="std")

            # Reorder rows based on model order
            heatmap_data = heatmap_data.reindex(index=model_order[::-1])

            # Plot the heatmap
            sns.heatmap(
                heatmap_data,
                ax=ax,
                annot=False,
                cmap="RdYlGn",
                fmt=".2f",
                cbar=(i == n_metrics - 1),  # Add a colorbar only to the last subplot
                vmin=vmin,
                vmax=vmax,
                linewidths=0.5
            )
            ax.set_title(metric, fontsize=14)
            ax.set_xlabel("Proportion of Target Compute", fontsize=10)
            ax.set_ylabel("Model Size", fontsize=10)

            # Annotate with both avg and std
            for row, model in enumerate(heatmap_data.index):
                for col, proportion_target in enumerate(heatmap_data.columns):
                    avg = heatmap_data.loc[model, proportion_target]
                    std = std_data.loc[model, proportion_target]
                    text = f"{avg:.2f}\n(±{std:.2f})" if seed == 'aggregate_seeds' else f"{avg:.2f}"
                    if pd.notnull(avg):
                        ax.text(
                            col + 0.5, row + 0.5,
                            text,
                            ha="center", va="center", fontsize=8, color="white"
                        )

            # Apply hatch pattern to highlight best-performing cells (including ties)
            for row, model in enumerate(heatmap_data.index):
                for col, proportion_target in enumerate(heatmap_data.columns):
                    if (model, proportion_target) in best_cells and metric in best_cells[(model, proportion_target)]:
                        rect = patches.Rectangle(
                            (col, row), 1, 1,
                            linewidth=1, edgecolor='black', facecolor='none', hatch='XXX'
                        )
                        ax.add_patch(rect)

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        # Adjust layout and save the figure
        fig.tight_layout()
        save_path_png = f"{self.plot_dir}/{save_name}.png"
        save_path_pdf = f"{self.plot_dir}/{save_name}.pdf"
        plt.savefig(save_path_png)
        plt.savefig(save_path_pdf)
        logging.info(f"Saved pivot heatmaps: '{save_path_png}' and '{save_path_pdf}'")
        plt.close()

    @staticmethod
    def calc_compute(model_size_str, tokens):
        """Calculate compute based on model size and tokens"""
        scale = 1e6 if model_size_str[-1].lower() == "m" else 1e9
        num_parameters = int(float(model_size_str[:-1]) * scale)
        return 6 * num_parameters * tokens

    def plot_heatmap_multi_seed(self, data, metrics, save_name="seed_heatmap", value_col="binary_accuracy"):
        import math
        import matplotlib.patches as patches

        # Calculate grid dimensions
        n_metrics = len(metrics)
        grid_size = math.ceil(math.sqrt(n_metrics))  # Determine nxn grid dimensions

        # Create a shared color scale
        df = data.groupby(["seed_count", "proportion", "metric"])[value_col].agg(['mean', 'std']).reset_index()
        vmin, vmax = df["mean"].min(), df["mean"].max()

        # Set up the grid for plotting
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 6, grid_size * 5))
        axes = axes.flatten()

        # Precompute the best-performing cells with ties allowed
        best_cells = {}
        for (num_seeds, proportion_target), group in df.groupby(["seed_count", "proportion"]):
            if group.empty:
                continue
            # Identify the maximum value(s) for this (seed_count, proportion_target)
            max_value = group["mean"].max()
            best_metrics = group[group["mean"] == max_value]["metric"].tolist()
            best_cells[(num_seeds, proportion_target)] = best_metrics

        for i, metric in enumerate(metrics):
            ax = axes[i]
            metric_data = df[df["metric"] == metric]

            # Pivot data for heatmap plotting
            heatmap_data = metric_data.pivot(index="seed_count", columns="proportion", values="mean")
            std_data = metric_data.pivot(index="seed_count", columns="proportion", values="std")

            # Reorder rows based on seed count
            heatmap_data = heatmap_data.reindex(index=sorted(heatmap_data.index, reverse=True))

            # Plot the heatmap
            sns.heatmap(
                heatmap_data,
                ax=ax,
                annot=False,
                cmap="RdYlGn",
                fmt=".2f",
                cbar=(i == n_metrics - 1),  # Add a colorbar only to the last subplot
                vmin=vmin,
                vmax=vmax,
                linewidths=0.5
            )
            ax.set_title(metric, fontsize=14)
            ax.set_xlabel("Proportion of Target Compute", fontsize=10)
            ax.set_ylabel("Number of Seeds", fontsize=10)

            # Annotate with both avg and std
            for row, num_seeds in enumerate(heatmap_data.index):
                for col, proportion_target in enumerate(heatmap_data.columns):
                    avg = heatmap_data.iloc[row, col]
                    std = std_data.iloc[row, col]
                    if pd.notnull(avg):
                        ax.text(
                            col + 0.5, row + 0.5,
                            f"{avg:.2f}\n(±{std:.2f})",
                            ha="center", va="center", fontsize=8, color="white"
                        )

            # Apply hatch pattern to highlight best-performing cells (including ties)
            for row, num_seeds in enumerate(heatmap_data.index):
                for col, proportion_target in enumerate(heatmap_data.columns):
                    if (num_seeds, proportion_target) in best_cells and metric in best_cells[(num_seeds, proportion_target)]:
                        rect = patches.Rectangle(
                            (col, row), 1, 1,
                            linewidth=1, edgecolor='black', facecolor='none', hatch='XXX'
                        )
                        ax.add_patch(rect)

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        # Adjust layout and save the figure
        fig.tight_layout()
        save_path_png = f"{self.plot_dir}/{save_name}.png"
        save_path_pdf = f"{self.plot_dir}/{save_name}.pdf"
        plt.savefig(save_path_png)
        plt.savefig(save_path_pdf)
        logging.info(f"Saved seed heatmaps: '{save_path_png}' and '{save_path_pdf}'")
        plt.close()

if __name__ == "__main__":
    from utils import RC_METRICS, LOSS_COLS, ACC_METRICS
    import argparse

    task_metrics = RC_METRICS + LOSS_COLS + ACC_METRICS

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="outputs2")
    parser.add_argument("--plot_dir", type=str, default="figures")
    args = parser.parse_args()



    df_model_scale = pd.read_csv(f"{args.out_dir}/2_prediction_model_scale.csv")
    # df_seeds = pd.read_csv(f"{args.out_dir}/2_prediction_seeds.csv")

    # Viz
    viz = Visualization(plot_dir=args.plot_dir)
    viz.plot_heatmap_model_scale(data=df_model_scale, metrics=task_metrics, value_col="binary_accuracy", save_name="heatmap_model_scale")
    viz.plot_heatmap_model_scale(data=df_model_scale, metrics=task_metrics, value_col="magnitude_correlation", save_name="heatmap_model_scale_correlation")
    viz.plot_heatmap_model_scale(data=df_model_scale, metrics=task_metrics, value_col="pearson_correlation", save_name="heatmap_model_scale_pearson_correlation")
    # viz.plot_heatmap_model_scale(data=df_seeds[df_seeds['seed_count'] == 3], metrics=task_metrics, value_col="three_way_accuracy", save_name="heatmap_model_scale_all_seeds_3_class", seed='all seeds')
    # viz.plot_heatmap_multi_seed(data=df_seeds, metrics=task_metrics, value_col="binary_accuracy", save_name="heatmap_multi_seed")
