"""
Drift Plot Module - Draw Time Dynamic Curves

Display ASR trends over time (number of trials)
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class DriftPlotter:
    """
    Draw drift plots (time dynamics)
    """

    # Scenario color configuration
    PALETTE = {
        'Baseline': '#95a5a6',
        'Optimism': '#e74c3c',
        'Optimism-Neg': '#c0392b',
        'Punishment': '#c0392b',
        'Threat': '#2c3e50',
        'Authority': '#8e44ad',
        'Sycophancy': '#e67e22',
        'Regret': '#16a085',
        'Stimulus': '#f1c40f',
        'Magnitude': '#d35400'
    }

    def __init__(self, output_dir: str = "logs/images", palette: dict = None):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Use custom palette if provided, otherwise use default
        if palette:
            self.PALETTE = palette
        # Otherwise, use default PALETTE defined at class level

        # Set plot style
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)

    def _process_single_file(self, filepath: str) -> pd.DataFrame | None:
        """
        Process single CSV file
        """
        try:
            df = pd.read_csv(filepath)
        except:
            return None

        if 'action' not in df.columns or 'group' not in df.columns:
            return None

        # Filter valid actions
        df = df[df['action'].isin(['Compliance', 'Refusal'])].copy()

        # Binary flag
        df['is_jailbreak'] = df['action'].apply(lambda x: 1 if x == 'Compliance' else 0)

        # Add filename
        df['source_file'] = os.path.basename(filepath)

        # Calculate relative trial index
        df = df.sort_values('trial')
        df['relative_trial'] = df.groupby('group').cumcount() + 1

        return df

    def load_batch_data(self, folder_path: str) -> pd.DataFrame:
        """
        Load and merge all CSV files
        Supports both legacy format (CSVs directly in folder) and new format (subdirectories per scenario group)
        """
        # Check if folder_path is a run directory (new format) or direct CSV folder (legacy)
        subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

        # New format: subdirectories for each scenario group
        if subdirs and any(d in ['Baseline', 'Optimism', 'Authority', 'Threat', 'Stimulus', 'Magnitude', 'Punishment', 'Regret', 'Sycophancy'] for d in subdirs):
            files = []
            for subdir in subdirs:
                subdir_path = os.path.join(folder_path, subdir)
                if os.path.isdir(subdir_path):
                    csv_files = glob.glob(os.path.join(subdir_path, "*.csv"))
                    files.extend(csv_files)
        else:
            # Legacy format: CSV files directly in folder
            files = glob.glob(os.path.join(folder_path, "*.csv"))

        print(f"Found {len(files)} CSV files")

        df_list = []
        for f in tqdm(files, desc="Loading CSV"):
            processed = self._process_single_file(f)
            if processed is not None and not processed.empty:
                df_list.append(processed)

        if not df_list:
            return pd.DataFrame()

        return pd.concat(df_list, ignore_index=True)

    def calculate_trends(self, full_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate cumulative and rolling ASR for each file
        """
        if full_df.empty:
            return pd.DataFrame()

        print("Calculating trends...")

        full_df = full_df.sort_values(['source_file', 'group', 'relative_trial'])

        def calc_metrics(chunk):
            chunk['cum_asr'] = chunk['is_jailbreak'].expanding().mean()
            chunk['rolling_asr'] = chunk['is_jailbreak'].rolling(window=5, min_periods=1).mean()
            return chunk

        augmented_df = full_df.groupby(['source_file', 'group'], group_keys=False).apply(calc_metrics)
        return augmented_df

    def plot_drift(self, df: pd.DataFrame, output_filename: str = "drift_analysis.png"):
        """
        Draw drift plot
        """
        if df.empty:
            print("No data to plot")
            return

        # Get unique groups and ensure all have colors
        unique_groups = df['group'].unique()

        # Build palette for all groups (use existing or generate)
        plot_palette = {}
        for group in unique_groups:
            if group in self.PALETTE:
                plot_palette[group] = self.PALETTE[group]
            else:
                # Generate color for unknown group
                import matplotlib.colors as mcolors
                base_colors = list(mcolors.TABLEAU_COLORS)
                idx = len(plot_palette) % len(base_colors)
                plot_palette[group] = base_colors[idx]

        fig, axes = plt.subplots(1, 2, figsize=(20, 7), dpi=150)

        # Left plot: Cumulative ASR
        sns.lineplot(
            data=df,
            x='relative_trial',
            y='cum_asr',
            hue='group',
            palette=plot_palette,
            linewidth=3,
            ax=axes[0],
            errorbar=('ci', 95)
        )

        axes[0].set_title("Average Cumulative ASR", fontsize=16, fontweight='bold')
        axes[0].set_xlabel("Relative Interaction Turns", fontsize=14)
        axes[0].set_ylabel("Cumulative ASR (Mean)", fontsize=14)
        axes[0].set_ylim(-0.05, 1.05)
        axes[0].grid(True, linestyle=':', alpha=0.6)

        max_trial = df['relative_trial'].max()
        axes[0].set_xlim(1, max_trial)

        # Right plot: Rolling ASR
        sns.lineplot(
            data=df,
            x='relative_trial',
            y='rolling_asr',
            hue='group',
            palette=plot_palette,
            linewidth=3,
            ax=axes[1],
            errorbar=('ci', 95)
        )

        axes[1].set_title("Average Instantaneous Compliance (Window=5)", fontsize=16, fontweight='bold')
        axes[1].set_xlabel("Relative Interaction Turns", fontsize=14)
        axes[1].set_ylabel("Rolling ASR (Mean)", fontsize=14)
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].set_xlim(1, max_trial)
        axes[1].grid(True, linestyle=':', alpha=0.6)

        # Unified legend
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].get_legend().remove()
        axes[1].get_legend().remove()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(labels), frameon=False)

        plt.tight_layout()

        save_path = os.path.join(self.output_dir, output_filename)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Drift plot saved: {save_path}")
        plt.close()

    def run(self, folder_path: str, output_filename: str = "drift_analysis.png"):
        """
        Complete workflow: Load -> Calculate -> Plot
        """
        df = self.load_batch_data(folder_path)
        if df.empty:
            print("No valid data")
            return

        df_trends = self.calculate_trends(df)
        self.plot_drift(df_trends, output_filename)