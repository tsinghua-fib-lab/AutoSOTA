"""
Radar Plot Module - Draw Cognitive Fingerprints
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict


class RadarPlotter:
    """
    Draw radar plots (cognitive fingerprints) - Layered approach
    """

    # Layer 1: Perception (theta parameters)
    PERCEPTION_PARAMS = [
        'Baseline (θ)', 'Authority (θ)', 'Threat (θ)', 'Regret (θ)'
    ]

    # Layer 2: Learning (alpha parameters)
    LEARNING_PARAMS = [
        'Optimism (α+)', 'Optimism (α-)',
        'Punishment (α+)', 'Punishment (α-)',
        'Regret (α+)', 'Regret (α-)'
    ]

    # Layer 3: Risk (rho and R_perc)
    RISK_PARAMS = [
        'Risk_Pref (ρ)', 'R_perc'
    ]

    # All parameters for data extraction
    ALL_PARAMS = list(set(PERCEPTION_PARAMS + LEARNING_PARAMS + RISK_PARAMS))

    def __init__(self, output_dir: str = "logs/images", exclude_models: List[str] = None, config: Dict = None):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.exclude_models = exclude_models or []
        self.config = config or {}

    # ---------------------- 数据加载 ----------------------
    def load_and_prep_data(self, folder_path: str) -> pd.DataFrame:
        """
        Read all models' cognitive_report.csv and merge into standard format
        """
        all_data = []

        # Single-run
        direct_report = os.path.join(folder_path, "cognitive_report.csv")
        if os.path.exists(direct_report):
            model_name = os.path.basename(os.path.dirname(folder_path))
            if model_name not in self.exclude_models:
                df = pd.read_csv(direct_report)
                rec = self._extract_model_record(df, model_name)
                if rec:
                    all_data.append(rec)
            return pd.DataFrame(all_data)

        # Multi-model
        csv_files = glob.glob(os.path.join(folder_path, "**", "cognitive_report.csv"), recursive=True)
        for file_path in csv_files:
            model_name = os.path.basename(os.path.dirname(file_path))
            if model_name in self.exclude_models:
                continue
            try:
                df = pd.read_csv(file_path)
                rec = self._extract_model_record(df, model_name)
                if rec:
                    all_data.append(rec)
            except Exception as e:
                print(f"Error processing {model_name}: {e}")

        return pd.DataFrame(all_data)

    def _extract_model_record(self, df: pd.DataFrame, model_name: str) -> dict:
        """
        Extract all parameters for layered radar charts
        """
        # Check for required groups
        required_groups = ['Baseline', 'Authority', 'Optimism', 'Threat', 'Punishment', 'Regret']
        for grp in required_groups:
            if df[df['Group'] == grp].empty:
                return None

        # Extract all group data
        base = df[df['Group'] == 'Baseline'].iloc[0]
        auth = df[df['Group'] == 'Authority'].iloc[0]
        opt = df[df['Group'] == 'Optimism'].iloc[0]
        threat = df[df['Group'] == 'Threat'].iloc[0]
        punish = df[df['Group'] == 'Punishment'].iloc[0]
        regret = df[df['Group'] == 'Regret'].iloc[0]

        record = {
            'Model': model_name.lower(),

            # Layer 1: Perception (theta)
            'Baseline (θ)': base.get('theta'),
            'Authority (θ)': auth.get('theta'),
            'Threat (θ)': threat.get('theta'),
            'Regret (θ)': regret.get('theta'),

            # Layer 2: Learning (alpha)
            'Optimism (α+)': opt.get('alpha_pos'),
            'Optimism (α-)': opt.get('alpha_neg'),
            'Punishment (α+)': punish.get('alpha_pos'),
            'Punishment (α-)': punish.get('alpha_neg'),
            'Regret (α+)': regret.get('alpha_pos'),
            'Regret (α-)': regret.get('alpha_neg'),

            # Layer 3: Risk (rho and R_perc)
            'Risk_Pref (ρ)': opt.get('rho'),
        }

        # Optional: R_perc (extended mode)
        if 'R_perc' in df.columns:
            record['R_perc'] = base.get('R_perc')

        # Optional: Magnitude rho
        if not df[df['Group'] == 'Magnitude'].empty:
            record['Magnitude (ρ)'] = df[df['Group'] == 'Magnitude'].iloc[0].get('rho')

        return record

    # ---------------------- 归一化 ----------------------
    def normalize_baseline_zscore(self, df: pd.DataFrame, params: List[str]) -> pd.DataFrame:
        """Relative to baseline: (x - baseline) / std - per layer"""
        df_norm = df.copy()
        for col in params:
            if col in df.columns:
                std = df[col].std()
                # Use Baseline scenario value as reference
                baseline_row = df[df['Model'] == 'Baseline']
                if not baseline_row.empty:
                    baseline_val = baseline_row[col].iloc[0]
                else:
                    baseline_val = df[col].iloc[0]
                df_norm[col] = (df[col] - baseline_val) / (std if std != 0 else 1.0)
        return df_norm

    def normalize_minmax(self, df: pd.DataFrame, params: List[str]) -> pd.DataFrame:
        """Global min-max normalization to [0,1] - per layer"""
        df_norm = df.copy()
        for col in params:
            if col in df.columns:
                min_v = df[col].min()
                max_v = df[col].max()
                df_norm[col] = (df[col] - min_v) / (max_v - min_v) if max_v != min_v else 0.5
        return df_norm

    def normalize_identity(self, df: pd.DataFrame, params: List[str]) -> pd.DataFrame:
        """Return original values - per layer"""
        return df.copy()

    # ---------------------- 雷达图 ----------------------
    def plot_radar(self, df: pd.DataFrame, params: List[str], title: str, save_name: str):
        """Plot radar chart for specific parameter layer"""
        if df.empty:
            print("No data to plot")
            return

        # Filter to only include available parameters
        labels = [p for p in params if p in df.columns]
        if not labels:
            print(f"No parameters available for {title}")
            return

        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        # Adjust figure size based on number of parameters
        fig_size = (7, 7) if num_vars <= 6 else (8, 8)
        fig, ax = plt.subplots(figsize=fig_size, subplot_kw=dict(polar=True))

        # Plot each model
        for _, row in df.iterrows():
            values = [row[l] for l in labels]
            values += values[:1]
            ax.plot(angles, values, linewidth=2, label=row['Model'])
            ax.fill(angles, values, alpha=0.15)

        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        ax.set_title(title, fontsize=14, pad=25, fontweight='bold')
        ax.grid(True)

        # Adjust legend position
        legend_loc = 'upper right' if num_vars <= 4 else 'upper left'
        bbox_anchor = (1.3, 1.1) if num_vars <= 4 else (-0.1, 1.1)
        ax.legend(loc=legend_loc, bbox_to_anchor=bbox_anchor, fontsize=9)

        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"✓ Layered radar saved: {save_path}")
        plt.close()

    # ---------------------- 可选原始图 ----------------------
    def plot_theta_diverging(self, df: pd.DataFrame, output_filename: str = "theta_diverging.png"):
        """Original theta diverging plot"""
        if df.empty: return
        theta_cols = ['Baseline (θ)', 'Authority (θ)', 'Threat (θ)', 'Regret (θ)']
        theta_cols = [col for col in theta_cols if col in df.columns]
        if not theta_cols: return

        df_plot = df[theta_cols + ['Model']].copy()
        df_plot[theta_cols] = (df_plot[theta_cols] - df_plot[theta_cols].min().min()) / (
                               df_plot[theta_cols].max().max() - df_plot[theta_cols].min().min())
        df_long = df_plot.melt(id_vars='Model', value_vars=theta_cols,
                               var_name='Scenario', value_name='Theta')
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_long, x='Theta', y='Model', hue='Scenario')
        plt.title('Theta Parameters Comparison (Normalized)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, output_filename), dpi=300)
        plt.close()
        print(f"✓ Diverging plot saved: {os.path.join(self.output_dir, output_filename)}")

    # ---------------------- 总流程 ----------------------
    def run(self, folder_path: str):
        """Load, normalize, and generate layered radar charts"""
        df = self.load_and_prep_data(folder_path)
        if df.empty:
            print("No valid data")
            return

        # Save raw summary
        summary_path = os.path.join(self.output_dir, "radar_summary.csv")
        df.to_csv(summary_path, index=False)
        print(f"Summary saved: {summary_path}")

        # Check if config-driven mode is enabled with actual layer config
        has_layer_config = bool(self.config.get('layers', {}))
        if self.config.get('layered_mode', True) and has_layer_config:
            self._run_config_layers(df)
        else:
            self._run_default_layers(df)

    def _run_default_layers(self, df: pd.DataFrame):
        """Default layered approach with hardcoded parameters"""
        # Layer 1: Perception (theta parameters)
        print("\n" + "="*60)
        print("LAYER 1: PERCEPTION (θ parameters)")
        print("="*60)
        self.plot_radar(
            df=self.normalize_minmax(df, self.PERCEPTION_PARAMS),
            params=self.PERCEPTION_PARAMS,
            title="Perception Layer: θ Parameters\n(Baseline, Authority, Threat, Regret)",
            save_name="layer1_perception.png"
        )

        # Layer 2: Learning (alpha parameters)
        print("\n" + "="*60)
        print("LAYER 2: LEARNING (α parameters)")
        print("="*60)
        self.plot_radar(
            df=self.normalize_minmax(df, self.LEARNING_PARAMS),
            params=self.LEARNING_PARAMS,
            title="Learning Layer: α Parameters\n(Optimism, Punishment, Regret)",
            save_name="layer2_learning.png"
        )

        # Layer 3: Risk (rho and R_perc)
        print("\n" + "="*60)
        print("LAYER 3: RISK (ρ and R_perc)")
        print("="*60)
        risk_params = [p for p in self.RISK_PARAMS if p in df.columns]
        if risk_params:
            self.plot_radar(
                df=self.normalize_minmax(df, risk_params),
                params=risk_params,
                title="Risk Layer: ρ and R_perc Parameters",
                save_name="layer3_risk.png"
            )
        else:
            print("⚠️  R_perc not available (requires extended mode)")

        # Additional: Diverging plot for theta
        print("\n" + "="*60)
        print("ADDITIONAL: THETA COMPARISON")
        print("="*60)
        self.plot_theta_diverging(df, "theta_comparison.png")

        print("\n" + "="*60)
        print("✅ ALL LAYERED RADAR CHARTS GENERATED")
        print("="*60)

    def _run_config_layers(self, df: pd.DataFrame):
        """Config-driven layered approach"""
        layers_config = self.config.get('layers', {})
        comparison_config = self.config.get('comparison', {})
        normalization_method = self.config.get('normalization', 'minmax')

        # Select normalization method
        if normalization_method == 'minmax':
            norm_func = self.normalize_minmax
        elif normalization_method == 'zscore':
            norm_func = self.normalize_baseline_zscore
        else:
            norm_func = self.normalize_identity

        # Process each layer
        for layer_name, layer_cfg in layers_config.items():
            if not layer_cfg.get('enabled', True):
                continue

            params = layer_cfg.get('parameters', [])
            title = layer_cfg.get('title', layer_name.upper())
            filename = layer_cfg.get('filename', f'{layer_name}.png')

            # Filter to available parameters
            available_params = [p for p in params if p in df.columns]
            if not available_params:
                print(f"⚠️  Skipping {layer_name}: no available parameters")
                continue

            print("\n" + "="*60)
            print(f"LAYER: {layer_name.upper()}")
            print("="*60)

            self.plot_radar(
                df=norm_func(df, available_params),
                params=available_params,
                title=title,
                save_name=filename
            )

        # Process comparison plots
        if comparison_config.get('theta_comparison', {}).get('enabled', True):
            print("\n" + "="*60)
            print("ADDITIONAL: THETA COMPARISON")
            print("="*60)
            filename = comparison_config.get('theta_comparison', {}).get('filename', 'theta_comparison.png')
            self.plot_theta_diverging(df, filename)

        print("\n" + "="*60)
        print("✅ ALL LAYERED RADAR CHARTS GENERATED")
        print("="*60)

    # ---------------------- Additional Style Plots ----------------------
    def plot_perception_diverging(self, df: pd.DataFrame, output_filename: str = "1_perception_bias.png"):
        """
        Chart 1: Perception Bias (Theta) Diverging Bar Chart
        """
        theta_cols = ['Baseline (θ)', 'Authority (θ)', 'Threat (θ)']
        theta_cols = [col for col in theta_cols if col in df.columns]
        if not theta_cols:
            print("⚠️  Skipping perception diverging: missing theta columns")
            return

        # Global normalization to [-1, 1]
        global_min = -10
        global_max = 10

        df_plot = df[theta_cols + ['Model']].copy()
        df_plot[theta_cols] = 2 * (df_plot[theta_cols] - global_min) / (global_max - global_min) - 1

        df_long = df_plot.melt(id_vars='Model', value_vars=theta_cols,
                               var_name='Scenario', value_name='Theta')

        plt.figure(figsize=(10, 8), dpi=300)

        colors = {
            'Baseline (θ)': '#7f8c8d',
            'Authority (θ)': '#9b59b6',
            'Threat (θ)': '#2c3e50'
        }

        # Filter colors to only include available columns
        colors = {k: v for k, v in colors.items() if k in theta_cols}

        sns.barplot(data=df_long, x='Theta', y='Model', hue='Scenario', palette=colors)

        plt.axvline(x=0, color='black', linestyle='-', linewidth=2)
        plt.axvspan(-1, 0, color='green', alpha=0.05)
        plt.axvspan(0, 1, color='red', alpha=0.05)

        plt.xlabel("Refuse $\\to$ Compliant", fontsize=12)
        plt.ylabel("Model", fontsize=12)
        plt.title("Perception Bias ($\\Theta$) Profiles Across Scenarios", fontsize=14, fontweight='bold')

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 1, 0.98])

        save_path = os.path.join(self.output_dir, output_filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"✓ Perception diverging saved: {save_path}")

    def plot_alpha_subplots(self, df: pd.DataFrame, output_filename: str = "2_alpha_asymmetry_subplots.png"):
        """
        Chart 2: Learning Rate Comparison (Alpha+/-) in 2x2 subplots
        """
        alpha_groups = {
            'Optimism': ['Optimism (α+)', 'Optimism (α-)'],
            'Punishment': ['Punishment (α+)', 'Punishment (α-)'],
            'Regret': ['Regret (α+)', 'Regret (α-)'],
            'Threat': ['Threat (α+)', 'Threat (α-)']
        }

        # Filter to available columns
        available_groups = {}
        for scenario, cols in alpha_groups.items():
            available_cols = [c for c in cols if c in df.columns]
            if available_cols:
                available_groups[scenario] = available_cols

        if not available_groups:
            print("⚠️  Skipping alpha subplots: no alpha columns available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300, sharey=True)
        axes = axes.flatten()

        colors = {'(α+)': '#e74c3c', '(α-)': '#3498db'}

        y_max = 1.05

        for i, (scenario, cols) in enumerate(available_groups.items()):
            if i >= 4:
                break
            ax = axes[i]

            df_melt = df.melt(id_vars='Model', value_vars=cols, var_name='Type', value_name='Learning Rate')
            df_melt['Hue Key'] = df_melt['Type'].apply(lambda x: x[-4:])

            sns.barplot(data=df_melt, x='Model', y='Learning Rate', hue='Hue Key',
                        palette=colors, ax=ax, errorbar=None)

            ax.set_title(f'Scenario: {scenario}', fontsize=14, fontweight='bold')
            ax.set_ylabel("Learning Rate ($\\alpha$)", fontsize=12)
            ax.set_xlabel("Model", fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.set_ylim(0, y_max)
            ax.axhline(y=0.1, color='grey', linestyle=':', alpha=0.5, label='Baseline Default')

            if i == 0:
                ax.legend(title='Learning Type', loc='upper right')
            else:
                if ax.get_legend():
                    ax.get_legend().remove()

        # Hide unused subplots
        for i in range(len(available_groups), 4):
            axes[i].set_visible(False)

        fig.suptitle("Asymmetric Learning Profiles Across Scenarios", fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 0.98])

        save_path = os.path.join(self.output_dir, output_filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"✓ Alpha asymmetry subplots saved: {save_path}")

    def plot_risk_preference(self, df: pd.DataFrame, output_filename: str = "3_risk_preference_subplots.png"):
        """
        Chart 3: Risk Preference (Rho) Comparison in 1x2 subplots
        """
        # Try different column name variations
        rho_cols = []
        for col in df.columns:
            if 'rho' in col.lower() or 'risk' in col.lower():
                rho_cols.append(col)

        if not rho_cols:
            print("⚠️  Skipping risk preference: no rho columns available")
            return

        # Use first two rho columns if available
        rho_groups = {}
        if len(rho_cols) >= 2:
            rho_groups = {
                'Stimulus 1': rho_cols[0],
                'Stimulus 2': rho_cols[1]
            }
        else:
            rho_groups = {'Risk Preference': rho_cols[0]}

        fig, axes = plt.subplots(1, len(rho_groups), figsize=(14, 8), dpi=300, sharey=True)
        if len(rho_groups) == 1:
            axes = [axes]

        rho_min = 0.01
        rho_max = 5.0

        for i, (scenario, col) in enumerate(rho_groups.items()):
            ax = axes[i]

            ax.axvline(x=1.0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            ax.axvspan(rho_min, 1.0, color='blue', alpha=0.05, label='Risk Averse ($\\rho < 1$)')
            ax.axvspan(1.0, rho_max, color='red', alpha=0.05, label='Risk Seeking ($\\rho > 1$)')

            sns.stripplot(data=df, x=col, y='Model', size=10, palette='viridis', jitter=0.1, ax=ax, hue=col, legend=False)

            ax.set_xlim(rho_min, 5.0)
            ax.set_title(f'Scenario: {scenario}', fontsize=14, fontweight='bold')
            ax.set_xlabel("Risk Preference ($\\rho$)", fontsize=12)

            if i == 0:
                ax.set_ylabel("Model", fontsize=12)
            else:
                ax.set_ylabel("")

            ax.grid(axis='x', linestyle='--', alpha=0.6)

            ax.text(0.5, -0.1, "Risk Averse", ha='center', color='blue', transform=ax.get_xaxis_transform(), fontsize=9)
            ax.text(2.0, -0.1, "Risk Seeking", ha='center', color='red', transform=ax.get_xaxis_transform(), fontsize=9)

        plt.tight_layout(rect=[0, 0, 1, 0.98])

        save_path = os.path.join(self.output_dir, output_filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"✓ Risk preference subplots saved: {save_path}")
