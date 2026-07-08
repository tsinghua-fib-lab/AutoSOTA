from collections import defaultdict

from substantive.faircp.structs.fairness_experiment_result import FairnessExperimentResult

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple, Iterable
from pathlib import Path
import warnings
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable
from scipy.special import expit
warnings.filterwarnings('ignore')

def calculate_accuracy_per_method(results: list[FairnessExperimentResult]):
    """
    Calculate accuracy per conformal method.

    Args:
        results (list[dict]): Output from run_llm_prediction

    Returns:
        dict: {method: accuracy_float}
    """
    correct_counts = defaultdict(int)
    total_counts = defaultdict(int)

    for r in results:
        method = r.method
        if r.result.startswith(r.label_text):
            correct_counts[method] += 1
        total_counts[method] += 1

    accuracies = {}
    for method in total_counts:
        accuracies[method] = correct_counts[method] / total_counts[method]

    print(accuracies)

# Set matplotlib style
plt.rcParams.update({
    'figure.figsize': (8, 6),
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.top': True,
    'xtick.bottom': True,
    'ytick.left': True,
    'ytick.right': True,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'xtick.major.pad': 6.0,
    'xtick.minor.pad': 6.0,
    'ytick.major.pad': 6.0,
    'ytick.minor.pad': 6.0,
    'xtick.major.size': 6.0,
    'xtick.minor.size': 3.0,
    'ytick.major.size': 6.0,
    'ytick.minor.size': 3.0,
    'text.usetex': False,
    'font.family': 'serif',
    'font.size': 18
})

# Default color scheme and mappings
COLORS = ['#3B528B', '#472D7B', '#5EC962', '#21918C', '#F0E442']
TREATMENT_NAMES = {
    'control': 'Control',
    'avgk': 'Avg-K',
    'marginal': 'Marginal',
    'conditional': 'Mondrian',
    'topk': 'Top-K',
    'clustered_label': 'Label-Clustered',
    'clustered_group': 'Group-Clustered',
    'backward': 'Backward',
}

# Dataset name dictionary
DATASET_NAMES = {
    'bios': 'BiosBias',
    'ravdess': 'RAVDESS',
    'facet': 'FACET',
    'acs-income': 'ACSIncome',
}

def debug_print(message: str, data: Any = None):
    """Helper function for consistent debugging output."""
    print(f"Stage (or messages): {message}")
    if data is not None:
        print(f"   Data: {data}")
    print()

def parse_conformal_set(conformal_set: Any) -> List[int]:
    """
    Parse a conformal set from many possible representations into a flat list of ints.

    """

    def is_scalar_number(x: Any) -> bool:
        return isinstance(x, (int, float, np.number)) and np.ndim(x) == 0

    def safe_isna_scalar(x: Any) -> bool:
        # Only call pd.isna for scalar values to avoid ambiguous truth errors
        try:
            # Treat strings like "NaN" as NA too
            if isinstance(x, str) and x.strip().lower() in {"nan", "none", ""}:
                return True
            # Scalars only
            if isinstance(x, (list, tuple, set, np.ndarray, pd.Series, pd.Index)):
                return False
            return bool(pd.isna(x))
        except Exception:
            return False

    def iter_like(x: Any) -> bool:
        # True if x is an iterable we should descend into (but not a string/bytes)
        if isinstance(x, (str, bytes)):
            return False
        return isinstance(x, (Iterable,))  # includes numpy arrays, lists, Series, etc.

    def flatten_to_ints(x: Any) -> List[int]:
        """
        Recursively flatten x and collect ints. Handles numpy arrays/Series of any shape.
        """
        out: List[int] = []

        # Numpy arrays / pandas objects: iterate over their flattened view
        if isinstance(x, (np.ndarray, pd.Series, pd.Index)):
            # ravel handles 0-d, 1-d, and higher
            for v in np.ravel(x):
                out.extend(flatten_to_ints(v))
            return out

        # Scalars: try to convert to int
        if is_scalar_number(x):
            try:
                out.append(int(x))
            except (ValueError, TypeError):
                pass
            return out

        # Strings: parse numbers out of comma/space/bracket formats
        if isinstance(x, str):
            s = x.strip()
            if s in {"", "[]"} or s.lower() in {"nan", "none"}:
                return out
            # remove surrounding brackets if present
            if s.startswith("[") and s.endswith("]"):
                s = s[1:-1]
            # split by comma or whitespace
            parts = [p for chunk in s.split(",") for p in chunk.split()]
            for p in parts:
                try:
                    out.append(int(p))
                except ValueError:
                    # ignore non-integer tokens
                    pass
            return out

        # Generic iterables (lists/tuples/sets, custom containers)
        if iter_like(x):
            for v in x:
                out.extend(flatten_to_ints(v))
            return out

        # Anything else: ignore
        return out

    # Early exit on None / scalar-NA
    if conformal_set is None or safe_isna_scalar(conformal_set):
        return []

    try:
        return flatten_to_ints(conformal_set)
    except Exception as e:
        debug_print(f"Error parsing conformal set: {conformal_set} (type: {type(conformal_set)}), Error: {e}")
        return []

def validate_and_clean_prediction(result: str, valid_labels: List[str]) -> str:
    """
    Validate prediction result and mark invalid ones.

    """
    if not result or pd.isna(result):
        return "INVALID"

    result_str = str(result).strip()

    # Check for explicit invalid patterns
    invalid_patterns = [
        "Invalid prediction:",
        "ERROR",
        "ERR_MISSING_RESPONSE"
    ]

    # Only mark as invalid if it explicitly contains invalid patterns
    for pattern in invalid_patterns:
        if pattern.lower() in result_str.lower():
            return "INVALID"

    # Check if result exactly matches any valid label OR starts with any valid label
    for label in valid_labels:
        if result_str == label or result_str.startswith(label):
            return label  # Return the exact label

    # Only mark as invalid if it's clearly not a label
    debug_print(f"Invalid prediction found: '{result_str}' not matching any valid pattern")
    return "INVALID"

def compute_basic_metrics(df: pd.DataFrame, label_map: Dict[int, str]) -> pd.DataFrame:

    df = df.copy()

    # Get list of valid labels
    valid_labels = list(label_map.values())
    debug_print("Valid labels", valid_labels)

    # Clean and validate prediction results
    debug_print("Sample raw results before cleaning", df['result'].head().tolist())
    df['result_clean'] = df['result'].apply(lambda x: validate_and_clean_prediction(x, valid_labels))
    #debug_print("Sample cleaned results", df['result_clean'].head().tolist())

    # Parse conformal sets with detailed debugging
    debug_print("Parsing conformal sets...")
    df['conformal_set_parsed'] = df['conformal_set'].apply(parse_conformal_set)
    debug_print("Sample conformal_set after parsing", df['conformal_set_parsed'].head().tolist())

    # Check if parsing worked
    empty_sets = (df['conformal_set_parsed'].apply(len) == 0).sum()
    debug_print(f"Empty conformal sets after parsing: {empty_sets}/{len(df)}")

    df['set_size'] = df['conformal_set_parsed'].apply(len)
    debug_print("Sample set sizes", df['set_size'].head().tolist())
    #debug_print("Set size statistics:", df['set_size'].describe())

    # Create reverse label map for faster lookup
    reverse_label_map = {label: idx for idx, label in label_map.items()}
    #debug_print("Reverse label map sample", dict(list(reverse_label_map.items())[:5]))

    # Compute coverage (whether true label is in conformal set)
    def is_covered(row):
        try:
            if row['method'] == 'control':
                return False  # Control group has no prediction set

            true_label_idx = reverse_label_map.get(row['label_text'])
            if true_label_idx is None:
                debug_print(f"True label not found in reverse_label_map: {row['label_text']}")
                return False

            # Check if conformal set is empty
            if len(row['conformal_set_parsed']) == 0:
                return False

            covered = true_label_idx in row['conformal_set_parsed']
            return covered
        except Exception as e:
            debug_print(f"Error in is_covered for row {row.get('index', 'unknown')}: {e}")
            return False

    df['covered'] = df.apply(is_covered, axis=1).astype(int)
    #debug_print("Coverage computed. Sample values", df['covered'].head().tolist())
    #debug_print("Coverage rate so far", f"{df['covered'].mean()*100:.2f}%")

    # Compute adoption (whether LLM prediction is in conformal set)
    def is_adopted(row):
        try:
            if row['method'] == 'control':
                return False  # Control group has no prediction set

            # Use cleaned result
            result_to_check = row['result_clean']
            if result_to_check == 'INVALID':
                return False

            # Check if conformal set is empty
            if len(row['conformal_set_parsed']) == 0:
                return False

            llm_pred_idx = reverse_label_map.get(result_to_check)
            if llm_pred_idx is None:
                # Try to find partial match
                for label, idx in reverse_label_map.items():
                    if result_to_check.startswith(label):
                        llm_pred_idx = idx
                        break

                if llm_pred_idx is None:
                    debug_print(f"LLM prediction not found in reverse_label_map: {result_to_check}")
                    return False

            adopted = llm_pred_idx in row['conformal_set_parsed']
            return adopted
        except Exception as e:
            debug_print(f"Error in is_adopted for row {row.get('index', 'unknown')}: {e}")
            return False

    df['adopted'] = df.apply(is_adopted, axis=1).astype(int)
    #debug_print("Adoption computed. Sample values", df['adopted'].head().tolist())
    #debug_print("Adoption rate so far", f"{df['adopted'].mean()*100:.2f}%")

    # Compute singleton (whether conformal set has exactly one element)
    df['singleton'] = (df['set_size'] == 1).astype(int)
    #debug_print("Singleton computed. Sample values", df['singleton'].head().tolist())

    # Compute accuracy (use cleaned result)
    df['accuracy'] = (df['result_clean'] == df['label_text']).astype(int)
    #debug_print("Accuracy computed. Sample values", df['accuracy'].head().tolist())
    #debug_print("Accuracy rate so far", f"{df['accuracy'].mean()*100:.2f}%")

    return df

def compute_group_statistics(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Compute statistics grouped by method and group.
    """
    #debug_print("Starting compute_group_statistics")
    #debug_print("Available methods", df['method'].unique().tolist())
    #debug_print("Available groups", df['group_text'].unique().tolist())

    def get_adoption_stats(group):
        if len(group) == 0:
            return pd.Series([0.0, 0.0], index=['Adoption', 'Adoption_std'])
        mean_adopt = group['adopted'].mean() * 100.0
        std_adopt = group['adopted'].std() * 100.0 if len(group) > 1 else 0.0
        return pd.Series([mean_adopt, std_adopt], index=['Adoption', 'Adoption_std'])

    def get_coverage_stats(group):
        if len(group) == 0:
            return pd.Series([0.0, 0.0], index=['Coverage', 'Coverage_std'])
        mean_cover = group['covered'].mean() * 100.0
        std_cover = group['covered'].std() * 100.0 if len(group) > 1 else 0.0
        return pd.Series([mean_cover, std_cover], index=['Coverage', 'Coverage_std'])

    def get_size_stats(group):
        if len(group) == 0:
            return pd.Series([0.0, 0.0], index=['Set size', 'Set size_std'])
        mean_size = group['set_size'].mean()
        std_size = group['set_size'].std() if len(group) > 1 else 0.0
        return pd.Series([mean_size, std_size], index=['Set size', 'Set size_std'])

    def get_singleton_stats(group):
        if len(group) == 0:
            return pd.Series([0, 0.0], index=['Singleton frequency', 'Singleton_std'])
        singleton_freq = group['singleton'].mean() * 100.0
        std_singleton = group['singleton'].std() * 100.0 if len(group) > 1 else 0.0
        return pd.Series([singleton_freq, std_singleton], index=['Singleton frequency', 'Singleton_std'])

    def get_accuracy_stats(group):
        if len(group) == 0:
            return pd.Series([0.0, 0.0], index=['Accuracy', 'Accuracy_std'])
        mean_acc = group['accuracy'].mean() * 100.0
        std_acc = group['accuracy'].std() * 100.0 if len(group) > 1 else 0.0
        return pd.Series([mean_acc, std_acc], index=['Accuracy', 'Accuracy_std'])

    # Compute grouped statistics
    group_stats = {}

    try:
        # Adoption rate
        #debug_print("Computing adoption statistics...")
        adoption_df = df.groupby(['method', 'group_text']).apply(get_adoption_stats).reset_index()
        adoption_df['Treatment'] = adoption_df['method'].map(TREATMENT_NAMES)
        group_stats['adoption'] = adoption_df
        #debug_print("Adoption statistics computed", adoption_df.shape)

        # Coverage rate
        #debug_print("Computing coverage statistics...")
        coverage_df = df.groupby(['method', 'group_text']).apply(get_coverage_stats).reset_index()
        coverage_df['Treatment'] = coverage_df['method'].map(TREATMENT_NAMES)
        group_stats['coverage'] = coverage_df
        #debug_print("Coverage statistics computed", coverage_df.shape)

        # Average set size
        #debug_print("Computing set size statistics...")
        size_df = df.groupby(['method', 'group_text']).apply(get_size_stats).reset_index()
        size_df['Treatment'] = size_df['method'].map(TREATMENT_NAMES)
        group_stats['size'] = size_df
        #debug_print("Set size statistics computed", size_df.shape)

        # Singleton frequency
        #debug_print("Computing singleton statistics...")
        singleton_df = df.groupby(['method', 'group_text']).apply(get_singleton_stats).reset_index()
        singleton_df['Treatment'] = singleton_df['method'].map(TREATMENT_NAMES)
        group_stats['singleton'] = singleton_df
        #debug_print("Singleton statistics computed", singleton_df.shape)

        # Accuracy
        #debug_print("Computing accuracy statistics...")
        accuracy_df = df.groupby(['method', 'group_text']).apply(get_accuracy_stats).reset_index()
        accuracy_df['Treatment'] = accuracy_df['method'].map(TREATMENT_NAMES)
        group_stats['accuracy'] = accuracy_df
        #debug_print("Accuracy statistics computed", accuracy_df.shape)

    except Exception as e:
        debug_print(f"Error in compute_group_statistics: {e}")
        import traceback
        traceback.print_exc()
        return {}

    return group_stats

def label_distribution(df: pd.DataFrame, label_map: Dict[int, str]) -> pd.DataFrame:
    """
    Compute label distribution for the test set
    """
    labels_by_index = [label_map[i] for i in sorted(label_map)]
    s = df["label_text"].astype(str).str.strip()
    counts = s.value_counts()
    counts = counts.reindex(labels_by_index, fill_value=0)

    freq_df = pd.DataFrame({
        "label_index": range(len(labels_by_index)),
        "label": labels_by_index,
        "count": counts.to_numpy()
    })

    return freq_df

def label_dist_by_group(df: pd.DataFrame, label_map: Dict[int, str]) -> pd.DataFrame:
    """
    Compute label distribution by group for the test set
    """
    labels_by_index = [label_map[i] for i in sorted(label_map)]
    groups = sorted(df["group_text"].unique())

    records = []
    for group in groups:
        group_df = df[df["group_text"] == group]
        s = group_df["label_text"].astype(str).str.strip()
        counts = s.value_counts()
        counts = counts.reindex(labels_by_index, fill_value=0)

        for label, count in zip(labels_by_index, counts):
            records.append({
                "group_text": group,
                "label": label,
                "count": count
            })

    freq_df = pd.DataFrame.from_records(records)
    return freq_df

def create_label_distribution_plot(df: pd.DataFrame, label_map: Dict[int, str],
                                   output_dir: str, dataset_name: str = "Dataset") -> Optional[str]:

    try:
        # Get label distribution
        freq_df = label_distribution(df, label_map)

        # Map dataset name using the official naming convention
        official_dataset_name = DATASET_NAMES.get(dataset_name, dataset_name)

        # Create the plot
        labels_by_index = freq_df["label"].tolist()
        plt.figure(figsize=(max(8, 0.8*len(labels_by_index)), 5))
        plt.bar(freq_df["label"], freq_df["count"], edgecolor="black", color='#3B528B', alpha=0.7)
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Label")
        plt.ylabel("Frequency")
        plt.title(f"Label frequency for {official_dataset_name}")
        plt.tight_layout()

        # Save the plot
        output_path = Path(output_dir) / "label_distribution.pdf"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        debug_print(f"Label distribution plot saved to: {output_path}")
        return str(output_path)

    except Exception as e:
        debug_print(f"Error creating label distribution plot: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_label_distribution_by_group_plot(df: pd.DataFrame, label_map: Dict[int, str],
                                            output_dir: str, dataset_name: str = "Dataset") -> Optional[str]:

    try:
        # Get label distribution by group
        freq_df = label_dist_by_group(df, label_map)

        # Map dataset name using the official naming convention
        official_dataset_name = DATASET_NAMES.get(dataset_name, dataset_name)

        # Convert to wide format for plotting
        wide = (freq_df.pivot(index="label", columns="group_text", values="count")
                .fillna(0).astype(int))

        labels = wide.index.to_list()
        groups = wide.columns.to_list()
        ng = len(groups)
        x = np.arange(len(labels), dtype=float)

        # Set up gaps between labels, no gap within label clusters
        cluster_width = 0.8
        bar_width = cluster_width / ng
        # Offsets so each cluster is centered at its label tick
        offsets = (np.arange(ng) - (ng - 1) / 2.0) * bar_width

        # Create the plot - widen figure to make room for outside legend
        fig, ax = plt.subplots(figsize=(max(8, 0.8*len(labels)) + 2, 5))

        # Use consistent colors from the COLORS palette
        colors = COLORS[:len(groups)] if len(groups) <= len(COLORS) else plt.cm.Set3(np.linspace(0, 1, len(groups)))

        # Plot bars for each group
        for j, g in enumerate(groups):
            color = colors[j] if isinstance(colors, list) else colors[j]
            ax.bar(x + offsets[j], wide[g].to_numpy(),
                   width=bar_width, label=g, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)

        # Customize the plot
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("Label")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Label frequency by group for {official_dataset_name}")

        # Put legend outside, to the right
        ax.legend(title="Group", loc="center left", bbox_to_anchor=(1.02, 0.5))

        ax.margins(x=0.02)            # a touch of horizontal padding
        fig.tight_layout()

        # Save the plot (include legend fully in the output)
        output_path = Path(output_dir) / "label_distribution_by_group.pdf"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        debug_print(f"Label distribution by group plot saved to: {output_path}")
        return str(output_path)

    except Exception as e:
        debug_print(f"Error creating label distribution by group plot: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_metric_plot(df: pd.DataFrame, metric: str, y_column: str,
                       title: str, ylabel: str, ylim: Tuple[float, float] = None,
                       save_path: Optional[str] = None,
                       include_control: bool = False) -> plt.Figure:
    """
    Create a bar plot for a specific metric with disparity annotations and arrows.
    """
    try:
        #debug_print(f"Creating plot for metric: {metric}")
        #debug_print(f"Input DataFrame shape: {df.shape}")

        # Exclude control group for plotting
        if include_control:
            plot_df = df.copy()
        else:
            plot_df = df[df['Treatment'] != 'Control'].copy()

        if plot_df.empty:
            debug_print(f"No data to plot for {metric} (all control group)")
            return None

        # Get unique groups for consistent ordering
        unique_groups = sorted(plot_df['group_text'].unique())
        #debug_print(f"Unique groups: {unique_groups}")

        desired_order = [
            "Marginal",
            "Mondrian",
            "Label-Clustered",
            "Group-Clustered",
            "Backward",
        ]

        if include_control:
            desired_order = ["Control"] + desired_order

        #if include_control:
            #treatment_order = ['Control'] + [
                #v for k, v in TREATMENT_NAMES.items()
                #if k != 'control' and v in plot_df['Treatment'].unique()
            #]
        #else:
            #treatment_order = [
                #v for k, v in TREATMENT_NAMES.items()
                #if k != 'control' and v in plot_df['Treatment'].unique()
            #]

        treatment_order = [t for t in desired_order if t in plot_df["Treatment"].unique()]

        plot_df['Treatment'] = pd.Categorical(plot_df['Treatment'], categories=treatment_order, ordered=True)
        plot_df = plot_df.sort_values(by='Treatment')

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        sns.barplot(data=plot_df, x="Treatment", y=y_column,
                    hue="group_text", hue_order=unique_groups,
                    palette=COLORS[:len(unique_groups)], ax=ax)

        plt.xticks(rotation=45, ha='right')

        # Set y-axis limits if provided
        if ylim:
            ax.set_ylim(ylim)

        # Add disparity annotations and arrows for each treatment
        for i, treatment in enumerate(treatment_order):
            treatment_data = plot_df[plot_df['Treatment'] == treatment]

            if len(unique_groups) >= 2 and len(treatment_data) >= 2:
                # Calculate disparity between groups
                group_values = {}
                for group in unique_groups:
                    group_data = treatment_data[treatment_data['group_text'] == group]
                    if not group_data.empty:
                        group_values[group] = group_data[y_column].iloc[0]

                if len(group_values) >= 2:
                    values = list(group_values.values())
                    max_val = max(values)
                    min_val = min(values)
                    disparate = max_val - min_val

                    # Position for annotations
                    x_pos = i

                    # Add disparity text annotation
                    if max_val + min_val > 0:
                        if max_val > min_val + 0.5:
                            ax.annotate(f"{disparate:.2f}",
                                       xy=(x_pos, 0.5 * (min_val + max_val)),
                                       xytext=(x_pos, 0.5 * (min_val + max_val)),
                                       ha='center', va='center',
                                       fontsize=14, color='#21918C', weight='bold')

                            # Add arrow showing the gap
                            x_pos_arrow = x_pos - 0.08 if unique_groups[0] in group_values and group_values[unique_groups[0]] == min_val else x_pos + 0.08

                            # Calculate arrow dimensions to fit within disparity gap
                            head_length = disparate * 0.1
                            line_offset = max_val * 0.0001 if max_val > 1 else 0.0001

                            # Adjust arrow length so head touches the dashed line
                            arrow_length = disparate - line_offset - head_length

                            ax.arrow(x_pos_arrow, min_val, 0, arrow_length,
                                    head_width=0.06, head_length=head_length,
                                    color='#21918C', alpha=0.8)

                            # Add horizontal dashed line at the top of the arrow
                            ax.hlines(max_val - line_offset, x_pos - 0.4, x_pos + 0.4,
                                     linestyles='--', color='#21918C')

        # Customize plot
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        # Move legend to the right and change title from "group_text" to "Group"
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, title='Group',
                 bbox_to_anchor=(1.05, 1), loc='upper left')

        # Adjust layout to prevent legend cutoff
        plt.tight_layout()

        # Save plot if path provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            debug_print(f"Plot saved to: {save_path}")

        return fig

    except Exception as e:
        debug_print(f"Error creating plot for {metric}: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_all_plots(group_stats: Dict[str, pd.DataFrame],
                       output_dir: str,
                       dataset_name: str = "Dataset",
                       df: Optional[pd.DataFrame] = None,
                       label_map: Optional[Dict[int, str]] = None) -> Dict[str, str]:
    """
    Generate all metric plots and save them.
    """
    #debug_print("Starting generate_all_plots")
    #debug_print(f"Output directory: {output_dir}")
    #debug_print(f"Available metrics: {list(group_stats.keys())}")
    #debug_print(f"Input dataset name: {dataset_name}")

    # Map dataset name using the official naming convention
    official_dataset_name = DATASET_NAMES.get(dataset_name, dataset_name)
    #debug_print(f"Mapped dataset name: {official_dataset_name}")

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    saved_plots = {}

    if df is not None and label_map is not None:
        label_plot_path = create_label_distribution_plot(df, label_map, str(output_dir), dataset_name)
        if label_plot_path:
            saved_plots['label_distribution'] = label_plot_path

        label_by_group_plot_path = create_label_distribution_by_group_plot(df, label_map, str(output_dir), dataset_name)
        if label_by_group_plot_path:
            saved_plots['label_distribution_by_group'] = label_by_group_plot_path

    # Plot configurations
    plot_configs = [
        {
            'metric': 'adoption',
            'y_column': 'Adoption',
            'title': f'Adoption by group and treatment for {official_dataset_name}',
            'ylabel': 'Adoption (%)',
            'ylim': (0, 100)
        },
        {
            'metric': 'coverage',
            'y_column': 'Coverage',
            'title': f'Coverage by group and treatment for {official_dataset_name}',
            'ylabel': 'Coverage (%)',
            'ylim': (0, 100)
        },
        {
            'metric': 'size',
            'y_column': 'Set size',
            'title': f'Average set size by group and treatment for {official_dataset_name}',
            'ylabel': 'Average set size',
            'ylim': None
        },
        {
            'metric': 'singleton',
            'y_column': 'Singleton frequency',
            'title': f'Singleton frequency by group and treatment for {official_dataset_name}',
            'ylabel': 'Singleton frequency (%)',
            'ylim': (0, 100)
        },
        {
            'metric': 'accuracy',
            'y_column': 'Accuracy',
            'title': f'Accuracy by group and treatment for {official_dataset_name}',
            'ylabel': 'Accuracy (%)',
            'ylim': (0, 100)
        }
    ]

    for config in plot_configs:
        metric = config['metric']
        if metric in group_stats:
            save_path = output_dir / f"{metric}_by_group_treatment.pdf"

            fig = create_metric_plot(
                df=group_stats[metric],
                metric=metric,
                y_column=config['y_column'],
                title=config['title'],
                ylabel=config['ylabel'],
                ylim=config['ylim'],
                save_path=str(save_path),
                include_control=(metric == "accuracy")
            )

            if fig is not None:
                saved_plots[metric] = str(save_path)
                plt.close(fig)
        else:
            debug_print(f"Metric {metric} not found in group_stats")

    #debug_print(f"Generated {len(saved_plots)} plots: {list(saved_plots.keys())}")
    return saved_plots

def print_statistics_summary(group_stats: Dict[str, pd.DataFrame],
                            df: pd.DataFrame) -> None:

    print("\n" + "="*60)
    print("LLM-IN-THE-LOOP STATISTICS SUMMARY")
    print("="*60)

    # Overall accuracy by method
    print("\n OVERALL ACCURACY BY METHOD:")
    print("-" * 40)
    overall_acc = df.groupby('method')['accuracy'].mean() * 100
    for method, acc in overall_acc.items():
        method_name = TREATMENT_NAMES.get(method, method)
        print(f"{method_name:>12}: {acc:6.2f}%")

    # Invalid predictions summary
    invalid_count = (df['result_clean'] == 'INVALID').sum()
    total_count = len(df)
    print(f"\n INVALID PREDICTIONS: {invalid_count}/{total_count} ({invalid_count/total_count*100:.2f}%)")

    # Statistics by group and method
    for metric_name, metric_df in group_stats.items():
        print(f"\n {metric_name.upper()} BY GROUP AND METHOD (TREATMENT)")
        print("-" * 40)

        # Get the main column name
        if metric_name == 'adoption':
            col_name = 'Adoption'
            unit = '%'
        elif metric_name == 'coverage':
            col_name = 'Coverage'
            unit = '%'
        elif metric_name == 'accuracy':
            col_name = 'Accuracy'
            unit = '%'
        elif metric_name == 'size':
            col_name = 'Set size'
            unit = ''
        elif metric_name == 'singleton':
            col_name = 'Singleton frequency'
            unit = '%'
        else:
            continue

        try:
            # Create pivot table for nice formatting
            pivot_df = metric_df.pivot(index='method', columns='group_text', values=col_name)

            for method in pivot_df.index:
                method_name = TREATMENT_NAMES.get(method, method)
                print(f"\n{method_name}:")
                for group in pivot_df.columns:
                    value = pivot_df.loc[method, group]
                    if pd.notna(value):
                        if unit == '%':
                            print(f"  {group:>10}: {value:6.2f}{unit}")
                        else:
                            print(f"  {group:>10}: {value:6.2f}{unit}")
        except Exception as e:
            debug_print(f"Error printing {metric_name} statistics: {e}")

    print("\n" + "="*60)

def fit_gee_llm(
    df: pd.DataFrame,
    outcome: str = "accuracy",
    treatment: str = "method",
    group: str = "group_text",
    item: str = "index",
    baseline_treatment: str = "control",
    difficulty: str = "difficulty"
) -> Tuple[Any, pd.DataFrame]:
    """
    Fit GEE clustered by prompt_idx (LLM evaluated on all tasks with five methods, including ``control``).

    """
    df = df.copy()

    # Ensure categorical encodings with control as baseline
    treatment_levels = list(df[treatment].unique())
    if baseline_treatment not in treatment_levels:
        raise ValueError(f"baseline_treatment='{baseline_treatment}' not found.")

    # Put baseline first in category order
    treatment_levels = [baseline_treatment] + [t for t in treatment_levels if t != baseline_treatment]
    df[treatment] = pd.Categorical(df[treatment], categories=treatment_levels, ordered=False)

    # Set up group categories (pandas will use first level as baseline)
    df[group] = pd.Categorical(df[group])

    # Build formula: outcome ~ treatment * group + difficulty
    formula = f"{outcome} ~ C({treatment}) * C({group}) + {difficulty}"

    debug_print(f"GEE Formula: {formula}")
    debug_print(f"Treatment levels: {treatment_levels}")
    debug_print(f"Group levels: {list(df[group].cat.categories)}")

    # Fit GEE model
    gee = smf.gee(
        formula=formula,
        groups=item,
        data=df,
        family=sm.families.Binomial(),
        cov_struct=Exchangeable()
    )

    res = gee.fit(cov_type='robust')
    return res, df

def fit_gee_llm_fairness(
    df: pd.DataFrame,
    outcome: str = "accuracy",
    treatment: str = "method",
    group: str = "group_text",
    item: str = "index",
    baseline_treatment: str = "control",
    difficulty: str = "difficulty",
    adopted: str = "adopted"
) -> Tuple[Any, pd.DataFrame]:
    """
    Fit GEE clustered by prompt_idx (LLM evaluated on all tasks with five methods, including ``control``).

    """
    df = df.copy()

    # Ensure categorical encodings with control as baseline
    treatment_levels = list(df[treatment].unique())
    if baseline_treatment not in treatment_levels:
        raise ValueError(f"baseline_treatment='{baseline_treatment}' not found.")

    # Put baseline first in category order
    treatment_levels = [baseline_treatment] + [t for t in treatment_levels if t != baseline_treatment]
    df[treatment] = pd.Categorical(df[treatment], categories=treatment_levels, ordered=False)

    # Set up group categories (pandas will use first level as baseline)
    df[group] = pd.Categorical(df[group])

    # Build formula: outcome ~ treatment * group + adopted + difficulty
    formula = f"{outcome} ~ C({treatment}) * C({group}) + {adopted} + {difficulty}"

    debug_print(f"GEE fairness Formula: {formula}")
    debug_print(f"Treatment levels: {treatment_levels}")
    debug_print(f"Group levels: {list(df[group].cat.categories)}")

    # Fit GEE model
    gee = smf.gee(
        formula=formula,
        groups=item,
        data=df,
        family=sm.families.Binomial(),
        cov_struct=Exchangeable()
    )

    res = gee.fit(cov_type='robust')
    return res, df

def _coef_lookup(params: pd.Series, a: str, b: Optional[str] = None) -> float:
    """
    Find a coefficient by name. For interactions, accept either order (a:b or b:a).

    """
    if b is None:
        return float(params.get(a, 0.0))

    # interaction may appear as "a:b" or "b:a"
    if a + ":" + b in params.index:
        return float(params[a + ":" + b])
    if b + ":" + a in params.index:
        return float(params[b + ":" + a])
    return 0.0

non_base_trt_mapping = {
    'ConformalMethod.MARGINAL': 'marginal',
    'ConformalMethod.CONDITIONAL': 'conditional',
    'ConformalMethod.BACKWARD': 'backward',
    'ConformalMethod.CLUSTERED_LABEL': 'clustered_label',
    'ConformalMethod.CLUSTERED_GROUP': 'clustered_group'
}

def compute_or_and_maxror(
    df: pd.DataFrame,
    res,
    treatment: str = "method",
    group: str = "group_text"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    From a fitted GEE, compute:
    - OR table: rows=group levels, columns=<treatment> (!= baseline)
    - maxROR per treatment: max_g OR_{t|g} / min_g OR_{t|g}

    """
    params = res.params

    # Extract treatment and group levels from the fitted model parameters
    debug_print("Available parameter names:", list(params.index))

    # Extract treatment levels from parameter names (main effects only)
    trt_params = [p for p in params.index if f'C({treatment})[T.' in p and ':' not in p]
    non_base_trts = []
    for p in trt_params:
        # Extract treatment name from parameter like "C(method)[T.avgk]"
        start = p.find('[T.') + 3
        end = p.find(']', start)
        if start > 2 and end > start:
            trt_name = p[start:end]
            non_base_trts.append(trt_name)

    # Extract group levels from parameter names (main effects only)
    grp_params = [p for p in params.index if f'C({group})[T.' in p and ':' not in p]

    # Get baseline group from the fitted model data
    try:
        # Access the categorical data from the fitted model
        group_categories = res.model.data.frame[group].cat.categories
        baseline_grp = group_categories[0]  # First category is baseline
        grp_levels = [baseline_grp]  # Start with baseline

        # Add non-baseline groups from parameters
        for p in grp_params:
            start = p.find('[T.') + 3
            end = p.find(']', start)
            if start > 2 and end > start:
                grp_name = p[start:end]
                if grp_name not in grp_levels:
                    grp_levels.append(grp_name)

    except Exception as e:
        debug_print(f"Could not extract group levels from model: {e}")
        # Fallback: reconstruct from parameters + common baseline
        grp_levels = []
        for p in grp_params:
            start = p.find('[T.') + 3
            end = p.find(']', start)
            if start > 2 and end > start:
                grp_name = p[start:end]
                if grp_name not in grp_levels:
                    grp_levels.append(grp_name)

    debug_print(f"Treatment levels (non-baseline): {non_base_trts}")
    debug_print(f"Group levels: {grp_levels}")

    if not non_base_trts:
        debug_print("No treatment parameters found - cannot compute OR")
        return pd.DataFrame(), pd.Series(dtype=float, name="maxROR")

    if len(grp_levels) < 2:
        debug_print("Less than 2 group levels found - cannot compute ROR")
        return pd.DataFrame(), pd.Series(dtype=float, name="maxROR")

    base_grp = grp_levels[0]

    # Helper to build coefficient names
    def trt_term(t):
        return f"C({treatment})[T.{t}]"

    def grp_term(g):
        return f"C({group})[T.{g}]"

    # Build OR table
    or_data: Dict[str, List[float]] = {}

    for t in non_base_trts:
        # Get main treatment effect (log-OR for baseline group)
        beta_t = _coef_lookup(params, trt_term(t))
        ors_per_group = []

        for g in grp_levels:
            if g == base_grp:
                # For baseline group, OR is just exp(main treatment effect)
                log_or = beta_t
            else:
                # For non-baseline groups, add interaction effect
                beta_int = _coef_lookup(params, trt_term(t), grp_term(g))
                log_or = beta_t + beta_int

            ors_per_group.append(np.exp(log_or))

        # Map treatment names to display names
        treatment_display_name = TREATMENT_NAMES.get(t, t)
        or_data[treatment_display_name] = ors_per_group

    or_df = pd.DataFrame(or_data, index=grp_levels)

    # Compute maxROR per treatment: max OR / min OR across groups
    maxror_data = {}
    for t in non_base_trts:
        treatment_display_name = TREATMENT_NAMES.get(t, t)
        col = treatment_display_name
        if col in or_df.columns:
            vals = or_df[col].values.astype(float)
            if len(vals) > 0 and np.min(vals) > 0:  # Avoid division by zero
                maxror_data[treatment_display_name] = float(np.max(vals) / np.min(vals))

    maxror = pd.Series(maxror_data, name="maxROR")

    debug_print(f"Analysis DataFrame shape (check for original GEE analysis): {df.shape}")

    disparity_data: Dict[str, List[float]] = {}

    # Compute probabilities for control
    p_hat_control_a = {}
    for g in grp_levels:
        if g == base_grp:
            sub_df = df[(df[treatment] == "control") & (df[group] == base_grp)].copy()
            task_diff = sub_df["difficulty"].to_numpy(dtype=float)
            eta = float(params['Intercept']) + float(params['difficulty']) * task_diff
        else:
            # For non-baseline groups
            sub_df = df[(df[treatment] == "control") & (df[group] == g)].copy()
            task_diff = sub_df["difficulty"].to_numpy(dtype=float)
            task_group = _coef_lookup(params, grp_term(g))
            eta = float(params['Intercept']) + task_group + float(params['difficulty']) * task_diff

        p_hat_control_a[g] = expit(eta).mean()
        #print(f"Original-model-based probability for control and group {g} is {p_hat_control_a[g]:.6f}")

    for t in non_base_trts:

        beta_t = _coef_lookup(params, trt_term(t))
        disparity_per_group = []

        for g in grp_levels:
            if g == base_grp:
                sub_df = df[(df[treatment] == non_base_trt_mapping.get(t, t)) & (df[group] == base_grp)].copy()
                task_diff = sub_df["difficulty"].to_numpy(dtype=float)
                eta = float(params['Intercept']) + beta_t + float(params['difficulty']) * task_diff
                p_hat = expit(eta).mean()
                #print(f"Original-model-based probability for treatment {t} and group {g} is {p_hat:.6f}")
                disp = p_hat - p_hat_control_a[g]
            else:
                sub_df = df[(df[treatment] == non_base_trt_mapping.get(t, t)) & (df[group] == g)].copy()
                task_diff = sub_df["difficulty"].to_numpy(dtype=float)
                eta = float(params['Intercept']) + beta_t + _coef_lookup(params, grp_term(g)) + _coef_lookup(params, trt_term(t), grp_term(g)) + float(params['difficulty']) * task_diff
                p_hat = expit(eta).mean()
                #print(f"Original-model-based probability for treatment {t} and group {g} is {p_hat:.6f}")
                disp = p_hat - p_hat_control_a[g]

            disparity_per_group.append(disp)

        # Map treatment names to display names
        treatment_display_name = TREATMENT_NAMES.get(t, t)
        disparity_data[treatment_display_name] = disparity_per_group

    disparity_df = pd.DataFrame(disparity_data, index=grp_levels)

    maxdisp_data = {}
    for t in non_base_trts:
        treatment_display_name = TREATMENT_NAMES.get(t, t)
        col = treatment_display_name
        if col in disparity_df.columns:
            vals = disparity_df[col].values.astype(float)
            if len(vals) > 0:
                maxdisp_data[treatment_display_name] = float(np.max(vals) - np.min(vals))

    maxdisp = pd.Series(maxdisp_data, name="maxDisparity")

    return or_df, maxror, disparity_df, maxdisp

def compute_or_and_maxror_marginalized(
    df: pd.DataFrame,
    res,
    treatment: str = "method",
    group: str = "group_text"
) -> Tuple[pd.DataFrame, pd.Series]:

    params = res.params

    # Extract treatment and group levels from the fitted model parameters
    # Group levels
    grp_params = [p for p in params.index if f'C({group})[T.' in p and ':' not in p]

    # Get baseline group from the fitted model data
    try:
        # Access the categorical data from the fitted model
        group_categories = res.model.data.frame[group].cat.categories
        baseline_grp = group_categories[0]  # First category is baseline
        grp_levels = [baseline_grp]  # Start with baseline

        # Add non-baseline groups from parameters
        for p in grp_params:
            start = p.find('[T.') + 3
            end = p.find(']', start)
            if start > 2 and end > start:
                grp_name = p[start:end]
                if grp_name not in grp_levels:
                    grp_levels.append(grp_name)

    except Exception as e:
        debug_print(f"Could not extract group levels from model: {e}")
        # Fallback: reconstruct from parameters + common baseline
        grp_levels = []
        for p in grp_params:
            start = p.find('[T.') + 3
            end = p.find(']', start)
            if start > 2 and end > start:
                grp_name = p[start:end]
                if grp_name not in grp_levels:
                    grp_levels.append(grp_name)

    debug_print(f"Group levels: {grp_levels}")
    base_grp = grp_levels[0] #baseline group

    #Treatment levels
    trt_params = [p for p in params.index if f'C({treatment})[T.' in p and ':' not in p]
    non_base_trts = []
    for p in trt_params:
        # Extract treatment name from parameter like "C(method)[T.ConformalMethod.MARGINAL]"
        start = p.find('[T.') + 3
        end = p.find(']', start)
        if start > 2 and end > start:
            trt_name = p[start:end]
            non_base_trts.append(trt_name)

    debug_print(f"Treatment levels (non-baseline): {non_base_trts}")

    if not non_base_trts:
        debug_print("No treatment parameters found - cannot compute OR")
        return pd.DataFrame(), pd.Series(dtype=float, name="maxROR")

    if len(grp_levels) < 2:
        debug_print("Less than 2 group levels found - cannot compute ROR")
        return pd.DataFrame(), pd.Series(dtype=float, name="maxROR")

    debug_print(f"Analysis DataFrame shape (check for marginalized analysis): {df.shape}")

    def trt_term(t):
        return f"C({treatment})[T.{t}]"

    def grp_term(g):
        return f"C({group})[T.{g}]"

    # Build OR table
    or_data: Dict[str, List[float]] = {}
    disparity_data: Dict[str, List[float]] = {}

    # Compute probabilities for control
    p_hat_control_a = {}
    for g in grp_levels:
        if g == base_grp:
            sub_df = df[(df[treatment] == "control") & (df[group] == base_grp)].copy()
            task_diff = sub_df["difficulty"].to_numpy(dtype=float)
            eta = float(params['Intercept']) + float(params['difficulty']) * task_diff
        else:
            # For non-baseline groups
            sub_df = df[(df[treatment] == "control") & (df[group] == g)].copy()
            task_diff = sub_df["difficulty"].to_numpy(dtype=float)
            task_group = _coef_lookup(params, grp_term(g))
            eta = float(params['Intercept']) + task_group + float(params['difficulty']) * task_diff

        p_hat_control_a[g] = expit(eta).mean()
        #print(f"Model-based probability for control and group {g} is {p_hat_control_a[g]:.6f}")

    for t in non_base_trts:

        beta_t = _coef_lookup(params, trt_term(t))
        ors_per_group = []
        disparity_per_group = []

        for g in grp_levels:
            if g == base_grp:
                sub_df = df[(df[treatment] == non_base_trt_mapping.get(t, t)) & (df[group] == base_grp)].copy()
                task_diff = sub_df["difficulty"].to_numpy(dtype=float)
                task_adopted = sub_df["adopted"].to_numpy(dtype=float)
                eta = float(params['Intercept']) + beta_t + float(params['difficulty']) * task_diff + float(params['adopted']) * task_adopted
                p_hat = expit(eta).mean()
                #print(f"Model-based probability for treatment {t} and group {g} is {p_hat:.6f}")
                orss = (p_hat / (1 - p_hat)) / (p_hat_control_a[g] / (1 - p_hat_control_a[g]))
                disp = p_hat - p_hat_control_a[g]
            else:
                sub_df = df[(df[treatment] == non_base_trt_mapping.get(t, t)) & (df[group] == g)].copy()
                task_diff = sub_df["difficulty"].to_numpy(dtype=float)
                task_adopted = sub_df["adopted"].to_numpy(dtype=float)
                eta = float(params['Intercept']) + beta_t + _coef_lookup(params, grp_term(g)) + _coef_lookup(params, trt_term(t), grp_term(g)) + float(params['difficulty']) * task_diff + float(params['adopted']) * task_adopted
                p_hat = expit(eta).mean()
                #print(f"Model-based probability for treatment {t} and group {g} is {p_hat:.6f}")
                orss = (p_hat / (1 - p_hat)) / (p_hat_control_a[g] / (1 - p_hat_control_a[g]))
                disp = p_hat - p_hat_control_a[g]

            ors_per_group.append(orss)
            disparity_per_group.append(disp)

        # Map treatment names to display names
        treatment_display_name = TREATMENT_NAMES.get(t, t)
        or_data[treatment_display_name] = ors_per_group
        disparity_data[treatment_display_name] = disparity_per_group

    or_df = pd.DataFrame(or_data, index=grp_levels)
    disparity_df = pd.DataFrame(disparity_data, index=grp_levels)

    # Compute maxROR per treatment: max OR / min OR across groups
    maxror_data = {}
    for t in non_base_trts:
        treatment_display_name = TREATMENT_NAMES.get(t, t)
        col = treatment_display_name
        if col in or_df.columns:
            vals = or_df[col].values.astype(float)
            if len(vals) > 0 and np.min(vals) > 0:  # Avoid division by zero
                maxror_data[treatment_display_name] = float(np.max(vals) / np.min(vals))

    maxror = pd.Series(maxror_data, name="maxROR")

    maxdisp_data = {}
    for t in non_base_trts:
        treatment_display_name = TREATMENT_NAMES.get(t, t)
        col = treatment_display_name
        if col in disparity_df.columns:
            vals = disparity_df[col].values.astype(float)
            if len(vals) > 0:
                maxdisp_data[treatment_display_name] = float(np.max(vals) - np.min(vals))

    maxdisp = pd.Series(maxdisp_data, name="maxDisparity")

    return or_df, maxror, disparity_df, maxdisp

def compute_gee_analysis(df: pd.DataFrame) -> Dict[str, Any]:

    try:
        # Check if we have the required columns
        required_columns = ['accuracy', 'method', 'group_text', 'index', 'difficulty', 'adopted']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            debug_print(f"Missing required columns for GEE analysis: {missing_columns}")
            return {}

        # Filter out rows with missing data
        analysis_df = df.dropna(subset=required_columns).copy()
        debug_print(f"Analysis DataFrame shape after removing NAs: {analysis_df.shape}")

        if analysis_df.empty:
            debug_print("No valid data for GEE analysis")
            return {}

        # Check if we have control group
        if 'control' not in analysis_df['method'].unique():
            debug_print("No control group found - cannot compute ORs")
            return {}

        # Check if we have multiple groups
        unique_groups = analysis_df['group_text'].nunique()
        if unique_groups < 2:
            debug_print(f"Only {unique_groups} group(s) found - need at least 2 for ROR analysis")
            return {}

        # Fit GEE model
        res, processed_df = fit_gee_llm(
            df=analysis_df,
            outcome="accuracy",
            treatment="method",
            group="group_text",
            item="index",
            baseline_treatment="control",
            difficulty="difficulty"
        )

        res_fair, processed_df_fair = fit_gee_llm_fairness(
            df=analysis_df,
            outcome="accuracy",
            treatment="method",
            group="group_text",
            item="index",
            baseline_treatment="control",
            difficulty="difficulty",
            adopted="adopted"
        )

        debug_print("GEE model fitted successfully")

        # Compute OR table and maxROR
        or_table, maxror, disparity_ori, maxdisp_ori = compute_or_and_maxror(
            df=analysis_df,
            res=res,
            treatment="method",
            group="group_text"
        )

        or_table_marginalized, maxror_marginalized, disparity_marginalized, maxdisp_marginalized = compute_or_and_maxror_marginalized(
            df=analysis_df,
            res=res_fair,
            treatment="method",
            group="group_text"
        )

        debug_print("OR and maxROR computed successfully")

        return {
            'or_table': or_table,
            'maxror': maxror,
            'or_table_marginalized': or_table_marginalized,
            'maxror_marginalized': maxror_marginalized,
            'gee_model': res,
            'gee_model_fair': res_fair,
            'processed_data': processed_df,
            'processed_data_fair': processed_df_fair,
            'disparity_original': disparity_ori,
            'maxdisp_original': maxdisp_ori,
            'disparity_marginalized': disparity_marginalized,
            'maxdisp_marginalized': maxdisp_marginalized
        }

    except Exception as e:
        debug_print(f"Error in GEE analysis: {e}")
        import traceback
        traceback.print_exc()
        return {}

def print_gee_results(gee_results: Dict[str, Any]) -> None:
    
    if not gee_results:
        print("\n WARNING: GEE ANALYSIS FAILED")
        return

    print("\n" + "="*80)
    print("GEE-BASED ODDS RATIO (OR) AND RATIO OF OR (ROR) ANALYSIS")
    print("="*80)

    # Print OR table
    if 'or_table' in gee_results:
        or_table = gee_results['or_table']
        print(f"\n ODDS RATIOS (OR) BY GROUP AND TREATMENT:")
        print("   (Relative to Control Group)")
        print("-" * 60)

        # Format OR table for nice printing
        print(or_table.round(3).to_string())

        print(f"\nInterpretation: OR > 1 indicates higher odds of correct prediction")
        print(f"compared to control group. OR < 1 indicates lower odds.")

    # Print maxROR table
    if 'maxror' in gee_results:
        maxror = gee_results['maxror']
        print(f"\n MAXIMUM RATIO OF ODDS RATIOS (maxROR) BY TREATMENT:")
        print("   (Disparity measure: maxROR around 1 indicates fairness)")
        print("-" * 60)

        # Create a nice table format for maxROR
        maxror_df = pd.DataFrame({
            'Treatment': maxror.index,
            'maxROR': maxror.values.round(3)
        })
        print(maxror_df.to_string(index=False))

        print(f"\nInterpretation: maxROR close to 1.0 indicates similar treatment")
        print(f"effects across groups. Higher values indicate more disparity.")

    if 'disparity_original' in gee_results:
        disparity_table = gee_results['disparity_original']
        print(f"\n IMPROVEMENT BY GROUP AND TREATMENT:")
        print("   (Relative to Control Group)")
        print("-" * 60)

        # Format OR table for nice printing
        print(disparity_table.round(3).to_string())

    if 'maxdisp_original' in gee_results:
        maxdisp = gee_results['maxdisp_original']
        print(f"\n MAXIMUM DISPARITY BY TREATMENT:")
        print("   (Disparity measure: maxDisparity around 0 indicates fairness)")
        print("-" * 60)

        # Create a nice table format for maxDisparity
        maxdisp_df = pd.DataFrame({
            'Treatment': maxdisp.index,
            'maxDisparity': maxdisp.values.round(3)
        })
        print(maxdisp_df.to_string(index=False))

        print(f"\n Interpretation: maxDisparity close to 0.0 indicates similar treatment improvement (relative to control)")
        print(f"in model-based probability of correctness across groups.")

    # Print OR table for assessing fairness (marginalized, with adoption adjustment)
    if 'or_table_marginalized' in gee_results:
        or_table = gee_results['or_table_marginalized']
        print(f"\n ODDS RATIOS (OR) BY GROUP AND TREATMENT (MARGINALIZED, ADJUSTED FOR ADOPTION):")
        print("   (Relative to Control Group)")
        print("-" * 60)

        # Format OR table for nice printing
        print(or_table.round(3).to_string())

    # Print maxROR table
    if 'maxror_marginalized' in gee_results:
        maxror = gee_results['maxror_marginalized']
        print(f"\n MAXIMUM RATIO OF ODDS RATIOS (maxROR) BY TREATMENT (MARGINALIZED, ADJUSTED FOR ADOPTION):")
        print("   (Disparity measure: maxROR around 1 indicates fairness)")
        print("-" * 60)

        # Create a nice table format for maxROR
        maxror_df = pd.DataFrame({
            'Treatment': maxror.index,
            'maxROR': maxror.values.round(3)
        })
        print(maxror_df.to_string(index=False))

        print(f"\nInterpretation: maxROR close to 1.0 indicates similar treatment")
        print(f"effects across groups. Higher values indicate more disparity.")

    print("\n" + "="*80)

    if 'disparity_marginalized' in gee_results:
        disparity_table = gee_results['disparity_marginalized']
        print(f"\n IMPROVEMENT BY GROUP AND TREATMENT (MARGINALIZED, ADJUSTED FOR ADOPTION):")
        print("   (Relative to Control Group)")
        print("-" * 60)

        # Format OR table for nice printing
        print(disparity_table.round(3).to_string())

    if 'maxdisp_marginalized' in gee_results:
        maxdisp = gee_results['maxdisp_marginalized']
        print(f"\n MAXIMUM DISPARITY BY TREATMENT (MARGINALIZED, ADJUSTED FOR ADOPTION):")
        print("   (Disparity measure: maxDisparity around 0 indicates fairness)")
        print("-" * 60)

        # Create a nice table format for maxDisparity
        maxdisp_df = pd.DataFrame({
            'Treatment': maxdisp.index,
            'maxDisparity': maxdisp.values.round(3)
        })
        print(maxdisp_df.to_string(index=False))

        print(f"\n Interpretation: maxDisparity close to 0.0 indicates similar treatment improvement (relative to control)")
        print(f"in model-based probability of correctness across groups.")

def print_gee_results_ravdess(gee_results: Dict[str, Any]) -> None:
    if not gee_results:
        print("\n WARNING: GEE ANALYSIS FAILED")
        return

    print("\n" + "="*80)
    print("GEE-BASED ODDS RATIO (OR) AND RATIO OF OR (ROR) ANALYSIS")
    print("="*80)

    # Print OR table
    if 'or_table' in gee_results:
        or_table = gee_results['or_table']
        print(f"\n ODDS RATIOS (OR) BY GROUP AND TREATMENT:")
        print("   (Relative to Control Group)")
        print("-" * 60)

        # Format OR table for nice printing
        print(or_table.round(3).to_string())

        print(f"\nInterpretation: OR > 1 indicates higher odds of correct prediction")
        print(f"compared to control group. OR < 1 indicates lower odds.")

    # Print maxROR table
    if 'maxror' in gee_results:
        maxror = gee_results['maxror']
        print(f"\n MAXIMUM RATIO OF ODDS RATIOS (maxROR) BY TREATMENT:")
        print("   (Disparity measure: maxROR around 1 indicates fairness)")
        print("-" * 60)

        # Create a nice table format for maxROR
        maxror_df = pd.DataFrame({
            'Treatment': maxror.index,
            'maxROR': maxror.values.round(3)
        })
        print(maxror_df.to_string(index=False))

        print(f"\nInterpretation: maxROR close to 1.0 indicates similar treatment")
        print(f"effects across groups. Higher values indicate more disparity.")

    if 'disparity_original' in gee_results:
        disparity_table = gee_results['disparity_original']
        print(f"\n IMPROVEMENT BY GROUP AND TREATMENT:")
        print("   (Relative to Control Group)")
        print("-" * 60)

        # Format OR table for nice printing
        print(disparity_table.round(3).to_string())

    if 'maxdisp_original' in gee_results:
        maxdisp = gee_results['maxdisp_original']
        print(f"\n MAXIMUM DISPARITY BY TREATMENT:")
        print("   (Disparity measure: maxDisparity around 0 indicates fairness)")
        print("-" * 60)

        # Create a nice table format for maxDisparity
        maxdisp_df = pd.DataFrame({
            'Treatment': maxdisp.index,
            'maxDisparity': maxdisp.values.round(3)
        })
        print(maxdisp_df.to_string(index=False))

        print(f"\n Interpretation: maxDisparity close to 0.0 indicates similar treatment improvement (relative to control)")
        print(f"in model-based probability of correctness across groups.")

def _has_missing_values(obj: Any) -> bool:
    
    if obj is None:
        return True

    if isinstance(obj, (pd.DataFrame, pd.Series)):
        if getattr(obj, "empty", False):
            return True
        return bool(obj.isna().to_numpy().any())

    try:
        s = pd.Series(obj)
        if s.empty:
            return True
        return bool(s.isna().any())
    except Exception:
        return True

def compute_comprehensive_fairness_statistics(predictions: List[Any],
                                              label_map: Dict[int, str],
                                              output_dir: str,
                                              dataset_name: str = "Dataset") -> Dict[str, Any]:
    """
    Main function to compute all LLM fairness statistics and generate plots. Called in run_llm_in_loop.py
    """
    official_dataset_name = DATASET_NAMES.get(dataset_name, dataset_name)

    debug_print("=== STARTING COMPREHENSIVE FAIRNESS STATISTICS ===")
    debug_print(f"Number of predictions: {len(predictions)}")
    debug_print(f"Dataset name: {official_dataset_name}")
    debug_print(f"Output directory: {output_dir}")

    # Convert predictions to DataFrame
    df_data = []
    for i, pred in enumerate(predictions):
        try:
            df_data.append({
                'index': pred.index,
                'method': pred.method,
                'group_text': pred.group_text,
                'label_text': pred.label_text,
                'result': pred.result,
                'conformal_set': pred.conformal_set,
                'difficulty': pred.difficulty
            })
        except Exception as e:
            debug_print(f"Error processing prediction {i}: {e}")
            continue

    df = pd.DataFrame(df_data)
    #debug_print(f"Created DataFrame with shape: {df.shape}")

    #df_marginal = df[df["method"].astype(str).str.strip().str.lower().eq("marginal")].copy()
    #df_marginal.reset_index(drop=True, inplace=True)

    # Extract rows with distinct index
    df_unique_idx = df.drop_duplicates(subset=['index'], keep='first').copy()
    df_unique_idx.reset_index(drop=True, inplace=True)

    if df.empty:
        debug_print("No predictions to analyze - DataFrame is empty")
        return {}

    if df_unique_idx.empty:
        debug_print("Error on extracting unique indices")
        return {}

    try:
        # Compute basic metrics
        df_with_metrics = compute_basic_metrics(df, label_map)
        #debug_print("Basic metrics computed successfully")

        # Compute grouped statistics
        group_stats = compute_group_statistics(df_with_metrics)
        #debug_print("Grouped statistics computed successfully")

        # Generate and save plots
        saved_plots = generate_all_plots(group_stats, output_dir, dataset_name,
                                         df=df_unique_idx, label_map=label_map)
        #debug_print("Plots generated successfully")

        # Compute GEE-based OR and ROR analysis
        gee_results = compute_gee_analysis(df_with_metrics)

        # Print summary
        print_statistics_summary(group_stats, df_with_metrics)

        if dataset_name.lower() == "ravdess":
            if _has_missing_values(gee_results['maxror_marginalized']):
                print_gee_results_ravdess(gee_results)
            else:
                print_gee_results(gee_results)
        else:
            print_gee_results(gee_results)

        # Save detailed statistics to CSV
        output_dir = Path(output_dir)
        for metric_name, metric_df in group_stats.items():
            csv_path = output_dir / f"{metric_name}_statistics.csv"
            metric_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            debug_print(f"Statistics saved to: {csv_path}")

        try:
            # Overall label distribution
            label_dist_df = label_distribution(df_unique_idx, label_map)
            label_dist_csv_path = output_dir / "label_distribution.csv"
            label_dist_df.to_csv(label_dist_csv_path, index=False, encoding='utf-8-sig')
            debug_print(f"Label distribution saved to: {label_dist_csv_path}")

            # Label distribution by group
            label_dist_by_group_df = label_dist_by_group(df_unique_idx, label_map)
            label_dist_by_group_csv_path = output_dir / "label_distribution_by_group.csv"
            label_dist_by_group_df.to_csv(label_dist_by_group_csv_path, index=False, encoding='utf-8-sig')
            debug_print(f"Label distribution by group saved to: {label_dist_by_group_csv_path}")

        except Exception as e:
            debug_print(f"Error saving label distribution CSV files: {e}")

        if gee_results and 'or_table' in gee_results and (not _has_missing_values(gee_results['or_table'])):
            or_csv_path = output_dir / "odds_ratios.csv"
            gee_results['or_table'].to_csv(or_csv_path, encoding='utf-8-sig')
            debug_print(f"OR table saved to: {or_csv_path}")

            maxror_csv_path = output_dir / "max_ror.csv"
            gee_results['maxror'].to_csv(maxror_csv_path, encoding='utf-8-sig')
            debug_print(f"maxROR saved to: {maxror_csv_path}")

        if gee_results and 'disparity_original' in gee_results and (not _has_missing_values(gee_results['disparity_original'])):
            disp_ori_csv_path = output_dir / "improvement_original.csv"
            gee_results['disparity_original'].to_csv(disp_ori_csv_path, encoding='utf-8-sig')
            debug_print(f"Improvement (original) saved to: {disp_ori_csv_path}")

            maxdisp_ori_csv_path = output_dir / "max_disparity_original.csv"
            gee_results['maxdisp_original'].to_csv(maxdisp_ori_csv_path, encoding='utf-8-sig')
            debug_print(f"maxDisparity (original) saved to: {maxdisp_ori_csv_path}")

        if gee_results and 'or_table_marginalized' in gee_results and (not _has_missing_values(gee_results['or_table_marginalized'])):
            or_marg_csv_path = output_dir / "odds_ratios_marginalized.csv"
            gee_results['or_table_marginalized'].to_csv(or_marg_csv_path, encoding='utf-8-sig')
            debug_print(f"OR table (marginalized, adjusted for adoption) saved to: {or_marg_csv_path}")

            maxror_marg_csv_path = output_dir / "max_ror_marginalized.csv"
            gee_results['maxror_marginalized'].to_csv(maxror_marg_csv_path, encoding='utf-8-sig')
            debug_print(f"maxROR (marginalized, adjusted for adoption) saved to: {maxror_marg_csv_path}")

        if gee_results and 'disparity_marginalized' in gee_results and (not _has_missing_values(gee_results['disparity_marginalized'])):
            disp_marg_csv_path = output_dir / "improvement_marginalized.csv"
            gee_results['disparity_marginalized'].to_csv(disp_marg_csv_path, encoding='utf-8-sig')
            debug_print(f"Improvement (marginalized, adjusted for adoption) saved to: {disp_marg_csv_path}")

            maxdisp_marg_csv_path = output_dir / "max_disparity_marginalized.csv"
            gee_results['maxdisp_marginalized'].to_csv(maxdisp_marg_csv_path, encoding='utf-8-sig')
            debug_print(f"maxDisparity (marginalized, adjusted for adoption) saved to: {maxdisp_marg_csv_path}")

        debug_print("=== COMPREHENSIVE FAIRNESS STATISTICS COMPLETED SUCCESSFULLY ===")

        return {
            'dataframe': df_with_metrics,
            'group_statistics': group_stats,
            'saved_plots': saved_plots,
            'gee_results': gee_results,
            'output_directory': str(output_dir)
        }

    except Exception as e:
        debug_print(f"ERROR in computing statistics: {e}")
        import traceback
        traceback.print_exc()
        return {}
