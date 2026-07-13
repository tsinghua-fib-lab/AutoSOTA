"""
LLM Solubility Evaluation Analysis Script
==========================================
Statistical analysis of LLM evaluation survey for solubility predictions
Comparing Gemini, Claude, DeepSeek, and ChatGPT

Requirements:
pip install pandas numpy scipy statsmodels scikit-learn matplotlib seaborn
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import wilcoxon, mannwhitneyu, kruskal, spearmanr, friedmanchisquare
import statsmodels.api as sm
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Set styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================================================
# PART 1: DATA LOADING AND PREPROCESSING
# ============================================================================

def load_survey_data(filepath):
    """
    Load LLM evaluation data from CSV
    Expected columns: Mixture info + (Q1 Prediction, Q1 Reasoning, Q2 Rating, Q2 Reasoning,
                      Q3 Explanation Rating, Q3 Prediction Agreement, Q3 Reasoning) × 4 LLMs
    """
    df = pd.read_csv(filepath)
    return df

def prepare_data(df):
    """
    Extract Q1, Q2, Q3, Q4 responses for each LLM and mixture
    Returns long-format DataFrame

    Q1 = LLM's own prediction (categorical)
    Q2 = Agreement with model prediction (1-5 scale)
    Q3 = Explanation quality rating (1-5 scale)
    Q4 = Agreement after reading explanation (1-5 scale, mapped from Q3 Prediction Agreement)
    """

    llms = ['Gemini', 'Claude', 'DeepSeek', 'ChatGPT']

    data_long = []

    for idx, row in df.iterrows():
        mixture_id = row['Mixture ID']
        # Process each mixture

        for llm_idx, llm in enumerate(llms):
            # Extract Q1, Q2, Q3, Q4 for this LLM
            q1_pred = row[f'{llm} Q1 Prediction']
            q2_rating = row[f'{llm} Q2 Rating']
            q3_rating = row[f'{llm} Q3 Explanation Rating']
            q4_rating = row[f'{llm} Q3 Prediction Agreement']

            data_long.append({
                'llm_id': llm_idx + 1,
                'llm_name': llm,
                'mixture': mixture_id + 1,  # 0-indexed to 1-indexed
                'Q1': q1_pred,
                'Q2': q2_rating,
                'Q3': q3_rating,
                'Q4': q4_rating
            })

    df_long = pd.DataFrame(data_long)

    # Convert Q1 categorical text responses to numeric values
    solubility_to_numeric = {
        'Highly Insoluble': 1,
        'Poorly Soluble': 2,
        'Moderately Soluble': 3,
        'Highly Soluble': 4,
        'Very Highly Soluble': 5
    }
    df_long['Q1'] = df_long['Q1'].map(solubility_to_numeric)

    # Convert Q2, Q3, Q4 to numeric
    df_long['Q2'] = pd.to_numeric(df_long['Q2'], errors='coerce')
    df_long['Q3'] = pd.to_numeric(df_long['Q3'], errors='coerce')
    df_long['Q4'] = pd.to_numeric(df_long['Q4'], errors='coerce')

    # Remove rows with NaN values
    df_long = df_long.dropna(subset=['Q1', 'Q2', 'Q3', 'Q4'])

    # Calculate differences
    df_long['Q4_minus_Q2'] = df_long['Q4'] - df_long['Q2']
    df_long['Q2_minus_Q1'] = df_long['Q2'] - df_long['Q1']
    df_long['Q4_minus_Q1'] = df_long['Q4'] - df_long['Q1']

    return df_long

# ============================================================================
# PART 2: DESCRIPTIVE STATISTICS
# ============================================================================

def descriptive_statistics(df_long):
    """
    Calculate mean, median, SD, IQR for each question
    """
    print("\n" + "="*80)
    print("DESCRIPTIVE STATISTICS")
    print("="*80)

    for question in ['Q1', 'Q2', 'Q3', 'Q4']:
        data = df_long[question]

        print(f"\n{question}:")
        print(f"  Mean ± SD: {data.mean():.2f} ± {data.std():.2f}")
        print(f"  Median: {data.median():.1f}")
        print(f"  IQR: {data.quantile(0.25):.1f} - {data.quantile(0.75):.1f}")
        print(f"  Range: {data.min():.0f} - {data.max():.0f}")

        # Response distribution
        print(f"  Response Distribution:")
        counts = data.value_counts().sort_index()
        total = len(data)
        for rating in range(1, 6):
            count = counts.get(rating, 0)
            pct = (count / total) * 100
            print(f"    {rating}: {count:3d} ({pct:5.1f}%)")

    return

def plot_distributions(df_long):
    """
    Visualize response distributions
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, question in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
        ax = axes[idx]

        counts = df_long[question].value_counts().sort_index()
        ax.bar(counts.index, counts.values, color=['#d62728', '#ff7f0e', '#bcbd22', '#2ca02c', '#1f77b4'])
        ax.set_xlabel('Rating', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'{question} Distribution', fontsize=14, fontweight='bold')
        ax.set_xticks(range(1, 6))
        ax.grid(axis='y', alpha=0.3)

        mean_val = df_long[question].mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.legend()

    plt.tight_layout()
    plt.savefig('llm_response_distributions.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: llm_response_distributions.png")

    return fig

# ============================================================================
# PART 3: INTER-RATER RELIABILITY (FLEISS' KAPPA)
# ============================================================================

def calculate_fleiss_kappa(df_long, question='Q2'):
    """
    Calculate Fleiss' Kappa for inter-rater agreement
    """
    unique_mixtures = sorted(df_long['mixture'].unique())
    n_mixtures = len(unique_mixtures)
    n_categories = 5

    ratings_matrix = np.zeros((n_mixtures, n_categories))

    for i, mixture in enumerate(unique_mixtures):
        mixture_data = df_long[df_long['mixture'] == mixture][question]
        for rating in range(1, 6):
            ratings_matrix[i, rating - 1] = (mixture_data == rating).sum()

    kappa = fleiss_kappa(ratings_matrix)

    return kappa, ratings_matrix

def inter_rater_reliability(df_long):
    """
    Calculate and report Fleiss' Kappa for all questions
    """
    print("\n" + "="*80)
    print("INTER-RATER RELIABILITY (FLEISS' KAPPA) - LLM Agreement")
    print("="*80)

    # Overall kappa
    print("\n--- OVERALL (All Predictions) ---")
    for question in ['Q2', 'Q3', 'Q4']:
        kappa, _ = calculate_fleiss_kappa(df_long, question)

        if kappa < 0.20:
            interp = "Slight agreement"
        elif kappa < 0.40:
            interp = "Fair agreement"
        elif kappa < 0.60:
            interp = "Moderate agreement"
        elif kappa < 0.80:
            interp = "Substantial agreement"
        else:
            interp = "Almost perfect agreement"

        print(f"\n{question}:")
        print(f"  Fleiss' κ = {kappa:.3f}")
        print(f"  Interpretation: {interp}")

    return

# ============================================================================
# PART 4: PAIRED COMPARISON TESTS (Q2 vs Q4)
# ============================================================================

def wilcoxon_signed_rank_test(df_long):
    """
    Test if reading explanation changes agreement (Q2 vs Q4)
    """
    print("\n" + "="*80)
    print("PAIRED COMPARISON: Q2 vs Q4 (Wilcoxon Signed-Rank Test)")
    print("="*80)

    q2 = df_long['Q2'].values
    q4 = df_long['Q4'].values

    statistic, p_value = wilcoxon(q2, q4, alternative='two-sided')

    n = len(q2)
    mean_w = n * (n + 1) / 4
    std_w = np.sqrt(n * (n + 1) * (2*n + 1) / 24)
    z_score = (statistic - mean_w) / std_w

    effect_size = abs(z_score) / np.sqrt(n)

    mean_diff = np.mean(q4 - q2)
    median_diff = np.median(q4 - q2)

    increased = np.sum(q4 > q2)
    decreased = np.sum(q4 < q2)
    unchanged = np.sum(q4 == q2)

    print(f"\nTest Statistics:")
    print(f"  W-statistic: {statistic:.2f}")
    print(f"  Z-score: {z_score:.2f}")
    print(f"  p-value: {p_value:.6f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
    print(f"  Effect size (r): {effect_size:.3f}")

    if effect_size < 0.1:
        effect_interp = "negligible"
    elif effect_size < 0.3:
        effect_interp = "small"
    elif effect_size < 0.5:
        effect_interp = "medium"
    else:
        effect_interp = "large"

    print(f"  Effect interpretation: {effect_interp}")

    print(f"\nDifference Metrics:")
    print(f"  Mean difference (Q4 - Q2): {mean_diff:+.3f}")
    print(f"  Median difference: {median_diff:+.1f}")
    print(f"  Percent improvement: {(mean_diff / df_long['Q2'].mean()) * 100:+.1f}%")

    print(f"\nDirection of Changes:")
    print(f"  Increased agreement (Q4 > Q2): {increased} ({increased/n*100:.1f}%)")
    print(f"  No change (Q4 = Q2): {unchanged} ({unchanged/n*100:.1f}%)")
    print(f"  Decreased agreement (Q4 < Q2): {decreased} ({decreased/n*100:.1f}%)")

    return z_score, p_value, effect_size

def plot_opinion_change(df_long):
    """
    Visualize opinion change from Q2 to Q4
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    ax1.scatter(df_long['Q2'], df_long['Q4'], alpha=0.5, s=50)
    ax1.plot([1, 5], [1, 5], 'r--', linewidth=2, label='No change')
    ax1.set_xlabel('Q2: Initial Agreement', fontsize=12)
    ax1.set_ylabel('Q4: Post-Explanation Agreement', fontsize=12)
    ax1.set_title('Opinion Change: Q2 → Q4', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0.5, 5.5)
    ax1.set_ylim(0.5, 5.5)

    ax2 = axes[1]
    changes = df_long['Q4_minus_Q2']
    counts = changes.value_counts().sort_index()
    colors = ['red' if x < 0 else 'gray' if x == 0 else 'green' for x in counts.index]
    ax2.bar(counts.index, counts.values, color=colors, alpha=0.7)
    ax2.set_xlabel('Opinion Change (Q4 - Q2)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Distribution of Opinion Changes', fontsize=14, fontweight='bold')
    ax2.axvline(0, color='black', linestyle='--', linewidth=2)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('llm_opinion_change_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: llm_opinion_change_analysis.png")

    return fig

# ============================================================================
# PART 5: LLM-LEVEL ANALYSIS
# ============================================================================

def llm_level_analysis(df_long):
    """
    Analyze individual LLM patterns
    """
    print("\n" + "="*80)
    print("LLM-LEVEL ANALYSIS")
    print("="*80)

    # Overall LLM profiles
    print("\n--- OVERALL LLM PROFILES ---")
    llm_summary = df_long.groupby('llm_name').agg({
        'Q2': 'mean',
        'Q3': 'mean',
        'Q4': 'mean',
        'Q4_minus_Q2': 'mean'
    }).round(2)

    print("\nLLM Response Profiles:")
    print(llm_summary.to_string())

    # Friedman test (non-parametric repeated measures)
    print("\n--- FRIEDMAN TEST: Overall ---")

    # Reshape data for Friedman test (each mixture is a subject, LLMs are conditions)
    mixtures = sorted(df_long['mixture'].unique())
    q4_minus_q2_matrix = []

    for mixture in mixtures:
        mixture_data = df_long[df_long['mixture'] == mixture].sort_values('llm_id')['Q4_minus_Q2'].values
        if len(mixture_data) == 4:  # All 4 LLMs responded
            q4_minus_q2_matrix.append(mixture_data)

    if len(q4_minus_q2_matrix) > 0:
        q4_minus_q2_matrix = np.array(q4_minus_q2_matrix)
        chi_stat, p_value = friedmanchisquare(*q4_minus_q2_matrix.T)

        print(f"\nFriedman Test (opinion change across LLMs):")
        print(f"  χ²-statistic: {chi_stat:.2f}")
        print(f"  p-value: {p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
        print(f"  Interpretation: {'Significant differences' if p_value < 0.05 else 'No significant differences'} in opinion change across LLMs")

    return llm_summary

# ============================================================================
# PART 7: CONFIRMATION BIAS ANALYSIS (Q1 vs Q2)
# ============================================================================

def confirmation_bias_analysis(df_long):
    """
    Analyze how LLMs' initial predictions (Q1) influence their agreement with model (Q2)
    """
    print("\n" + "="*80)
    print("CONFIRMATION BIAS ANALYSIS: Q1 → Q2 → Q4")
    print("="*80)

    # Overall change from Q1 to Q2
    print("\n--- OPINION CHANGE: Q1 → Q2 ---")
    q1_to_q2_diff = df_long['Q2_minus_Q1']

    print(f"\nChange from Q1 to Q2 (Q2 - Q1):")
    print(f"  Mean change: {q1_to_q2_diff.mean():+.3f}")
    print(f"  Median change: {q1_to_q2_diff.median():+.1f}")
    print(f"  SD: {q1_to_q2_diff.std():.3f}")

    increased_q1_q2 = (df_long['Q2'] > df_long['Q1']).sum()
    decreased_q1_q2 = (df_long['Q2'] < df_long['Q1']).sum()
    unchanged_q1_q2 = (df_long['Q2'] == df_long['Q1']).sum()
    total = len(df_long)

    print(f"\nDirection of changes (Q1 → Q2):")
    print(f"  Increased (Q2 > Q1): {increased_q1_q2} ({increased_q1_q2/total*100:.1f}%)")
    print(f"  Unchanged (Q2 = Q1): {unchanged_q1_q2} ({unchanged_q1_q2/total*100:.1f}%)")
    print(f"  Decreased (Q2 < Q1): {decreased_q1_q2} ({decreased_q1_q2/total*100:.1f}%)")

    stat_q1_q2, p_q1_q2 = wilcoxon(df_long['Q1'], df_long['Q2'], alternative='two-sided')
    print(f"\nWilcoxon signed-rank test (Q1 vs Q2):")
    print(f"  W-statistic: {stat_q1_q2:.2f}")
    print(f"  p-value: {p_q1_q2:.6f} {'***' if p_q1_q2 < 0.001 else '**' if p_q1_q2 < 0.01 else '*' if p_q1_q2 < 0.05 else 'ns'}")

    # Correlation between Q1 and Q2
    print("\n--- CONFIRMATION BIAS INDICATORS ---")
    rho_q1_q2, p_rho_q1_q2 = spearmanr(df_long['Q1'], df_long['Q2'])
    print(f"\nCorrelation Q1 vs Q2 (Spearman's ρ):")
    print(f"  ρ = {rho_q1_q2:.3f}, p = {p_rho_q1_q2:.6f} {'***' if p_rho_q1_q2 < 0.001 else '**' if p_rho_q1_q2 < 0.01 else '*' if p_rho_q1_q2 < 0.05 else 'ns'}")

    return

def plot_confirmation_bias(df_long):
    """
    Visualize confirmation bias and opinion changes
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Q1 vs Q2 scatter
    ax1 = axes[0, 0]
    ax1.scatter(df_long['Q1'], df_long['Q2'],
                alpha=0.5, s=50, c='#1f77b4', label='Predictions')
    ax1.plot([1, 5], [1, 5], 'k--', linewidth=2, alpha=0.5, label='No change')
    ax1.set_xlabel('Q1: LLM Prediction', fontsize=11)
    ax1.set_ylabel('Q2: Agreement with Model', fontsize=11)
    ax1.set_title('Confirmation Bias: Q1 → Q2', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0.5, 5.5)
    ax1.set_ylim(0.5, 5.5)

    rho, p = spearmanr(df_long['Q1'], df_long['Q2'])
    ax1.text(0.05, 0.95, f'ρ = {rho:.3f}\np = {p:.4f}',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel 2: Mean trajectory by LLM
    ax2 = axes[0, 1]
    for llm in df_long['llm_name'].unique():
        llm_data = df_long[df_long['llm_name'] == llm]
        means = [llm_data['Q1'].mean(), llm_data['Q2'].mean(), llm_data['Q4'].mean()]
        ax2.plot([1, 2, 3], means, 'o-', linewidth=2, markersize=8, label=llm)

    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(['Q1\n(LLM Pred)', 'Q2\n(See Model)', 'Q4\n(+ Explanation)'])
    ax2.set_ylabel('Mean Rating', fontsize=11)
    ax2.set_title('Mean Opinion Trajectory by LLM', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Panel 3: Distribution of Q1→Q2 change
    ax3 = axes[1, 0]
    changes_q1_q2 = df_long['Q2_minus_Q1']
    counts = changes_q1_q2.value_counts().sort_index()
    colors = ['red' if x < 0 else 'gray' if x == 0 else 'green' for x in counts.index]
    ax3.bar(counts.index, counts.values, color=colors, alpha=0.7)
    ax3.set_xlabel('Opinion Change (Q2 - Q1)', fontsize=11)
    ax3.set_ylabel('Count', fontsize=11)
    ax3.set_title('Distribution of Q1 → Q2 Changes', fontsize=13, fontweight='bold')
    ax3.axvline(0, color='black', linestyle='--', linewidth=2)
    ax3.grid(axis='y', alpha=0.3)

    # Panel 4: Overall mean trajectory
    ax4 = axes[1, 1]
    overall_means = [df_long['Q1'].mean(),
                     df_long['Q2'].mean(),
                     df_long['Q4'].mean()]

    ax4.plot([1, 2, 3], overall_means, 'o-', color='#1f77b4', linewidth=3,
             markersize=10, label='All predictions')
    ax4.set_xticks([1, 2, 3])
    ax4.set_xticklabels(['Q1\n(LLM Pred)', 'Q2\n(See Model)', 'Q4\n(+ Explanation)'])
    ax4.set_ylabel('Mean Rating', fontsize=11)
    ax4.set_title('Mean Opinion Trajectory', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('llm_confirmation_bias_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: llm_confirmation_bias_analysis.png")

    return fig

# ============================================================================
# PART 8: LLM COMPARISON ANALYSIS
# ============================================================================

def llm_comparison_analysis(df_long):
    """
    Compare LLMs across different metrics
    """
    print("\n" + "="*80)
    print("LLM COMPARISON ANALYSIS")
    print("="*80)

    # Pairwise comparisons for Q4_minus_Q2 (explanation effectiveness)
    print("\n--- PAIRWISE COMPARISONS: Explanation Effectiveness (Q4 - Q2) ---")

    llms = df_long['llm_name'].unique()

    for i, llm1 in enumerate(llms):
        for llm2 in llms[i+1:]:
            llm1_data = df_long[df_long['llm_name'] == llm1]['Q4_minus_Q2'].values
            llm2_data = df_long[df_long['llm_name'] == llm2]['Q4_minus_Q2'].values

            # Mann-Whitney U test
            stat, p_val = mannwhitneyu(llm1_data, llm2_data, alternative='two-sided')

            print(f"\n{llm1} vs {llm2}:")
            print(f"  {llm1}: {np.mean(llm1_data):+.3f} ± {np.std(llm1_data):.3f}")
            print(f"  {llm2}: {np.mean(llm2_data):+.3f} ± {np.std(llm2_data):.3f}")
            print(f"  U = {stat:.1f}, p = {p_val:.4f} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'}")

    return

def plot_llm_comparison(df_long):
    """
    Visualize LLM comparisons
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    metrics = ['Q2', 'Q3', 'Q4', 'Q4_minus_Q2']
    titles = ['Initial Agreement', 'Explanation Quality Rating', 'Post-Explanation Agreement', 'Opinion Change']

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]

        llms = df_long['llm_name'].unique()
        data_to_plot = [df_long[df_long['llm_name'] == llm][metric] for llm in llms]

        bp = ax.boxplot(data_to_plot, labels=llms, widths=0.6,
                        patch_artist=True, showmeans=True)

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_ylabel('Rating', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('llm_comparison_boxplots.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: llm_comparison_boxplots.png")

    return fig

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(filepath='Solubility Research Form (Responses) - Form responses 1 (2).csv'):
    """
    Run complete analysis pipeline
    """
    print("\n" + "="*80)
    print("LLM SOLUBILITY EVALUATION ANALYSIS")
    print("="*80)
    print(f"\nLoading data from: {filepath}")

    # Load and prepare data
    df = load_survey_data(filepath)
    df_long = prepare_data(df)

    n_mixtures = df_long['mixture'].nunique()
    n_llms = df_long['llm_name'].nunique()

    print(f"✓ Loaded {n_llms} LLMs × {n_mixtures} mixtures = {len(df_long)} observations")

    # Run analyses
    descriptive_statistics(df_long)
    plot_distributions(df_long)

    inter_rater_reliability(df_long)

    confirmation_bias_analysis(df_long)
    plot_confirmation_bias(df_long)

    wilcoxon_signed_rank_test(df_long)
    plot_opinion_change(df_long)

    llm_level_analysis(df_long)

    llm_comparison_analysis(df_long)
    plot_llm_comparison(df_long)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  • llm_response_distributions.png")
    print("  • llm_confirmation_bias_analysis.png")
    print("  • llm_opinion_change_analysis.png")
    print("  • llm_comparison_boxplots.png")

    return df_long

if __name__ == "__main__":
    # Run analysis
    df_long = main('Solubility Research Form (Responses) - Form responses 1 (2).csv')

    # Save processed data
    df_long.to_csv('llm_survey_data_long_format.csv', index=False)
    print("\n✓ Saved processed data: llm_survey_data_long_format.csv")
