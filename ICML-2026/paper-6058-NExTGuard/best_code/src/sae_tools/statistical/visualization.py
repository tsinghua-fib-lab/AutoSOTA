import numpy as np
import matplotlib.pyplot as plt
from .metrics import GlobalMetricsResult

def get_feature_info(feature_id: int, metrics_result: GlobalMetricsResult):
    """Get Full info of a feature from GlobalMetricsResult object"""
    idx = int(feature_id)
    return {
        'id': feature_id,
        'f1': float(metrics_result.f1_scores[idx]),
        'precision': float(metrics_result.precisions[idx]),
        'recall': float(metrics_result.recalls[idx]),
        'activation_ratio': float(metrics_result.activation_ratios[idx]),
        'feature_diff': float(metrics_result.feature_diff[idx])
    }

def print_top_features(metrics_result: GlobalMetricsResult, metric_key: str, title: str, top_n=10):
    """
    print top features of a given metric
    """
    key_mapping = {
        'high_f1': metrics_result.top_f1_ids,
        'high_precision': metrics_result.top_precision_ids,
        'high_recall': metrics_result.top_recall_ids,
        'high_diff': metrics_result.top_diff_ids,
        'pareto': metrics_result.pareto_front_ids
    }

    feature_ids = key_mapping.get(metric_key)

    if feature_ids is None:
        print(f"\n⚠️  {title}: unknown metric key '{metric_key}'")
        return

    if len(feature_ids) == 0:
        print(f"\n⚠️  {title}: no available data")
        return
    
    n = min(top_n, len(feature_ids))
    ids = feature_ids[:n]
    
    print(f"\n📊 {title}:")
    print(f"  Total {len(feature_ids)} features in this category")
    print(f"  Top {n} IDs: {ids}")
    print("-" * 80)
    
    for i, fid in enumerate(ids):
        info = get_feature_info(fid, metrics_result)
        print(f"  {i+1:2d}. Feature {fid:5d}: F1={info['f1']:.4f}, "
              f"Prec={info['precision']:.4f}, Rec={info['recall']:.4f}, "
              f"ActRatio={info['activation_ratio']:.4f}")

def print_metrics_overview(metrics_result: GlobalMetricsResult, preview_size=10):
    """
    show overview of GlobalMetricsResult
    """
    stats = metrics_result.stats
    
    print("\n" + "=" * 80)
    print("📊 overview of GlobalMetricsResult Object")
    print("=" * 80)
    
    print(f"total features: {stats.get('num_features', len(metrics_result.feature_indices))}")
    print(f"separation score: {metrics_result.separation_score}")
    
    print(f"\nTop {len(metrics_result.top_precision_ids)} high Precision features: {metrics_result.top_precision_ids[:preview_size]}...")
    print(f"Top {len(metrics_result.top_recall_ids)} high Recall features: {metrics_result.top_recall_ids[:preview_size]}...")
    print(f"Top {len(metrics_result.top_f1_ids)} high F1 features: {metrics_result.top_f1_ids[:preview_size]}...")
    print(f"Pareto front features: {len(metrics_result.pareto_front_ids)}")
    print(f"Top {len(metrics_result.top_diff_ids)} high difference features: {metrics_result.top_diff_ids[:preview_size]}...")
    
    print(f"\narray shapes:")
    print(f"  precisions: {metrics_result.precisions.shape}")
    print(f"  recalls: {metrics_result.recalls.shape}")
    print(f"  f1_scores: {metrics_result.f1_scores.shape}")
    print(f"  activation_ratios: {metrics_result.activation_ratios.shape}")
    print(f"  feature_diff: {metrics_result.feature_diff.shape}")

def plot_pr_space(metrics_result: GlobalMetricsResult, output_file=None, title=None, color_by='diff', highlight_indices=None):
    """
    plot all features on PR space
    """
    precisions = metrics_result.precisions
    recalls = metrics_result.recalls
    f1_scores = metrics_result.f1_scores
    activation_ratios = metrics_result.activation_ratios
    
    num_features = len(precisions)
    
    if color_by == 'diff':
        color_data = metrics_result.feature_diff
        color_label = 'Normalized Feature Difference (pos - neg) / (std(pos) + std(neg))'
        vmin, vmax = None, None
    elif color_by == 'ratio':
        color_data = activation_ratios
        color_label = 'Activation Ratio'
        vmin, vmax = 0, 1
    else:
        raise ValueError(f"color_by must be 'diff' or 'ratio', current value is: {color_by}")
    
    print(f"⚡ Generating Points-Only Plot (color by {color_by})...")
    fig, ax = plt.subplots(figsize=(10, 10))
    
    scatter = ax.scatter(
        x=recalls,
        y=precisions,
        c=color_data,
        cmap='RdBu_r',
        vmin=vmin,
        vmax=vmax,
        s=20,
        alpha=0.6,
        edgecolors='none',
        rasterized=True
    )
    
    if highlight_indices is not None:
        highlight_indices = np.array(highlight_indices)
        valid_mask = (highlight_indices >= 0) & (highlight_indices < len(recalls))
        if np.any(valid_mask):
            highlight_indices = highlight_indices[valid_mask]
            ax.scatter(
                x=recalls[highlight_indices],
                y=precisions[highlight_indices],
                s=150, c='yellow', edgecolors='red', linewidths=2, alpha=0.9, marker='*', zorder=10,
                label=f'Highlighted ({len(highlight_indices)} points)'
            )
            ax.legend(loc='upper right', framealpha=0.9)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(color_label, fontsize=12, rotation=270, labelpad=20)
    
    ax.set_xlabel('Recall', fontsize=14, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=14, fontweight='bold')
    
    if title is None:
        title = f'All Features on PR Space (Separation Score={metrics_result.separation_score:.4f})'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_aspect('equal', 'box')
    
    stats_text = f'Total Features: {num_features}\n'
    stats_text += f'F1 Score Mean: {np.mean(f1_scores):.4f}\n'
    stats_text += f'F1 Score Max: {np.max(f1_scores):.4f}\n'
    
    if color_by == 'diff':
        stats_text += f'Diff Range: [{np.min(color_data):.4f}, {np.max(color_data):.4f}]'
    
    ax.text(0.98, 0.02, stats_text, 
            transform=ax.transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), family='monospace')
    
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Points-only plot saved to: {output_file}")
    
    plt.show()
    return fig