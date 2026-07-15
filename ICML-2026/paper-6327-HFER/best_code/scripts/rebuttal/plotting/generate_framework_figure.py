"""
Spectral Diagnosis → Intervention Framework Figure
===================================================
Addresses: Reviewer 6GDW ("does spectral steering work?")
Provides a conceptual framework figure for the rebuttal showing the
Diagnosis → Intervention → Verification loop.

Usage:
    cd geometry-of-reason
    python scripts/rebuttal/generate_framework_figure.py
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})


def draw_box(ax, x, y, w, h, text, color='#E3F2FD', edge_color='#1565C0',
             fontsize=10, fontweight='normal', alpha=1.0):
    """Draw a rounded box with text."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.12",
                         facecolor=color, edgecolor=edge_color,
                         linewidth=1.5, alpha=alpha)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontweight=fontweight, color='#212121', wrap=True)


def draw_arrow(ax, x1, y1, x2, y2, color='#424242', style='->', lw=1.5):
    """Draw a curved arrow between two points."""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle=style,
                            connectionstyle='arc3,rad=0',
                            color=color, linewidth=lw,
                            mutation_scale=15)
    ax.add_patch(arrow)


def draw_curved_arrow(ax, x1, y1, x2, y2, color='#424242', rad=0.3, lw=1.5):
    """Draw a curved arrow."""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle='->',
                            connectionstyle=f'arc3,rad={rad}',
                            color=color, linewidth=lw,
                            mutation_scale=15)
    ax.add_patch(arrow)


def generate_framework_figure():
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 5.5)
    ax.axis('off')

    # Title
    ax.text(5.0, 5.2, 'Spectral Diagnosis → Intervention Pipeline',
            ha='center', va='center', fontsize=14, fontweight='bold',
            color='#0D47A1')

    # === Row 1: Main pipeline ===
    # Box 1: Proof Generation
    draw_box(ax, 1.5, 3.5, 2.2, 1.0,
             'Proof\nGeneration\n(LLM)', color='#FFF3E0', edge_color='#E65100',
             fontsize=10, fontweight='bold')

    # Arrow 1→2
    draw_arrow(ax, 2.6, 3.5, 3.4, 3.5, color='#424242')

    # Box 2: Spectral Diagnosis
    draw_box(ax, 5.0, 3.5, 2.6, 1.0,
             'Spectral Diagnosis\nHFER · Fiedler · Smoothness',
             color='#E8F5E9', edge_color='#2E7D32',
             fontsize=10, fontweight='bold')

    # Arrow 2→Decision
    draw_arrow(ax, 6.3, 3.5, 7.2, 3.5, color='#424242')

    # Decision diamond (as a rotated box)
    diamond = plt.Polygon([[8.0, 3.5], [8.5, 4.0], [9.0, 3.5], [8.5, 3.0]],
                          facecolor='#FCE4EC', edgecolor='#C62828',
                          linewidth=1.5)
    ax.add_patch(diamond)
    ax.text(8.5, 3.5, 'Valid?', ha='center', va='center', fontsize=9,
            fontweight='bold', color='#B71C1C')

    # YES arrow → Accept
    draw_arrow(ax, 9.0, 3.5, 9.5, 3.5, color='#2E7D32', lw=2)
    ax.text(9.8, 3.5, '✓ Accept\nProof', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#2E7D32',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#C8E6C9',
                      edgecolor='#2E7D32', linewidth=1.5))

    # NO arrow → Spectral Steering
    draw_arrow(ax, 8.5, 3.0, 8.5, 1.7, color='#C62828', lw=2)
    ax.text(8.9, 2.4, '✗', fontsize=12, color='#C62828', fontweight='bold')

    # Box: Spectral Steering
    draw_box(ax, 5.0, 1.2, 2.6, 0.9,
             'Spectral Steering\n(SVD weight edit)',
             color='#EDE7F6', edge_color='#4527A0',
             fontsize=10, fontweight='bold')

    # Arrow from decision to steering
    draw_arrow(ax, 8.5, 1.2, 6.3, 1.2, color='#C62828', lw=1.5)

    # Arrow from steering back to generation
    draw_curved_arrow(ax, 3.7, 1.2, 1.5, 3.0, color='#4527A0', rad=0.3, lw=1.5)
    ax.text(1.8, 1.8, 'Regenerate\n(steered model)', ha='center', va='center',
            fontsize=8, color='#4527A0', fontstyle='italic')

    # === Annotations ===
    # Layer info on Spectral Diagnosis
    ax.text(5.0, 2.8, 'Layer L (75th %-ile) · 1 parameter (threshold)',
            ha='center', va='center', fontsize=8, color='#666666',
            fontstyle='italic')

    # Steering details
    ax.text(5.0, 0.5, r'$W_{new} = U \, \mathrm{diag}(S_{sharp}) \, V^T$'
            + '  |  α = −0.3 (sharpening)',
            ha='center', va='center', fontsize=8, color='#666666',
            fontstyle='italic')

    # Reference box
    ref_text = ('This paper: Diagnosis (§3-4)\n'
                'Spectral Steering: Intervention (§5)')
    ax.text(1.5, 0.4, ref_text, ha='center', va='center', fontsize=8,
            color='#37474F',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ECEFF1',
                      edgecolor='#78909C', linewidth=1))

    os.makedirs('output/rebuttal', exist_ok=True)
    fig_path = 'output/rebuttal/framework_figure.pdf'
    plt.savefig(fig_path)
    plt.savefig(fig_path.replace('.pdf', '.png'))
    print(f"Saved: {fig_path}")
    print(f"Saved: {fig_path.replace('.pdf', '.png')}")


if __name__ == '__main__':
    generate_framework_figure()
