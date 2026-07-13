"""Plotting helpers for feasible-region and experiment-result figures."""

import numpy as np
from matplotlib.colors import to_rgb


REGION_COLORS = {
    'group': ['C0', 'C1'],
    'intersection': 'C2',
    'marker': 'black'}
REGION_ALPHA = 0.33

BOX_COLORS = {
    'violation': 'C0',
    'accuracy': 'C1',
    'unconstrained': 'gray'}
BOX_ALPHA = 0.33


def plot_feasible_regions(ax, result, show_breakpoints=False):
  for a, dist in enumerate(result.dists):
    p, q = dist.boundary()
    pi = dist.pi
    name = dist.name if dist.name is not None else f'Group {a}'
    c = REGION_COLORS['group'][a]

    ax.fill(np.r_[p, pi, pi, p[0]], np.r_[q, q[-1], pi, pi],
            c=c, lw=0, alpha=REGION_ALPHA, zorder=2)
    ax.plot(p, q, c=c, lw=1)
    ax.fill([], [], fc=(*to_rgb(c), REGION_ALPHA),
            ec=c, lw=1, label=fr'$\mathcal{{C}}^{{{a}}}$: {name}')

    if show_breakpoints:
      ax.plot(dist.pk, dist.qk, '.', c=c, ms=3, zorder=3)

  ax.plot(result.p, result.q,
          c=REGION_COLORS['intersection'], lw=3,
          label=r'$\partial(\mathcal{C}^0\cap\mathcal{C}^1)$')

  ax.plot([result.max_acc.p], [result.max_acc.q],
          '+', c=REGION_COLORS['marker'], ms=8,
          label=fr'max. acc. $= {result.max_acc.value:.4f}$')
  ax.plot([result.min_dsep.p], [result.min_dsep.q],
          'o', c=REGION_COLORS['marker'], ms=5, mfc='none',
          label='min. $\\Delta_{\\mathrm{sep}}$')

  ax.set_xlabel('PPV')
  ax.set_ylabel('FOR')
  ax.set_xlim(0, 1)
  ax.set_ylim(0, 1)
  ax.set_aspect('equal')
  ax.grid(alpha=REGION_ALPHA)
  ax.legend(loc='upper left')

  return ax


def styled_boxplot(ax, data, positions=None, color='violation', width=None):
  c = BOX_COLORS[color]
  if width is None:
    if positions is None:
      width = 0.6
    else:
      width = 0.6 * min(np.diff(positions))

  artists = ax.boxplot(
      data,
      positions=positions,
      widths=width,
      patch_artist=True,
      manage_ticks=False,
      showfliers=True,
      boxprops=dict(facecolor=(*to_rgb(c), BOX_ALPHA)),
      medianprops=dict(color=c, linewidth=2),
      flierprops=dict(marker='.', alpha=BOX_ALPHA),)

  return artists
