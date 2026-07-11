"""
Table-formatting helpers used by the LaTeX-table-emitting notebooks.

These were previously duplicated across five notebooks with slightly
different precision/scaling combinations.  They are consolidated here
as a small set of parametric helpers so that each table can choose its
own precision via the ``prec=`` argument and a consistent rendering
convention is preserved across the paper.
"""

from __future__ import annotations

import numpy as np


def fmt(mean: float, std: float | None = None, prec: int = 2) -> str:
    """Format a mean (and optionally a std) with ``prec`` decimals.

    Examples
    --------
    >>> fmt(0.87)
    '0.87'
    >>> fmt(0.871, 0.045, prec=3)
    '0.871 (0.045)'
    >>> fmt(7.45, 5.68, prec=2)
    '7.45 (5.68)'
    """
    s_mean = f"{mean:.{prec}f}"
    if std is None:
        return s_mean
    return f"{s_mean} ({std:.{prec}f})"


def fmt_pct(mean: float, std: float | None = None, prec: int = 1) -> str:
    """Format value(s) as percentages (multiplied by 100).

    Examples
    --------
    >>> fmt_pct(0.871, 0.045)
    '87.1 (4.5)'
    >>> fmt_pct(0.072)
    '7.2'
    """
    s_mean = f"{100*mean:.{prec}f}"
    if std is None:
        return s_mean
    return f"{s_mean} ({100*std:.{prec}f})"


def apply_deco(s: str, deco: str | None) -> str:
    """Wrap a string in a LaTeX decoration macro.

    ``deco`` is one of ``"bold"``, ``"underline"``, or anything falsy
    (which returns the string unchanged).
    """
    if deco == "bold":
        return r"\textbf{" + s + "}"
    if deco == "underline":
        return r"\underline{" + s + "}"
    return s


def rank_decor(values) -> list[str | None]:
    """Return decorations to highlight the top two values in a sequence.

    The maximum gets ``"bold"``, the second-highest gets ``"underline"``,
    everything else gets ``None``.  Suitable for direct use with
    :func:`apply_deco` to produce LaTeX tables that emphasise the best
    and runner-up entries per column.
    """
    values = np.asarray(values)
    order = np.argsort(values)[::-1]
    deco: list[str | None] = [None] * len(values)
    if len(values) > 0:
        deco[order[0]] = "bold"
    if len(values) > 1:
        deco[order[1]] = "underline"
    return deco
