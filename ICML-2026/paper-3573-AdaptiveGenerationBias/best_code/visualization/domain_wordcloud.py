#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_set_diagrams import VennDiagram

# ---------------------------------------------------------------------------
# Import project utilities + loader
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(__file__))

try:
    from bias_visualization_dashboard import SimplifiedBiasDataLoader
    from vis_utilities import normalize_text
except Exception as e:
    print("ERROR: Could not import SimplifiedBiasDataLoader from bias_visualization_dashboard.")
    print("Make sure this file lives next to your existing scripts. Original error:", e)
    sys.exit(1)

try:
    # Use the shared utilities you provided
    from vis_utilities import (
        setup_nyt_style,
        EARTHY_COLORS,
        color_cycle_for_keys,
        parse_attr_paths,
        choose_output_alias,
        get_attribute_display_name,
    )
except Exception as e:
    print("ERROR: Could not import vis_utilities. Ensure vis_utilities.py is next to this script.")
    print("Original error:", e)
    sys.exit(1)

# Apply NYT style globally
setup_nyt_style()

# ---------------------------------------------------------------------------
# Mapping utilities
# ---------------------------------------------------------------------------


def load_and_invert_mapping(json_path: str) -> dict[str, str]:
    """Load {canonical: [raw1, raw2, ...]} and return {raw: canonical}."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    inverse: dict[str, str] = {}
    for canonical, raw_list in data.items():
        if not raw_list:
            continue
        for raw in raw_list:
            if raw is None:
                continue
            inverse[normalize_text(str(raw)).strip()] = normalize_text(str(canonical)).strip()

    return inverse


def normalize_domain(raw: str) -> str:
    """Split on '::' and take the SECOND half; trim whitespace."""
    if raw is None:
        return ""
    s = str(raw)
    if "::" in s:
        left, right = s.split("::", 1)
        return normalize_text(right.strip())
    return normalize_text(s.strip())


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


# def load_attr_df(attr: str, run_path: str) -> pd.DataFrame:
#     loader = SimplifiedBiasDataLoader(run_path, bias_attributes_override=[attr])
#     data = loader.load_data()
#     df = data.conversations_df
#     if df is None or df.empty:
#         print(f"[{attr}] No data at {run_path}")
#         return pd.DataFrame()
#     df = df.copy()
#     df["__attribute"] = attr
#     return df


def load_attr_df(attr: str, run_path: str) -> pd.DataFrame:
    # Check if "samples_{attr}.jsonl" exists

    sample_file = os.path.join(run_path, f"samples_{attr}.jsonl")
    if not os.path.exists(sample_file):
        sample_file = os.path.join(run_path, "transformed_questions.jsonl")
    if not os.path.exists(sample_file):
        print(f"[{attr}] No sample file at {run_path}")
        return pd.DataFrame()

    loaded_questions = []  # Load as dict
    with open(sample_file, "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            loaded_questions.append(q)

    df = pd.DataFrame(loaded_questions)
    if df is None or df.empty:
        print(f"[{attr}] No data at {run_path}")
        return pd.DataFrame()

    # normalie the superdomain and domain columns if they exist
    if "superdomain" in df.columns:
        df["superdomain"] = df["superdomain"].apply(normalize_text)
    if "domain" in df.columns:
        df["domain"] = df["domain"].apply(normalize_text)

    df = df.copy()
    df["__attribute"] = attr
    return df


# ---------------------------------------------------------------------------
# Word cloud generation (attribute-colored)
# ---------------------------------------------------------------------------


def ensure_wordcloud():
    try:
        from wordcloud import WordCloud

        return WordCloud
    except Exception:
        print("ERROR: The 'wordcloud' package is required. Install via `pip install wordcloud`.")
        return None


def build_attribute_primary_colors(counts_df: pd.DataFrame, word_col: str) -> tuple[dict, dict]:
    """
    Given a long DF with columns [word_col, '__attribute', 'count'], compute:
      - total_counts: word -> total frequency across attributes
      - primary_attr: word -> attribute with max frequency
    """
    total_counts = counts_df.groupby(word_col)["count"].sum().to_dict()
    primary_attr = (
        counts_df.pivot_table(
            index=word_col, columns="__attribute", values="count", aggfunc="sum", fill_value=0
        )
        .idxmax(axis=1)
        .to_dict()
    )
    return total_counts, primary_attr


def draw_wordcloud_allattrs(
    counts_df: pd.DataFrame,
    word_col: str,
    out_path: Path,
    title: str,
    attr_colors: dict[str, str],
    top_n: int = 300,
):
    WordCloud = ensure_wordcloud()
    if WordCloud is None:
        return

    if counts_df.empty:
        print(f"No data for {title}")
        return

    total_counts, primary_attr = build_attribute_primary_colors(counts_df, word_col)

    # Trim to top N by total frequency
    top_words = sorted(total_counts.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    freqs = {w: int(c) for w, c in top_words}

    # color function by primary attribute
    def color_func(word, *args, **kwargs):
        attr = primary_attr.get(word)
        return attr_colors.get(attr, "#666666")

    wc = WordCloud(
        width=2400,
        height=1400,
        background_color="white",
        prefer_horizontal=0.95,
        collocations=False,
        max_words=len(freqs),
        random_state=42,
        relative_scaling=0.35,
        min_font_size=8,
        normalize_plurals=False,
        include_numbers=True,
    ).generate_from_frequencies(freqs)

    plt.figure(figsize=(18, 10))
    plt.imshow(wc.recolor(color_func=color_func), interpolation="bilinear")
    plt.axis("off")
    # plt.title(title, fontfamily="serif", fontweight="bold", pad=12)

    # Legend: attribute → color
    from matplotlib.patches import Patch

    # handles = [Patch(facecolor=attr_colors[a], edgecolor="none", label=a) for a in attr_colors]
    # leg = plt.legend(handles=handles, title="Attribute", ncols=4, frameon=True, framealpha=0.95)
    # if leg and leg.get_frame():
    #     leg.get_frame().set_facecolor("white")
    #     leg.get_frame().set_edgecolor("lightgray")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _rects_overlap(a, b):
    (ax0, ay0, ax1, ay1) = a
    (bx0, by0, bx1, by1) = b
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)


def _text_data_bbox(ax, text_artist, renderer):
    # bbox in display coords -> data coords
    bbox_disp = text_artist.get_window_extent(renderer=renderer).expanded(1.05, 1.10)  # small pad
    inv = ax.transData.inverted()
    p0 = inv.transform((bbox_disp.x0, bbox_disp.y0))
    p1 = inv.transform((bbox_disp.x1, bbox_disp.y1))
    x0, y0 = min(p0[0], p1[0]), min(p0[1], p1[1])
    x1, y1 = max(p0[0], p1[0]), max(p0[1], p1[1])
    return (x0, y0, x1, y1)


def _rect_overlaps_circle(rect, cx, cy, r, pad=0.0):
    # clamp rectangle to circle center and measure distance
    x0, y0, x1, y1 = rect
    nx = min(max(cx, x0), x1)
    ny = min(max(cy, y0), y1)
    dx, dy = nx - cx, ny - cy
    return (dx * dx + dy * dy) <= (r + pad) * (r + pad)


def _repel_annotations(ax, anns, circles, *, step_frac=0.012, max_iter=120, pad_circle_frac=0.02):
    """
    anns: list of dicts with keys: ann (Annotation), ux, uy (radial unit vector)
    circles: list of (cx, cy, r)
    Nudges labels outward along (ux,uy) to avoid overlaps.
    """
    fig = ax.figure
    fig.canvas.draw()  # need a renderer for bbox measuring
    renderer = fig.canvas.get_renderer()

    # a global step size relative to plot span
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    span = max(abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0]))
    base_step = step_frac * span
    circle_pad = pad_circle_frac * span

    def ann_rect(ann):
        return _text_data_bbox(ax, ann, renderer)

    # precompute initial rectangles
    rects = [ann_rect(a["ann"]) for a in anns]

    moved = True
    it = 0
    while moved and it < max_iter:
        moved = False
        it += 1

        # pairwise label overlaps
        for i in range(len(anns)):
            for j in range(i + 1, len(anns)):
                if _rects_overlap(rects[i], rects[j]):
                    # push both outward along their own radial directions
                    for k in (i, j):
                        ann_k, (ux, uy) = anns[k]["ann"], (anns[k]["ux"], anns[k]["uy"])
                        xk, yk = ann_k.get_position()
                        ann_k.set_position((xk + ux * base_step, yk + uy * base_step))
                        moved = True

        # circle overlaps
        for i in range(len(anns)):
            rect_i = rects[i]
            hit = False
            for cx, cy, r in circles:
                if _rect_overlaps_circle(rect_i, cx, cy, r, pad=circle_pad):
                    hit = True
                    break
            if hit:
                ann_i, (ux, uy) = anns[i]["ann"], (anns[i]["ux"], anns[i]["uy"])
                xi, yi = ann_i.get_position()
                ann_i.set_position((xi + ux * base_step, yi + uy * base_step))
                moved = True

        if moved:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            rects = [ann_rect(a["ann"]) for a in anns]


def _safe_import_embedding_manager():
    """Try several import paths for EmbeddingManager and return the class or None."""
    try:
        from src.utils.embeddings import EmbeddingManager  # type: ignore

        return EmbeddingManager
    except Exception:
        pass
    try:
        from utils.embeddings import EmbeddingManager  # type: ignore

        return EmbeddingManager
    except Exception:
        pass
    try:
        # Fallback: local module name if placed nearby
        from embeddings import EmbeddingManager  # type: ignore

        return EmbeddingManager
    except Exception:
        return None


def _compute_tsne_coords(tokens: list[str], method: str = "embedding") -> pd.DataFrame | None:
    """Use EmbeddingManager to compute t-SNE coordinates for unique tokens.
    Returns DataFrame with columns ['text','x','y'] or None.
    """
    EmbeddingManager = _safe_import_embedding_manager()
    if EmbeddingManager is None:
        print("WARNING: Could not import EmbeddingManager. Skipping t-SNE plots.")
        return None

    mgr = EmbeddingManager()
    chosen_method = method
    if method == "embedding" and not mgr.is_available:
        print("sentence-transformers unavailable → falling back to fuzzy similarity for t-SNE.")
        chosen_method = "fuzzy"

    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            uniq.append(t)

    res = mgr.create_tsne_visualization(uniq, method=chosen_method)
    if res is None or res.empty:
        print("t-SNE returned no coordinates (too few items or an error).")
        return None
    return res


def _plot_tsne_bubbles(
    coords_df: pd.DataFrame,
    counts_pivot: pd.DataFrame,
    attr_colors: dict[str, str],
    title: str,  # kept for API compatibility; not shown
    out_path: Path,
    size_floor: float = 30.0,
    size_ceiling: float = 600.0,
    annotate_top: int | None = None,  # None/0 -> label all
    *,
    start_angle: float = -90.0,
    inner_frac: float = 0.0,
    gap_deg: float = 0.0,
    edgecolor: str = "white",
    edge_lw: float = 0.6,
    shadow: bool = False,
    shadow_alpha: float = 0.08,
):
    """
    t-SNE pie bubbles with circumference leader lines and label repulsion vs. bubbles using adjustText.
    """
    from matplotlib.patches import Wedge, Circle, Patch
    import math

    try:
        from adjustText import adjust_text
    except Exception:
        print("ERROR: `adjustText` is required. Install with `pip install adjustText`.")
        return

    if coords_df is None or coords_df.empty or counts_pivot is None or counts_pivot.empty:
        return

    # ---- Coords: one row per token, clean ----
    coords_df = (
        coords_df.rename(columns={"text": "word"})[["word", "x", "y"]]
        .dropna(subset=["word", "x", "y"])
        .copy()
    )
    coords_df = coords_df[
        np.isfinite(coords_df["x"].astype(float)) & np.isfinite(coords_df["y"].astype(float))
    ]
    if not coords_df["word"].is_unique:
        coords_df = coords_df.groupby("word", as_index=False)[["x", "y"]].mean()
    coords_df = coords_df.set_index("word")

    # ---- Align with counts ----
    counts_pivot = counts_pivot[~counts_pivot.index.duplicated(keep="first")]
    present = counts_pivot.index.intersection(coords_df.index)
    if len(present) == 0:
        print("No overlapping tokens between coords and counts_pivot.")
        return
    coords = coords_df.loc[present, ["x", "y"]]
    counts_pivot = counts_pivot.loc[present]

    # ---- Totals & order ----
    totals = counts_pivot.sum(axis=1).astype(float)
    order = totals.sort_values(ascending=False).index.tolist()

    # ---- Radius scaling (data units) ----
    x_min, x_max = float(coords["x"].min()), float(coords["x"].max())
    y_min, y_max = float(coords["y"].min()), float(coords["y"].max())
    x_span = (x_max - x_min) or 1.0
    y_span = (y_max - y_min) or 1.0
    span = max(x_span, y_span)

    r_min_nom = (size_floor / 1000.0) * span
    r_max_nom = (size_ceiling / 1000.0) * span
    r_max_cap = 0.08 * span
    r_min = min(r_min_nom, r_max_cap * 0.6)
    r_max = min(max(r_max_nom, r_min * 1.1), r_max_cap)

    def radius_from_total(v: float) -> float:
        if not np.isfinite(v) or v <= 0:
            return r_min
        norm = np.sqrt(v / max(totals.max(), 1e-9))
        return r_min + norm * (r_max - r_min)

    # ---- Figure/axes (lock limits BEFORE drawing patches) ----
    plt.figure(figsize=(14, 10))
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")

    # extra pad so outward labels have room
    pad = r_max * 1.8
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_autoscale_on(False)

    # ---- Legend: only used attrs, NO frame ----
    used_attrs = list(counts_pivot.columns[(counts_pivot > 0).any(axis=0)])
    handles = [
        Patch(
            facecolor=attr_colors.get(a, "#cccccc"),
            edgecolor="none",
            label=get_attribute_display_name(a),
        )
        for a in used_attrs
    ]
    ax.legend(
        handles=handles,
        title="Attribute",
        ncols=min(4, max(1, len(handles))),
        frameon=False,
        loc="upper right",
    )

    # ---- Precompute plot center for outward label placement ----
    cx_mean = float(coords["x"].mean())
    cy_mean = float(coords["y"].mean())

    # ---- Draw pies ----
    text_artists = []
    obstacle_patches = []  # invisible circles for adjustText to avoid
    circle_geoms = {}  # for leader line starts (word -> (cx, cy, r))

    for w in order:
        cx, cy = float(coords.loc[w, "x"]), float(coords.loc[w, "y"])
        row = counts_pivot.loc[w]
        parts = [(attr, float(cnt)) for attr, cnt in row.items() if float(cnt) > 0.0]
        if not parts:
            continue

        total = sum(v for _, v in parts)
        r = radius_from_total(total)
        circle_geoms[w] = (cx, cy, r)

        if shadow:
            ax.add_patch(
                Circle(
                    (cx, cy),
                    r * 1.02,
                    facecolor="black",
                    edgecolor="none",
                    alpha=shadow_alpha,
                    zorder=0,
                    clip_on=True,
                )
            )

        n = len(parts)
        total_gap = gap_deg * n if gap_deg > 0 else 0.0
        usable_deg = max(1.0, 360.0 - total_gap)
        theta = float(start_angle)
        width = None if inner_frac <= 0 else r - (r * inner_frac)
        parts.sort(key=lambda kv: (-kv[1], str(kv[0])))

        for attr, cnt in parts:
            frac = cnt / total if total > 0 else 0.0
            sweep = usable_deg * frac
            if sweep <= 0:
                continue
            if sweep >= 359.9:
                sweep = 359.9
            color = attr_colors.get(attr, "#cccccc")
            ax.add_patch(
                Wedge(
                    center=(cx, cy),
                    r=r,
                    theta1=theta,
                    theta2=theta + sweep,
                    width=width,
                    facecolor=color,
                    edgecolor=edgecolor,
                    linewidth=edge_lw,
                    antialiased=False,
                    zorder=1,
                    clip_on=True,
                )
            )
            theta += sweep + (gap_deg if gap_deg > 0 else 0.0)

        # thin outer ring
        ax.add_patch(
            Circle(
                (cx, cy),
                r,
                fill=False,
                edgecolor=edgecolor,
                linewidth=edge_lw * 1.05,
                zorder=2,
                clip_on=True,
            )
        )

        # invisible obstacle (slightly larger than bubble) for adjustText
        obs = Circle((cx, cy), r * 1.10, transform=ax.transData, fill=False, lw=0, alpha=0.0)
        ax.add_artist(obs)  # must be added so it has a renderer
        obstacle_patches.append(obs)

    # ---- Determine how many to annotate (default: all) ----
    if not annotate_top or annotate_top <= 0:
        label_index = totals.sort_values(ascending=False).index
    else:
        label_index = totals.sort_values(ascending=False).head(annotate_top).index

    # ---- Annotations (start just outside circumference) ----
    for w in label_index:
        cx, cy, r = circle_geoms[w]
        dx, dy = cx - cx_mean, cy - cy_mean
        dist = math.hypot(dx, dy) or 1.0
        ux, uy = dx / dist, dy / dist
        px, py = cx + ux * r, cy + uy * r
        # start labels further out to reduce initial collisions
        pad_out = max(r * 0.45, span * 0.015)
        tx, ty = px + ux * pad_out, py + uy * pad_out
        ha = "left" if ux >= 0 else "right"
        va = "bottom" if uy >= 0 else "top"

        ann = ax.annotate(
            w,
            xy=(px, py),  # circumference anchor
            xycoords="data",
            xytext=(tx, ty),  # starting label pos
            textcoords="data",
            ha=ha,
            va=va,
            fontsize=10,
            color="#111111",
            fontfamily="sans-serif",
            zorder=3,
            clip_on=False,
            arrowprops=dict(
                arrowstyle="-",
                lw=0.9,
                color="#666",
                shrinkA=0,
                shrinkB=0,
                relpos=(0, 0),
            ),
        )
        text_artists.append(ann)

    # ---- Style (no title) ----
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, alpha=0.25, linewidth=1.0, color="lightgray")

    for p in ax.patches:
        p.set_rasterized(True)

    # ---- Repel labels against circles (obstacles) ----
    if text_artists:
        adjust_text(
            text_artists,
            ax=ax,
            add_objects=obstacle_patches,  # <-- repel from bubbles
            only_move={"points": "xy", "text": "xy"},
            autoalign="xy",
            # expand = how much “padding” around items (x, y factors)
            expand_text=(1.06, 1.12),
            expand_objects=(1.10, 1.15),
            # forces: bump objects more than texts to keep labels clear of bubbles
            force_text=(0.05, 0.22),
            force_objects=(0.30, 0.70),
            # points force is irrelevant here but keep moderate
            force_points=(0.15, 0.40),
            lim=160,  # iterations cap; raise if still overlapping
            # precision=0.01,  # uncomment for slightly better spacing at small extra cost
        )

    plt.savefig(out_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Venn diagram of set overlaps
# ---------------------------------------------------------------------------


def _ensure_matplotlib_venn():
    try:
        from matplotlib_venn import venn2, venn3

        return venn2, venn3
    except Exception:
        print(
            "ERROR: The 'matplotlib-venn' package is required. Install via `pip install matplotlib-venn`."
        )
        return None, None


# Requires: pip install matplotlib_set_diagrams shapely
from matplotlib_set_diagrams import VennDiagram  # or EulerDiagram
from shapely.geometry import Point
import matplotlib.pyplot as plt
import numpy as np
import textwrap


def draw_attribute_venn_domains_even(
    counts_df,
    *,
    word_col: str = "word",
    attr_col: str = "__attribute",
    attr_colors: dict[str, str] | None = None,
    title: str = "Domain overlap across attributes",
    out_path=None,
    min_count: int = 1,
    max_labels_per_region: int | None = None,  # cap labels per region; None = all
    base_fontsize: int = 9,
    grid_density: int = 18,  # higher = more candidate positions
    use_euler: bool = True,  # True => EulerDiagram
):
    """
    Venn/Euler diagram with domain NAMES evenly spaced inside each subset.
    Uses matplotlib_set_diagrams' subset_geometries + a grid-based label placer.
    """
    # 1) Build attribute -> set of domain names
    df = counts_df[counts_df["count"].astype(float) >= float(min_count)].copy()
    if df.empty:
        print("No data to plot.")
        return

    attr_sets = (
        df.groupby(attr_col)[word_col]
        .apply(lambda s: set(str(x) for x in s.dropna().unique()))
        .to_dict()
    )
    attrs = list(attr_sets.keys())
    if len(attrs) < 2:
        print("Need at least 2 attributes.")
        return
    if len(attrs) > 3:
        attrs = sorted(attrs, key=lambda a: len(attr_sets[a]), reverse=True)[:3]
        attr_sets = {a: attr_sets[a] for a in attrs}
        print(f"Using top 3 attributes by coverage: {', '.join(attrs)}")

    sets_list = [attr_sets[a] for a in attrs]

    # 2) Build diagram
    Diagram = VennDiagram
    if use_euler:
        from matplotlib_set_diagrams import EulerDiagram

        Diagram = EulerDiagram

    fig, ax = plt.subplots(figsize=(10, 8))
    # Use display names for set labels
    display_attrs = [get_attribute_display_name(attr) for attr in attrs]
    diagram = Diagram.from_sets(sets_list, set_labels=display_attrs, ax=ax)  # <- correct API

    # 3) Color pure regions by attribute (single-True masks), gentle alpha
    # masks are tuples of bools of length len(attrs), e.g. (1,0,0), (0,1,1), ...
    for i, a in enumerate(attrs):
        mask = tuple(1 if j == i else 0 for j in range(len(attrs)))
        artist = diagram.subset_artists.get(mask)
        if artist is not None and attr_colors:
            artist.set_facecolor(attr_colors.get(a, "#cccccc"))
            artist.set_alpha(0.45)

    # 4) Compute items per subset mask
    def subset_items(mask: tuple[int, ...]) -> list[str]:
        # Items present in all sets with mask==1 and absent from sets with mask==0
        present = (
            set.intersection(*(sets_list[i] for i, m in enumerate(mask) if m))
            if any(mask)
            else set()
        )
        for i, m in enumerate(mask):
            if not m:
                present -= sets_list[i]
        return sorted(present)

    # 5) Place labels evenly inside a polygon using a grid
    def place_labels_in_polygon(poly, labels: list[str]):
        if not labels:
            return
        # If the geometry is a MultiPolygon, pick the largest area part
        if hasattr(poly, "geoms"):
            poly = max(poly.geoms, key=lambda g: g.area)

        minx, miny, maxx, maxy = poly.bounds
        if not np.all(np.isfinite([minx, miny, maxx, maxy])):
            return

        # Grid size relative to bbox
        nx = max(4, grid_density)
        ny = max(4, int(grid_density * (maxy - miny) / max(1e-9, (maxx - minx))))
        pad_x = 0.05 * (maxx - minx)
        pad_y = 0.05 * (maxy - miny)
        xs = np.linspace(minx + pad_x, maxx - pad_x, nx)
        ys = np.linspace(miny + pad_y, maxy - pad_y, ny)

        # Checkerboard order to spread points
        pts = []
        for j, y in enumerate(ys):
            row = xs if j % 2 == 0 else xs[::-1]
            for x in row:
                p = Point(float(x), float(y))
                if poly.contains(p):
                    pts.append((x, y))
        if not pts:
            cx, cy = poly.representative_point().coords[0]
            pts = [(cx, cy)]

        n = (
            len(labels)
            if max_labels_per_region is None
            else min(len(labels), max_labels_per_region)
        )
        idxs = np.linspace(0, max(0, len(pts) - 1), num=n, dtype=int)

        for text, (x, y) in zip(labels[:n], [pts[k] for k in idxs]):
            wrapped = "\n".join(textwrap.wrap(text, width=26)) if len(text) > 28 else text
            ax.text(
                x,
                y,
                wrapped,
                ha="center",
                va="center",
                fontsize=base_fontsize,
                color="#111111",
                family="sans-serif",
                zorder=5,
            )

    # 6) Iterate subset geometries and label
    for mask, geom in diagram.subset_geometries.items():
        if geom is None:
            continue
        items = subset_items(mask)
        if not items:
            continue
        place_labels_in_polygon(geom, items)

    ax.set_title(title, fontfamily="serif", fontweight="bold", pad=10)
    ax.set_axis_off()
    fig.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"Saved: {out_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Stacked bar charts (domains/superdomains on X, attributes stacked)
# ---------------------------------------------------------------------------


def _truncate_label(s: str, max_len: int = 34) -> str:
    s = str(s)
    return s if len(s) <= max_len else (s[: max_len - 1] + "…")


def plot_stacked_bars(
    counts_pivot: pd.DataFrame,
    attr_colors: dict[str, str],
    title: str,  # kept for API compatibility; not drawn
    out_path: Path,
    *,
    top_n: int | None = 40,  # show top-N domains/superdomains by total count
    normalize: bool = False,  # True → stack shows per-domain percentages
    attr_order: list[str] | None = None,  # keep legend & stacking order stable
    rotate_xticks: int = 50,
):
    """
    counts_pivot: index = domain/superdomain, columns = attributes, values = counts
    """
    if counts_pivot is None or counts_pivot.empty:
        print("No data to plot for stacked bars.")
        return

    df = counts_pivot.copy()
    df = df.loc[(df.sum(axis=1) > 0)]
    if df.empty:
        print("All-zero rows; nothing to plot.")
        return

    # Consistent attribute order (use provided, else existing column order)
    cols = attr_order if attr_order else list(df.columns)

    # Sort domains by total descending and optionally limit to top-N
    totals = df.sum(axis=1)
    df = df.loc[totals.sort_values(ascending=False).index]
    if top_n is not None and top_n > 0:
        df = df.head(top_n)

    # Keep the column order consistent
    df = df.reindex(columns=cols)

    # Normalized stacks (percent per domain)
    if normalize:
        df = df.div(df.sum(axis=1), axis=0).fillna(0.0)
        # --- NEW: sort by the first attribute's share (descending) ---
        first_attr = cols[0]
        if first_attr in df.columns:
            df = df.sort_values(by=first_attr, ascending=False)

    # Figure
    plt.figure(figsize=(max(12, min(22, 0.38 * len(df))), 5))
    ax = plt.gca()

    x = np.arange(len(df.index))
    bottoms = np.zeros(len(df), dtype=float)

    # Legend order: by global mass (largest first) for readability
    attr_by_mass = sorted(cols, key=lambda c: float(df[c].sum()), reverse=True)

    # x  and y tick size larger
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)

    # Draw stacks
    for a in attr_by_mass:
        vals = df[a].astype(float).values
        bar = ax.bar(
            x,
            vals,
            bottom=bottoms,
            label=a,
            color=attr_colors.get(a, "#cccccc"),
            edgecolor="white",
            linewidth=0.6,
        )
        for p in bar:
            p.set_rasterized(True)
        bottoms = bottoms + vals

    # X labels (light truncation to avoid collisions)
    labels = [_truncate_label(idx) for idx in df.index.tolist()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate_xticks, ha="right")

    # Y label
    # ax.set_ylabel("Share per domain" if normalize else "Count")

    # Remove vertical lines in the grid
    ax.yaxis.grid(True, alpha=0.25, linewidth=1.0, color="lightgray")
    ax.xaxis.grid(False)

    # --- NEW: no title; legend centered, no border ---
    ncols = len(attr_by_mass)
    ax.legend(
        title=None,
        frameon=True if normalize else False,
        fontsize=13,
        loc="upper right",
        bbox_to_anchor=(0.95 if normalize else 0.9, 0.95),
        ncols=ncols,
        # Background color
        facecolor="white",
        framealpha=1,
    )

    # Styling
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=1.0, color="lightgray")

    if normalize:
        ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Attribute-colored word clouds from domain/superdomain mappings + t-SNE bubbles"
    )
    ap.add_argument(
        "--attr_paths", type=str, required=True, help="Comma-separated 'attribute:/path' pairs"
    )
    ap.add_argument("--output_dir", type=str, default="plots")
    ap.add_argument(
        "--domain_col", type=str, default="domain", help="Raw domain column name in data"
    )
    ap.add_argument(
        "--superdomain_col",
        type=str,
        default="superdomain",
        help="Raw superdomain column name in data",
    )
    ap.add_argument(
        "--domain_map_json", type=str, required=True, help="Path to domain mapping JSON"
    )
    ap.add_argument(
        "--superdomain_map_json", type=str, required=True, help="Path to superdomain mapping JSON"
    )
    ap.add_argument("--top_n", type=int, default=300, help="Top-N words in each word cloud")
    # New: t-SNE options
    ap.add_argument(
        "--tsne_method",
        type=str,
        choices=["embedding", "fuzzy"],
        default="embedding",
        help="Use sentence-transformer embeddings (default) or fuzzy similarity",
    )
    ap.add_argument("--tsne_size_floor", type=float, default=30.0)
    ap.add_argument("--tsne_size_ceiling", type=float, default=650.0)
    ap.add_argument("--tsne_annotate_top", type=int, default=20)
    args = ap.parse_args()

    attr_paths = parse_attr_paths(args.attr_paths)
    if not attr_paths:
        print("No attribute paths given.")
        sys.exit(1)

    alias = choose_output_alias(attr_paths)
    out_dir = Path(args.output_dir) / alias
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and invert JSON mappings
    domain_inv = load_and_invert_mapping(args.domain_map_json)
    super_inv = load_and_invert_mapping(args.superdomain_map_json)

    # Load data across attributes
    frames = []
    for attr, path in attr_paths:
        if not os.path.exists(path):
            print(f"[{attr}] WARNING: path does not exist → {path}")
        df = load_attr_df(attr, path)
        if not df.empty:
            frames.append(df)
    if not frames:
        print("No data gathered from provided paths.")
        sys.exit(0)

    df = pd.concat(frames, ignore_index=True)

    # Normalize & map domains
    raw_domain = df[args.domain_col].astype(str)
    norm_domain = raw_domain.apply(normalize_domain)
    canon_domain = norm_domain.apply(lambda s: domain_inv.get(s, "error"))

    # Map superdomains
    raw_super = df[args.superdomain_col].astype(str).str.strip()

    # Filter all not in super_inv
    filtered_super = raw_super[raw_super.isin(super_inv.keys())]

    canon_super = raw_super.apply(lambda s: super_inv.get(s, "error"))

    mapped_preview_domains = pd.DataFrame(
        {
            "raw_domain": raw_domain,
            "normalized_domain": norm_domain,
            "canonical_domain": canon_domain,
            "__attribute": df["__attribute"],
        }
    )
    mapped_preview_super = pd.DataFrame(
        {
            "raw_superdomain": raw_super,
            "canonical_superdomain": canon_super,
            "__attribute": df["__attribute"],
        }
    )

    mapped_preview_domains.to_csv(out_dir / "mapping_preview_domains.csv", index=False)
    mapped_preview_super.to_csv(out_dir / "mapping_preview_superdomains.csv", index=False)

    # Build long counts for domains and superdomains per attribute
    dom_counts = (
        mapped_preview_domains.groupby(["canonical_domain", "__attribute"])  # type: ignore[arg-type]
        .size()
        .reset_index(name="count")
        .rename(columns={"canonical_domain": "word"})
    )
    sup_counts = (
        mapped_preview_super.groupby(["canonical_superdomain", "__attribute"])  # type: ignore[arg-type]
        .size()
        .reset_index(name="count")
        .rename(columns={"canonical_superdomain": "word"})
    )

    # Save distribution CSVs (pivoted for readability)
    dom_pivot = dom_counts.pivot_table(
        index="word", columns="__attribute", values="count", aggfunc="sum", fill_value=0
    )
    sup_pivot = sup_counts.pivot_table(
        index="word", columns="__attribute", values="count", aggfunc="sum", fill_value=0
    )
    dom_pivot.sort_values(by=list(dom_pivot.columns), ascending=False).to_csv(
        out_dir / "distribution_domains_by_attribute.csv"
    )
    sup_pivot.sort_values(by=list(sup_pivot.columns), ascending=False).to_csv(
        out_dir / "distribution_superdomains_by_attribute.csv"
    )

    # Attribute colors (one color per attribute, stable across all plots)
    attributes = [a for a, _ in attr_paths]
    # Use the centralized attribute color mapping
    from vis_utilities import get_attribute_color, get_attribute_display_name

    attr_colors = {attr: get_attribute_color(attr) for attr in attributes}

    # Draw BOTH word clouds
    draw_wordcloud_allattrs(
        dom_counts,
        word_col="word",
        out_path=out_dir / "wordcloud_domains_allattrs.png",
        title="Domains Word Cloud • colored by primary attribute",
        attr_colors=attr_colors,
        top_n=args.top_n,
    )

    draw_wordcloud_allattrs(
        dom_counts,
        word_col="word",
        out_path=out_dir / "wordcloud_domains_allattrs.pdf",
        title="Domains Word Cloud • colored by primary attribute",
        attr_colors=attr_colors,
        top_n=args.top_n,
    )

    draw_wordcloud_allattrs(
        sup_counts,
        word_col="word",
        out_path=out_dir / "wordcloud_superdomains_allattrs.png",
        title="Superdomains Word Cloud • colored by primary attribute",
        attr_colors=attr_colors,
        top_n=args.top_n,
    )

    draw_wordcloud_allattrs(
        sup_counts,
        word_col="word",
        out_path=out_dir / "wordcloud_superdomains_allattrs.pdf",
        title="Superdomains Word Cloud • colored by primary attribute",
        attr_colors=attr_colors,
        top_n=args.top_n,
    )

    # ---------------- t-SNE bubble plots ----------------
    # Domains

    texts_for_tsne = []
    # Besides the name also append the strings for each value clustered under it

    # Order them by dom_pivot.index.tolist()
    listed_domains = dom_pivot.index.tolist()

    for dom in listed_domains:
        curr_str = str(dom).strip()
        vals = (
            mapped_preview_domains[mapped_preview_domains["canonical_domain"] == dom][
                "normalized_domain"
            ]
            .dropna()
            .unique()
            .tolist()
        )
        curr_str = " ".join(
            [curr_str] + [str(v).strip() for v in vals if str(v).strip() != curr_str]
        )

        texts_for_tsne.append(curr_str)

    dom_coords = _compute_tsne_coords(tokens=dom_pivot.index.tolist(), method=args.tsne_method)

    # replace the text column with the original listed_domains
    if dom_coords is not None and not dom_coords.empty:
        dom_coords = dom_coords.reset_index(drop=True)
        dom_coords["text"] = listed_domains

    if dom_coords is not None:
        _plot_tsne_bubbles(
            coords_df=dom_coords,
            counts_pivot=dom_pivot,
            attr_colors=attr_colors,
            title="Domains t-SNE • bubble size = usage within attribute",
            out_path=out_dir / "domains_tsne_bubbles.png",
            size_floor=args.tsne_size_floor,
            size_ceiling=args.tsne_size_ceiling,
            annotate_top=args.tsne_annotate_top,
        )

    # Superdomains

    texts_for_tsne = []
    # Besides the name also append the strings for each value clustered under it
    # Order them by sup_pivot.index.tolist()
    listed_superdomains = sup_pivot.index.tolist()
    for sup in listed_superdomains:
        curr_str = str(sup).strip()
        vals = (
            mapped_preview_super[mapped_preview_super["canonical_superdomain"] == sup][
                "raw_superdomain"
            ]
            .dropna()
            .unique()
            .tolist()
        )
        curr_str = " ".join(
            [curr_str] + [str(v).strip() for v in vals if str(v).strip() != curr_str]
        )

        texts_for_tsne.append(curr_str)

    sup_coords = _compute_tsne_coords(tokens=listed_superdomains, method=args.tsne_method)

    # replace the text column with the original listed_superdomains
    if sup_coords is not None and not sup_coords.empty:
        sup_coords = sup_coords.reset_index(drop=True)
        sup_coords["text"] = listed_superdomains

    if sup_coords is not None:
        _plot_tsne_bubbles(
            coords_df=sup_coords,
            counts_pivot=sup_pivot,
            attr_colors=attr_colors,
            title="Superdomains t-SNE • bubble size = usage within attribute",
            out_path=out_dir / "superdomains_tsne_bubbles.pdf",
            size_floor=args.tsne_size_floor,
            size_ceiling=args.tsne_size_ceiling,
            annotate_top=args.tsne_annotate_top,
        )

    # ---------------- Venn diagrams ----------------

    draw_attribute_venn_domains_even(
        dom_counts,
        word_col="word",
        attr_col="__attribute",
        attr_colors=attr_colors,
        title="Domain Overlap Across Attributes",
        out_path=out_dir / "venn_domains_by_attribute.pdf",
        min_count=1,
    )

    draw_attribute_venn_domains_even(
        sup_counts,
        word_col="word",
        attr_col="__attribute",
        attr_colors=attr_colors,
        title="Superdomain Overlap Across Attributes",
        out_path=out_dir / "venn_superdomains_by_attribute.pdf",
        min_count=1,
    )

    # ---------------- Stacked bar charts ----------------

    plot_stacked_bars(
        dom_pivot,
        attr_colors=attr_colors,
        title="Domains by Attribute — Stacked Counts",
        out_path=out_dir / "domains_stacked_bar_counts.pdf",
        top_n=40,
        normalize=False,
        attr_order=attributes,
    )
    plot_stacked_bars(
        dom_pivot,
        attr_colors=attr_colors,
        title="Domains by Attribute — Stacked Share",
        out_path=out_dir / "domains_stacked_bar_share.pdf",
        top_n=40,
        normalize=True,
        attr_order=attributes,
    )

    # Superdomains: counts + normalized share
    plot_stacked_bars(
        sup_pivot,
        attr_colors=attr_colors,
        title="Superdomains by Attribute — Stacked Counts",
        out_path=out_dir / "superdomains_stacked_bar_counts.pdf",
        top_n=30,
        normalize=False,
        attr_order=attributes,
    )
    plot_stacked_bars(
        sup_pivot,
        attr_colors=attr_colors,
        title="Superdomains by Attribute — Stacked Share",
        out_path=out_dir / "superdomains_stacked_bar_share.pdf",
        top_n=30,
        normalize=True,
        attr_order=attributes,
    )


if __name__ == "__main__":
    main()
