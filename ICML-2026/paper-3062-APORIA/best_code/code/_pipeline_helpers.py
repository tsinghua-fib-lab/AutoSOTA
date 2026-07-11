"""
Figure-1 (pipeline overview) helpers.

This module exists solely to keep ``Pipeline.ipynb`` readable: the
bespoke plotting and projection code for Figure 1 lives here, while
the notebook only orchestrates data loading, the structural-analysis
call, and the panel drawing.  Nothing here is reused in other
notebooks; promote to ``aporia`` if that changes.

Public surface: every top-level def/class is re-exported.
"""

from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.manifold import TSNE
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import pdist, cdist
from sklearn.model_selection import StratifiedShuffleSplit

import aporia as ap

from aporia.plotting import EDGE_COLORS, VERTEX_COLORS, BORDER_COLOR

def train_test_split_80_20(X, y, random_state=42):
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=0.2, random_state=random_state
    )
    trn_idx, tst_idx = next(sss.split(X, y))
    return X[trn_idx], X[tst_idx], y[trn_idx], y[tst_idx], trn_idx, tst_idx


def get_distance_block(geometry_store, key, space):
    """
    Returns a dict with keys D_GG, D_HH, D_GH
    """
    d = geometry_store[key]

    key = {'embedding': '', 'fisher': '_z'}

    if space in d:  # nested
        return d[space]

    # flattened fallback
    return {
        "D_GG": d[f"D_GG{key[space]}"],
        "D_HH": d[f"D_HH{key[space]}"],
        "D_GH": d[f"D_GH{key[space]}"],
    }


def plot_distance_violin(
    geometry_store,
    key,
    space="embedding",
    colors=None,
    figsize=(5, 4),
    savepath=None,
    transparent=False,
    rotate=False
):
    """
    Half-violin plot for GG / HH and boxplot for GH
    in the selected space ('embedding' or 'fisher')
    """
    if colors is None:
        colors = EDGE_COLORS

    d = get_distance_block(geometry_store, key, space)

    fig, ax = plt.subplots(figsize=figsize)

    # ---- GG: upper half violin ----
    vp = ax.violinplot(
        d["D_GG"],
        positions=[0],
        widths=0.5,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        side="high",
        orientation="horizontal",
    )
    for pc in vp["bodies"]:
        pc.set_facecolor(colors["GG"])
        pc.set_edgecolor(BORDER_COLOR)
        pc.set_linewidth(1.0)
        pc.set_alpha(0.5)

    # ---- HH: lower half violin ----
    vp = ax.violinplot(
        d["D_HH"],
        positions=[0],
        widths=0.5,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        side="low",
        orientation="horizontal",
    )
    for pc in vp["bodies"]:
        pc.set_facecolor(colors["HH"])
        pc.set_edgecolor(BORDER_COLOR)
        pc.set_linewidth(1.0)
        pc.set_alpha(0.5)

    # ---- GH: boxplot ----
    bp = ax.boxplot(
        d["D_GH"],
        positions=[0],
        widths=0.1,
        patch_artist=True,
        showfliers=False,
        orientation="horizontal",
    )
    for box in bp["boxes"]:
        box.set_facecolor(colors["GH"])
        box.set_edgecolor(BORDER_COLOR)
        box.set_alpha(0.9)

    for elem in ["whiskers", "caps", "medians"]:
        for artist in bp[elem]:
            artist.set_color(BORDER_COLOR)

    xmin, xmax = ax.get_xlim()
    y0 = 0

    ax.annotate(
        r"$L_2$",
        xy=(-.001, y0),
        xytext=(xmax*1.1, y0),
        # textcoords="offset points",
        arrowprops=dict(
            arrowstyle="<-",
            color="0.6",
            linewidth=1.2,
            shrinkA=0,
            shrinkB=0,
        ),
        ha="left",
        va="center",
        color="0.4",
        rotation=-90 if rotate else 0,
        fontsize=40,
        zorder=0,
    )

    ax.axvline(0, 0.25, 0.75, color='grey')

    ax.set_axis_off()

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", pad_inches=0, transparent=transparent)

    return fig, ax


def tsne_projection(
    X_train, y_train,
    X_test, y_test,
    perplexity=30,
    random_state=42
):
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )

    Z_all = tsne.fit_transform(X_all)

    n_tr = len(X_train)
    Z_tr, Z_te = Z_all[:n_tr], Z_all[n_tr:]

    return Z_tr, Z_te, y_train, y_test


def build_tsne_complete_graph(Z, y):
    """
    Z : (n,2) t-SNE coordinates
    y : (n,) class labels {0,1}
    """
    G = nx.Graph()

    # ---- nodes ----
    for i, (pos, cls) in enumerate(zip(Z, y)):
        G.add_node(
            i,
            pos=tuple(pos),
            cls="G" if cls == 0 else "H",
        )

    # ---- complete edges ----
    for i, j in itertools.combinations(range(len(Z)), 2):
        ci, cj = y[i], y[j]

        if ci == 0 and cj == 0:
            etype = "GG"
        elif ci == 1 and cj == 1:
            etype = "HH"
        else:
            etype = "GH"

        G.add_edge(i, j, etype=etype)

    return G


def plot_tsne_complete_graph(
    G,
    ax=None,
    node_size=30,
    edge_alphas={
        "GG": 0.1,
        "HH": 0.08,
        "GH": 0.04,
    },
    edge_lw=0.6,
    colors=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure

    if colors is None:
        colors = VERTEX_COLORS

    pos = nx.get_node_attributes(G, "pos")
    cls = nx.get_node_attributes(G, "cls")

    # ---- edges by type ----
    if edge_lw is not None:
        for etype, color in EDGE_COLORS.items():
            edges = [
                (u, v)
                for u, v, d in G.edges(data=True)
                if d["etype"] == etype
            ]

            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=edges,
                edge_color=color,
                alpha=edge_alphas[etype],
                width=edge_lw,
                ax=ax,
            )

    # ---- nodes ----
    for c in ["G", "H"]:
        nodes = [n for n in G.nodes if cls[n] == c]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodes,
            node_color=colors[c],
            node_size=node_size,
            node_shape='s' if c == "G" else 'o',
            alpha=0.8,
            ax=ax,
        )

    ax.set_axis_off()
    return fig, ax


def build_tsne_star_graph(
    Z_train,
    y_train,
    z_test,
):
    """
    Star graph centred at test point

    Z_train : (n,2)
    y_train : (n,)
    z_test  : (2,)
    """
    G = nx.Graph()

    test_id = "test"

    # ---- test node ----
    G.add_node(
        test_id,
        pos=tuple(z_test),
        cls="test",
    )

    # ---- training nodes + star edges ----
    for i, (pos, cls) in enumerate(zip(Z_train, y_train)):
        cls = "G" if cls == 0 else "H"
        G.add_node(
            i,
            pos=tuple(pos),
            cls=cls,
        )

        G.add_edge(
            test_id,
            i,
            etype=cls,   # 0=G, 1=H
        )

    return G


def plot_tsne_star_graph(
    G,
    ax=None,
    node_size=30,
    test_size=400,
    edge_alpha=0.5,
    edge_lw=0.9,
    title=None
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure

    pos = nx.get_node_attributes(G, "pos")
    cls = nx.get_node_attributes(G, "cls")

    # ---- edges (test → train) ----
    if edge_lw is not None:
        for cls_id, color in VERTEX_COLORS.items():
            edges = [
                (u, v)
                for u, v, d in G.edges(data=True)
                if d["etype"] == cls_id
            ]

            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=edges,
                edge_color=color,
                alpha=edge_alpha,
                width=edge_lw,
                ax=ax,
            )

    # ---- training nodes ----
    for c in ["G", "H"]:
        nodes = [n for n in G.nodes if cls.get(n) == c]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodes,
            node_color=VERTEX_COLORS[c],
            node_size=node_size,
            node_shape='s' if c == "G" else 'o',
            alpha=0.8,
            ax=ax,
        )

    # ---- test node (black star) ----
    if test_size is not None:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=["test"],
            node_color="black",
            node_size=test_size,
            node_shape="*",
            linewidths=1.2,
            edgecolors="white",
            ax=ax,
        )
    
    if title is not None:
        ax.set_title(title)

    ax.set_axis_off()
    return fig, ax


def fisher_point_to_oblique(z, angle_deg=-30):
    theta = np.deg2rad(angle_deg)
    d = np.array([np.cos(theta), np.sin(theta)])
    return z * d


def fisher_to_oblique(z, angle_deg=-30):
    """
    Map 1D Fisher coordinates to 2D points on an oblique line
    """
    theta = np.deg2rad(angle_deg)
    direction = np.array([np.cos(theta), np.sin(theta)])
    return z[:, None] * direction[None, :]


def add_orthogonal_jitter(Z, scale=0.01):
    direction = Z.mean(axis=0)
    direction /= np.linalg.norm(direction)
    perp = np.array([-direction[1], direction[0]])
    return Z + scale * np.random.randn(len(Z), 1) * perp


def plot_fisher_with_optional_test(
    Z_train_j, y_train,
    z_test=None,
    angle_deg=-30,
    ax=None,
    train_size=22,
    test_size=400,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure

    # ---- training points (jittered) ----
    for cls_id, cls in enumerate(["G", "H"]):
        mask = (y_train == cls_id)
        ax.scatter(
            Z_train_j[mask, 0],
            Z_train_j[mask, 1],
            s=train_size,
            alpha=0.6,
            c=VERTEX_COLORS[cls],
            marker='s' if cls == "G" else 'o',
            zorder=2,
        )

    # ---- optional test point (NO jitter, on axis) ----
    if z_test is not None:
        Zt = fisher_point_to_oblique(z_test, angle_deg=angle_deg)
        ax.scatter(
            Zt[0],
            Zt[1],
            s=test_size,
            c="black",
            marker="*",
            edgecolor="white",
            linewidth=1.2,
            zorder=5,
        )

    ax.set_axis_off()
    return fig, ax


def draw_fisher_axis(ax, z, angle_deg=-30, lw=1.5):
    """
    Draw the Fisher axis corresponding to 1D coordinates z
    """
    theta = np.deg2rad(angle_deg)
    direction = np.array([np.cos(theta), np.sin(theta)])

    t_min, t_max = z.min(), z.max()
    t_min-=.05
    t_max+=.05
    line = np.vstack([
        t_min * direction,
        t_max * direction
    ])

    ax.plot(
        line[:, 0],
        line[:, 1],
        "--",
        color="grey",
        lw=lw,
        alpha=0.6,
        zorder=0,
    )


def draw_fisher_violin(
    ax,
    z,
    angle_deg=-30,
    width=0.08,
    color="C0",
    alpha=0.35,
    zorder=1,
    dotop=True,
    dobot=True
):
    """
    Draw a half-symmetric violin along the Fisher axis
    """
    from scipy.stats import gaussian_kde
    
    theta = np.deg2rad(angle_deg)
    d = np.array([np.cos(theta), np.sin(theta)])
    d_perp = np.array([-d[1], d[0]])

    # KDE in Fisher space
    kde = gaussian_kde(z)

    z_grid = np.linspace(z.min(), z.max(), 300)
    density = kde(z_grid)
    density = density / density.max() * width

    centerline = z_grid[:, None] * d[None, :]

    upper = centerline + density[:, None] * d_perp
    lower = centerline - density[:, None] * d_perp

    if dotop and dobot:
        violin = np.vstack([upper, lower[::-1]])
    elif dotop:
        violin = np.vstack([(upper[0]+lower[0])/2, upper, (upper[-1]+lower[-1])/2])
    elif dobot:
        violin = np.vstack([(upper[0]+lower[0])/2, lower, (upper[-1]+lower[-1])/2])
    else:
        raise Exception()

    ax.fill(
        violin[:, 0],
        violin[:, 1],
        color=color,
        alpha=alpha,
        linewidth=0,
        zorder=zorder,
    )


class FisherDistanceExtractor:
    def __init__(self, lambda_reg=1e-3, normalise=True, normalise_by_trace=True):
        self.lambda_reg = lambda_reg
        self.normalise = normalise
        self.normalise_by_trace = normalise_by_trace

    def fit(self, X_train, y_train):
        X_G, X_H = ap.split_by_label(X_train, y_train)

        # store original embeddings
        self.X_G = X_G
        self.X_H = X_H

        # Fisher direction
        self.v = ap.fisher_direction(
            X_G, X_H,
            lambda_reg=self.lambda_reg,
            normalise=self.normalise,
            normalise_by_trace=self.normalise_by_trace
        )

        # Fisher projections
        self.Z_G = (X_G @ self.v)[:, None]
        self.Z_H = (X_H @ self.v)[:, None]

        # reference intra-class distances
        self.ref_G_fisher = pdist(self.Z_G)
        self.ref_H_fisher = pdist(self.Z_H)

        self.ref_G_embed = pdist(self.X_G)
        self.ref_H_embed = pdist(self.X_H)

    def extract_test_distances(self, X_test, y_test):
        rows = []

        for test_id, (x, y_true) in enumerate(zip(X_test, y_test)):
            z = (x @ self.v).reshape(1, 1)

            # Fisher distances
            dG_f = cdist(z, self.Z_G).ravel()
            dH_f = cdist(z, self.Z_H).ravel()

            # Embedding distances
            dG_e = cdist(x[None, :], self.X_G).ravel()
            dH_e = cdist(x[None, :], self.X_H).ravel()

            for j in range(len(dG_f)):
                rows.append({
                    "test_id": test_id,
                    "y_true": int(y_true),
                    "target_class": "G",
                    "train_id": j,
                    "distance_fisher": dG_f[j],
                    "distance_embed": dG_e[j],
                })

            for j in range(len(dH_f)):
                rows.append({
                    "test_id": test_id,
                    "y_true": int(y_true),
                    "target_class": "H",
                    "train_id": j,
                    "distance_fisher": dH_f[j],
                    "distance_embed": dH_e[j],
                })

        # ---- reference geometry (optional but useful) ----
        for j, d in enumerate(self.ref_G_fisher):
            rows.append({
                "test_id": None,
                "y_true": None,
                "target_class": "G_ref",
                "train_id": j,
                "distance_fisher": d,
                "distance_embed": self.ref_G_embed[j],
            })

        for j, d in enumerate(self.ref_H_fisher):
            rows.append({
                "test_id": None,
                "y_true": None,
                "target_class": "H_ref",
                "train_id": j,
                "distance_fisher": d,
                "distance_embed": self.ref_H_embed[j],
            })

        return pd.DataFrame(rows)
    
    def extract_wasserstein_scores(self, X_test, y_test):
        rows = []

        for test_id, (x, y_true) in enumerate(zip(X_test, y_test)):
            z = (x @ self.v).reshape(1, 1)

            # Fisher distances
            D_G = cdist(z, self.Z_G).ravel()
            D_H = cdist(z, self.Z_H).ravel()

            # Wasserstein distances
            W_G = (
                wasserstein_distance(D_G, self.ref_G_fisher)
                if len(self.ref_G_fisher) > 0 else np.inf
            )
            W_H = (
                wasserstein_distance(D_H, self.ref_H_fisher)
                if len(self.ref_H_fisher) > 0 else np.inf
            )

            y_pred = 0 if W_G <= W_H else 1
            correct = int(y_pred == y_true)

            rows.append({
                "test_id": test_id,
                "y_true": int(y_true),
                "y_pred": int(y_pred),
                "correct": correct,
                "W_G": W_G,
                "W_H": W_H,
                "margin": W_H - W_G,  # positive → G
            })

        return pd.DataFrame(rows)


def generate_fisher_distance_table(
    df,
    model_id,
    prompt_id,
    cfg,
    lambda_reg=1,
    random_state=42
):
    X, y = ap.extract_prompt_data(df, model_id, prompt_id, cfg)

    X_trn, X_tst, y_trn, y_tst, _, tstIdxs = train_test_split_80_20(
        X, y, random_state=random_state
    )

    extractor = FisherDistanceExtractor(lambda_reg=lambda_reg)
    extractor.fit(X_trn, y_trn)

    dist_df = extractor.extract_test_distances(X_tst, y_tst)
    wass_df = extractor.extract_wasserstein_scores(X_tst, y_tst)

    for d in (dist_df, wass_df):
        d["model_id"] = model_id
        d["prompt_id"] = prompt_id

    return dist_df, wass_df


def get_test_distance_block(dist_df, test_id, space="fisher"):
    """
    space ∈ {"fisher", "embedding"}
    """
    col = "distance_fisher" if space == "fisher" else "distance_embed"

    dG = dist_df[
        (dist_df["test_id"] == test_id) &
        (dist_df["target_class"] == "G")
    ][col].values

    dH = dist_df[
        (dist_df["test_id"] == test_id) &
        (dist_df["target_class"] == "H")
    ][col].values

    return {
        "D_G": dG,
        "D_H": dH,
    }


def plot_test_distance_violin(
    dist_df,
    test_id,
    space="fisher",
    colors=None,
    figsize=(5, 4),
    savepath=None,
    title="",
    transparent=False,
    rotate=False
):
    if colors is None:
        colors = VERTEX_COLORS

    d = get_test_distance_block(dist_df, test_id, space=space)

    fig, ax = plt.subplots(figsize=figsize)

    # upper: G
    vp = ax.violinplot(
        d["D_G"],
        positions=[0],
        widths=0.5,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        side="high",
        orientation="horizontal",
    )
    for pc in vp["bodies"]:
        pc.set_facecolor(colors["G"])
        print(BORDER_COLOR)
        pc.set_edgecolor(BORDER_COLOR)
        pc.set_linewidth(1.0)
        pc.set_alpha(0.5)

    # lower: H
    vp = ax.violinplot(
        d["D_H"],
        positions=[0],
        widths=0.5,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        side="low",
        orientation="horizontal",
    )
    for pc in vp["bodies"]:
        pc.set_facecolor(colors["H"])
        pc.set_edgecolor(BORDER_COLOR)
        pc.set_linewidth(1.0)
        pc.set_alpha(0.5)

    # ---- axis arrow ----
    xmin, xmax = ax.get_xlim()
    ax.annotate(
        r"$L_2$",
        xy=(-0.001, 0),
        xytext=(xmax * 1.1, 0),
        arrowprops=dict(
            arrowstyle="<-",
            color="0.6",
            linewidth=1.2,
            shrinkA=0,
            shrinkB=0,
        ),
        ha="left",
        va="center",
        color="0.4",
        rotation=-90 if rotate else 0,
        fontsize=40,
        zorder=0,
    )

    if title is not None:
        ax.set_title(title)

    # ---- zero reference ----
    ax.axvline(0, 0.25, 0.75, color="grey")

    ax.set_axis_off()

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", pad_inches=0, transparent=transparent)

    return fig, ax
