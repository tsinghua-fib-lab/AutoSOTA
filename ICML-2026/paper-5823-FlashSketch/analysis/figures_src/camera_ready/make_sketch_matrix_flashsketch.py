from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from pathlib import Path
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

import globals as g
from analysis.figures_src.camera_ready.utils import maybe_copy_pdf_to_paper
from io_utils import ensure_dir, write_json
from logging_utils import get_logger
from provenance import get_git_state, get_tree_hashes
from sketches.flashsketch import FlashSketchConfig, sketch as flashsketch


_LOGGER = get_logger("fig.camera-ready.sketch-matrix")
TREE_HASH_PATHS = ("bench", "sketches", "kernels", "data")

M_BLOCKS = 16
BR = 64
BC = 128
KAPPA = 4
S = 2
SEED = 2
BASE_HEIGHT_IN = 6.5


def _block_nonzero_mask(matrix, br, bc):
    """Return a boolean mask of nonzero blocks."""
    k, d = matrix.shape
    if k % br != 0 or d % bc != 0:
        raise ValueError("Matrix shape must be divisible by block size.")
    m_rows = k // br
    m_cols = d // bc
    blocks = matrix.reshape(m_rows, br, m_cols, bc)
    return np.any(blocks != 0, axis=(1, 3))


def _hopcroft_karp(adj, n_left, n_right):
    """Return a maximum matching using Hopcroft-Karp."""
    match_left = [-1] * n_left
    match_right = [-1] * n_right
    dist = [0] * n_left
    inf = n_left + n_right + 5

    def bfs():
        queue = deque()
        for u in range(n_left):
            if match_left[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = inf
        dist_nil = inf
        while queue:
            u = queue.popleft()
            if dist[u] >= dist_nil:
                continue
            for v in adj[u]:
                mate = match_right[v]
                if mate == -1:
                    dist_nil = dist[u] + 1
                elif dist[mate] == inf:
                    dist[mate] = dist[u] + 1
                    queue.append(mate)
        return dist_nil != inf

    def dfs(u):
        for v in adj[u]:
            mate = match_right[v]
            if mate == -1 or (dist[mate] == dist[u] + 1 and dfs(mate)):
                match_left[u] = v
                match_right[v] = u
                return True
        dist[u] = inf
        return False

    matching = 0
    while bfs():
        for u in range(n_left):
            if match_left[u] == -1 and dfs(u):
                matching += 1
    return match_left, match_right, matching


def _decompose_permutations(mask, kappa):
    """Return permutation labels (1..kappa) for each nonzero block."""
    n_rows, n_cols = mask.shape
    adj_sets = [set(np.where(mask[row])[0]) for row in range(n_rows)]
    perm_labels = np.zeros((n_rows, n_cols), dtype=int)

    for perm_idx in range(1, kappa + 1):
        adj = [sorted(adj_sets[row]) for row in range(n_rows)]
        match_left, _, matching = _hopcroft_karp(adj, n_rows, n_cols)
        if matching != n_rows:
            raise ValueError("Failed to find perfect matching for permutation decomposition.")
        for row, col in enumerate(match_left):
            perm_labels[row, col] = perm_idx
            adj_sets[row].remove(col)
    return perm_labels


def _plot_colored_perms(sign_matrix, perm_labels, br, bc, output_path):
    """Plot sketch matrix with permutation-colored blocks."""
    k, d = sign_matrix.shape
    aspect = d / max(k, 1)
    fig_width = BASE_HEIGHT_IN * aspect
    fig, ax = plt.subplots(figsize=(fig_width, BASE_HEIGHT_IN), constrained_layout=True)

    cmap = LinearSegmentedColormap.from_list(
        "red_white_green", [(1.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 0.6, 0.0)], N=256
    )
    ax.imshow(
        sign_matrix,
        interpolation="nearest",
        aspect="equal",
        vmin=-1.0,
        vmax=1.0,
        cmap=cmap,
    )

    kappa = int(perm_labels.max())
    color_map = plt.colormaps.get_cmap("tab10").resampled(max(kappa, 1))
    overlay = np.zeros((k, d, 4), dtype=float)
    for row in range(perm_labels.shape[0]):
        r0 = row * br
        r1 = r0 + br
        for col in range(perm_labels.shape[1]):
            perm = perm_labels[row, col]
            if perm <= 0:
                continue
            c0 = col * bc
            c1 = c0 + bc
            color = list(color_map(perm - 1))
            color[3] = 0.25
            overlay[r0:r1, c0:c1, :] = color

    ax.imshow(overlay, interpolation="nearest", aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])

    output_path = Path(output_path)
    fig.savefig(output_path, bbox_inches="tight")
    png_path = output_path.with_suffix(".png")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return output_path, png_path


def run():
    """Generate a camera-ready sketch matrix example."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for sketch matrix visualization.")

    d = M_BLOCKS * BC
    k = M_BLOCKS * BR

    A = torch.eye(d, device="cuda", dtype=torch.float32)
    output_dir = g.FIGURES_DIR() / "camera_ready_sketch_matrices"
    ensure_dir(output_dir)

    cfg = FlashSketchConfig(
        k=k,
        kappa=KAPPA,
        s=S,
        seed=SEED,
        dtype=g.DTYPE_FP32,
        block_rows=BR,
    )
    SA = flashsketch(A, cfg)
    SA_cpu = SA.detach().cpu().numpy()
    sign_matrix = np.sign(SA_cpu)
    nonzero_per_col = np.count_nonzero(sign_matrix, axis=0)
    avg_nonzeros_per_col = float(nonzero_per_col.mean())

    mask = _block_nonzero_mask(sign_matrix, BR, BC)
    perm_labels = _decompose_permutations(mask, KAPPA)

    output_path = output_dir / "fig_camera_ready_flashsketch_m16_k4_seed2.pdf"
    pdf_path, png_path = _plot_colored_perms(sign_matrix, perm_labels, BR, BC, output_path)
    _LOGGER.info("Generated %s", pdf_path)
    _LOGGER.info("Generated %s", png_path)
    maybe_copy_pdf_to_paper(pdf_path)

    manifest = {
        "git": get_git_state(),
        "tree_hashes": get_tree_hashes(TREE_HASH_PATHS),
        "config": {
            "m_blocks": M_BLOCKS,
            "br": BR,
            "bc": BC,
            "kappa": KAPPA,
            "s": S,
            "seed": SEED,
            "k": k,
            "d": d,
        },
        "avg_nonzeros_per_col": avg_nonzeros_per_col,
        "outputs": [str(pdf_path), str(png_path)],
    }
    write_json(output_dir / "manifest.json", manifest)


if __name__ == "__main__":
    run()
