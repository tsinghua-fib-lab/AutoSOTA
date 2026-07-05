# =========================================================
# PC vs LocPC – Heatmaps and Animation Script
# =========================================================

# =========================================================
# Imports and General Setup
# =========================================================
import random
from collections import deque

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# =========================================================
# Reproducibility
# =========================================================
SEED = 2026
random.seed(SEED)
np.random.seed(SEED)


# =========================================================
# External Imports
# =========================================================
from reproducibility.clear2026.dags_generator import (
    random_DAG_identifiable_CDE,
    random_DAG_nonidentifiable_CDE
)

from pyciphod.causal_discovery.basic.pc.pc import PC
from pyciphod.causal_discovery.local.locpc import LocPC


# =========================================================
# Linear SCM Simulation
# =========================================================
def simulate_linear_scm_from_dag(dag, n_obs=10000,
                                 coef_range=(-1, 1),
                                 sigma_range=(0.5, 1.0)):
    """
    Simulate data from a linear SCM defined by a DAG.
    """
    nodes = list(nx.topological_sort(dag))
    p = len(nodes)

    adj = nx.to_numpy_array(dag, nodelist=nodes)
    coef = np.random.uniform(*coef_range, size=(p, p)) * adj.T
    sigma_vec = np.random.uniform(*sigma_range, size=p)

    coef_inv = np.linalg.inv(np.eye(p) - coef)
    data = np.random.normal(0, sigma_vec, size=(n_obs, p)) @ coef_inv.T

    return pd.DataFrame(data, columns=nodes)


# =========================================================
# Linear SCM Simulation (coef + data version)
# =========================================================
def simulate_linearSCM_from_dag(dag, n_obs=1,
                                coef_range=(-1,1),
                                sigma_range=(0.5,1)):
    nodes = list(nx.topological_sort(dag))
    p = len(nodes)

    adj = nx.to_numpy_array(dag, nodelist=nodes)
    coef = np.random.uniform(coef_range[0], coef_range[1], size=(p,p)) * adj.T
    sigma_vec = np.random.uniform(*sigma_range, size=p)

    coef_inv = np.linalg.inv(np.eye(p) - coef)
    data = np.random.normal(0, sigma_vec, size=(n_obs, p)) @ coef_inv.T

    return (
        pd.DataFrame(coef, index=nodes, columns=nodes),
        pd.DataFrame(data, columns=nodes)
    )


# =========================================================
# Neighborhood computation
# =========================================================
def compute_neighborhood(graph: nx.DiGraph, target: str):
    """
    Compute hop-distance from target in undirected graph.
    Returns dict {node: hop_distance}.
    """
    undirected = graph.to_undirected()

    visited = {target}
    distances = {target: 0}
    queue = deque([(target, 0)])

    while queue:
        node, dist = queue.popleft()
        for neigh in undirected.neighbors(node):
            if neigh not in visited:
                visited.add(neigh)
                distances[neigh] = dist + 1
                queue.append((neigh, dist + 1))

    return distances


def neighborhood(g: nx.DiGraph, target: str):
    """
    Alternative neighborhood function (same logic).
    """
    if target not in g:
        raise ValueError(f"Target node '{target}' must be in the graph")

    undirected = g.to_undirected()
    visited = {target}
    neighborhood_dict = {target: 0}
    queue = deque([(target, 0)])

    while queue:
        node, h = queue.popleft()
        for neighbor in undirected.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                neighborhood_dict[neighbor] = h + 1
                queue.append((neighbor, h + 1))

    return neighborhood_dict


# =========================================================
# Heatmap Plotting
# =========================================================
def plot_pc_vs_locpc_heatmaps(pc_obj, locpc_list, target, graph):
    neighborhood = compute_neighborhood(graph, target)
    nodes = list(graph.nodes)

    # Tri des noeuds selon la distance au target
    sorted_nodes = sorted(
        nodes,
        key=lambda x: neighborhood.get(x, max(neighborhood.values()) + 1)
    )
    n = len(sorted_nodes)
    node_to_idx = {node: i for i, node in enumerate(sorted_nodes)}

    objects = [pc_obj] + locpc_list

    # Construire une grille complète (pas triangulaire)
    def build_grid(test_list):
        grid = np.zeros((n, n))
        for i, j, _ in test_list:
            xi = node_to_idx[i]
            yi = node_to_idx[j]
            grid[yi, xi] += 1
        return grid

    grids = [build_grid(obj.performed_tests) for obj in objects]

    # Valeur max globale pour uniformiser les couleurs
    global_max = max(grid.max() for grid in grids)
    vmax = global_max if global_max > 0 else 1

    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    titles = ["PC", "LocPC hop=1", "LocPC hop=2", "LocPC hop=3"]

    for ax, grid, title, obj in zip(axes.flat, grids, titles, objects):
        im = ax.imshow(grid, cmap="Reds", vmin=0, vmax=vmax)
        ax.invert_yaxis()
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(sorted_nodes, rotation=90, fontsize=8)
        ax.set_yticklabels(sorted_nodes, fontsize=8)
        ax.set_xlabel("Node i")
        ax.set_ylabel("Node j")
        ax.set_title(title, fontsize = 20)
        ax.invert_yaxis()

        ax.text(
            0, -0.2,
            f"Total # of tests: {obj.nb_ci_tests}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=16,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
        )
    
    plt.tight_layout()
    plt.show()



# =========================================================
# Animation
# =========================================================
def animate_pc_vs_locpc_hops(pc_obj, locpc_objs, target, g,
                             interval=100, decay=0.5):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import numpy as np

    neighborhood_dict = neighborhood(g, target)
    all_nodes = list(g.nodes)

    sorted_nodes = sorted(
        all_nodes,
        key=lambda x: neighborhood_dict.get(
            x, max(neighborhood_dict.values(), default=0)+1
        )
    )

    n = len(sorted_nodes)
    node_to_idx = {node: i for i, node in enumerate(sorted_nodes)}

    fig, axes = plt.subplots(1, 4, figsize=(18,5))
    ims = []
    grids = [np.zeros((n,n)) for _ in range(4)]
    
    names = ["PC"] + [f"LocPC hop={h}" for h in [1,2,3]]
    counters_text = []

    for ax, name in zip(axes, names):
        im = ax.imshow(np.zeros((n,n)), cmap='Reds', vmin=0, vmax=1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(sorted_nodes, rotation=90, fontsize=8)
        ax.set_yticklabels(sorted_nodes, fontsize=8)
        ax.set_xlabel("Node i")
        ax.set_ylabel("Node j")
        ax.set_title(name)
        ax.invert_yaxis()
        ims.append(im)
        
        # Texte compteur en dessous de l'axe x
        txt = ax.text(0.8, 0.92, "Tests: 0", transform=ax.transAxes,
                      ha="center", fontsize=10, color='blue')
        counters_text.append(txt)

    pc_tests = [(node_to_idx[i], node_to_idx[j])
                for (i,j,_) in pc_obj.performed_tests]

    locpc_tests_list = [
        [(node_to_idx[i], node_to_idx[j])
         for (i,j,_) in loc_obj.performed_tests]
        for loc_obj in locpc_objs
    ]

    n_frames = max(len(pc_tests),
                   *(len(tests) for tests in locpc_tests_list))

    def update(frame):

        for k in range(4):
            grids[k] *= decay

        if frame < len(pc_tests):
            x, y = pc_tests[frame]
            grids[0][y, x] = 1
            counters_text[0].set_text(f"Tests: {frame+1}")

        for k, loc_tests in enumerate(locpc_tests_list):
            if frame < len(loc_tests):
                x, y = loc_tests[frame]
                grids[k+1][y, x] = 1
                counters_text[k+1].set_text(f"Tests: {frame+1}")

        for im_idx, im in enumerate(ims):
            im.set_data(grids[im_idx])

        return ims + counters_text

    ani = animation.FuncAnimation(
        fig, update,
        frames=n_frames,
        interval=interval,
        blit=True,
        repeat=False
    )


    plt.show()


# =========================================================
# Main Execution
# =========================================================
def main():

    # ---------------------------------
    # Generate identifiable DAG
    # ---------------------------------
    n = 25
    m = 4
    g, y, x = random_DAG_identifiable_CDE(n, m/(n+1))

    # ---------------------------------
    # Simulate data
    # ---------------------------------
    data = simulate_linear_scm_from_dag(g, n_obs=10000)

    # ---------------------------------
    # Run PC
    # ---------------------------------
    pc = PC(data=data)
    pc.run()

    # ---------------------------------
    # Run LocPC (hop 1,2,3)
    # ---------------------------------
    locpc_list = []
    for hop in [1,2,3]:
        locpc_obj = LocPC(data)
        locpc_obj.run(target=y, hop=hop)
        locpc_list.append(locpc_obj)

    # ---------------------------------
    # Static heatmaps
    # ---------------------------------
    plot_pc_vs_locpc_heatmaps(
        pc,
        locpc_list,
        target=y,
        graph=g
    )



# =========================================================
if __name__ == "__main__":
    main()