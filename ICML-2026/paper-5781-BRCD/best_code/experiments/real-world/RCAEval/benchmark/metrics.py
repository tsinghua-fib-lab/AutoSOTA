from itertools import combinations

import networkx as nx

from RCAEval.classes.graph import MemoryGraph, Node


def _has_both_edges(graph, x, y):
    return graph.has_edge(x, y) and graph.has_edge(y, x)


def _has_any_edge(graph, x, y):
    return graph.has_edge(x, y) or graph.has_edge(y, x)


def _has_only_edge(graph, x, y):
    return graph.has_edge(x, y) and (not graph.has_edge(y, x))


def _has_no_edge(graph, x, y):
    return not (graph.has_edge(x, y) or graph.has_edge(y, x))


def _has_one_edge(graph, x, y):
    return graph.has_edge(x, y) ^ graph.has_edge(y, x)


def SHD(G1: MemoryGraph, G2: MemoryGraph):
    """Calculates Structural Hamming Distance between two graphs.
    Args:
        G1: a graph (as a networkx.Graph or networkx.DiGraph)
        G2: a graph (as a networkx.Graph or networkx.DiGraph)
    Returns:
        SHD between graphs G1 and G2
    """
    G1 = G1._graph
    G2 = G2._graph

    shd = 0
    for i, j in combinations(G1.nodes(), 2):
        # Edge present in G1, but missing in G2.
        if _has_any_edge(G1, i, j) and _has_no_edge(G2, i, j):
            shd += 1
        # Edge missing in G1, but present in G2.
        elif _has_no_edge(G1, i, j) and _has_any_edge(G2, i, j):
            shd += 1
        # Edge undirected in G1, but directed in G2.
        elif _has_both_edges(G1, i, j) and _has_one_edge(G2, i, j):
            shd += 1
        # Edge directed in G1, but undirected or reversed in G2.
        elif (_has_only_edge(G1, i, j) and G2.has_edge(j, i)) or (
            _has_only_edge(G1, j, i) and G2.has_edge(i, j)
        ):
            shd += 1
    return shd


def F1(true_graph: MemoryGraph, est_graph: MemoryGraph):
    tp = len(set(true_graph.str_edges) & set(est_graph.str_edges))

    # print(f"True edges: {len(true_graph.str_edges)}")
    # print(f"Estimated edges: {len(est_graph.str_edges)}")
    # print(f"Corrected edges: {tp}")

    if tp == 0:
        return {"precision": 0, "recall": 0, "f1": 0}

    pre = tp / len(est_graph.str_edges)
    rec = tp / len(true_graph.str_edges)
    f1 = 2 * pre * rec / (pre + rec)
    return {"precision": pre, "recall": rec, "f1": f1}


def F1_Skeleton(true_graph: MemoryGraph, est_graph: MemoryGraph):
    # double the single edge
    true_edges = []
    for node1, node2 in true_graph.str_edges:
        true_edges.append((node1, node2))
        true_edges.append((node2, node1))
    true_edges = set(true_edges)

    est_edges = []
    for node1, node2 in est_graph.str_edges:
        est_edges.append((node1, node2))
        est_edges.append((node2, node1))
    est_edges = set(est_edges)

    tp = len(true_edges & est_edges)

    # print(f"True edges: {len(true_edges) / 2}")
    # print(f"Estimated edges: {len(est_edges) / 2}")
    # print(f"Corrected edges: {tp / 2}")

    if tp == 0:
        return {"precision": 0, "recall": 0, "f1": 0}

    pre = tp / len(est_edges)
    rec = tp / len(true_edges)
    f1 = 2 * pre * rec / (pre + rec)
    return {"precision": pre, "recall": rec, "f1": f1}