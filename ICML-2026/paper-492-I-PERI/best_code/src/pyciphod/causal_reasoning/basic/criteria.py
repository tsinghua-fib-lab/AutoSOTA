from pyciphod.utils.graphs.graphs import DirectedMixedGraph
from typing import Type, Optional, Set, Tuple, Iterable, FrozenSet
from pyciphod.utils.graphs.separation import d_separated


def back_door_criterion(graph:DirectedMixedGraph, X: Iterable, Y: Iterable, Z: Iterable):
    """Back-door criterion: Z blocks all back-door paths from X to Y, and no node in Z is a descendant of X.
    Returns True if the criterion is satisfied.
    """

    # Check that no node in Z is a descendant of X

    # Check that Z blocks all back-door paths from X to Y
    for x in X:
        for z in Z:
            if z in graph.get_descendants(x):
                return False

        g_manipulated = graph.copy()
        g_manipulated.remove_outgoing_edges(x)

        for y in Y:
            if not d_separated(g_manipulated, {x}, {y}, Z):
                return False
    return True


def front_door_criterion(graph, X, Y, Z):
    """Front-door criterion: Z intercepts all directed paths from X to Y, there is no back-door path from X to Z, and all back-door paths from Z to Y are blocked by X.
    Returns True if the criterion is satisfied.
    """
    # Check that Z intercepts all directed paths from X to Y
    g_manipulated = graph.copy()
    g_manipulated.remove_incoming_edges(X)
    for x in X:
        for y in Y:
            # remove incoming edges to X
            paths = g_manipulated.get_active_paths(x, y, max_path_length = len(graph.get_vertices()))
            test_interception = True
            #TODO
            for path in paths:
                path

    # # Check that Z intercepts all directed paths from X to Y
    # for x in X:
    #     for y in Y:
    #         if nx.has_path(graph, x, y) and not any(nx.has_path(graph, x, z) and nx.has_path(graph, z, y) for z in Zs):
    #             return False

    # Check that there is no back-door path from X to Z
    g_manipulated = graph.copy()
    g_manipulated.remove_outgoing_edges(X)
    for x in X:
        for z in Z:
            if not d_separated(g_manipulated, {x}, {z}, set()):
                return False

    # Check that all back-door paths from Z to Y are blocked by X
    g_manipulated = graph.copy()
    g_manipulated.remove_outgoing_edges(Z)
    # g_manipulated.draw_graph()
    for z in Z:
        for y in Y:
            if not d_separated(g_manipulated, {z}, {y}, X):
                return False

    return True
