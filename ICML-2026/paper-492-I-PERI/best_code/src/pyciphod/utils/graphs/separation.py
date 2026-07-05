from typing import Iterable, List, Set
from pyciphod.utils.graphs.graphs import Graph


def d_separated(graph: Graph, X: Iterable[str], Y: Iterable[str], Z: Iterable[str] = None, max_path_length: int = 100) -> bool:
    """Return True if sets X and Y are d-separated given Z in `graph`.

    The function searches for any active path between any x in X and y in Y; if none exists, returns True.

    Parameters:
    - graph: Graph or subclass
    - X, Y: iterables of node labels
    - Z: iterable of conditioning node labels
    - max_path_length: cutoff for path search (avoids combinatorial explosion)
    """
    Xs = set(X)
    Ys = set(Y)
    Zs = set(Z or [])

    vertices = set(graph.get_vertices())
    # sanity: restrict to existing nodes
    Xs = Xs & vertices
    Ys = Ys & vertices
    Zs = Zs & vertices

    if not Xs or not Ys:
        # trivially separated if one is empty
        return True

    # For each pair, search for a simple path up to cutoff that is active
    for x in Xs:
        for y in Ys:
            if x == y:
                # same node: not separated unless conditioning removes (we consider not separated)
                return False
            try:
                for path in graph.get_simple_paths(x, y, allowed_nodes=None, cutoff=max_path_length):
                    if graph.is_active_path(path, Zs):
                        # print(path)
                        # found an active path => not d-separated
                        return False
            except Exception:
                # on errors conservatively assume not separated
                return False
    # no active path found
    return True





def graph_to_latent_graph(graph, latent_prefix: str = "U"):
    """
    Convert an ADMG into a DAG by replacing each bidirected edge X <-> Y
    with a fresh latent node U_xy and edges:

        U_xy -> X
        U_xy -> Y

    Directed edges are copied unchanged.

    Returns
    -------
    latent_graph
        A DAG-like graph containing:
        - all original observed nodes
        - one fresh latent node per bidirected edge
    latent_nodes
        The set of newly created latent node names.
    """
    latent_graph = graph.copy()  # start with a copy of the original graph (or create a new empty graph if needed)

    used_names = set(list(graph.get_vertices()))

    latent_nodes: Set[str] = set()

    for i, (x, y) in enumerate(sorted(graph.get_confounded_edges())):
        base_name = f"{latent_prefix}_{x}_{y}"
        u = _fresh_name(base_name, used_names)
        used_names.add(u)
        latent_nodes.add(u)

        latent_graph.add_vertex(u)
        latent_graph.add_directed_edge(u, x)
        latent_graph.add_directed_edge(u, y)
        latent_graph.remove_confounded_edge(x, y)

    return latent_graph, latent_nodes


def _fresh_name(base_name: str, used_names: Set[str]) -> str:
    if base_name not in used_names:
        return base_name

    k = 1
    while f"{base_name}_{k}" in used_names:
        k += 1
    return f"{base_name}_{k}"


def m_separated(graph: Graph, X: Iterable[str], Y: Iterable[str], Z: Iterable[str] = None, max_path_length: int = 100) -> bool:
    """Return True if sets X and Y are m-separated given Z in `graph`.

    The function searches for any active path between any x in X and y in Y; if none exists, returns True.

    Parameters:
    - graph: Graph or subclass
    - X, Y: iterables of node labels
    - Z: iterable of conditioning node labels
    - max_path_length: cutoff for path search (avoids combinatorial explosion)
    """
    # For m-separation, we can treat all edges as undirected for path search, but still apply the same active path rules.
    # We can use the same d_separated function but with a modified graph that treats all edges as undirected.
    latent_graph, _ = graph_to_latent_graph(graph)
    return d_separated(latent_graph, X, Y, Z, max_path_length)