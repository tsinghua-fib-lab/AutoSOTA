import networkx as nx
from itertools import product


def enumerate_amos(graph):
    """
    Faithful implementation of Algorithm 1 from the attached PDF.
    Enumerates all Acyclic Moral Orientations (AMOs) of a chordal graph.

    Args:
        graph (nx.Graph): A chordal graph (UCCG) as an undirected NetworkX graph.

    Returns:
        list: A list of nx.DiGraph objects representing all AMOs.
    """
    def mcs_enum(graph, visited, labels, oriented_edges, amos, unique_dags):
        """
        Recursive function to enumerate AMOs using MCS and dynamic edge orientation.

        Args:
            graph (nx.Graph): The chordal graph.
            visited (set): Set of visited vertices.
            labels (dict): Cardinality labels for vertices.
            oriented_edges (list): Directed edges constructed so far.
            amos (list): List to store all AMOs.
            unique_dags (set): Set to track unique DAG edge sets.
        """
        # Base case: All vertices visited
        if len(visited) == len(graph):
            dag = nx.DiGraph()
            dag.add_edges_from(oriented_edges)
            if len(oriented_edges) == len(graph.edges):  # All edges must be oriented
                edge_set = frozenset(dag.edges())
                if edge_set not in unique_dags:  # Avoid duplicates
                    unique_dags.add(edge_set)
                    amos.append(dag)
            return

        # Compute reachable vertices R in G[S]
        reachable = compute_reachable(graph, visited)

        # Pick the vertex with the maximum label
        max_label = max(labels[v] for v in reachable)
        candidates = [v for v in reachable if labels[v] == max_label]

        for v in candidates:
            visited.add(v)
            current_labels = labels.copy()
            current_oriented_edges = oriented_edges[:]

            # Orient edges between v and its neighbors
            for neighbor in graph.neighbors(v):
                if neighbor not in visited:
                    labels[neighbor] += 1  # Increment label for unvisited neighbors
                elif (neighbor, v) not in current_oriented_edges and (v, neighbor) not in current_oriented_edges:
                    if not creates_invalid_collider(v, neighbor, current_oriented_edges, graph):
                        current_oriented_edges.append((neighbor, v))

            # Recursive call
            mcs_enum(graph, visited, labels, current_oriented_edges, amos, unique_dags)

            # Backtrack
            visited.remove(v)
            labels = current_labels

    def compute_reachable(graph, visited):
        """
        Compute the reachable set R in the subgraph G[S], where S = V \ visited.

        Args:
            graph (nx.Graph): The full chordal graph.
            visited (set): Set of visited vertices.

        Returns:
            set: The set of reachable vertices in G[S].
        """
        unvisited = set(graph.nodes) - visited
        reachable = set()
        for node in unvisited:
            if node not in reachable:
                reachable |= nx.node_connected_component(graph.subgraph(unvisited), node)
        return reachable

    def creates_invalid_collider(v, neighbor, oriented_edges, graph):
        """
        Check if orienting (neighbor -> v) creates an invalid unshielded collider.

        Args:
            v (str): The current vertex.
            neighbor (str): The neighbor vertex.
            oriented_edges (list): The current list of directed edges.
            graph (nx.Graph): The chordal graph.

        Returns:
            bool: True if an invalid collider is created, False otherwise.
        """
        for other in graph.neighbors(v):
            if other != neighbor and other not in graph.neighbors(neighbor):
                if (other, v) in oriented_edges and (v, other) not in oriented_edges:
                    return True
        return False

    # Initialize the recursive variables
    labels = {v: 0 for v in graph.nodes}
    visited = set()
    amos = []
    unique_dags = set()

    # Start recursion
    mcs_enum(graph, visited, labels, [], amos, unique_dags)

    return amos


def uccg_to_nx_graph(uccg):
    """
    Convert a UCCG (from PDAG.chain_components) to a NetworkX undirected graph.

    Args:
        uccg (PDAG): A UCCG represented as a PDAG object.

    Returns:
        nx.Graph: The corresponding NetworkX undirected graph.
    """
    nx_graph = nx.Graph()
    for u, v in uccg.edges:
        nx_graph.add_edge(u, v)
    return nx_graph

def enumerate_dags(cpdag, list_of_amos):
    """
    Generate all possible DAGs by combining the directed edges of the CPDAG
    with combinations of AMOs from each UCCG.

    Args:
        cpdag (PDAG): The CPDAG.
        list_of_amos (list): List of lists, where each inner list contains AMOs
                             (as nx.DiGraph objects) for a UCCG.

    Returns:
        list: A list of nx.DiGraph objects representing all possible DAGs.
    """
    # Step 1: Extract directed edges from the CPDAG
    directed_edges = list(cpdag.arcs)
    # Step 2: Generate all combinations of AMOs from the UCCGs
    if len(list_of_amos) == 1:
        # Special case: single UCCG
        all_combinations = [[amo] for amo in list_of_amos[0]]
    else:
        # General case: multiple UCCGs
        all_combinations = product(*list_of_amos)

    # Step 3: Merge directed edges and AMOs to create DAGs
    final_dags = []
    for amos_combination in all_combinations:
        dag = nx.DiGraph()
        dag.add_edges_from(directed_edges)  # Add fixed directed edges
        for amo in amos_combination:
            dag.add_edges_from(amo.edges())  # Add edges from the selected AMO
        if nx.is_directed_acyclic_graph(dag):  # Ensure the result is a valid DAG
            final_dags.append(dag)

    return final_dags