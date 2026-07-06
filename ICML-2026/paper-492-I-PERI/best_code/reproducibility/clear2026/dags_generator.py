# this file was created before developping our own function to transfer a dag to a cpdag so we used the implementation of the causal-learn package
import networkx as nx
import random as rd
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.utils.DAG2CPDAG import dag2cpdag


def generate_ER_oriented_DAG(size, edge_prob):
    """
    Generate a DAG as a GeneralGraph using an oriented Erdős–Rényi random graph.

    Steps:
    - Generate an undirected Erdős–Rényi graph with given size and edge probability.
    - Shuffle nodes to get a random topological order.
    - Orient edges from nodes with smaller order index to larger to ensure acyclicity.
    - Construct and return a GeneralGraph with directed edges.

    Args:
        size (int): Number of nodes in the graph.
        edge_prob (float): Probability of an edge between any two nodes.

    Returns:
        GeneralGraph: Directed acyclic graph represented as a GeneralGraph.
    """
    # Generate an undirected Erdős–Rényi graph
    undirected = nx.erdos_renyi_graph(n=size, p=edge_prob, directed=False)

    # Random topological ordering of nodes to orient edges
    nodes = list(undirected.nodes())
    rd.shuffle(nodes)
    order = {node: i for i, node in enumerate(nodes)}

    # Create graph nodes with string names
    node_names = [str(i) for i in range(size)]
    graph_nodes = [GraphNode(name) for name in node_names]

    # Initialize GeneralGraph with these nodes
    g = GeneralGraph(graph_nodes)
    g.graph[:, :] = 0  # clear any existing edges

    # Add directed edges respecting the topological order to avoid cycles
    for u, v in undirected.edges():
        src, tgt = (u, v) if order[u] < order[v] else (v, u)
        g.add_edge(Edge(g.nodes[src], g.nodes[tgt], Endpoint.TAIL, Endpoint.ARROW))

    return g


def general_graph_to_nx(dag):
    """
    Convert a causal-learn GeneralGraph (assumed DAG) to a NetworkX DiGraph.

    Args:
        dag (GeneralGraph): The input causal-learn graph.

    Returns:
        networkx.DiGraph: Equivalent directed graph with same nodes and edges.
    """
    g = nx.DiGraph()

    # Add nodes by their names
    g.add_nodes_from(node.get_name() for node in dag.get_nodes())

    # Add edges (assumes edges are directed, as typical for a DAG)
    for edge in dag.get_graph_edges():
        g.add_edge(edge.get_node1().get_name(), edge.get_node2().get_name())

    return g


def random_DAG_identifiable_CDE(size, prob, max_tries=1000):
    """
    Find a random DAG that is CDE-identifiable with the following properties:
    - There exists an outcome node whose adjacency is fully directed into it (no undirected edges).
    - There is at least one node with a directed edge into the outcome (the treatment).
    
    Attempts up to max_tries graphs, each time generating a random DAG.

    Args:
        size (int): Number of nodes in the DAG.
        prob (float): Probability of edges in the Erdős–Rényi model.
        max_tries (int): Maximum attempts before failure.

    Returns:
        tuple: (dag_nx, outcome_name, treatment_name)
            - dag_nx (networkx.DiGraph): The found DAG as a NetworkX graph.
            - outcome_name (str): Name of the outcome node.
            - treatment_name (str): Name of the treatment node.

    Raises:
        RuntimeError: If no suitable DAG is found after max_tries attempts.
    """
    for _ in range(max_tries):
        g = generate_ER_oriented_DAG(size, prob)
        c = dag2cpdag(g)

        candidates_outcome = [str(i) for i in range(size)]

        # Randomly pick outcomes until a valid one is found or candidates exhausted
        while candidates_outcome:
            outcome_name = candidates_outcome.pop(rd.randrange(len(candidates_outcome)))
            outcome_node = c.get_node(outcome_name)
            adj_outcome = c.get_adjacent_nodes(outcome_node)

            # Skip if any adjacent edge is undirected (not fully identifiable)
            if any(c.is_undirected_from_to(a, outcome_node) for a in adj_outcome):
                continue

            # Collect nodes with directed edge into outcome (valid treatments)
            valid_treatments = [a for a in adj_outcome if c.is_directed_from_to(a, outcome_node)]

            if not valid_treatments:
                continue

            # Pick a random treatment node from valid candidates
            treatment_node = rd.choice(valid_treatments)
            treatment_name = treatment_node.get_name()

            # Convert causal-learn DAG to networkx DiGraph for output
            dag_nx = general_graph_to_nx(g)

            return dag_nx, outcome_name, treatment_name

    raise RuntimeError(f"Failed to find a CDE identifiable DAG after {max_tries} tries.")


def random_DAG_nonidentifiable_CDE(size, prob, max_tries=10000):
    """
    Find a random DAG whose CPDAG is non-CDE-identifiable with these properties:
    - There exists an outcome node Y with at least one undirected edge in its adjacency.
    - There is at least one node X with a directed edge X -> Y (treatment).
    """
    import random as rd

    for attempt in range(1, max_tries + 1):
        g = generate_ER_oriented_DAG(size, prob)
        c = dag2cpdag(g)
        candidates_outcome = [str(i) for i in range(size)]

        while candidates_outcome:
            outcome_name = candidates_outcome.pop(rd.randrange(len(candidates_outcome)))
            outcome_node = c.get_node(outcome_name)
            adj_outcome = c.get_adjacent_nodes(outcome_node)
            adj_names = [a.get_name() for a in adj_outcome]

            has_undirected = any(
                c.is_undirected_from_to(a, outcome_node) or c.is_undirected_from_to(outcome_node, a)
                for a in adj_outcome
            )
            if not has_undirected:
                continue

            valid_treatments = [a for a in adj_outcome if c.is_directed_from_to(a, outcome_node)]
            valid_treat_names = [a.get_name() for a in valid_treatments]

            if not valid_treatments:
                continue

            treatment_node = rd.choice(valid_treatments)
            treatment_name = treatment_node.get_name()
            dag_nx = general_graph_to_nx(g)
            return dag_nx, outcome_name, treatment_name

    raise RuntimeError(f"Failed to find a NON CDE identifiable DAG after {max_tries} tries.")


