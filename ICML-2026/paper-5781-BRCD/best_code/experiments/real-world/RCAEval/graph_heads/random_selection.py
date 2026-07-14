from random import sample

from causallearn.graph.GraphClass import CausalGraph


def random_selection(causal_graph: CausalGraph, n: int):
    """
    Randomly select n nodes from the causal graph.
    """
    nodes = sample(causal_graph.G.nodes, n)
    return [n.get_name() for n in nodes]
