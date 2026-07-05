from pyciphod.utils.graphs.partially_specified_graphs import SummaryCausalGraph
from pyciphod.causal_reasoning.cluster_graph.scg.micro_queries.total_effect import (
    id_identifiable_by_adjustment_from_scg,
)


# def get_graph_with_removed_incoming_edges(x):
#     g2 = self._g.copy()
#     in_edges = list(g2.in_edges(x))
#     for a, b in in_edges:
#         g2.remove_edge(a, b)
#     return SCGWrapper(g2)


def example_1_singleton_scc_identifiable():
    # Simple acyclic SCG: X -> Y (no cycles) -> should be trivially identifiable
    g = SummaryCausalGraph()
    g.add_directed_edge('X', 'Y')

    print("Example 1: simple X->Y (acyclic)")
    res = id_identifiable_by_adjustment_from_scg(g, 'X', 'Y', gamma=0)
    print("Identifiable by adjustment?", bool(res))
    print()


def example_2_cycle_non_identifiable():
    g = SummaryCausalGraph()
    g.add_directed_edge('X', 'Y')
    # make an SCC containing X
    g.add_directed_edge('X', 'U')
    g.add_directed_edge('U', 'X')
    # connect to Y through a node that participates in a cycle
    g.add_directed_edge('U', 'A')
    g.add_directed_edge('A', 'Y')
    # create a cycle that includes A (so cyclic nodes intersect ancestors of Y)
    g.add_directed_edge('A', 'B')
    g.add_directed_edge('B', 'A')

    print("Example 2: X in non-singleton SCC and cycle intersecting ancestors(Y)")
    res = id_identifiable_by_adjustment_from_scg(g, 'X', 'Y', gamma=0)
    print("Identifiable by adjustment?", bool(res))
    print()


def example_3_gamma_one_special_case():
    # Special case: gamma == 1 and small bidirected cycle between X and Y
    g = SummaryCausalGraph()
    g.add_directed_edge('X', 'Y')
    g.add_directed_edge('Y', 'X')

    print("Example 3: bidirected X<->Y and gamma=1 special check")
    res = id_identifiable_by_adjustment_from_scg(g, 'X', 'Y', gamma=0)
    print("Identifiable by adjustment?", bool(res))
    print()


if __name__ == '__main__':
    example_1_singleton_scc_identifiable()
    example_2_cycle_non_identifiable()
    example_3_gamma_one_special_case()
