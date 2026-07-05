import copy

from pyciphod.utils.graphs.partially_specified_graphs import SummaryCausalGraph

# TODO
def id_identifiable_by_adjustment_from_scg(g:SummaryCausalGraph, x:str, y:str, gamma=0):
    sccs_x = g.get_strongly_connected_components(x)
    g_x = copy.deepcopy(g)
    g_x.remove_incoming_edges(x)
    g_x.remove_outgoing_edges(x)
    ancs_y = g_x.get_ancestors(y)
    cyc_y = g.get_simple_cycles(y)

    if sccs_x == {x}:
        return True

    if sccs_x != {x} and gamma == 0 and cyc_y.intersection(ancs_y) == set():
        return True

    bid_xy = {(x, y), (y, x)}
    if gamma == 1 and cyc_y == bid_xy :
        return True


def get_all_valid_adjustment_sets(g, x, y, gamma=0):
    1


def get_adjustment_formula(g, x, y, gamma=0):
    1


def is_s_orientable_from_scg_with_faithful_distributions():
    1


def is_s_identifiable_from_scg_with_faithful_distributions():
    1