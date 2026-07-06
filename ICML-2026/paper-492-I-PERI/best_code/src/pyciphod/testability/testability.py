import pandas as pd

from cyphod.utils.graphs.d_separated import Separation
from cyphod.utils.stat_tests.independence_tests import CiTests


def testability_via_markov_condition(graph, data, ci: CiTests, sig):
    # TODO check is vertices matchs columns names

    for vertex_i in graph.get_vertices():
        pa = graph.get_parents(vertex_i)
        nd = graph.get_vertices() - graph.get_descendants(vertex_i)
        for vertex_j in nd:
            c = ci(vertex_i, vertex_j, cond_list=pa, drop_na=True)
            try:
                pval = c.get_pvalue()
            except:
                pval= c.get_pvalue_by_permutation()

