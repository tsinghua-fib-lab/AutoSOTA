from pyciphod.causal_reasoning.summary_causal_graph.micro_queries.direct_effect import CDE_is_identifiable, NDE_is_identifiable
from pyciphod.utils.graphs.partially_specified_graphs import SummaryCausalGraph

def test_CDE():
    '''
    Examples from "Average Controlled and Average Natural Micro Direct Effects in Summary
    Causal Graphs" by Ferreira and Assaad UAI 2026.
    '''
    #Fig 1
    Gs = SummaryCausalGraph()
    Gs.add_vertices(['W','X','Y'])
    Gs.add_directed_edges_from([('X','Y'),('X','W'),('W','Y'),('Y','W')])
    Gs.add_directed_edges_from([('X','X'),('W','W'),('Y','Y')])
    Gs.add_lag_max(1)
    #Gs.draw_graph()
    assert (not CDE_is_identifiable(Gs, 'X', 'Y', 0))
    assert (not CDE_is_identifiable(Gs, 'X', 'Y', 1))


    #Fig 2
    Gs = SummaryCausalGraph()
    Gs.add_vertices(['W','X','Y'])
    Gs.add_directed_edges_from([('X','Y'),('X','W'),('W','Y')])
    Gs.add_directed_edges_from([('X','X'),('W','W'),('Y','Y')])
    Gs.add_confounded_edges_from([('W','Y')])
    Gs.add_confounded_edges_from([('W','W'),('Y','Y')])
    Gs.add_lag_max(1)
    #Gs.draw_graph()
    assert (not CDE_is_identifiable(Gs, 'X', 'Y', 0)) 
    assert (not CDE_is_identifiable(Gs, 'X', 'Y', 1))

    #Fig 3
    Gs = SummaryCausalGraph()
    Gs.add_vertices(['W','X','Y'])
    Gs.add_directed_edges_from([('X','Y'),('X','W'),('W','X'),('W','Y')])
    Gs.add_directed_edges_from([('X','X'),('W','W'),('Y','Y')])
    Gs.add_lag_max(1)
    #Gs.draw_graph()
    assert CDE_is_identifiable(Gs, 'X', 'Y', 0) 
    assert CDE_is_identifiable(Gs, 'X', 'Y', 1)

    #Fig 4
    Gs = SummaryCausalGraph()
    Gs.add_vertices(['W','X','Y'])
    Gs.add_directed_edges_from([('X','Y'),('X','W'),('W','Y')])
    Gs.add_directed_edges_from([('X','X'),('W','W'),('Y','Y')])
    Gs.add_confounded_edges_from([('W','X')])
    Gs.add_confounded_edges_from([('X','X'),('W','W')])
    Gs.add_lag_max(1)
    #Gs.draw_graph()
    assert CDE_is_identifiable(Gs, 'X', 'Y', 0) 
    assert CDE_is_identifiable(Gs, 'X', 'Y', 1)

    #Fig 5.a
    Gs = SummaryCausalGraph()
    Gs.add_vertices(['W','X','Y','Z'])
    Gs.add_directed_edges_from([('X','Y'),('W','X'),('X','W'),('X','Z'),('Z','X'),('W','Y'),('Z','Y')])
    Gs.add_directed_edges_from([('X','X'),('W','W'),('Y','Y'),('Z','Z')])
    Gs.add_confounded_edges_from([('W','X'),('X','Z')])
    Gs.add_confounded_edges_from([('X','X'),('Z','Z')])
    Gs.add_lag_max(1)
    #Gs.draw_graph()
    assert CDE_is_identifiable(Gs, 'X', 'Y', 0) 
    assert CDE_is_identifiable(Gs, 'X', 'Y', 1)

    #Fig 5.b
    Gs = SummaryCausalGraph()
    Gs.add_vertices(['W','X','Y','Z'])
    Gs.add_directed_edges_from([('X','Y'),('X','W'),('W','Y'),('Z','Y'),('W','Z'),('Z','W')])
    Gs.add_directed_edges_from([('X','X'),('W','W'),('Y','Y'),('Z','Z')])
    Gs.add_confounded_edges_from([('W','Z')])
    Gs.add_lag_max(1)
    #Gs.draw_graph()
    assert CDE_is_identifiable(Gs, 'X', 'Y', 0) 
    assert CDE_is_identifiable(Gs, 'X', 'Y', 1)

    #Fig 5.c
    Gs = SummaryCausalGraph()
    Gs.add_vertices(['W','X','Y','Z'])
    Gs.add_directed_edges_from([('X','Y'),('X','W'),('W','Y'),('Z','Y'),('W','Z'),('Z','W')])
    Gs.add_directed_edges_from([('W','W'),('Y','Y'),('Z','Z')])
    Gs.add_confounded_edges_from([('W','Z')])
    Gs.add_lag_max(1)
    #Gs.draw_graph()
    assert CDE_is_identifiable(Gs, 'X', 'Y', 0) 
    assert CDE_is_identifiable(Gs, 'X', 'Y', 1)

    #Fig 6
    Gs = SummaryCausalGraph()
    Gs.add_vertices(['A','E','X','Y'])
    Gs.add_directed_edges_from([('X','Y'),('A','Y'),('E','Y'),('X','A'),('A','X'),('X','E'),('E','X')])
    Gs.add_directed_edges_from([('A','A'),('Y','Y'),('E','E'),('X','X')])
    Gs.add_lag_max(1)
    #Gs.draw_graph()
    assert CDE_is_identifiable(Gs, 'X', 'Y', 0) 
    assert CDE_is_identifiable(Gs, 'X', 'Y', 1)


def test_NDE():
    '''
    Examples from "Average Controlled and Average Natural Micro Direct Effects in Summary
    Causal Graphs" by Ferreira and Assaad UAI 2026.
    '''
    #Fig 1
    Gs = SummaryCausalGraph()
    Gs.add_vertices(['W','X','Y'])
    Gs.add_directed_edges_from([('X','Y'),('X','W'),('W','Y'),('Y','W')])
    Gs.add_directed_edges_from([('X','X'),('W','W'),('Y','Y')])
    Gs.add_lag_max(1)
    #Gs.draw_graph()
    assert (not NDE_is_identifiable(Gs, 'X', 'Y', 0)) 
    assert (not NDE_is_identifiable(Gs, 'X', 'Y', 1))


    #Fig 2
    Gs = SummaryCausalGraph()
    Gs.add_vertices(['W','X','Y'])
    Gs.add_directed_edges_from([('X','Y'),('X','W'),('W','Y')])
    Gs.add_directed_edges_from([('X','X'),('W','W'),('Y','Y')])
    Gs.add_confounded_edges_from([('W','Y')])
    Gs.add_confounded_edges_from([('W','W'),('Y','Y')])
    Gs.add_lag_max(1)
    #Gs.draw_graph()
    assert (not NDE_is_identifiable(Gs, 'X', 'Y', 0)) 
    assert (not NDE_is_identifiable(Gs, 'X', 'Y', 1))

    #Fig 3
    Gs = SummaryCausalGraph()
    Gs.add_vertices(['W','X','Y'])
    Gs.add_directed_edges_from([('X','Y'),('X','W'),('W','X'),('W','Y')])
    Gs.add_directed_edges_from([('X','X'),('W','W'),('Y','Y')])
    Gs.add_lag_max(1)
    #Gs.draw_graph()
    assert (not NDE_is_identifiable(Gs, 'X', 'Y', 0)) 
    assert (not NDE_is_identifiable(Gs, 'X', 'Y', 1))

    #Fig 4
    Gs = SummaryCausalGraph()
    Gs.add_vertices(['W','X','Y'])
    Gs.add_directed_edges_from([('X','Y'),('X','W'),('W','Y')])
    Gs.add_directed_edges_from([('X','X'),('W','W'),('Y','Y')])
    Gs.add_confounded_edges_from([('W','X')])
    Gs.add_confounded_edges_from([('X','X'),('W','W')])
    Gs.add_lag_max(1)
    #Gs.draw_graph()
    assert (not NDE_is_identifiable(Gs, 'X', 'Y', 0)) 
    assert (not NDE_is_identifiable(Gs, 'X', 'Y', 1))

    #Fig 5.a
    Gs = SummaryCausalGraph()
    Gs.add_vertices(['W','X','Y','Z'])
    Gs.add_directed_edges_from([('X','Y'),('W','X'),('X','W'),('X','Z'),('Z','X'),('W','Y'),('Z','Y')])
    Gs.add_directed_edges_from([('X','X'),('W','W'),('Y','Y'),('Z','Z')])
    Gs.add_confounded_edges_from([('W','X'),('X','Z')])
    Gs.add_confounded_edges_from([('X','X'),('Z','Z')])
    Gs.add_lag_max(1)
    #Gs.draw_graph()
    assert (not NDE_is_identifiable(Gs, 'X', 'Y', 0)) 
    assert (not NDE_is_identifiable(Gs, 'X', 'Y', 1))

    #Fig 5.b
    Gs = SummaryCausalGraph()
    Gs.add_vertices(['W','X','Y','Z'])
    Gs.add_directed_edges_from([('X','Y'),('X','W'),('W','Y'),('Z','Y'),('W','Z'),('Z','W')])
    Gs.add_directed_edges_from([('X','X'),('W','W'),('Y','Y'),('Z','Z')])
    Gs.add_confounded_edges_from([('W','Z')])
    Gs.add_lag_max(1)
    #Gs.draw_graph()
    assert (not NDE_is_identifiable(Gs, 'X', 'Y', 0))
    assert NDE_is_identifiable(Gs, 'X', 'Y', 1)

    #Fig 5.c
    Gs = SummaryCausalGraph()
    Gs.add_vertices(['W','X','Y','Z'])
    Gs.add_directed_edges_from([('X','Y'),('X','W'),('W','Y'),('Z','Y'),('W','Z'),('Z','W')])
    Gs.add_directed_edges_from([('W','W'),('Y','Y'),('Z','Z')])
    Gs.add_confounded_edges_from([('W','Z')])
    Gs.add_lag_max(1)
    #Gs.draw_graph()
    assert NDE_is_identifiable(Gs, 'X', 'Y', 0) 
    assert NDE_is_identifiable(Gs, 'X', 'Y', 1)

    #Fig 6
    Gs = SummaryCausalGraph()
    Gs.add_vertices(['A','E','X','Y'])
    Gs.add_directed_edges_from([('X','Y'),('A','Y'),('E','Y'),('X','A'),('A','X'),('X','E'),('E','X')])
    Gs.add_directed_edges_from([('A','A'),('Y','Y'),('E','E'),('X','X')])
    Gs.add_lag_max(1)
    #Gs.draw_graph()
    assert (not NDE_is_identifiable(Gs, 'X', 'Y', 0)) 
    assert (not NDE_is_identifiable(Gs, 'X', 'Y', 1))

if __name__ == "__main__":
    test_CDE()
    test_NDE()