import pytest

from pyciphod.utils.graphs.graphs import Graph, AcyclicDirectedMixedGraph, DirectedAcyclicGraph, create_random_admg, create_random_dag
from pyciphod.utils.graphs.separation import d_separated, m_separated
from pyciphod.utils.graphs.partially_specified_graphs import CompletedPartiallyDirectedAcyclicDifferenceGraph
from pyciphod.utils.graphs.background_knowledge import BackgroundKnowledge


@pytest.fixture
def graph():
    """Fixture returning a fresh Graph instance."""
    return Graph()


def test_add_vertices_and_edges(graph):
    g = graph
    # ajouter sommets et différents types d'arêtes
    g.add_vertices(["A", "B", "X", "Y", "C", "D", "E", "F"])

    # directed X -> Y
    g.add_directed_edge("X", "Y")
    # undirected A - B
    g.add_undirected_edge("A", "B")
    # confounded C <-> D
    g.add_confounded_edge("C", "D")
    # uncertain E *-o F
    g.add_uncertain_edge("E", "F", edge_type='o-o')

    verts = g.get_vertices()
    assert set(["A", "B", "X", "Y", "C", "D", "E", "F"]).issubset(verts)

    # directed edges
    directed = g.get_directed_edges()
    assert ("X", "Y") in directed

    # undirected edges should contain both orientations
    undirected = g.get_undirected_edges()
    assert ("A", "B") in undirected and ("B", "A") in undirected

    # confounded: check either order present
    conf = g.get_confounded_edges()
    assert any(e == ("C", "D") or e == ("D", "C") for e in conf)

    # uncertain edges recorded in uncertain graph (directed representation)
    uncertain = g.get_uncertain_edges()
    assert ("E", "F") in uncertain

    # edge types for uncertain should include 'o-o'
    edge_types = g.get_edge_types("E", "F")
    assert 'o-o' in edge_types

    # parents/children
    assert "X" in g.get_parents("Y")
    assert "Y" in g.get_children("X")

    # adjacencies include neighbors from all edge types
    adj_A = g.get_adjacencies("A")
    assert "B" in adj_A
    adj_C = g.get_adjacencies("C")
    assert "D" in adj_C


def test_remove_edges_and_uncertain_update(graph):
    g = graph
    g.add_vertices(["U", "V"])
    g.add_undirected_edge("U", "V")
    assert ("U", "V") in g.get_undirected_edges()

    g.remove_undirected_edge("U", "V")
    assert ("U", "V") not in g.get_undirected_edges() and ("V", "U") not in g.get_undirected_edges()

    # uncertain update: add then remove
    g.add_uncertain_edge("P", "Q", edge_type='o-o')
    assert ("P", "Q") in g.get_uncertain_edges()
    g.remove_uncertain_edge("P", "Q")
    assert ("P", "Q") not in g.get_uncertain_edges()





def test_acyclicity_detection():
    g = Graph()
    g.add_vertices(["X", "Y", "Z"])
    g.add_directed_edge("X", "Y")
    g.add_directed_edge("Y", "Z")
    assert g.is_acyclic() is True

    # add back edge to create cycle
    g.add_directed_edge("Z", "X")
    assert g.is_acyclic() is False


def test_get_adjacencies_mixed_edges():
    g = Graph()
    g.add_vertices(["A", "B", "C", "D"])
    g.add_directed_edge("A", "B")
    g.add_undirected_edge("A", "C")
    g.add_confounded_edge("A", "D")

    adj = g.get_adjacencies("A")
    # adjacency should include B (child), C (undirected), D (confounded)
    assert {"B", "C", "D"}.issubset(adj)


def test_graph_parents_children_and_adjacencies():
    g = Graph()
    g.add_directed_edge('X', 'Y')
    g.add_directed_edge('Z', 'Y')
    g.add_undirected_edge('X', 'Z')

    parents_y = g.get_parents('Y')
    children_x = g.get_children('X')
    adj_x = g.get_adjacencies('X')

    assert 'X' in parents_y or 'Z' in parents_y
    assert 'Y' in children_x
    # adjacencies of X should include Y and Z
    assert 'Y' in adj_x
    assert 'Z' in adj_x

def test_directed_acyclic_graph_prevents_cycles():
    dag = DirectedAcyclicGraph()
    dag.add_vertices(["A", "B", "C"])
    # add A->B and B->C should be fine
    dag.add_directed_edge("A", "B")
    dag.add_directed_edge("B", "C")
    assert dag.is_acyclic()

    # adding C->A should raise ValueError and not be present
    with pytest.raises(ValueError):
        dag.add_directed_edge("C", "A")
    assert ("C", "A") not in dag.get_directed_edges()


def test_acmg_allows_confounded_and_directed_edges_and_is_acyclic():
    admg = AcyclicDirectedMixedGraph()
    admg.add_vertices(["X", "Y", "Z"])
    admg.add_directed_edge("X", "Y")
    admg.add_confounded_edge("Y", "Z")
    # Directed edges should be acyclic
    assert admg.is_acyclic()
    # adjacencies should include confounded neighbor
    assert "Z" in admg.get_adjacencies("Y")


def test_colliders_and_unshielded_colliders():
    g = Graph()
    g.add_directed_edge('M', 'B')
    g.add_directed_edge('A', 'B')
    g.add_directed_edge('B', 'C')
    # collider at (A, B, M) and B is connected to C
    colliders = g.get_all_colliders()
    # find (A, B, M) normalized ordering (function sorts a,b)
    found = any((a == 'A' and b == 'B' and c == 'M') or (a == 'M' and b == 'B' and c == 'A') for (a, b, c) in colliders)
    assert found
    unshielded = g.get_all_unshielded_colliders()
    # (A,B,M) might be unshielded if A and M are not adjacent
    if 'A' not in g.get_adjacencies('M'):
        assert ('A', 'B', 'M') in unshielded or ('M', 'B', 'A') in unshielded



def test_create_random_admg_and_dag_have_vertices_and_no_directed_cycles():
    admg = create_random_admg(num_v=5, p_edge=0.8, seed=42)
    assert len(admg.get_vertices()) == 5
    assert admg.is_acyclic()

    dag = create_random_dag(num_v=6, p_edge=0.5, seed=7)
    assert len(dag.get_vertices()) == 6
    assert dag.is_acyclic()




def test_construct_from_dag_v_structure():
    # DAG: A -> C <- B  (v-structure)
    dag = DirectedAcyclicGraph()
    for n in ['A', 'B', 'C']:
        dag.add_vertex(n)
    dag.add_directed_edge('A', 'C')
    dag.add_directed_edge('B', 'C')

    cpdag = CompletedPartiallyDirectedAcyclicDifferenceGraph()
    cpdag.construct_from_dag(dag)

    directed = set(cpdag.get_directed_edges())
    undirected = set(cpdag.get_undirected_edges())

    # Expect A->C and B->C oriented
    assert ('A', 'C') in directed
    assert ('B', 'C') in directed
    # A and B should not be adjacent
    assert 'B' not in cpdag.get_adjacencies('A')


def test_construct_from_dag_chain_remains_undirected():
    # DAG: A -> B -> C (no collider)
    dag = DirectedAcyclicGraph()
    for n in ['A', 'B', 'C']:
        dag.add_vertex(n)
    dag.add_directed_edge('A', 'B')
    dag.add_directed_edge('B', 'C')

    cpdag = CompletedPartiallyDirectedAcyclicDifferenceGraph()
    cpdag.construct_from_dag(dag)

    # In this implementation the chain edges remain undirected in the CPDAG
    undirected = set(cpdag.get_undirected_edges())
    directed = set(cpdag.get_directed_edges())

    assert ('A', 'B') in undirected or ('B', 'A') in undirected
    assert ('B', 'C') in undirected or ('C', 'B') in undirected
    # No directed edges should be present for these pairs
    assert ('A', 'B') not in directed and ('B', 'A') not in directed
    assert ('B', 'C') not in directed and ('C', 'B') not in directed


def test_construct_from_dag_preserves_vertices():
    dag = DirectedAcyclicGraph()
    for n in ['X', 'Y', 'Z']:
        dag.add_vertex(n)
    dag.add_directed_edge('X', 'Y')

    cpdag = CompletedPartiallyDirectedAcyclicDifferenceGraph()
    cpdag.construct_from_dag(dag)

    assert set(cpdag.get_vertices()) == set(dag.get_vertices())



def test_chain_and_collider():
    # Chain: A -> B -> C, A and C are d-separated given B
    g = Graph()
    g.add_directed_edge('A', 'B')
    g.add_directed_edge('B', 'C')

    assert not d_separated(g, ['A'], ['C'])  # without conditioning path active
    assert d_separated(g, ['A'], ['C'], ['B'])  # conditioning on B blocks the chain

    # Collider: A -> B <- C, A and C are d-separated unless B (or its descendant) is conditioned
    g2 = Graph()
    g2.add_directed_edge('A', 'B')
    g2.add_directed_edge('C', 'B')

    assert d_separated(g2, ['A'], ['C'])
    assert not d_separated(g2, ['A'], ['C'], ['B'])


def test_fork_structure_blocks_when_conditioned():
    # Fork: Z -> X and Z -> Y (Z is common cause). Conditioning on Z blocks X--Y
    g = Graph()
    g.add_directed_edge('Z', 'X')
    g.add_directed_edge('Z', 'Y')

    assert not d_separated(g, ['X'], ['Y'])
    assert d_separated(g, ['X'], ['Y'], ['Z'])


def test_collider_activated_by_descendant():
    # Collider: A -> B <- C and B -> D (D is a descendant of collider B)
    # Without conditioning, A and C are d-separated; conditioning on D activates path
    g = Graph()
    g.add_directed_edge('A', 'B')
    g.add_directed_edge('C', 'B')
    g.add_directed_edge('B', 'D')

    # initially separated
    assert d_separated(g, ['A'], ['C'])
    # conditioning on descendant D opens path through collider B
    assert not d_separated(g, ['A'], ['C'], ['D'])


def test_multiple_paths_one_blocked_other_open():
    # Two parallel paths between S and T:
    # S -> A -> T  (path1)
    # S -> B -> T  (path2)
    # Conditioning on A blocks path1 but path2 remains, so S and T are dependent
    g = Graph()
    g.add_directed_edge('S', 'A')
    g.add_directed_edge('A', 'T')
    g.add_directed_edge('S', 'B')
    g.add_directed_edge('B', 'T')

    assert not d_separated(g, ['S'], ['T'])
    # condition on A should not render S and T independent because the S->B->T path remains
    assert not d_separated(g, ['S'], ['T'], ['A'])
    # condition on both A and B blocks both paths
    assert d_separated(g, ['S'], ['T'], ['A', 'B'])


def test_long_chain_with_collider_and_conditioning():
    # Complex chain: X -> M -> N <- O -> Y and also X -> P -> Y
    # This creates both collider at N and an alternate path through P.
    g = Graph()
    g.add_directed_edge('X', 'M')
    g.add_directed_edge('M', 'N')
    g.add_directed_edge('O', 'N')
    g.add_directed_edge('O', 'Y')
    g.add_directed_edge('X', 'P')
    g.add_directed_edge('P', 'Y')

    # Without conditioning, several paths exist -> dependent
    assert not d_separated(g, ['X'], ['Y'])
    # Conditioning on P blocks the X->P->Y path, but the path via N  remain blocked
    assert d_separated(g, ['X'], ['Y'], ['P'])
    # Conditioning on N's descendant (none here) would open collider; conditioning on N itself opens collider
    assert not d_separated(g, ['X'], ['Y'], ['N'])
    # Conditioning on both P and N blocks both routes
    assert not d_separated(g, ['X'], ['Y'], ['P', 'N'])


def test_m_separation():
    g = Graph()
    g.add_vertices(["X", "Y", "Z", "M"])
    g.add_directed_edge("X", "M")
    g.add_confounded_edge("M", "Y")

    assert m_separated(g, X=["X"], Y=["Y"]) is True
    assert m_separated(g, X=["X"], Y=["Y"], Z=["M"]) is False


def test_mandatory_and_forbidden_edges_and_orientations():
    bk = BackgroundKnowledge()

    bk.add_mandatory_edge('A', 'B')
    bk.add_forbidden_edge('C', 'D')
    bk.add_mandatory_orientation('E', 'F')
    bk.add_forbidden_orientation('G', 'H')

    assert ('A', 'B') in bk.get_mandatory_edges()
    assert ('C', 'D') in bk.get_forbidden_edges()
    assert ('E', 'F') in bk.get_mandatory_orientations()
    assert ('G', 'H') in bk.get_forbidden_orientations()


def test_add_edges_from_and_orientations_from_and_non_descendants_from():
    bk = BackgroundKnowledge()
    bk.add_mandatory_edges_from({('X', 'Y'), ('U', 'V')})
    bk.add_forbidden_edges_from({('M', 'N')})
    bk.add_mandatory_orientations_from({('P', 'Q')})
    bk.add_forbidden_orientations_from({('R', 'S')})

    assert ('X', 'Y') in bk.get_mandatory_edges()
    assert ('M', 'N') in bk.get_forbidden_edges()
    assert ('P', 'Q') in bk.get_mandatory_orientations()
    assert ('R', 'S') in bk.get_forbidden_orientations()

    # non_descendants_from should add forbidden orientations for each nd
    bk.add_non_descendants_from('Z', {'A', 'B'})
    nd = bk.get_non_descendants()
    assert 'Z' in nd
    assert nd['Z'] == {'A', 'B'}
    # forbidden orientations should include (A,Z) and (B,Z) per implementation
    f_or = bk.get_forbidden_orientations()
    assert ('Z', 'A') in f_or and ('Z', 'B') in f_or


def test_add_non_descendant_and_remove():
    bk = BackgroundKnowledge()
    bk.add_non_descendant('T', 'S')
    nd = bk.get_non_descendants()
    assert 'S' in nd
    assert 'T' in nd['S']

    # adding non-descendant also adds forbidden orientation (node, non_descendant) in implementation
    f_or = bk.get_forbidden_orientations()
    assert ('T', 'S') in f_or

    # remove_non_descendant should remove the mapping and the forbidden orientation
    bk.remove_non_descendant('S', 'T')
    nd2 = bk.get_non_descendants()
    assert 'S' not in nd2 or 'T' not in nd2.get('S', set())
    # forbidden orientation should no longer include (T,S)
    assert ('T', 'S') not in bk.get_forbidden_orientations()


def test_getters_return_copies():
    bk = BackgroundKnowledge()
    bk.add_mandatory_edge('A', 'B')
    me = bk.get_mandatory_edges()
    me.add(('X', 'Y'))
    # original should not be modified
    assert ('X', 'Y') not in bk.get_mandatory_edges()

    bk.add_non_descendants_from('N', {'L'})
    nd = bk.get_non_descendants()
    nd['N'].add('Z')
    # original non_descendants should not contain 'Z'
    assert 'Z' not in bk.get_non_descendants().get('N', set())
