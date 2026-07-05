import pytest

from pyciphod.causal_reasoning.basic.criteria import back_door_criterion, front_door_criterion
from pyciphod.causal_reasoning.basic.do_calculus import (
    rule1_applies,
    rule2_applies,
    rule3_applies,
)
from pyciphod.utils.graphs.graphs import DirectedMixedGraph


def test_rule1_backdoor():
    g = DirectedMixedGraph()
    g.add_vertex('X')
    g.add_vertex('Y')
    g.add_vertex('Z')
    g.add_vertex('W')
    g.add_directed_edge('X', 'Y')
    g.add_directed_edge('W', 'X')
    g.add_directed_edge('Z', 'Y')
    g.add_directed_edge('W', 'Z')

    assert back_door_criterion(g, X={'X'}, Y={'Y'},  Z={'Z', 'W'}) is True


# def test_rule1_frontdoor():
#     1


def test_rule1_simple_disconnected_Z():
    # Graph: X -> Y, Z isolated. After removing incoming to X, Z is still isolated => rule1 should apply
    g = DirectedMixedGraph()
    g.add_vertex('X')
    g.add_vertex('Y')
    g.add_vertex('Z')
    g.add_directed_edge('X', 'Y')

    assert rule1_applies(g, Y='Y', X='X', Z='Z') is True


def test_rule1_negative_connected_Z():
    # Graph: X -> Y <- Z. Z directly affects Y, so rule1 should NOT apply
    g = DirectedMixedGraph()
    g.add_vertices(['X', 'Y', 'Z'])
    g.add_directed_edge('X', 'Y')
    g.add_directed_edge('Z', 'Y')

    assert rule1_applies(g, Y='X', X={}, Z='Z',  W='Y') is False


def test_rule2_simple_remove_outgoing():
    # Graph: Z -> X -> Y. In G_barX (remove incoming to X) and removing outgoing from Z removes Z->X,
    # leaving Z disconnected from Y: rule2 should apply.
    g = DirectedMixedGraph()
    g.add_vertices(['X', 'Y', 'Z'])
    g.add_directed_edge('Z', 'X')
    g.add_directed_edge('X', 'Y')

    assert rule2_applies(g, Y='Y',X='X', Z='Z') is True


def test_rule2_negative_when_path_remains():
    # Graph: X -> Y -> Z. Removing outgoing from Z doesn't remove Y->Z, so dependency remains -> rule2 false
    g = DirectedMixedGraph()
    g.add_vertices(['X', 'Y', 'Z'])
    g.add_directed_edge('X', 'Y')
    g.add_directed_edge('Y', 'Z')

    assert rule2_applies(g, Y='Y', X='X',  Z='Z') is False


def test_rule3_simple_true():
    # Graph: Z -> X -> Y. For rule3 with W empty, Z* = Z (not ancestor of any W), removing incoming to X (G_barX)
    # and incoming to Z* removes Z->X, so Z is separated from Y given X: rule3 should apply.
    g = DirectedMixedGraph()
    g.add_vertices(['X', 'Y', 'Z'])
    g.add_directed_edge('Z', 'X')
    g.add_directed_edge('X', 'Y')

    assert rule3_applies(g, X='X', Y='Y', Z='Z', W=None) is True


def test_rule3_negative_when_ancestor_of_W():
    # Graph: Z -> W -> Y and X -> Y. Here Z is an ancestor of W; for rule3 Z* excludes Z, so rule3 should not apply
    g = DirectedMixedGraph()
    g.add_vertices(['X', 'Y', 'Z', 'W'])
    g.add_directed_edge('X', 'Y')
    g.add_directed_edge('Z', 'W')
    g.add_directed_edge('W', 'Y')

    assert rule3_applies(g, X='X', Y='Y', Z='Z', W='W') is False
