import pytest
import pandas as pd

from pyciphod.causal_discovery.basic.constraint_based import PC
from pyciphod.utils.graphs.background_knowledge import BackgroundKnowledge


@pytest.fixture
def pc():
    """Fixture returning a fresh PC instance for tests."""
    return PC()


class DummyTest:
    """Dummy CI test class to contrôler les p-values retournées pour (x,y,S)."""

    def __init__(self, x, y, cond_list=None, drop_na=False):
        self.x = x
        self.y = y
        self.cond_list = cond_list or []
        self.drop_na = drop_na

    P_MAP = {}

    def get_pvalue(self, df):
        key = (self.x, self.y, tuple(self.cond_list), self.drop_na)
        # Essayer la clé symétrique si absente
        if key in DummyTest.P_MAP:
            return DummyTest.P_MAP[key]
        key_sym = (self.y, self.x, tuple(self.cond_list))
        return DummyTest.P_MAP.get(key_sym, 0.0)


def test_skeleton_removes_and_keeps_edges():
    df = pd.DataFrame({"A": [1, 2, 3, 4], "B": [1, 1, 1, 1], "C": [4, 3, 2, 1]})

    # Définir les p-values : enlever A-B (p > 0.05), garder A-C (p <= 0.05), garder B-C
    DummyTest.P_MAP = {
        ("A", "B", ()): 0.5,
        ("A", "C", ()): 0.01,
        ("B", "C", ()): 0.01,
    }

    pc_alg = PC(sparsity=0.05, ci_test=DummyTest)
    pc_alg._skeleton(data=df)

    # Vérifier que des tests CI ont été effectués et que la paire A-B a été séparée (edge supprimée)
    assert pc_alg.nb_ci_tests > 0
    assert any((t[0] == 'A' and t[1] == 'B') or (t[0] == 'B' and t[1] == 'A') for t in pc_alg.performed_tests)
    assert ('A', 'B') in pc_alg.sepset or ('B', 'A') in pc_alg.sepset

    # Pour les arêtes qui doivent rester, il ne devrait pas y avoir d'entrée dans sepset
    assert ('A', 'C') not in pc_alg.sepset and ('C', 'A') not in pc_alg.sepset
    assert ('B', 'C') not in pc_alg.sepset and ('C', 'B') not in pc_alg.sepset


def test_apply_background_knowledge_orients_and_adds_edges(pc):
    bk = BackgroundKnowledge()

    # Mandatory undirected edge U-V (sera ajouté si manquant)
    bk.add_mandatory_edge("U", "V")

    # Mandatory orientation W -> X
    bk.add_mandatory_orientation("W", "X")

    # Forbidden orientation Y -> Z (donc si Y-Z non orienté existera il doit être orienté Z -> Y)
    bk.add_forbidden_orientation("Y", "Z")

    pc._bk = bk

    # Préparer le graphe: ajouter sommets et arêtes non-orientées
    pc.g_hat.add_vertices(["U", "V", "W", "X", "Y", "Z"])
    pc.g_hat.add_undirected_edge("W", "X")
    pc.g_hat.add_undirected_edge("Y", "Z")

    # Appliquer la connaissance a priori
    pc._apply_background_knowledge()

    # Mandatory orientation W -> X doit produire l'arête dirigée (W,X)
    directed = pc.g_hat.get_directed_edges()
    assert ("W", "X") in directed

    # Forbidden orientation Y -> Z doit avoir orienté comme Z -> Y
    assert ("Z", "Y") in directed


def test_uc_rule_orients_unshielded_collider(pc):
    # Ajouter sommets et chemin non-orienté X - Y - Z
    pc.g_hat.add_vertices(["X", "Y", "Z"])
    pc.g_hat.add_undirected_edge("X", "Y")
    pc.g_hat.add_undirected_edge("Y", "Z")

    # S'assurer que X et Z ne sont pas adjacents
    assert ("X", "Z") not in pc.g_hat.get_undirected_edges()

    # sepset (X,Z) doit être vide donc Y non présent
    pc.sepset[("X", "Z")] = []

    pc._uc_rule()

    directed = pc.g_hat.get_directed_edges()
    # S'attend à X -> Y et Z -> Y
    assert ("X", "Y") in directed
    assert ("Z", "Y") in directed


def test_apply_meek_rules_rule1_and_rule2_and_rule3():
    pc = PC()
    # Rule1: X -> Y et Y - Z (non-orientée) et X et Z non adjacents => orienter Y -> Z
    pc.g_hat.add_vertices(["X1", "Y1", "Z1"])
    pc.g_hat.add_directed_edge("X1", "Y1")
    pc.g_hat.add_undirected_edge("Y1", "Z1")
    assert ("Y1", "Z1") in pc.g_hat.get_undirected_edges()

    changed = pc._apply_meek_rules()
    assert changed is True
    assert ("Y1", "Z1") in pc.g_hat.get_directed_edges()

    # Rule2: X -> Y et Y -> Z et X - Z non-orienté => orienter X -> Z
    pc2 = PC()
    pc2.g_hat.add_vertices(["X2", "Y2", "Z2"])
    pc2.g_hat.add_directed_edge("X2", "Y2")
    pc2.g_hat.add_directed_edge("Y2", "Z2")
    pc2.g_hat.add_undirected_edge("X2", "Z2")

    changed2 = pc2._apply_meek_rules()
    assert changed2 is True
    assert ("X2", "Z2") in pc2.g_hat.get_directed_edges()

    # Rule3: X -> Y, Z -> Y, X - Z non-orienté, et existe W avec W-Y, W-X, W-Z non-orientées => orienter W -> Y
    pc3 = PC()
    pc3.g_hat.add_vertices(["X3", "Y3", "Z3", "W3"])
    pc3.g_hat.add_directed_edge("X3", "Y3")
    pc3.g_hat.add_directed_edge("Z3", "Y3")
    pc3.g_hat.add_undirected_edge("X3", "Z3")
    # Connexions de W non-orientées
    pc3.g_hat.add_undirected_edge("W3", "Y3")
    pc3.g_hat.add_undirected_edge("W3", "X3")
    pc3.g_hat.add_undirected_edge("W3", "Z3")

    changed3 = pc3._apply_meek_rules()
    assert changed3 is True
    # W3 -> Y3 devrait être dirigée maintenant
    assert ("W3", "Y3") in pc3.g_hat.get_directed_edges()
