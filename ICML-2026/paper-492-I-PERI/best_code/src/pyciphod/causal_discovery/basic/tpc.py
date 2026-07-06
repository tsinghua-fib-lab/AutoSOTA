import pandas as pd
from itertools import combinations

from PyCIPHOD.utils.graphs.partially_specified_graphs import TemporalPartiallyDirectedAcyclicGraph
from PyCIPHOD.utils.independence_tests.basic import CiTests, FisherZ
from PyCIPHOD.utils.background_knowledge.background_knowledge import BackgroundKnowledge


#### TO IMPLEMENT (Now : copy past of PC with only CPDAG -> TPDAG)

class TPC:
    """
    Implements the PC algorithm for causal discovery from observational data.

    Attributes:
        _data (pd.DataFrame): Observational data.
        _sparsity (float): Significance threshold for conditional independence tests.
        _ci_test (class): A test of the conditional independence test class (e.g., FisherZ).
        _nodes (list): List of variable names in the data.
        tpdag (TemporalPartiallyDirectedAcyclicGraph).
        nb_ci_tests (int): Number of CI tests performed.
        sepset (dict): Separation sets for node pairs in the skeleton.
    """

    def __init__(self, data: pd.DataFrame,  sparsity: float = 0.05, ci_test: CiTests = FisherZ, background_knowledge: BackgroundKnowledge = None):
        """Initialize PC algorithm with data, sparsity threshold, and CI test."""
        self._data = data
        self._sparsity = sparsity
        self._ci_test = ci_test
        self._bk = background_knowledge

        self._nodes = list(data.columns)
        self.tpdag = TemporalPartiallyDirectedAcyclicGraph()

        self.nb_ci_tests = 0
        self.sepset = dict()

    def _skeleton(self):
        """
        Construct the skeleton of the graph using an order-independent approach
        following Colombo & Maathuis (2014). Iteratively removes edges based on CI tests.
        """
        self.tpdag.add_undirected_edges_from(list(combinations(self._nodes, 2)))
        s = 0
        repeat = True
        while repeat:
            repeat = False
            adj = {x: self.tpdag.get_adjacencies(x) for x in self._nodes}
            for x in self._nodes:
                if len(adj[x]) - 1 >= s:
                    repeat = True
                    for y in adj[x]:
                        for S in combinations([a for a in adj[x] if a != y], s):
                            test = self._ci_test(x, y, list(S))
                            self.nb_ci_tests += 1
                            if test.get_pvalue(self._data) > self._sparsity:
                                self.tpdag.remove_undirected_edge(x, y)
                                self.sepset[(x, y)] = self.sepset[(y, x)] = S
                                break
            s += 1
            
    
    def _apply_background_knowledge(self):
        """
        Apply background knowledge constraints to the tpdag:
        1) Remove forbidden edges and add mandatory edges in the skeleton.
        2) Orient edges according to mandatory and forbidden orientations if present.
        """
        if not self._bk:
            return 

        # --- Step 1: enforce mandatory and forbidden edges ---
        # Remove forbidden edges if present
        # for u, v in self._bk.get_forbidden_edges():
        #     if (u, v) in self.tpdag.get_undirected_edges() or (v, u) in self.tpdag.get_undirected_edges():
        #         self.tpdag.remove_undirected_edge(u, v)
        #         self.tpdag.remove_undirected_edge(v, u)

        # Add mandatory edges if missing
        for u, v in self._bk.get_mandatory_edges():
            if (u, v) not in self.tpdag.get_undirected_edges() and (v, u) not in self.tpdag.get_undirected_edges():
                self.tpdag.add_undirected_edge(u, v)

        # --- Step 2: enforce orientations ---
        # Mandatory orientations
        for u, v in self._bk.get_mandatory_orientations():
            if (u, v) in self.tpdag.get_undirected_edges() or (v, u) in self.tpdag.get_undirected_edges():
                self.tpdag.remove_undirected_edge(u, v)
                self.tpdag.remove_undirected_edge(v, u)
                self.tpdag.add_directed_edge(u, v)

        # Forbidden orientations
        for u, v in self._bk.get_forbidden_orientations():
            if (u, v) in self.tpdag.get_undirected_edges():
                # Orient as v -> u instead
                self.tpdag.remove_undirected_edge(u, v)
                self.tpdag.add_directed_edge(v, u)
            elif (v, u) in self.tpdag.get_undirected_edges():
                # Orient as u -> v instead
                self.tpdag.remove_undirected_edge(v, u)
                self.tpdag.add_directed_edge(u, v)
            

    def _uc_rule(self):
        """
        Apply the Unshielded Collider (UC) rule:
        For each unshielded triple x - y - z with x and z not adjacent, 
        orient x -> y <- z if y not in sepset(x, z).
        """
        adj = {x: self.tpdag.get_adjacencies(x) for x in self._nodes}
        for x in self._nodes:
            for y in adj[x]:
                for z in adj[y]:
                    if z == x or z in adj[x]:
                        continue
                    if y not in self.sepset.get((x, z), []):
                        self.tpdag.remove_undirected_edge(x, y)
                        self.tpdag.remove_undirected_edge(y, z)
                        self.tpdag.add_directed_edges_from([(x, y), (z, y)])

    def _meek_rules(self):
        """
        Apply Meek's orientation rules iteratively using only edge sets:
        Rule 1, Rule 2, Rule 3 for propagating orientations in a tpdag.
        Returns True if any edge was oriented, False otherwise.
        """
        changed = False
        adj = {x: self.tpdag.get_adjacencies(x) for x in self._nodes}
        
        for x in self._nodes:
            for y in adj[x]:
                # Rule 1:
                for z in set(adj[y]) - set(adj[x]):
                    if (x,y) in self.tpdag.get_directed_edges() and (y,z) in self.tpdag.get_undirected_edges():
                        changed = True
                        self.tpdag.remove_undirected_edge(z, y)
                        self.tpdag.add_directed_edge(y, z)
                # Rule 2:
                for z in set(adj[y]) & set(adj[x]):
                    if (x,y) in self.tpdag.get_directed_edges() and (y,z) in self.tpdag.get_directed_edges() and (x,z) in self.tpdag.get_undirected_edges():
                        changed = True
                        self.tpdag.remove_undirected_edge(x, z)
                        self.tpdag.add_directed_edge(x, z)
        
        # Rule 3: 
        for x in self._nodes:
            for y in adj[x]:
                if (x, y) not in self.tpdag.get_directed_edges():
                    continue
                for z in set(adj[y]) - set(adj[x]):
                    if (z, y) not in self.tpdag.get_directed_edges():
                        continue
                    for w in set(adj[x]) & set(adj[y]) & set(adj[z]):
                        undirected_triplet = [(w, y), (w, x), (z, w)]
                        if all(edge in self.tpdag.get_undirected_edges() for edge in undirected_triplet):
                            changed = True
                            self.tpdag.remove_undirected_edge(w, y)
                            self.tpdag.add_directed_edge(w, y)

        return changed

    def _orientation(self):
        """Orient edges using the UC rule and iterative Meek rules until convergence."""
        self._uc_rule()
        repeat = True
        while repeat:
            repeat = self._meek_rules()

    def run(self):
        """
        Execute the full PC algorithm:
        1. Construct skeleton (order-independent)
        2. Apply UC and Meek orientation rules
        """
        self._skeleton()
        self._apply_background_knowledge()
        self._orientation()