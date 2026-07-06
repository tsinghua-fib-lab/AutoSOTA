from abc import ABC, abstractmethod
from typing import Type, Optional
import pandas as pd
from itertools import combinations

from pyciphod.utils.graphs.partially_specified_graphs import CompletedPartiallyDirectedAcyclicGraph, PartialAncestralGraphs
from pyciphod.utils.graphs.orientation_rules import meek_rule_1, meek_rule_2, meek_rule_3
from pyciphod.utils.stat_tests.independence_tests import CiTests, FisherZTest as FisherZ
from pyciphod.utils.graphs.background_knowledge import BackgroundKnowledge


class ConstraintBased(ABC):
    def __init__(self, sparsity: float = 0.05, ci_test: Type[CiTests] = FisherZ, background_knowledge: Optional[BackgroundKnowledge] = None, twd: Optional[bool] = False):
        # self._data = data
        self._sparsity = sparsity
        self._ci_test = ci_test
        self._bk = background_knowledge if background_knowledge is not None else BackgroundKnowledge()
        self._twd = twd # Test wise deletion
        self.performed_tests = set()
        self.nb_ci_tests = 0
        self.sepset = dict()
        self.g_hat = self._initialize_graph() # Graph representation (e.g., CPDAG for PC, PAG for FCI)

    @abstractmethod
    def _initialize_graph(self):
        """Return the graph object used by the algorithm."""
        pass

    @abstractmethod
    def _apply_background_knowledge(self):
        """Apply background knowledge constraints to the graph structure."""
        pass

    @abstractmethod
    def _skeleton(self, data: pd.DataFrame = None, max_sepset_size: int = None):
        """Construct the skeleton of the graph using CI tests."""
        pass

    @abstractmethod
    def _orientation(self):
        """Orient edges using rules based on the skeleton and separation sets."""
        pass

    def run(self, data: pd.DataFrame = None):
        """Execute the full constraint-based algorithm."""
        self._skeleton(data)
        self._apply_background_knowledge()
        self._orientation()


class PC(ConstraintBased):
    """
    Implements the PC algorithm for causal discovery from observational data.

    Attributes:
        _data (pd.DataFrame): Observational data.
        _sparsity (float): Significance threshold for conditional independence tests.
        _ci_test (class): A test of the conditional independence test class (e.g., FisherZ).
        _nodes (list): List of variable names in the data.
        cpdag (CompletedPartiallyDirectedAcyclicGraph).
        nb_ci_tests (int): Number of CI tests performed.
        sepset (dict): Separation sets for node pairs in the skeleton.
    """

    def __init__(self, sparsity: float = 0.05, ci_test: CiTests = FisherZ, background_knowledge: BackgroundKnowledge = None, twd = False):
        super().__init__(sparsity, ci_test, background_knowledge, twd)
        # Additional attributes for FCI can be defined here, such as the PAG representation and rules for handling latent confounders and selection bias.

    def _initialize_graph(self):
        """Return the graph object used by the algorithm."""
        return CompletedPartiallyDirectedAcyclicGraph()

    def _skeleton(self, data: pd.DataFrame = None, max_sepset_size: int = None):
        """
        Construct the skeleton of the graph using an order-independent approach
        following Colombo & Maathuis (2014). Iteratively removes edges based on CI tests.
        """
        if max_sepset_size is None:
            max_sepset_size = len(data.columns) - 2
        nodes = list(data.columns)
        self.g_hat.add_undirected_edges_from(list(combinations(nodes, 2)))
        data_test = data
        s = 0
        repeat = True
        while repeat and s <= max_sepset_size:
            repeat = False
            adj = {x: self.g_hat.get_adjacencies(x) for x in nodes}
            treated = []
            for x in nodes:
                if len(adj[x]) - 1 >= s:
                    repeat = True
                    for y in sorted(adj[x]):
                        for S in combinations([a for a in sorted(adj[x]) if a != y], s):
                            if (y, x, S) not in treated:
                                treated.append((x, y, S))
                                test = self._ci_test(x, y, list(S), self._twd)
                                self.performed_tests.add((x,y,S))
                                self.nb_ci_tests += 1
                                if self._twd:
                                    data_test = data.dropna(subset=[x, y] + list(S))
                                try:
                                    pval = test.get_pvalue(data_test)
                                except:
                                    print(test, "relies on get_pvalue_by_permutation for p-value estimation")
                                    pval = test.get_pvalue_by_permutation(data_test)
                                if pval > self._sparsity:
                                    self.g_hat.remove_undirected_edge(x, y)
                                    self.sepset[(x, y)] = self.sepset[(y, x)] = S
                                    break
            s += 1

    def _apply_background_knowledge(self):
        """
        Apply background knowledge constraints to the CPDAG:
        1) Remove forbidden edges and add mandatory edges in the skeleton.
        2) Orient edges according to mandatory and forbidden orientations if present.
        """
        if not self._bk:
            return 

        # --- Step 1: enforce mandatory edges ---

        # Add mandatory edges if missing
        for u, v in self._bk.get_mandatory_edges():
            if (u, v) not in self.g_hat.get_undirected_edges() and (v, u) not in self.g_hat.get_undirected_edges():
                self.g_hat.add_undirected_edge(u, v)

        # --- Step 2: enforce orientations ---
        # Mandatory orientations
        for u, v in self._bk.get_mandatory_orientations():
            if (u, v) in self.g_hat.get_undirected_edges() or (v, u) in self.g_hat.get_undirected_edges():
                self.g_hat.remove_undirected_edge(u, v)
                self.g_hat.remove_undirected_edge(v, u)
                self.g_hat.add_directed_edge(u, v)

        # Forbidden orientations
        for u, v in self._bk.get_forbidden_orientations():
            if (u, v) in self.g_hat.get_undirected_edges():
                # Orient as v -> u instead
                self.g_hat.remove_undirected_edge(u, v)
                self.g_hat.add_directed_edge(v, u)
            elif (v, u) in self.g_hat.get_undirected_edges():
                # Orient as u -> v instead
                self.g_hat.remove_undirected_edge(v, u)
                self.g_hat.add_directed_edge(u, v)

    def _uc_rule(self):
        """
        Apply the Unshielded Collider (UC) rule:
        For each unshielded triple x - y - z with x and z not adjacent, 
        orient x -> y <- z if y not in sepset(x, z).
        """
        nodes = sorted(self.g_hat.get_vertices())
        adj = {x: self.g_hat.get_adjacencies(x) for x in nodes}
        for x in nodes:
            for y in sorted(adj[x]):
                for z in sorted(adj[y]):
                    if z == x or z in sorted(adj[x]):
                        continue
                    if (y not in self.sepset.get((x, z), [])):# and (y not in self.sepset.get((z, x), [])):
                        # TODO: add many ways to handle conflics
                        if ((x, y) in self.g_hat.get_undirected_edges()) and ((z, y) in self.g_hat.get_undirected_edges()):
                            # if ((x, y) not in self.g_hat.get_directed_edges()) and ((z, y) not in self.g_hat.get_directed_edges()):
                            self.g_hat.remove_undirected_edge(x, y)
                            self.g_hat.remove_undirected_edge(y, z)
                            self.g_hat.remove_undirected_edge(y, x)
                            self.g_hat.remove_undirected_edge(z, y)
                            self.g_hat.add_directed_edges_from([(x, y), (z, y)])

    def _apply_meek_rules(self):
        """
        Apply Meek's orientation rules iteratively using only edge sets:
        Rule 1, Rule 2, Rule 3 for propagating orientations in a CPDAG.
        Returns True if any edge was oriented, False otherwise.
        """
        nodes = self.g_hat.get_vertices()
        changed = False
        adj = {x: self.g_hat.get_adjacencies(x) for x in nodes}
        
        for x in nodes:
            for y in sorted(adj[x]):
                # Rule 1:
                for z in sorted(set(adj[y]) - set(adj[x])):
                    if meek_rule_1(self.g_hat, x, y, z):
                    # if (x,y) in self.g_hat.get_directed_edges() and (y,z) in self.g_hat.get_undirected_edges():
                        changed = True
                        self.g_hat.remove_undirected_edge(z, y)
                        self.g_hat.add_directed_edge(y, z)
                # Rule 2:
                for z in sorted(set(adj[y]) & set(adj[x])):
                    if meek_rule_2(self.g_hat, x, y, z):
                    # if (x,y) in self.g_hat.get_directed_edges() and (y,z) in self.g_hat.get_directed_edges() and (x,z) in self.g_hat.get_undirected_edges():
                        changed = True
                        self.g_hat.remove_undirected_edge(x, z)
                        self.g_hat.add_directed_edge(x, z)
        
        # Rule 3: 
        for x in nodes:
            for y in adj[x]:
                if (x, y) not in self.g_hat.get_directed_edges():
                    continue
                for z in sorted(set(adj[y]) - set(adj[x])):
                    if (z, y) not in self.g_hat.get_directed_edges():
                        continue
                    for w in sorted(set(adj[x]) & set(adj[y]) & set(adj[z])):
                        if meek_rule_3(self.g_hat, x, y, z, w):
                        # undirected_triplet = [(w, y), (w, x), (z, w)]
                        # if all(edge in self.g_hat.get_undirected_edges() for edge in undirected_triplet):
                            changed = True
                            self.g_hat.remove_undirected_edge(w, y)
                            self.g_hat.add_directed_edge(w, y)
        return changed

    def _orientation(self):
        """Orient edges using the UC rule and iterative Meek rules until convergence."""
        self._uc_rule()
        repeat = True
        while repeat:
            # self.g_hat, changed = apply_meek_rules(self.g_hat)
            repeat = self._apply_meek_rules()


class RestPC(PC):
    def __init__(self, sparsity: float = 0.05, ci_test: CiTests = FisherZ, background_knowledge: BackgroundKnowledge = None, twd = False):
        super().__init__(sparsity, ci_test, background_knowledge, twd)
        # Additional attributes for RestPC can be defined here, such as the representation of the rest graph and rules for handling selection bias.

    def _skeleton(self, data: pd.DataFrame = None, max_sepset_size: int = 1):
        super()._skeleton(data, max_sepset_size=0)  # Run only one iteration of the skeleton phase of PC

    def _orientation(self):
        """Orient edges using the UC rule."""
        self._uc_rule()


class FCI(ConstraintBased):
    def __init__(
        self,
        sparsity: float = 0.05,
        ci_test: CiTests = FisherZ,
        background_knowledge: BackgroundKnowledge = None,
        twd=False,
    ):
        super().__init__(sparsity, ci_test, background_knowledge, twd)

    def _initialize_graph(self):
        return PartialAncestralGraphs()

    def _skeleton(self, data: pd.DataFrame = None, max_sepset_size: int = None):
        """
        FCI skeleton phase:
        1) Run the PC/FAS adjacency search on a CPDAG skeleton.
        2) Convert the surviving adjacencies into a PAG skeleton (o-o).
        3) Orient temporary unshielded colliders to compute Possible-D-Sep.
        4) Refine the skeleton using Possible-D-Sep.
        5) Rebuild a clean PAG skeleton (all remaining edges as o-o).
        """
        if max_sepset_size is None:
            max_sepset_size = len(data.columns) - 2

        nodes = list(data.columns)

        # ----------------------------------------------------------
        # Step 1: Use PC skeleton code (FAS)
        # ----------------------------------------------------------
        pc_tmp = PC(
            sparsity=self._sparsity,
            ci_test=self._ci_test,
            background_knowledge=None,
            twd=self._twd,
        )
        pc_tmp._skeleton(data=data, max_sepset_size=max_sepset_size)

        self.performed_tests = set(pc_tmp.performed_tests)
        self.nb_ci_tests = pc_tmp.nb_ci_tests
        self.sepset = dict(pc_tmp.sepset)

        # ----------------------------------------------------------
        # Step 2: Convert CPDAG skeleton -> PAG skeleton (o-o)
        # ----------------------------------------------------------
        self.g_hat = PartialAncestralGraphs()
        self.g_hat.add_vertices(nodes)

        for x, y in combinations(nodes, 2):
            if y in pc_tmp.g_hat.get_adjacencies(x):
                self.g_hat.add_uncertain_edge(x, y, edge_type="o-o")

        # ----------------------------------------------------------
        # Step 3: Temporary collider orientation for Possible-D-Sep
        # ----------------------------------------------------------
        adj = {x: self.g_hat.get_adjacencies(x) for x in nodes}
        for y in nodes:
            neigh = sorted(adj[y])
            for i in range(len(neigh)):
                x = neigh[i]
                for j in range(i + 1, len(neigh)):
                    z = neigh[j]
                    if self.g_hat.is_adjacent(x, z):
                        continue
                    if y not in self.sepset.get((x, z), []):
                        # orient x *-> y
                        self.g_hat.remove_uncertain_edge(x, y)
                        self.g_hat.remove_undirected_edge(x, y)
                        self.g_hat.remove_directed_edge(x, y)
                        self.g_hat.remove_directed_edge(y, x)
                        self.g_hat.remove_confounded_edge(x, y)
                        self.g_hat.add_uncertain_edge(x, y, edge_type="o->")

                        # orient z *-> y
                        self.g_hat.remove_uncertain_edge(z, y)
                        self.g_hat.remove_undirected_edge(z, y)
                        self.g_hat.remove_directed_edge(z, y)
                        self.g_hat.remove_directed_edge(y, z)
                        self.g_hat.remove_confounded_edge(z, y)
                        self.g_hat.add_uncertain_edge(z, y, edge_type="o->")

        # ----------------------------------------------------------
        # Step 4: Possible-D-Sep refinement
        # ----------------------------------------------------------
        # Possible-D-Sep(x, y): all nodes v such that there exists a path
        # x = v0, v1, ..., vk = v where every internal node vi on the path
        # is either a collider on that path or lies in a triangle.
        for x, y in combinations(nodes, 2):
            if not self.g_hat.is_adjacent(x, y):
                continue

            possible_dsep_xy = set()
            allowed_nodes = set(self.g_hat.get_vertices()) - {y}
            cutoff = len(nodes)

            for v in self.g_hat.get_vertices():
                if v in {x, y}:
                    continue

                found_valid_path = False
                for path in self.g_hat.get_simple_paths(
                    x, v, allowed_nodes=allowed_nodes, cutoff=cutoff
                ):
                    valid = True
                    if len(path) >= 3:
                        for k in range(1, len(path) - 1):
                            a = path[k - 1]
                            b = path[k]
                            c = path[k + 1]

                            is_collider = (
                                self.g_hat.is_pointed_edge(a, b)
                                and self.g_hat.is_pointed_edge(c, b)
                            )
                            is_triangle = self.g_hat.is_adjacent(a, c)

                            if not (is_collider or is_triangle):
                                valid = False
                                break
                    if valid:
                        found_valid_path = True
                        break

                if found_valid_path:
                    possible_dsep_xy.add(v)

            # also add the reverse Possible-D-Sep(y, x)
            allowed_nodes = set(self.g_hat.get_vertices()) - {x}
            for v in self.g_hat.get_vertices():
                if v in {x, y} or v in possible_dsep_xy:
                    continue

                found_valid_path = False
                for path in self.g_hat.get_simple_paths(
                    y, v, allowed_nodes=allowed_nodes, cutoff=cutoff
                ):
                    valid = True
                    if len(path) >= 3:
                        for k in range(1, len(path) - 1):
                            a = path[k - 1]
                            b = path[k]
                            c = path[k + 1]

                            is_collider = (
                                self.g_hat.is_pointed_edge(a, b)
                                and self.g_hat.is_pointed_edge(c, b)
                            )
                            is_triangle = self.g_hat.is_adjacent(a, c)

                            if not (is_collider or is_triangle):
                                valid = False
                                break
                    if valid:
                        found_valid_path = True
                        break

                if found_valid_path:
                    possible_dsep_xy.add(v)

            possible_dsep_xy = sorted(possible_dsep_xy)
            if len(possible_dsep_xy) == 0:
                continue

            removed = False
            max_k = min(max_sepset_size, len(possible_dsep_xy))
            for s in range(max_k + 1):
                for S in combinations(possible_dsep_xy, s):
                    test = self._ci_test(x, y, list(S), self._twd)
                    self.performed_tests.add((x, y, tuple(S)))
                    self.nb_ci_tests += 1

                    data_test = data
                    if self._twd:
                        data_test = data.dropna(subset=[x, y] + list(S))

                    try:
                        pval = test.get_pvalue(data_test)
                    except Exception:
                        pval = test.get_pvalue_by_permutation(data_test)

                    # print("FCI PDS", x, y, S, pval)

                    if pval > self._sparsity:
                        self.g_hat.remove_uncertain_edge(x, y)
                        self.g_hat.remove_undirected_edge(x, y)
                        self.g_hat.remove_directed_edge(x, y)
                        self.g_hat.remove_directed_edge(y, x)
                        self.g_hat.remove_confounded_edge(x, y)
                        self.sepset[(x, y)] = self.sepset[(y, x)] = S
                        removed = True
                        break
                if removed:
                    break

        # ----------------------------------------------------------
        # Step 5: rebuild a clean PAG skeleton with only o-o edges
        # ----------------------------------------------------------
        remaining_edges = []
        for x, y in combinations(nodes, 2):
            if self.g_hat.is_adjacent(x, y):
                remaining_edges.append((x, y))

        self.g_hat = PartialAncestralGraphs()
        self.g_hat.add_vertices(nodes)
        for x, y in remaining_edges:
            self.g_hat.add_uncertain_edge(x, y, edge_type="o-o")

    def _apply_background_knowledge(self):
        if not self._bk:
            return

        # Add mandatory adjacencies
        for u, v in self._bk.get_mandatory_edges():
            if not self.g_hat.is_adjacent(u, v):
                self.g_hat.add_uncertain_edge(u, v, edge_type="o-o")

        # Mandatory orientations
        for u, v in self._bk.get_mandatory_orientations():
            if self.g_hat.is_adjacent(u, v):
                self.g_hat.remove_uncertain_edge(u, v)
                self.g_hat.remove_undirected_edge(u, v)
                self.g_hat.remove_directed_edge(u, v)
                self.g_hat.remove_directed_edge(v, u)
                self.g_hat.remove_confounded_edge(u, v)
                self.g_hat.add_directed_edge(u, v)

        # Forbidden orientations: orient the reverse if still adjacent
        for u, v in self._bk.get_forbidden_orientations():
            if self.g_hat.is_adjacent(u, v):
                self.g_hat.remove_uncertain_edge(u, v)
                self.g_hat.remove_undirected_edge(u, v)
                self.g_hat.remove_directed_edge(u, v)
                self.g_hat.remove_directed_edge(v, u)
                self.g_hat.remove_confounded_edge(u, v)
                self.g_hat.add_directed_edge(v, u)

    def _orientation(self):
        """ TODO: implement the full FCI orientation phase with all rules (R1-R10) and handling of selection bias. For now, we implement a simplified version that only applies the unshielded collider rule and a few propagation rules inspired by R1-R3. The full implementation would require careful handling of edge types and additional rules for selection bias and confounding. """
        nodes = sorted(self.g_hat.get_vertices())

        # ----------------------------------------------------------
        # Unshielded colliders
        # ----------------------------------------------------------
        adj = {x: self.g_hat.get_adjacencies(x) for x in nodes}
        for y in nodes:
            neigh = sorted(adj[y])
            for i in range(len(neigh)):
                x = neigh[i]
                for j in range(i + 1, len(neigh)):
                    z = neigh[j]
                    if self.g_hat.is_adjacent(x, z):
                        continue
                    if y not in self.sepset.get((x, z), []):
                        self.g_hat.remove_uncertain_edge(x, y)
                        self.g_hat.remove_undirected_edge(x, y)
                        self.g_hat.remove_directed_edge(x, y)
                        self.g_hat.remove_directed_edge(y, x)
                        self.g_hat.remove_confounded_edge(x, y)
                        self.g_hat.add_uncertain_edge(x, y, edge_type="o->")

                        self.g_hat.remove_uncertain_edge(z, y)
                        self.g_hat.remove_undirected_edge(z, y)
                        self.g_hat.remove_directed_edge(z, y)
                        self.g_hat.remove_directed_edge(y, z)
                        self.g_hat.remove_confounded_edge(z, y)
                        self.g_hat.add_uncertain_edge(z, y, edge_type="o->")

        # ----------------------------------------------------------
        # Simple propagation rules
        # ----------------------------------------------------------
        changed = True
        while changed:
            changed = False
            adj = {x: self.g_hat.get_adjacencies(x) for x in nodes}

            # R1-like: x *-> y o-o z and x not adj z  => y -> z
            for x in nodes:
                for y in sorted(adj[x]):
                    if not self.g_hat.is_pointed_edge(x, y):
                        continue
                    for z in sorted(adj[y] - {x}):
                        if self.g_hat.is_adjacent(x, z):
                            continue
                        if self.g_hat.is_pointed_edge(y, z) or self.g_hat.is_pointed_edge(z, y):
                            continue
                        self.g_hat.remove_uncertain_edge(y, z)
                        self.g_hat.remove_undirected_edge(y, z)
                        self.g_hat.remove_directed_edge(y, z)
                        self.g_hat.remove_directed_edge(z, y)
                        self.g_hat.remove_confounded_edge(y, z)
                        self.g_hat.add_directed_edge(y, z)
                        changed = True

            # R2-like: x -> y -> z and x o-o z => x -> z
            adj = {x: self.g_hat.get_adjacencies(x) for x in nodes}
            for x in nodes:
                for y in sorted(self.g_hat.get_children(x)):
                    for z in sorted(self.g_hat.get_children(y)):
                        if x == z:
                            continue
                        if not self.g_hat.is_adjacent(x, z):
                            continue
                        if self.g_hat.is_pointed_edge(x, z) or self.g_hat.is_pointed_edge(z, x):
                            continue
                        self.g_hat.remove_uncertain_edge(x, z)
                        self.g_hat.remove_undirected_edge(x, z)
                        self.g_hat.remove_directed_edge(x, z)
                        self.g_hat.remove_directed_edge(z, x)
                        self.g_hat.remove_confounded_edge(x, z)
                        self.g_hat.add_directed_edge(x, z)
                        changed = True

            # R3-like: x o-o y and z -> y <- w, x adj z, x adj w, z not adj w => x -> y
            adj = {x: self.g_hat.get_adjacencies(x) for x in nodes}
            parents = {y: self.g_hat.get_parents(y) for y in nodes}

            for y in nodes:
                py = sorted(parents[y])
                if len(py) < 2:
                    continue
                for z, w in combinations(py, 2):
                    if self.g_hat.is_adjacent(z, w):
                        continue
                    for x in sorted(adj[y] - {z, w}):
                        if not self.g_hat.is_adjacent(x, z):
                            continue
                        if not self.g_hat.is_adjacent(x, w):
                            continue
                        if self.g_hat.is_pointed_edge(x, y) or self.g_hat.is_pointed_edge(y, x):
                            continue
                        self.g_hat.remove_uncertain_edge(x, y)
                        self.g_hat.remove_undirected_edge(x, y)
                        self.g_hat.remove_directed_edge(x, y)
                        self.g_hat.remove_directed_edge(y, x)
                        self.g_hat.remove_confounded_edge(x, y)
                        self.g_hat.add_directed_edge(x, y)
                        changed = True

