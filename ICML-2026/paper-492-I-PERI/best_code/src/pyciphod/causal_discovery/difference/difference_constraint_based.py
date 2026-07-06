from abc import ABC, abstractmethod
from typing import Type, Optional
import pandas as pd
from itertools import combinations
import warnings
import numpy as np

from pyciphod.utils.graphs.partially_specified_graphs import (
    CompletedPartiallyDirectedAcyclicDifferenceGraph,
)
from pyciphod.utils.stat_tests.equality_tests import CeTests, PartialCorrelationEqualityTest, LinearRegressionCoefficientEqualityTest
from pyciphod.utils.graphs.background_knowledge import BackgroundKnowledge


class DifferenceConstraintBased(ABC):
    """
    Base class for difference-oriented constraint-based algorithms.

    This mirrors `ConstraintBased` but operates on two populations (df1, df2)
    and uses equality tests (classes derived from `CeTests`) to decide whether
    a dependence/statistic differs across the two datasets.
    """

    def __init__(
        self,
        sparsity: float = 0.05,
        eq_test: Type[CeTests] = LinearRegressionCoefficientEqualityTest,
        background_knowledge: Optional[BackgroundKnowledge] = None,
        twd: Optional[bool] = False,
        n_permutations: int = 1000,
        seed: Optional[int] = None,
    ):
        self._sparsity = sparsity
        self._eq_test = eq_test
        self._bk = background_knowledge if background_knowledge is not None else BackgroundKnowledge()
        self._twd = twd
        self.performed_tests = set()
        # reuse the existing name for counting tests
        self.nb_ci_tests = 0
        self.sepset = dict()
        self._n_permutations = n_permutations
        self._seed = seed
        self.g_hat = self._initialize_graph()

    @abstractmethod
    def _initialize_graph(self):
        """Return the graph object used by the algorithm."""
        pass

    @abstractmethod
    def _apply_background_knowledge(self):
        """Apply background knowledge constraints to the graph structure."""
        pass

    @abstractmethod
    def _skeleton(self, df1: pd.DataFrame = None, df2: pd.DataFrame = None, max_sepset_size: int = None):
        """Construct the skeleton of the difference graph using equality tests."""
        pass

    @abstractmethod
    def _orientation(self):
        """Orient edges using rules based on the skeleton and separation sets."""
        pass

    def run(self, df1: pd.DataFrame = None, df2: pd.DataFrame = None):
        """Execute the full difference constraint-based algorithm."""
        self._skeleton(df1, df2)
        self._apply_background_knowledge()
        self._orientation()


class LinearDifferencePC(DifferenceConstraintBased):
    """
    Difference analogue of the PC algorithm: learns a
    CompletedPartiallyDirectedAcyclicDifferenceGraph using equality tests
    between two datasets.
    """

    def __init__(
        self,
        sparsity: float = 0.05,
        eq_test: Type[CeTests] = LinearRegressionCoefficientEqualityTest,
        background_knowledge: BackgroundKnowledge = None,
        twd: bool = False,
        n_permutations: int = 1000,
        seed: Optional[int] = None,
    ):
        super().__init__(sparsity, eq_test, background_knowledge, twd, n_permutations, seed)

    def _initialize_graph(self):
        return CompletedPartiallyDirectedAcyclicDifferenceGraph()

    def _skeleton(self, df1: pd.DataFrame = None, df2: pd.DataFrame = None, max_sepset_size: int = None):
        """
        Construct the skeleton of the difference graph by testing, for each pair
        of variables (x, y) and conditioning set S, whether the dependence
        statistic differs across the two populations. If the equality null
        (no difference) is not rejected (pval > sparsity) we remove the
        undirected edge (no difference detected under S).
        """
        if df1 is None or df2 is None:
            raise ValueError("Both df1 and df2 must be provided to learn a difference graph.")

        if max_sepset_size is None:
            max_sepset_size = len(df1.columns) - 2

        nodes = list(df1.columns)
        # Start with complete undirected graph
        self.g_hat.add_undirected_edges_from(list(combinations(nodes, 2)))

        s = 0
        repeat = True

        while repeat and s < max_sepset_size:
            repeat = False
            adj = {x: self.g_hat.get_adjacencies(x) for x in nodes}
            for x in nodes:
                if len(adj[x]) - 1 >= s:
                    repeat = True
                    for y in sorted(adj[x]):
                        possible_S = [a for a in nodes if a != y and a!= x]
                        for S in combinations(possible_S, s):
                            # instantiate equality-test for (x,y) conditioned on S
                            test = self._eq_test(x, y, list(S), self._twd)
                            self.performed_tests.add((x, y, S))
                            self.nb_ci_tests += 1

                            # possibly drop rows if test-wise-deletion is enabled
                            df1_test = df1
                            df2_test = df2
                            if self._twd:
                                cols = [x, y] + list(S)
                                df1_test = df1.dropna(subset=cols)
                                df2_test = df2.dropna(subset=cols)

                            try:
                                pval = test.get_pvalue(
                                    df1_test,
                                    df2_test,
                                )
                                # pval = test.get_pvalue_by_permutation(
                                #     df1_test,
                                #     df2_test,
                                #     n_permutations=self._n_permutations,
                                #     seed=self._seed,
                                # )
                            except Exception as e:
                                warnings.warn(f"Equality test failed for ({x},{y},S={S}): {e}")
                                pval = None

                            # If pval is undefined, skip
                            if pval is None or (isinstance(pval, float) and np.isnan(pval)):
                                continue

                            print(pval, " for test ", (x, y, S))
                            # If we cannot reject equality (pval > alpha), remove edge
                            if pval > self._sparsity:
                                self.g_hat.remove_undirected_edge(x, y)
                                self.sepset[(x, y)] = self.sepset[(y, x)] = S
                                break
            s += 1

    def _apply_background_knowledge(self):
        if not self._bk:
            return

        # Add mandatory edges if missing
        for u, v in self._bk.get_mandatory_edges():
            if (u, v) not in self.g_hat.get_undirected_edges() and (v, u) not in self.g_hat.get_undirected_edges():
                self.g_hat.add_undirected_edge(u, v)

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
        nodes = self.g_hat.get_vertices()
        adj = {x: self.g_hat.get_adjacencies(x) for x in nodes}
        for x in nodes:
            for y in sorted(adj[x]):
                for z in sorted(adj[y]):
                    if z == x or z in sorted(adj[x]):
                        continue
                    if y not in self.sepset.get((x, z), []):
                        self.g_hat.remove_undirected_edge(x, y)
                        self.g_hat.remove_undirected_edge(y, z)
                        self.g_hat.add_directed_edges_from([(x, y), (z, y)])

    def _meek_rule_1(self, X, Y, Z):
        if (X, Y) in self.g_hat.get_directed_edges() and (Y, Z) in self.g_hat.get_undirected_edges() and Z not in self.g_hat.get_adjacencies(X):
            return True
        return False

    def _meek_rule_2(self, X, Y, Z):
        if (X, Y) in self.g_hat.get_directed_edges() and (Y, Z) in self.g_hat.get_directed_edges() and (X, Z) in self.g_hat.get_undirected_edges():
            return True
        return False

    def _meek_rule_3(self, X, Y, Z, W):
        if (X, Y) in self.g_hat.get_directed_edges() and (Z, Y) in self.g_hat.get_directed_edges() and Z not in self.g_hat.get_adjacencies(X):
            if (W, Y) in self.g_hat.get_undirected_edges() and (W, X) in self.g_hat.get_undirected_edges() and (W, Z) in self.g_hat.get_undirected_edges():
                return True
        return False

    def _apply_meek_rules(self):
        nodes = self.g_hat.get_vertices()
        changed = False
        adj = {x: self.g_hat.get_adjacencies(x) for x in nodes}

        for x in nodes:
            for y in sorted(adj[x]):
                # Rule 1:
                for z in sorted(set(adj[y]) - set(adj[x])):
                    if self._meek_rule_1(x, y, z):
                        changed = True
                        self.g_hat.remove_undirected_edge(z, y)
                        self.g_hat.add_directed_edge(y, z)
                # Rule 2:
                for z in sorted(set(adj[y]) & set(adj[x])):
                    if self._meek_rule_2(x, y, z):
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
                        if self._meek_rule_3(x, y, z, w):
                            changed = True
                            self.g_hat.remove_undirected_edge(w, y)
                            self.g_hat.add_directed_edge(w, y)
        return changed

    def _orientation(self):
        self._uc_rule()
        repeat = True
        while repeat:
            repeat = self._apply_meek_rules()
