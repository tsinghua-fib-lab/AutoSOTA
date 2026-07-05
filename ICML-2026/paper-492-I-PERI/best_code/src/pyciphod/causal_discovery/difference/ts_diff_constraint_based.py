from abc import ABC, abstractmethod
from itertools import combinations
from typing import Optional, Type
import warnings

import numpy as np
import pandas as pd
from scipy import stats

## TODO : modifier les chemins
# from pyciphod.causal_discovery.time_series.ts_constraint_based import (
#     FtCompletedPartiallyDirectedAcyclicGraph,
#     TsPC,
# )
# from pyciphod.causal_discovery.time_series.ts_constraint_orientation import (
#     time_orientation, uc_rule, apply_meek_rules
# )

from pyciphod.utils.graphs.partially_specified_graphs import FtCompletedPartiallyDirectedAcyclicGraph

from pyciphod.causal_discovery.basic.ts_constraint_based import TsPC
from pyciphod.utils.graphs.orientation_rules import (
    time_orientation, uc_rule, apply_ts_meek_rules
)
from pyciphod.utils.graphs.background_knowledge import BackgroundKnowledge
from pyciphod.utils.stat_tests.equality_tests import (
    CeTests,
    LinearRegressionCoefficientEqualityTest,
)
from pyciphod.utils.stat_tests.independence_tests import CiTests
from pyciphod.utils.time_series.data_format import DTimeVar

# utils

def node_key(node):
    return (str(node.name), int(node.time))

def edge_key(edge):
    u, v = edge
    return (
        str(u.name), int(u.time),
        str(v.name), int(v.time),
    )

def sorted_nodes(nodes):
    return sorted(nodes, key=node_key)

def sorted_edges(edges):
    return sorted(edges, key=edge_key)



class TsDifferenceConstraintBased(ABC):
    """Base class for temporal difference constraint-based algorithms."""

    def __init__(
        self,
        sparsity: float = 0.05,
        eq_test: Type[CeTests] = LinearRegressionCoefficientEqualityTest,
        background_knowledge: Optional[BackgroundKnowledge] = None,
        twd: bool = False,
        seed: Optional[int] = None,
    ):
        self._sparsity = sparsity
        self._eq_test = eq_test
        self._bk = background_knowledge if background_knowledge is not None else BackgroundKnowledge()
        self._twd = twd
        self._seed = seed
        self.performed_tests = set()
        self.nb_ci_tests = 0
        self.sepset = {}
        self.g_hat = self._initialize_graph()

    def _initialize_graph(self):
        """Return the temporal difference graph object used by the algorithm."""
        return FtCompletedPartiallyDirectedAcyclicGraph()

    @abstractmethod
    def _skeleton(self, df1: pd.DataFrame = None, df2: pd.DataFrame = None, max_sepset_size: int = None, ):
        """Construct the skeleton of the temporal difference graph."""
        pass

    def _validate_temporal_data(self, data: pd.DataFrame):
        """Check that the input data contains DTimeVar columns."""
        if data is None:
            raise ValueError("data must be provided.")

        nodes = list(data.columns)
        if not all(isinstance(node, DTimeVar) for node in nodes):
            raise TypeError("All columns must be DTimeVar objects.")

        return nodes

    def _validate_two_temporal_dataframes(self, df1: pd.DataFrame, df2: pd.DataFrame):
        """Check that both regimes contain the same DTimeVar columns."""
        nodes1 = self._validate_temporal_data(df1)
        nodes2 = self._validate_temporal_data(df2)

        if list(df1.columns) != list(df2.columns):
            raise ValueError("df1 and df2 must have the same columns in the same order.")

        if nodes1 != nodes2:
            raise ValueError("df1 and df2 must contain the same DTimeVar nodes.")

        return nodes1

    def _apply_background_knowledge(self):
        """Apply background knowledge constraints to the temporal difference graph."""
        # TODO
        pass

    @abstractmethod
    def _orientation(self):
        """Orient edges using temporal order, UC rule and Meek rules."""
        pass

    def run(self, df1: pd.DataFrame = None, df2: pd.DataFrame = None, max_sepset_size: int = None, ):
        """Execute the full temporal difference constraint-based algorithm."""
        self._skeleton(df1=df1, df2=df2, max_sepset_size=max_sepset_size)
        self._apply_background_knowledge()
        self._orientation()
        return self.g_hat


class TsLDiffPC(TsDifferenceConstraintBased):
    """Temporal Linear Difference PC."""

    def _skeleton(self, df1: pd.DataFrame = None, df2: pd.DataFrame = None, max_sepset_size: int = None,):
        """Construct the skeleton of the temporal difference graph using equality tests."""
        nodes = self._validate_two_temporal_dataframes(df1, df2)

        candidate_edges = [
            (x, y)
            for x, y in combinations(sorted(nodes), 2)
            if (x.time <= 0 and y.time == 0) or (y.time <= 0 and x.time == 0)
        ]
        self.g_hat.add_undirected_edges_from(candidate_edges)

        if max_sepset_size is None:
            max_sepset_size = len(nodes) - 2

        s = 0
        repeat = True

        while repeat and s <= max_sepset_size:
            repeat = False
            adj = {x: self.g_hat.get_adjacencies(x) for x in nodes}

            for x in nodes:
                for y in sorted(adj[x]):
                    if not ((x.time <= 0 and y.time == 0) or (y.time <= 0 and x.time == 0)):
                        continue

                    possible_S = [a for a in sorted(nodes) if a != x and a != y]

                    if len(possible_S) >= s:
                        repeat = True

                    for S in combinations(possible_S, s):
                        test = self._eq_test(x, y, list(S), self._twd)
                        self.performed_tests.add((x, y, S))
                        self.nb_ci_tests += 1

                        df1_test = df1
                        df2_test = df2

                        if self._twd:
                            cols = [x, y] + list(S)
                            df1_test = df1.dropna(subset=cols)
                            df2_test = df2.dropna(subset=cols)

                        try:
                            pval = test.get_pvalue(df1_test, df2_test)
                        except Exception as e:
                            warnings.warn(f"Equality test failed for ({x}, {y}, S={S}): {e}")
                            pval = None

                        if pval is None:
                            continue

                        if isinstance(pval, float) and np.isnan(pval):
                            continue

                        if pval > self._sparsity:
                            self.g_hat.remove_undirected_edge(x, y)
                            self.sepset[(x, y)] = S
                            self.sepset[(y, x)] = S
                            break

            s += 1

    def _orientation(self):
        """Orient edges using temporal order, UC rule and Meek rules."""
        time_orientation(self.g_hat)
        uc_rule(g=self.g_hat, sepset=self.sepset)

        repeat = True
        while repeat:
            repeat = apply_ts_meek_rules(self.g_hat)

    def _has_directed_path(self, src, tgt):
        """Return True if there is a directed path src -> ... -> tgt."""
        children = {}
        for u, v in sorted_edges(self.g_hat.get_directed_edges()):
            children.setdefault(u, set()).add(v)
        stack, seen = [src], set()
        while stack:
            node = stack.pop()
            if node == tgt:
                return True
            if node not in seen:
                seen.add(node)
                stack.extend(sorted_nodes(children.get(node, [])))

        return False

    def _orient_with_tspc(self, tspc):
        """Orient unresolved edges using compatible TsPC orientations."""
        tspc_directed = set(sorted_edges(tspc.g_hat.get_directed_edges()))

        for x, y in sorted_edges(self.g_hat.get_undirected_edges()):
            for src, tgt in ((x, y), (y, x)):
                if (
                    (src, tgt) in tspc_directed
                    and src.time <= tgt.time
                    and not self._has_directed_path(tgt, src)
                ):
                    self.g_hat.remove_undirected_edge(src, tgt)
                    self.g_hat.add_directed_edge(src, tgt)
                    break

    def add_tspc_orientation(
        self,
        df1: pd.DataFrame = None,
        pc_sparsity: Optional[float] = None,
        pc_ci_test: Optional[Type[CiTests]] = None,
        pc_max_sepset_size: Optional[int] = None,
    ):
        """Orient unresolved edges using TsPC on the normal regime."""
        if df1 is None:
            raise ValueError("df1 must be provided.")
        
        if len(self.g_hat.get_vertices()) == 0:
            raise ValueError(
                "No difference graph available. "
                "Run TsLDiffPC before calling add_pc_orientation()."
            )

        pc_kwargs = {
            "sparsity": (pc_sparsity if pc_sparsity is not None else self._sparsity),
            "twd": self._twd,
        }

        if pc_ci_test is not None:
            pc_kwargs["ci_test"] = pc_ci_test

        tspc = TsPC(**pc_kwargs)

        tspc.run(data=df1,max_sepset_size=pc_max_sepset_size,)

        self._orient_with_tspc(tspc)

        return self.g_hat
    
class TsDCI(TsDifferenceConstraintBased):
    """Temporal Difference Causal Inference."""

    def _powerset(self, nodes, max_set_size=None):
        nodes = list(nodes)

        if max_set_size is None:
            max_set_size = len(nodes)

        for s in range(max_set_size + 1):
            for S in combinations(sorted(nodes), s):
                yield S

    def _ols_residual_variance(self, df, target, regressors):
        cols = [target] + list(regressors)
        d = df[cols].dropna()

        y = d[target].to_numpy(dtype=float)

        if len(regressors) == 0:
            residuals = y - y.mean()
            dof = len(y) - 1
            if dof <= 0:
                return np.nan, dof
            return np.sum(residuals ** 2) / dof, dof

        X = d[list(regressors)].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(X)), X])

        n, q = X.shape
        if n <= q:
            return np.nan, n - q

        beta = np.linalg.pinv(X.T @ X) @ X.T @ y
        residuals = y - X @ beta
        dof = n - q

        return np.sum(residuals ** 2) / dof, dof

    def _residual_variance_equality_pvalue(self, df1, df2, target, regressors):
        var1, dof1 = self._ols_residual_variance(df1, target, regressors)
        var2, dof2 = self._ols_residual_variance(df2, target, regressors)

        if (
            var1 is None
            or var2 is None
            or np.isnan(var1)
            or np.isnan(var2)
            or dof1 <= 0
            or dof2 <= 0
            or var2 == 0
        ):
            return None

        ratio = var1 / var2
        p_left = stats.f.cdf(ratio, dof1, dof2)

        return 2 * min(p_left, 1 - p_left)

    def _skeleton(
        self,
        df1: pd.DataFrame = None,
        df2: pd.DataFrame = None,
        max_sepset_size: int = None,
    ):
        """Construct the skeleton of the temporal difference graph."""
        nodes = self._validate_two_temporal_dataframes(df1, df2)

        self._df1 = df1
        self._df2 = df2

        candidate_edges = [
            (x, y)
            for x, y in combinations(sorted(nodes), 2)
            if (x.time <= 0 and y.time == 0)
            or (y.time <= 0 and x.time == 0)
        ]

        self.g_hat.add_undirected_edges_from(candidate_edges)

        if max_sepset_size is None:
            max_sepset_size = len(nodes) - 2

        changed_nodes = set(nodes)

        for x, y in candidate_edges:
            possible_S = sorted_nodes(changed_nodes - {x, y})

            for S in self._powerset(possible_S, max_set_size=max_sepset_size):
                df1_test = df1
                df2_test = df2

                if self._twd:
                    cols = [x, y] + list(S)
                    df1_test = df1.dropna(subset=cols)
                    df2_test = df2.dropna(subset=cols)

                try:
                    test_xy = self._eq_test(x, y, list(S), self._twd)
                    pval_xy = test_xy.get_pvalue(df1_test, df2_test)
                except Exception as e:
                    warnings.warn(
                        f"TsDCI skeleton test failed for {x}->{y}, S={S}: {e}"
                    )
                    pval_xy = None

                self.performed_tests.add((x, y, S, "skeleton_x_to_y"))
                self.nb_ci_tests += 1

                if pval_xy is not None and not np.isnan(pval_xy):
                    if pval_xy > self._sparsity:
                        self.g_hat.remove_undirected_edge(x, y)
                        self.sepset[(x, y)] = S
                        self.sepset[(y, x)] = S
                        break

                try:
                    test_yx = self._eq_test(y, x, list(S), self._twd)
                    pval_yx = test_yx.get_pvalue(df1_test, df2_test)
                except Exception as e:
                    warnings.warn(
                        f"TsDCI skeleton test failed for {y}->{x}, S={S}: {e}"
                    )
                    pval_yx = None

                self.performed_tests.add((y, x, S, "skeleton_y_to_x"))
                self.nb_ci_tests += 1

                if pval_yx is not None and not np.isnan(pval_yx):
                    if pval_yx > self._sparsity:
                        self.g_hat.remove_undirected_edge(x, y)
                        self.sepset[(x, y)] = S
                        self.sepset[(y, x)] = S
                        break

    def _dci_orient_contemporaneous_edges(self, max_set_size=None):
        """Orient contemporaneous edges using residual variance invariance."""
        nodes = sorted_nodes(self.g_hat.get_vertices())
        changed_nodes = set(nodes)

        if max_set_size is None:
            max_set_size = len(nodes) - 1

        for i, j in sorted_edges(self.g_hat.get_undirected_edges()):
            if i.time != j.time:
                continue

            not_i = sorted_nodes(changed_nodes - {i})
            not_j = sorted_nodes(changed_nodes - {j})

            powersets = zip(
                self._powerset(not_i, max_set_size=max_set_size),
                self._powerset(not_j, max_set_size=max_set_size),
            )

            for m_i, m_j in powersets:
                try:
                    p_i = self._residual_variance_equality_pvalue(
                        df1=self._df1,
                        df2=self._df2,
                        target=i,
                        regressors=list(m_i),
                    )
                except Exception as e:
                    warnings.warn(
                        f"TsDCI orientation failed for target={i}, M={m_i}: {e}"
                    )
                    p_i = None

                self.performed_tests.add((i, j, m_i, "orient_i"))
                self.nb_ci_tests += 1

                if p_i is not None and not np.isnan(p_i) and p_i > self._sparsity:
                    if j in m_i:
                        src, tgt = j, i
                    else:
                        src, tgt = i, j

                    if (src.time <= tgt.time and (src, tgt) in self.g_hat.get_undirected_edges()):
                        self.g_hat.remove_undirected_edge(src, tgt)
                        self.g_hat.add_directed_edge(src, tgt)
                    break

                try:
                    p_j = self._residual_variance_equality_pvalue(
                        df1=self._df1,
                        df2=self._df2,
                        target=j,
                        regressors=list(m_j),
                    )
                except Exception as e:
                    warnings.warn(
                        f"TsDCI orientation failed for target={j}, M={m_j}: {e}"
                    )
                    p_j = None

                self.performed_tests.add((j, i, m_j, "orient_j"))
                self.nb_ci_tests += 1

                if p_j is not None and not np.isnan(p_j) and p_j > self._sparsity:
                    if i in m_j:
                        src, tgt = i, j
                    else:
                        src, tgt = j, i

                    if (src.time <= tgt.time and (src, tgt) in self.g_hat.get_undirected_edges()):
                        self.g_hat.remove_undirected_edge(src, tgt)
                        self.g_hat.add_directed_edge(src, tgt)
                    break

    def _orientation(self):
        """Orient lagged edges by time and contemporaneous edges with DCI."""
        time_orientation(self.g_hat)
        self._dci_orient_contemporaneous_edges()

    def _has_directed_path(self, src, tgt):
        """Return True if there is a directed path src -> ... -> tgt."""
        children = {}

        for u, v in sorted_edges(self.g_hat.get_directed_edges()):
            children.setdefault(u, set()).add(v)

        stack, seen = [src], set()

        while stack:
            node = stack.pop()

            if node == tgt:
                return True

            if node not in seen:
                seen.add(node)
                stack.extend(sorted_nodes(children.get(node, [])))

        return False

    def _orient_with_tspc(self, tspc):
        """Orient unresolved edges using compatible TsPC orientations."""
        tspc_directed = set(sorted_edges(tspc.g_hat.get_directed_edges()))

        for x, y in sorted_edges(self.g_hat.get_undirected_edges()):
            for src, tgt in ((x, y), (y, x)):
                if (
                    (src, tgt) in tspc_directed
                    and src.time <= tgt.time
                    and not self._has_directed_path(tgt, src)
                ):
                    self.g_hat.remove_undirected_edge(src, tgt)
                    self.g_hat.add_directed_edge(src, tgt)
                    break

    def add_tspc_orientation(
        self,
        df1: pd.DataFrame = None,
        pc_sparsity: Optional[float] = None,
        pc_ci_test: Optional[Type[CiTests]] = None,
        pc_max_sepset_size: Optional[int] = None,
    ):
        """Orient unresolved edges using TsPC on the normal regime."""
        if df1 is None:
            raise ValueError("df1 must be provided.")

        if len(self.g_hat.get_vertices()) == 0:
            raise ValueError(
                "No difference graph available. "
                "Run TsDCI before calling add_tspc_orientation()."
            )

        pc_kwargs = {
            "sparsity": pc_sparsity if pc_sparsity is not None else self._sparsity,
            "twd": self._twd,
        }

        if pc_ci_test is not None:
            pc_kwargs["ci_test"] = pc_ci_test

        tspc = TsPC(**pc_kwargs)

        tspc.run(
            data=df1,
            max_sepset_size=pc_max_sepset_size,
        )

        self._orient_with_tspc(tspc)

        return self.g_hat