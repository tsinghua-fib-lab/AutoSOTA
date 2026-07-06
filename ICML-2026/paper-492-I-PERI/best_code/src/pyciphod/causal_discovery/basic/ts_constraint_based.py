from abc import ABC, abstractmethod
from itertools import combinations
from typing import Optional, Type

import pandas as pd


from pyciphod.utils.graphs.orientation_rules import (
    time_orientation, uc_rule, apply_ts_meek_rules
)
from pyciphod.utils.graphs.background_knowledge import BackgroundKnowledge
from pyciphod.utils.graphs.partially_specified_graphs import PartiallyDirectedGraphs
from pyciphod.utils.stat_tests.independence_tests import (
    CiTests,
    FisherZTest as FisherZ,
)
from pyciphod.utils.time_series.data_format import DTimeVar

from pyciphod.utils.graphs.partially_specified_graphs import FtCompletedPartiallyDirectedAcyclicGraph


# helpers
def node_key(node):
    return (str(node.name), int(node.time))

def edge_key(edge):
    u, v = edge
    return (
        str(u.name), int(u.time),
        str(v.name), int(v.time),
    )

def sorted_edges(edges):
    return sorted(edges, key=edge_key)



class TsConstraintBased(ABC):
    """Base class for temporal constraint-based causal discovery algorithms."""

    def __init__(
        self,
        sparsity: float = 0.05,
        ci_test: Type[CiTests] = FisherZ,
        background_knowledge: Optional[BackgroundKnowledge] = None,
        twd: bool = False,
    ):
        self._sparsity = sparsity
        self._ci_test = ci_test
        self._bk = background_knowledge if background_knowledge is not None else BackgroundKnowledge()
        self._twd = twd
        self.performed_tests = set()
        self.nb_ci_tests = 0
        self.sepset = {}
        self.g_hat = self._initialize_graph()

    def _initialize_graph(self):
        """Return the temporal graph object used by the algorithm."""
        return FtCompletedPartiallyDirectedAcyclicGraph()

    @abstractmethod
    def _skeleton(self, data: pd.DataFrame = None, max_sepset_size: int = None):
        """Construct the skeleton of the temporal graph."""
        pass

    def _validate_temporal_data(self, data: pd.DataFrame):
        """Check that the input data contains DTimeVar columns."""
        if data is None:
            raise ValueError("data must be provided.")

        nodes = list(data.columns)
        if not all(isinstance(node, DTimeVar) for node in nodes):
            raise TypeError("All columns must be DTimeVar objects.")

        return nodes


    def _apply_background_knowledge(self):
        """Apply background knowledge constraints to the temporal graph."""
        # TODO
        pass

    @abstractmethod
    def _orientation(self):
        """Orient edges using temporal order, UC rule and Meek rules."""
        pass
        
    def _stationary_expand_graph(self, nodes):
        """Expand learned temporal edges across the lag window by stationarity."""
        node_lookup = {(node.name, node.time): node for node in nodes}
        time_points = sorted({node.time for node in nodes})

        directed_edges = sorted_edges(self.g_hat.get_directed_edges())
        undirected_edges = sorted_edges(self.g_hat.get_undirected_edges())

        for x, y in directed_edges:
            for shift in time_points:
                x_new = node_lookup.get((x.name, x.time + shift))
                y_new = node_lookup.get((y.name, y.time + shift))

                if x_new is None or y_new is None or x_new == y_new:
                    continue

                if not self.g_hat.is_adjacent(x_new, y_new):
                    self.g_hat.add_directed_edge(x_new, y_new)

        for x, y in undirected_edges:
            for shift in time_points:
                x_new = node_lookup.get((x.name, x.time + shift))
                y_new = node_lookup.get((y.name, y.time + shift))

                if x_new is None or y_new is None or x_new == y_new:
                    continue

                if not self.g_hat.is_adjacent(x_new, y_new):
                    self.g_hat.add_undirected_edge(x_new, y_new)

    def run(self, data: pd.DataFrame = None, max_sepset_size: int = None):
        """Execute the full temporal constraint-based algorithm."""
        nodes = self._validate_temporal_data(data)

        self._skeleton(data=data, max_sepset_size=max_sepset_size)
        self._apply_background_knowledge()
        self._orientation()
        self._stationary_expand_graph(nodes)
        return self.g_hat


class TsPC(TsConstraintBased):
    """Temporal PC algorithm for lag-augmented time-series data."""

    def _skeleton(self, data: pd.DataFrame = None, max_sepset_size: int = None,):
        """Construct the skeleton of the temporal graph using CI tests."""
        nodes = self._validate_temporal_data(data)

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
                        test = self._ci_test(x, y, list(S), self._twd)
                        self.performed_tests.add((x, y, S))
                        self.nb_ci_tests += 1

                        data_test = data
                        if self._twd:
                            data_test = data.dropna(subset=[x, y] + list(S))

                        try:
                            pval = test.get_pvalue(data_test)
                        except Exception:
                            pval = test.get_pvalue_by_permutation(data_test)

                        if pval > self._sparsity:
                            self.g_hat.remove_undirected_edge(x, y)
                            self.g_hat.remove_undirected_edge(y, x)
                            self.sepset[(x, y)] = S
                            self.sepset[(y, x)] = S
                            break

            s += 1

    def _orientation(self):
        """Orient edges using temporal order, UC rule and Meek rules."""
        time_orientation(self.g_hat)
        uc_rule(g=self.g_hat,sepset=self.sepset,)
        repeat = True
        while repeat:
            repeat = apply_ts_meek_rules(self.g_hat)