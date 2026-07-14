"""
Causal Inference-based Root Cause Analysis (CIRCA)
"""
import logging
from abc import ABC
from datetime import timedelta
from typing import Callable, Dict, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model._base import LinearModel
from sklearn.preprocessing import StandardScaler

from RCAEval.classes.data import CaseData, MemoryDataLoader
from RCAEval.classes.graph import Graph, MemoryGraph, Node
from RCAEval.graph_heads.random_walk import Score, Scorer


def zscore(train_y: np.ndarray, test_y: np.ndarray) -> np.ndarray:
    """
    Estimate to what extend each value in test_y violates
    the normal distribution defined by train_y
    """
    scaler = StandardScaler().fit(train_y.reshape(-1, 1))
    return scaler.transform(test_y.reshape(-1, 1))[:, 0]


def zscore_conf(score: float) -> float:
    """
    Convert z-score into confidence about the hypothesis the score is abnormal
    """
    return 1 - 2 * norm.cdf(-abs(score))


class DecomposableScorer(Scorer):
    """
    Score each node separately
    """

    def score_node(
        self,
        graph: Graph,
        series: Dict[Node, Sequence[float]],
        node: Node,
        data: CaseData,
    ) -> Score:
        """
        Estimate how suspicious a node is
        """
        raise NotImplementedError

    def _score(
        self,
        candidates: Sequence[Node],
        series: Dict[Node, Sequence[float]],
        graph: Graph,
        data: CaseData,
    ):
        results: Dict[Node, Score] = {}
        for node in candidates:
            score = self.score_node(graph, series, node, data)
            if score is not None:
                results[node] = score
        return results

    def score(
        self,
        graph: Graph,
        data: CaseData,
        current: float,
        scores: Dict[Node, Score] = None,
    ) -> Dict[Node, Score]:
        series = data.load_data(graph, current)
        candidates = list(series.keys()) if scores is None else list(scores.keys())

        results = self._score(candidates=candidates, series=series, graph=graph, data=data)

        if scores is None:
            return results
        return {node: scores[node].update(score) for node, score in results.items()}


class Regressor(ABC):
    """
    Regress one node on its parents, assuming x ~ P(x | pa(X))
    """

    def __init__(self):
        klass = self.__class__
        self._logger = logging.getLogger(f"{klass.__module__}.{klass.__name__}")

    @staticmethod
    def _zscore(train_y: np.ndarray, test_y: np.ndarray) -> np.ndarray:
        return zscore(train_y=train_y, test_y=test_y)

    def _score(
        self,
        train_x: np.ndarray,
        test_x: np.ndarray,
        train_y: np.ndarray,
        test_y: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError

    def score(
        self,
        train_x: np.ndarray,
        test_x: np.ndarray,
        train_y: np.ndarray,
        test_y: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate to what extend each value in test_y violates regression
        """
        if len(train_x) == 0:
            return self._zscore(train_y=train_y, test_y=test_y)
        try:
            return self._score(train_x=train_x, test_x=test_x, train_y=train_y, test_y=test_y)
        except ValueError as err:
            self._logger.warning(err, exc_info=True)
            return self._zscore(train_y=train_y, test_y=test_y)


class ANMRegressor(Regressor):
    """
    Regress one node on its parents with an Additive Noise Model
    assuming x = f(pa(X)) + e and e follows a normal distribution
    """

    def __init__(self, regressor: LinearModel = None, **kwargs):
        super().__init__(**kwargs)
        self._regressor = regressor if regressor else LinearRegression()

    def _score(
        self,
        train_x: np.ndarray,
        test_x: np.ndarray,
        train_y: np.ndarray,
        test_y: np.ndarray,
    ) -> np.ndarray:
        self._regressor.fit(train_x, train_y)
        train_err: np.ndarray = train_y - self._regressor.predict(train_x)
        test_err: np.ndarray = test_y - self._regressor.predict(test_x)
        return self._zscore(train_y=train_err, test_y=test_err)


class RHTScorer(DecomposableScorer):
    """
    Scorer with regression-based hypothesis testing
    """

    def __init__(
        self,
        tau_max: int = 0,
        regressor: Regressor = None,
        use_confidence: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._tau_max = max(tau_max, 0)
        self._regressor = regressor if regressor else ANMRegressor()
        self._use_confidence = use_confidence

    @staticmethod
    def _split_train_test(
        series_x: np.ndarray,
        series_y: np.ndarray,
        train_window: int,
        test_window: int,
    ):
        train_x: np.ndarray = series_x[:train_window, :]
        train_y: np.ndarray = series_y[:train_window]
        test_x: np.ndarray = series_x[-test_window:, :]
        test_y: np.ndarray = series_y[-test_window:]
        return train_x, test_x, train_y, test_y

    def split_data(
        self,
        data: Dict[Node, Sequence[float]],
        node: Node,
        parents: Sequence[Node],
        case_data: CaseData,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data for training and testing

        Return (train_x, test_x, train_y, test_y)
        """
        series = np.array([data[parent] for parent in parents if parent in data]).T
        if len(series) == 0:
            series = np.zeros((0, 0))
        series_x = np.hstack([np.roll(series, i, 0) for i in range(self._tau_max + 1)])
        series_x: np.ndarray = series_x[self._tau_max :, :]
        series_y = np.array(data[node][self._tau_max :])

        return self._split_train_test(
            series_x=series_x,
            series_y=series_y,
            train_window=case_data.train_window - self._tau_max,
            test_window=case_data.test_window,
        )

    def score_node(
        self,
        graph: Graph,
        series: Dict[Node, Sequence[float]],
        node: Node,
        data: CaseData,
    ) -> Score:
        parents = list(graph.parents(node))

        train_x, test_x, train_y, test_y = self.split_data(series, node, parents, data)
        z_scores = self._regressor.score(
            train_x=train_x, test_x=test_x, train_y=train_y, test_y=test_y
        )
        z_score = self._aggregator(abs(z_scores))
        confidence = zscore_conf(z_score)
        if self._use_confidence:
            score = Score(confidence)
            score.key = (score.score, z_score)
        else:
            score = Score(z_score)
        score["z-score"] = z_score
        score["Confidence"] = confidence

        return score


class DAScorer(Scorer):
    """
    Scorer with descendant adjustment
    """

    def __init__(
        self,
        threshold: float = 0,
        aggregator: Callable[[Sequence[float]], float] = max,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._threshold = max(threshold, 0.0)
        self._aggregator = aggregator

    def score(
        self,
        graph: Graph,
        data: CaseData,
        current: float,
        scores: Dict[Node, Score] = None,
    ) -> Dict[Node, Score]:
        sorted_nodes = [
            {node for node in nodes if node in scores} for nodes in graph.topological_sort
        ]
        # 0. Set topological rank
        for index, nodes in enumerate(sorted_nodes):
            for node in nodes:
                score = scores[node]
                score["index"] = index

        # 1. Gather child scores
        child_scores: Dict[Node, Dict[Node, float]] = {}
        for nodes in reversed(sorted_nodes):
            for node in nodes:
                child_score: Dict[Node, float] = {}
                for child in graph.children(node):
                    if child in scores:
                        child_score[child] = scores[child].score
                        if scores[child].score < self._threshold:
                            child_score.update(child_scores.get(child, {}))
                child_scores[node] = child_score

        # 2. Set child_score
        for node, score in scores.items():
            if score.score >= self._threshold:
                child_score = child_scores[node]
                if child_score:
                    child_score = self._aggregator(child_score.values())
                    score.score += child_score
                    score["child_score"] = child_score

        # 3. Set key
        for score in scores.values():
            score.key = (score.score, -score["index"], score.get("z-score", 0))
        return scores


def rht(
    adj: np.ndarray,
    inject_time: int,
    data: pd.DataFrame,
    sli: Node = None,
    num_loop=None,
    previous_scores=None,
):
    """
    adj: np.ndarray
    data: pd.DataFrame

    return:
    ranks : List[str]
    """
    node_names = data.columns.to_list()

    # == prepare the graph
    nodes = [Node(name.split("_")[0], name.split("_")[1]) for name in node_names if name != "time"]
    graph = nx.DiGraph()

    # convert adj to edges
    node_num = len(adj)
    for a in range(node_num):
        for b in range(node_num):
            # case 1 no edge: a b
            if adj[a, b] == adj[b, a] == 0:
                pass
            # case 2 undirected a -- b
            elif adj[a, b] == adj[b, a] == -1:
                # eck eck, what should we do???
                # afaik, RW doesn't care about direction?
                # just add an edge
                graph.add_edge(nodes[b], nodes[a])
            # case 3 directed a->b
            elif adj[a, b] == 1 and adj[b, a] == -1:
                graph.add_edge(nodes[b], nodes[a])
            # case 4 directed b->a
            elif adj[a, b] == -1 and adj[b, a] == 1:
                graph.add_edge(nodes[a], nodes[b])
            # elif adj[a, b] == 0 and adj[b, a] == 1:
            #     # already ok
            #     pr_input[a, b] = 0
            #     pr_input[b, a] = 1
            # elif adj[a, b] == 1 and adj[b, a] == 0:
            #     # already ok
            #     pr_input[a, b] = 1
            #     pr_input[b, a] = 0
            else:
                raise ValueError(f"Unexpected value: {adj[a, b]}, {adj[b, a]}")

    graph = graph.reverse()
    mem_graph = MemoryGraph(graph)
    # == end prepare the graph

    scorer = RHTScorer()
    scores: Dict[Node, Score] = None

    timestamps = data["time"]
    sli = np.random.choice(nodes)
    services = list(set([c.split("_")[0] for c in data.columns if c != "time"]))
    metrics = list(set([c.split("_")[1] for c in data.columns if c != "time"]))
    out_data = {s: {} for s in services}
    for s in services:
        for m in metrics:
            try:
                out_data[s][m] = list(zip(timestamps, data[f"{s}_{m}"]))
            except Exception:
                pass
    mem_data_loader = MemoryDataLoader(out_data)

    data = CaseData(
        data_loader=mem_data_loader, sli=sli, detect_time=inject_time, interval=timedelta(seconds=1)
    )

    scores = scorer.score(graph=mem_graph, data=data, current=inject_time + 300, scores=scores)
    scores = sorted(scores.items(), key=lambda item: item[1].key, reverse=True)
    output = []
    for item in scores:
        output.append((f"{item[0].entity}_{item[0].metric}", item[1].score))

    return output
