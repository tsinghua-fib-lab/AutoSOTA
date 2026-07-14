from abc import ABC
from collections import defaultdict
from typing import Callable, Dict, List, Sequence, Union

import networkx as nx
import numpy as np
import pandas as pd

from RCAEval.classes.data import CaseData
from RCAEval.classes.graph import Graph, MemoryGraph, Node


def _times(num: int, multiplier: int = 10) -> int:
    return num * multiplier


class Score:
    """
    The more suspicious a node, the higher the score.
    """

    def __init__(self, score: float, info: dict = None, key: tuple = None):
        self._score = score
        self._key = key
        self._info = {} if info is None else info

    def __eq__(self, obj) -> bool:
        if isinstance(obj, Score):
            return self.score == obj.score
        return False

    def __getitem__(self, key: str):
        return self._info[key]

    def __setitem__(self, key: str, value):
        self._info[key] = value

    def get(self, key: str, default=None):
        """
        Return the value for key if key has been set, else default
        """
        return self._info.get(key, default)

    @property
    def score(self) -> float:
        """
        The overall score
        """
        return self._score

    @score.setter
    def score(self, value: float):
        """
        Update score
        """
        self._score = value

    @property
    def key(self) -> float:
        """
        key for sorting
        """
        return self.score if self._key is None else self._key

    @key.setter
    def key(self, value: tuple):
        """
        Update key
        """
        self._key = value

    @property
    def info(self) -> dict:
        """
        Extra information
        """
        return self._info

    def update(self, score: "Score") -> "Score":
        """
        Update score and info
        """
        self._info.update(score.info)
        self.score = score.score
        self.key = score.key
        return self

    def asdict(self) -> Dict[str, Union[float, dict, tuple]]:
        """
        Serialized as a dict
        """
        data = {"score": self._score, "info": {**self._info}}
        if self._key is not None:
            data["key"] = self._key
        return data

    def __repr__(self) -> str:
        return str(self.asdict())


class Scorer(ABC):
    """
    The abstract interface to score nodes
    """

    def __init__(
        self,
        aggregator: Callable[[Sequence[float]], float] = max,
        max_workers: int = 1,
        seed: int = 0,
        cuda: bool = False,
    ):
        self._aggregator = aggregator
        self._max_workers = max_workers
        self._seed = seed
        self._cuda = cuda

    def score(
        self,
        graph: Graph,
        data: CaseData,
        current: float,
        scores: Dict[Node, Score] = None,
    ) -> Dict[Node, Score]:
        """
        Estimate suspicious nodes
        """
        raise NotImplementedError


class RandomWalkScorer(Scorer):
    """
    Scorer based on random walk
    """

    def __init__(
        self,
        rho: float = 0.5,
        remove_sli: bool = False,
        num_loop: Union[int, Callable[[int], int]] = None,
        **kwargs,
    ):
        # pylint: disable=too-many-arguments
        super().__init__(**kwargs)
        self._rho = rho
        self._remove_sli = remove_sli
        self._num_loop = num_loop if num_loop is not None else _times
        self._rng = np.random.default_rng(self._seed)

    def generate_transition_matrix(
        self, graph: Graph, data: CaseData, scores: Dict[Node, Score]
    ) -> pd.DataFrame:
        """
        Generate the transition matrix
        """
        nodes = list(scores.keys())
        size = len(nodes)
        matrix = pd.DataFrame(np.zeros([size, size]), index=nodes, columns=nodes)
        for node in scores:
            for child in graph.children(node):
                if child in scores:
                    matrix[node][child] = self._rho * abs(scores[child].score)

            parents = graph.parents(node)
            if self._remove_sli:
                parents -= {data.sli}
            for parent in parents:
                if parent in scores:
                    matrix[node][parent] = abs(scores[parent].score)

            matrix[node][node] = max(abs(scores[node].score) - matrix[node].max(), 0)

            total_weight = matrix[node].sum()
            if total_weight > 0:
                matrix[node] /= total_weight
            else:
                matrix[node] = 1 / size
        return matrix

    def _walk(self, start: Node, num_loop: int, matrix: pd.DataFrame) -> Dict[Node, int]:
        node: Node = start
        counter = defaultdict(int)
        for _ in range(num_loop):
            node = self._rng.choice(matrix.index, p=matrix[node])
            counter[node] += 1
        return counter

    def score(
        self,
        graph: Graph,
        data: CaseData,
        current: float,
        scores: Dict[Node, Score] = None,
    ) -> Dict[Node, Score]:
        if not scores:
            return scores

        matrix = self.generate_transition_matrix(graph=graph, data=data, scores=scores)
        if isinstance(self._num_loop, int):
            num_loop = self._num_loop
        else:
            num_loop = self._num_loop(len(scores))

        # print("==== matrix ===")
        # print(matrix)
        # print("==== end matrix ===")
        counter = self._walk(start=data.sli, num_loop=num_loop, matrix=matrix)
        # print(f"{num_loop=}")
        # print(dict(counter))

        for node, score in scores.items():
            score["pagerank"] = score.score = counter[node] / num_loop
        return scores


class SecondOrderRandomWalkScorer(RandomWalkScorer):
    """
    Scorer based on second-order random walk
    """

    def __init__(self, beta: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self._beta = beta

    def _walk(self, start: Node, num_loop: int, matrix: pd.DataFrame) -> Dict[Node, int]:
        node: Node = start
        node_pre: Node = start
        counter = defaultdict(int)
        for _ in range(num_loop):
            prob_pre = matrix[node_pre][node]
            node_pre = node

            candidates: List[Node] = []
            weights: List[float] = []
            # for key, value in matrix[node].iteritems():
            for key, value in matrix[node].items():
                if value > 0:
                    candidates.append(key)
                    weights.append((1 - self._beta) * prob_pre + self._beta * value)
            total_weight = sum(weights)
            if total_weight == 0:
                node = self._rng.choice(candidates)
            else:
                node = self._rng.choice(candidates, p=[weight / total_weight for weight in weights])
            counter[node] += 1
        return counter


def random_walk(
    adj: np.ndarray,
    node_names: List[str] = None,
    sli: Node = None,
    num_loop=None,
    previous_scores=None,
):
    """
    adj: np.ndarray
    node_names:
    """
    if node_names is None:
        node_names = [f"X{i}" for i in range(len(adj))]

    # == build the graph
    nodes = [Node(name, None) for name in node_names]
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
            elif adj[a, b] == 0 and adj[b, a] == 1:
                graph.add_edge(nodes[a], nodes[b])
            elif adj[a, b] == 1 and adj[b, a] == 0:
                graph.add_edge(nodes[b], nodes[a])
            elif adj[a, b] == 1 and adj[b, a] == 1:
                graph.add_edge(nodes[a], nodes[b])
                graph.add_edge(nodes[b], nodes[a])
            else:
                raise ValueError(f"Unexpected value: {adj[a, b]}, {adj[b, a]}")

    graph = graph.reverse()
    mem_graph = MemoryGraph(graph)
    # == end build the graph

    # prepare data for RandomWalk
    sli = np.random.choice(nodes)
    data = CaseData(data_loader=None, sli=sli, detect_time=0)

    if previous_scores is None:
        scores = {node: Score(0) for node in nodes}
    else:
        # transfer previous_scores (maybe pearsonr)
        scores = {node: Score(previous_scores[node.entity]) for node in nodes}

    # init the scorer
    rw = RandomWalkScorer(num_loop=num_loop)

    # run the scorer
    scores = rw.score(mem_graph, data, current=0, scores=scores)

    scores = sorted(scores.items(), key=lambda item: item[1].key, reverse=True)

    output = []
    for item in scores:
        output.append((item[0].entity, item[1].score))
    return output


def second_order_random_walk(
    adj: np.ndarray,
    node_names: List[str] = None,
    sli: Node = None,
    num_loop=None,
    previous_scores=None,
):
    if node_names is None:
        node_names = [f"X{i}" for i in range(len(adj))]

    # == build the graph
    nodes = [Node(name, None) for name in node_names]
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
    # == end build the graph

    # prepare data for RandomWalk
    sli = np.random.choice(nodes)
    data = CaseData(data_loader=None, sli=sli, detect_time=0)

    if previous_scores is None:
        scores = {node: Score(0) for node in nodes}
    else:
        # transfer previous_scores (maybe pearsonr)
        scores = {node: Score(previous_scores[node.entity]) for node in nodes}

    # init the scorer
    rw = SecondOrderRandomWalkScorer(num_loop=num_loop)

    # run the scorer
    scores = rw.score(mem_graph, data, current=0, scores=scores)

    scores = sorted(scores.items(), key=lambda item: item[1].key, reverse=True)

    output = []
    for item in scores:
        output.append((item[0].entity, item[1].score))
    return output
