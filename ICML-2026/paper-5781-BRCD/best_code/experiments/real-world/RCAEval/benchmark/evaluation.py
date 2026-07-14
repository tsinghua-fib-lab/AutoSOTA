from typing import List, Set, Sequence
from RCAEval.classes.graph import Node
from RCAEval.utility import dump_json, load_json


class Evaluator:
    """"""

    def __init__(self):
        self._accuracy = {k: 0.0 for k in range(1, 6)}
        self._accuracy_service = {k: 0.0 for k in range(1, 6)}
        self._ranks: List[List[Node]] = []

    def add_case(self, ranks: Sequence[Node], answer: Node):
        self._ranks.append(ranks[: 5])

        service_ranks = [n.entity for n in ranks]
        service_answer = answer.entity

        for k in range(1, 6):
            # fine-grained accuracy
            self._accuracy[k] += int(answer in ranks[:k])

            # coarse-grained
            self._accuracy_service[k] += int(service_answer in service_ranks[:k])

    @property
    def num(self) -> int:
        """
        Number of cases
        """
        return len(self._ranks)

    def accuracy(self, k: int) -> float:
        """
        AC@k is the average of accuracy@k among cases

        For each case, accuracy@k = |ranks[:k] \\cap answers| / |answers|
        """
        if k not in self._accuracy or not self._ranks:
            return None
        return self._accuracy[k] / self.num

    def accuracy_service(self, k: int) -> float:
        """
        AC@k is the average of accuracy@k among cases

        For each case, accuracy@k = |ranks[:k] \\cap answers| / |answers|
        """
        if k not in self._accuracy_service or not self._ranks:
            return None
        return self._accuracy_service[k] / self.num

    def average(self, k: int) -> float:
        """
        Avg@k = \\sum_{j=1}^{k} AC@j / k
        """
        if k not in self._accuracy or not self._ranks:
            return None
        return sum(self.accuracy(i) for i in range(1, k + 1)) / k

    def average_service(self, k: int) -> float:
        """
        Avg@k = \\sum_{j=1}^{k} AC@j / k
        """
        if k not in self._accuracy_service or not self._ranks:
            return None
        return sum(self.accuracy_service(i) for i in range(1, k + 1)) / k