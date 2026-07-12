from abc import ABC, abstractmethod
from feedback import PartialOrderFeedback

class PositionalFilter(ABC):
    def __init__(self, n: int) -> None:
        self.n = n
    
    @abstractmethod
    def filter(self, ranking: list[int]) -> PartialOrderFeedback:
        pass

