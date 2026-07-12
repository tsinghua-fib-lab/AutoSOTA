from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Sequence
from feedback import Feedback

F = TypeVar('F', bound=Feedback)

class Algorithm(ABC, Generic[F]):
    def __init__(self, name: str) -> None:
        self.name = name
    
    @abstractmethod
    def feedback(self, feedback: F) -> None:
        pass

    @abstractmethod
    def predict(self) -> list[int]:
        pass
