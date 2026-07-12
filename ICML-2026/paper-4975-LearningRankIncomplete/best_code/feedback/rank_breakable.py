from abc import ABC, abstractmethod
from .feedback import Feedback
from .preference import PreferenceFeedback

class RankBreakableFeedback(Feedback):
    @abstractmethod
    def rank_breaking(self) -> list[PreferenceFeedback]:
        pass
