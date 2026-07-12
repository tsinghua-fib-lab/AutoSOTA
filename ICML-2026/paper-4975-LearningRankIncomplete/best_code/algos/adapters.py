from typing import TypeVar, Generic
from feedback import PreferenceFeedback, PartialOrderFeedback, RankBreakableFeedback
from .algo import Algorithm

# Define algorithm types based on feedback
Algo_PO = TypeVar('Algo_PO', bound=Algorithm[PartialOrderFeedback])
Algo_Pref = TypeVar('Algo_Pref', bound=Algorithm[PreferenceFeedback])

class PreferenceToPartialOrderAdapter(Algorithm[PreferenceFeedback], Generic[Algo_PO]):
    """
    Adapts an algorithm that expects PartialOrderFeedback to accept PreferenceFeedback.
    """
    def __init__(self, algorithm: Algo_PO):
        super().__init__(name=algorithm.name)
        self._algorithm = algorithm

    def feedback(self, feedback: PreferenceFeedback) -> None:
        # A single preference is a valid partial order with one relation.
        # We need to create a PartialOrderFeedback object.
        po_feedback = PartialOrderFeedback(preferences=[feedback])
        self._algorithm.feedback(po_feedback)

    def predict(self) -> list[int]:
        return self._algorithm.predict()

class PartialOrderToPreferenceAdapter(Algorithm[PartialOrderFeedback], Generic[Algo_Pref]):
    """
    Adapts an algorithm that expects PreferenceFeedback to accept PartialOrderFeedback.
    """
    def __init__(self, algorithm: Algo_Pref):
        super().__init__(name=algorithm.name)
        self._algorithm = algorithm

    def feedback(self, feedback: PartialOrderFeedback) -> None:
        # Feed all pairwise preferences from the partial order to the underlying algorithm.
        for pref in feedback.rank_breaking():
            self._algorithm.feedback(pref)

    def predict(self) -> list[int]:
        return self._algorithm.predict()

class RankBreakableToPreferenceAdapter(Algorithm[RankBreakableFeedback], Generic[Algo_Pref]):
    """
    Adapts an algorithm that expects PreferenceFeedback to accept any RankBreakableFeedback.
    """
    def __init__(self, algorithm: Algo_Pref):
        super().__init__(name=algorithm.name)
        self._algorithm = algorithm

    def feedback(self, feedback: RankBreakableFeedback) -> None:
        # rank_breaking() breaks the complex feedback down into pairwise preferences.
        # We feed each of these pairwise preferences to the underlying algorithm.
        for pref in feedback.rank_breaking():
            self._algorithm.feedback(pref)

    def predict(self) -> list[int]:
        return self._algorithm.predict()


class RankBreakableToPartialOrderAdapter(Algorithm[RankBreakableFeedback], Generic[Algo_PO]):
    """
    Adapts an algorithm that expects PartialOrderFeedback to accept any RankBreakableFeedback.
    """
    def __init__(self, algorithm: Algo_PO):
        super().__init__(name=algorithm.name)
        self._algorithm = algorithm

    def feedback(self, feedback: RankBreakableFeedback) -> None:
        # 1. Break the feedback down into a list of PreferenceFeedback objects.
        prefs = feedback.rank_breaking()
        
        # 2. Construct a new PartialOrderFeedback using those preferences.
        # Note: This relies on the output of rank_breaking() being a valid, 
        # explicitly transitive partial order, which your TopHRankingFeedback correctly produces.
        po_feedback = PartialOrderFeedback(preferences=prefs)
        
        # 3. Feed the newly constructed PartialOrderFeedback to the underlying algorithm.
        self._algorithm.feedback(po_feedback)

    def predict(self) -> list[int]:
        return self._algorithm.predict()
