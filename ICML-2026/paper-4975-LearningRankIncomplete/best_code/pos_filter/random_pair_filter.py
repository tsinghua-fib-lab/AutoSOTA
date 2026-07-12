from .pos_filter import PositionalFilter
import numpy as np
from feedback import PartialOrderFeedback
from feedback import PreferenceFeedback

class RandomPairPositionalFilter(PositionalFilter):
    def __init__(self, n: int) -> None:
        super().__init__(n)

    def filter(self, ranking: list[int]) -> PartialOrderFeedback:
        pair_pos = np.random.choice(self.n, 2, replace=False)
        prec_pos, succ_pos = np.sort(pair_pos)

        return PartialOrderFeedback([PreferenceFeedback(prec_id=ranking[prec_pos], succ_id=ranking[succ_pos])])
