# top_h_ranking.py
from .rank_breakable import RankBreakableFeedback
from .preference import PreferenceFeedback

class TopHRankingFeedback(RankBreakableFeedback):
    def __init__(self, top_h_ids: list[int], others_ids: list[int]) -> None:
        # 1. Check for duplicates in top_h
        if len(top_h_ids) != len(set(top_h_ids)):
            raise ValueError("Invalid top_h_ids: Contains duplicate item IDs, which violates strict ordering.")

        # 2. Check for contradictions between top_h and others
        overlap = set(top_h_ids).intersection(others_ids)
        if overlap:
            raise ValueError(f"Contradiction: Items with IDs {overlap} cannot be in both top_h and others.")

        self.top_h_ids = top_h_ids
        self.h = len(top_h_ids)
        self.others_ids = others_ids

    def rank_breaking(self) -> list[PreferenceFeedback]:
        prefs: list[PreferenceFeedback] = []

        for l in range(self.h):
            for m in range(l+1, self.h):
                prefs.append(PreferenceFeedback(self.top_h_ids[l], self.top_h_ids[m]))
            for i in self.others_ids:
                prefs.append(PreferenceFeedback(self.top_h_ids[l], i))
        
        return prefs
