
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from AgentTailor.ATNetwork.Critics import Encoder
from AgentTailor.ATNetwork.edge_judge import EdgeJudge
from experiments_util.text_similarity import compute_text_similarity


class TextQEdgeJudge(EdgeJudge):


    def __init__(self, min_score: float = -0.5, max_score: float = 0.95) -> None:
        self.min_score = min_score
        self.max_score = max_score

    def score_edge(
        self,
        encoder: Encoder,
        edge_input: Dict[str, Any],
        task_text: str,
        feedback_summary: str,
        pass_ratio: float,
        unit_tests: List[str],
        test_state: Tuple[bool, ...],
    ) -> float:
        _ = (feedback_summary, pass_ratio, unit_tests, test_state)
        edge_output: str = str(edge_input.get("edge_info", "") or "")
        final_answer: str = str(edge_input.get("ans_info", "") or "")
        task_text = task_text or ""

        if edge_output.strip():
            sim_answer = compute_text_similarity(encoder, edge_output, final_answer) if final_answer.strip() else 0.0
            sim_task = compute_text_similarity(encoder, edge_output, task_text) if task_text else 0.0

            base_score = 0.85 * sim_answer + 0.15 * sim_task
        else:
            base_score = -0.1

        if not edge_input.get("selected", True):
            base_score -= 0.05

        # Cancel clamp: return raw base_score (may be outside [min_score, max_score]).
        return float(base_score)


__all__ = ["TextQEdgeJudge"]


