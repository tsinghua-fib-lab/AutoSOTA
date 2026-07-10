
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from AgentTailor.ATNetwork.Critics import Encoder
from AgentTailor.ATNetwork.edge_judge import EdgeJudge
from experiments_util.text_similarity import compute_text_similarity

TEXT_SIM_EPS = 1e-6


def compute_edge_soft_score(
    encoder: Encoder,
    edge_input: Dict[str, Any],
    task_text: str,
    feedback_summary: str,
    pass_ratio: float,
    unit_tests: List[str],
    test_state: Tuple[bool, ...],
) -> float:

    edge_output = edge_input.get("edge_info", "")
    node_context = f"{edge_input.get('node1_info', '')}\n{edge_input.get('node2_info', '')}"

    sim_edge_task = compute_text_similarity(encoder, edge_output, task_text)
    sim_edge_feedback = compute_text_similarity(encoder, edge_output, feedback_summary)
    sim_node_task = compute_text_similarity(encoder, node_context, task_text)

    pos_edge_task = max(sim_edge_task, 0.0)
    pos_edge_feedback = max(sim_edge_feedback, 0.0)
    pos_node_task = max(sim_node_task, 0.0)

    passed_sims: List[float] = []
    failed_sims: List[float] = []
    if edge_output and unit_tests and test_state:
        matched = min(len(unit_tests), len(test_state))
        for idx in range(matched):
            sim_val = compute_text_similarity(encoder, edge_output, unit_tests[idx])
            if test_state[idx]:
                passed_sims.append(max(sim_val, 0.0))
            else:
                failed_sims.append(max(sim_val, 0.0))

    avg_pass_sim = float(np.mean(passed_sims)) if passed_sims else 0.0
    avg_fail_sim = float(np.mean(failed_sims)) if failed_sims else 0.0
    pass_coverage = (sum(test_state) / len(test_state)) if test_state else 0.0

    positive_scale = 1.0
    if pass_ratio < 0.4:
        positive_scale = pass_ratio / 0.4

    if edge_input.get("selected", False):


        base_score = (
            0.15 * pos_edge_task
            + 0.1 * pos_edge_feedback
            + 0.08 * pos_node_task
            + 0.2 * avg_pass_sim
            + 0.1 * pass_coverage
        )
        penalty = 0.35 * avg_fail_sim + 0.05 * (1.0 - pass_ratio)
        score = positive_scale * base_score - penalty
    else:
        score = (
            0.1 * pos_node_task
            + 0.05 * pos_edge_task
            - 0.2
            - 0.25 * avg_fail_sim
            - 0.05 * (1.0 - pass_ratio)
        )

    # Cancel clamp: return raw score (may be outside [-0.4, 0.9]).
    return float(score)


class Train4SoftJudge(EdgeJudge):

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
        raw_score = compute_edge_soft_score(
            encoder=encoder,
            edge_input=edge_input,
            task_text=task_text,
            feedback_summary=feedback_summary,
            pass_ratio=pass_ratio,
            unit_tests=unit_tests,
            test_state=test_state,
        )
        # Cancel clamp: return raw_score (may be outside [min_score, max_score]).
        return float(raw_score)


__all__ = ["compute_edge_soft_score", "Train4SoftJudge"]

