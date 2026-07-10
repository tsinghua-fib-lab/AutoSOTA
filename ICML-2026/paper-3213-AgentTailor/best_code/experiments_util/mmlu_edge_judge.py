from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from AgentTailor.ATNetwork.edge_judge import EdgeJudge
from AgentTailor.ATNetwork.Critics import Encoder
from experiments_util.text_similarity import compute_text_similarity

"""
class MMLUEdgeJudge(EdgeJudge):

    def __init__(self, min_score: float = -0.5, max_score: float = 0.95) -> None:
        self.min_score = min_score
        self.max_score = max_score

    def score_edge(
        self,
        encoder: Encoder,
        edge_input: Dict[str, str],
        task_text: str,
        feedback_summary: str,
        pass_ratio: float,
        unit_tests: List[str],
        test_state: Tuple[bool, ...],
    ) -> float:
        edge_output: str = edge_input.get("edge_info", "") or ""
        option_map = _parse_options(task_text)
        expected_letter = _extract_expected_letter(feedback_summary)
        declared_letter = _extract_letter(edge_output)


        correct_text = option_map.get(expected_letter, "")
        sim_correct = _similarity(encoder, edge_output, correct_text)
        sim_wrong = _max_wrong_similarity(encoder, edge_output, option_map, expected_letter)
        sim_question = _similarity(encoder, edge_output, task_text)

        score = 0.0


        score += 0.2 * pass_ratio - 0.05 * (1.0 - pass_ratio)


        if expected_letter:
            if declared_letter == expected_letter:
                score += 0.45 + 0.15 * pass_ratio
            elif declared_letter:
                score -= 0.3 + 0.1 * (1.0 - pass_ratio)
        elif declared_letter:
            score += 0.05


        score += 0.35 * sim_correct
        score -= 0.25 * sim_wrong
        score += 0.1 * sim_question


        round_idx = edge_input.get("round", 0)
        try:
            round_idx = int(round_idx)
        except (TypeError, ValueError):
            round_idx = 0
        score += 0.02 * max(0, round_idx)

        # Cancel clamp: return raw score (may be outside [min_score, max_score]).
        return float(score)
"""
class MMLUEdgeJudge(EdgeJudge):

    def __init__(self, min_score: float = -0.5, max_score: float = 0.95) -> None:
        self.min_score = min_score
        self.max_score = max_score

    def score_edge(
        self,
        encoder: Encoder,
        edge_input: Dict[str, str],
        task_text: str,
        feedback_summary: str,
        pass_ratio: float,
        unit_tests: List[str],
        test_state: Tuple[bool, ...],
    ) -> float:
        edge_output: str = edge_input.get("edge_info", "") or ""
        option_map = _parse_options(task_text)
        expected_letter = _extract_expected_letter(feedback_summary)
        declared_letter = _extract_letter(edge_output)


        correct_text = option_map.get(expected_letter, "")
        sim_correct = _similarity(encoder, edge_output, correct_text)
        sim_wrong = _max_wrong_similarity(encoder, edge_output, option_map, expected_letter)
        sim_question = _similarity(encoder, edge_output, task_text)

        score = 0.0
        score += 1* sim_correct
        round_idx = edge_input.get("round", 0)
        try:
            round_idx = int(round_idx)
        except (TypeError, ValueError):
            round_idx = 0
        score += 0.02 * max(0, round_idx)

        return float(max(self.min_score, min(self.max_score, score)))

def _extract_letter(text: str) -> Optional[str]:
    match = re.search(r"\b([A-Da-d])\b", text or "")
    return match.group(1).upper() if match else None


def _extract_expected_letter(feedback_summary: str) -> Optional[str]:


    match = re.search(r"expected\s+([A-Da-d])", feedback_summary or "")
    if match:
        return match.group(1).upper()

    if "Incorrect" not in (feedback_summary or ""):

        match = re.search(r"choice\s*==\s*([A-Da-d])", feedback_summary or "")
        if match:
            return match.group(1).upper()
    return None


def _parse_options(task_text: str) -> Dict[str, str]:


    option_map: Dict[str, str] = {}
    for line in (task_text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) > 2 and line[1] == "." and line[0].upper() in "ABCD":
            option_map[line[0].upper()] = line[2:].strip()
            continue

        match = re.match(r"Option\s+([A-Da-d])\s*[:.]\s*(.*)", line)
        if match:
            option_map[match.group(1).upper()] = match.group(2).strip()
    return option_map


def _similarity(encoder: Encoder, text_a: str, text_b: str) -> float:
    if not text_a or not text_b:
        return 0.0
    sim = compute_text_similarity(encoder, text_a, text_b)
    return max(0.0, float(sim))


def _max_wrong_similarity(
    encoder: Encoder,
    edge_output: str,
    option_map: Dict[str, str],
    expected_letter: Optional[str],
) -> float:
    wrong_texts = [
        text for letter, text in option_map.items()
        if letter != expected_letter and text
    ]
    if not wrong_texts or not edge_output:
        return 0.0
    return max(_similarity(encoder, edge_output, text) for text in wrong_texts)


__all__ = ["MMLUEdgeJudge"]



