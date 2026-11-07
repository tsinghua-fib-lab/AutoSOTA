"""Quality-diversity residuals over verified Countdown expression trees."""

from __future__ import annotations

import ast
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import torch


ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


@dataclass(frozen=True)
class StrategyArchiveConfig:
    coefficient: float = 0.15


def _canonical(node):
    if isinstance(node, ast.Expression):
        return _canonical(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return ("n", str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return ("neg", _canonical(node.operand))
    if isinstance(node, ast.BinOp):
        names = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
        operation = names.get(type(node.op))
        if operation is None:
            raise ValueError("unsupported operation")
        left, right = _canonical(node.left), _canonical(node.right)
        if operation in ("+", "*") and repr(left) > repr(right):
            left, right = right, left
        return (operation, left, right)
    raise ValueError("unsupported expression")


def expression_signature(text: str):
    matches = list(ANSWER_PATTERN.finditer(text))
    if not matches:
        return None
    expression = matches[-1].group(1).strip()
    try:
        return repr(_canonical(ast.parse(expression, mode="eval")))
    except (SyntaxError, ValueError, TypeError):
        return None


class StrategyArchiveController:
    """Reward rare verified strategies only relative to other correct ones."""

    def __init__(self, config: StrategyArchiveConfig):
        if config.coefficient < 0:
            raise ValueError("coefficient must be non-negative")
        self.config = config
        self._archive: dict[int, Counter] = defaultdict(Counter)

    def observe(self, sample_indices, correctness: torch.Tensor, response_texts):
        residual = torch.zeros_like(correctness, dtype=torch.float32)
        groups = defaultdict(list)
        for position, sample_id in enumerate(sample_indices):
            groups[int(sample_id)].append(position)

        active_groups = 0
        parsed_correct = 0
        correct_total = int(correctness.sum().detach().cpu().item())
        unique_counts = []
        pending = []
        for sample_id, positions in groups.items():
            signatures = {}
            for position in positions:
                if not bool(correctness[position].item()):
                    continue
                signature = expression_signature(str(response_texts[position]))
                if signature is not None:
                    signatures[position] = signature
                    parsed_correct += 1
                    pending.append((sample_id, signature))
            if not signatures:
                continue
            current_counts = Counter(signatures.values())
            unique_counts.append(len(current_counts))
            if len(signatures) < 2 or len(current_counts) < 2:
                continue
            scores = []
            positions_with_signature = list(signatures)
            for position in positions_with_signature:
                signature = signatures[position]
                within_group = 1.0 / current_counts[signature]
                historical = 1.0 / math.sqrt(1.0 + self._archive[sample_id][signature])
                scores.append(within_group + historical)
            score_tensor = torch.tensor(scores, device=residual.device, dtype=torch.float32)
            std = torch.std(score_tensor, unbiased=False)
            if float(std.detach().cpu().item()) <= 1e-6:
                continue
            normalized = (score_tensor - score_tensor.mean()) / (std + 1e-6)
            residual[positions_with_signature] = normalized
            active_groups += 1

        # Cross-fit the history: current responses enter only after residuals.
        for sample_id, signature in pending:
            self._archive[sample_id][signature] += 1

        metrics = {
            "strategy_archive/active_group_fraction": float(
                active_groups / max(len(groups), 1)
            ),
            "strategy_archive/parsed_correct_fraction": float(
                parsed_correct / max(correct_total, 1)
            ),
            "strategy_archive/unique_strategy_mean": float(
                sum(unique_counts) / max(len(unique_counts), 1)
            ) if unique_counts else 0.0,
            "strategy_archive/residual_abs_mean": float(
                residual.abs().mean().detach().cpu().item()
            ),
            "strategy_archive/covered_problem_count": float(len(self._archive)),
        }
        return residual, metrics
