
from __future__ import annotations

import asyncio
import copy
import os
import sys
import random
import functools
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set

from pathlib import Path

# Add project root to Python path (allow running this file directly)
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from AgentTailor.ATNetwork.Actor import Actor
from AgentTailor.ATNetwork.Critics import Critics, Encoder
from AgentTailor.ATNetwork.edge_judge import EdgeJudge
from dataset.humaneval_dataset import HumanEvalDataset
from experiments_util.soft_judge import Train4SoftJudge
from experiments_util.textqa_edge_judge import TextQEdgeJudge
from experiments_util.mmlu_edge_judge import MMLUEdgeJudge
from experiments_util.edge_utils import prepare_edge_inputs
from AgentTailor.utils.globals import PromptTokens, CompletionTokens, Cost, ApiCalls
device = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def epn_concat_input_dim(n_segments: int = 5, embed_dim: int = 384) -> int:
    """EPN first-layer input width: concatenated encoded segment length (default 5×MiniLM-L6-v2)."""
    return n_segments * embed_dim


def epn_head_hidden_sizes() -> List[int]:
    """Hidden layer widths after the concatenated input (final dim 1 is scalar output)."""
    return [1]


EPS = 1e-6
UNSELECTED_PENALTY =-0.1
MAX_REWARD_ABS = 1.0
MIN_VIRTUAL_LR = 5e-5
MAX_EDGE_PROMPT_LEN = None  # None means no truncation
MAX_EDGE_OUTPUT_LEN = None  # None means no truncation
MAX_EDGE_HISTORY_LEN = None  # None means no truncation


@dataclass
class TrainingState:
    cumulative_correct: int = 0
    cumulative_total: int = 0
    initial_prompt_tokens: int = 0
    initial_completion_tokens: int = 0

    @property
    def accuracy(self) -> float:
        if self.cumulative_total == 0:
            return 0.0
        return self.cumulative_correct / max(1, self.cumulative_total)

    def update(self, is_correct: bool) -> None:
        self.cumulative_total += 1
        if is_correct:
            self.cumulative_correct += 1

    def get_token_stats(self) -> Tuple[int, int, int]:
        prompt_tokens = int(PromptTokens.instance().value) - self.initial_prompt_tokens
        completion_tokens = int(CompletionTokens.instance().value) - self.initial_completion_tokens
        total_tokens = prompt_tokens + completion_tokens
        return prompt_tokens, completion_tokens, total_tokens

    def reset_token_baseline(self) -> None:
        self.initial_prompt_tokens = int(PromptTokens.instance().value)
        self.initial_completion_tokens = int(CompletionTokens.instance().value)


def _set_optimizer_lr(optimizer: optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _get_role(actor: Actor, node_id: str) -> str:
    if node_id in actor.nodes:
        node = actor.nodes[node_id]
        if getattr(node, "role", None):
            return str(node.role)
        if getattr(node, "agent_name", None):
            return str(node.agent_name)
    return node_id


def _edge_key(out_id: str, in_id: str, edge_type: str) -> Tuple[str, str, str]:
    return (out_id, in_id, edge_type)


def _find_edge_index(actor: Actor, out_node_id: str, in_node_id: str, edge_type: str) -> int:
    edge_list = (
        actor.potential_spatial_edges if edge_type == "spatial" else actor.potential_temporal_edges
    )
    for idx, edge in enumerate(edge_list):
        if edge[0] == out_node_id and edge[1] == in_node_id:
            return idx
    return -1


def _get_edge_logit(
        actor: Actor,
        edge_type: str,
        out_id: str,
        in_id: str,
        spatial_edge_map: Dict[Tuple[str, str], int],
        temporal_edge_map: Dict[Tuple[str, str], int],
) -> Optional[torch.Tensor]:
    if edge_type == "spatial":
        idx = spatial_edge_map.get((out_id, in_id))
        if idx is None:
            return None
        return actor.spatial_logits[idx]
    elif edge_type == "temporal":
        idx = temporal_edge_map.get((out_id, in_id))
        if idx is None:
            return None
        if not actor.optimized_temporal:
            return None
        return actor.temporal_logits[idx]
    return None


def _clamp_reward(value: float) -> float:
    return max(-MAX_REWARD_ABS, min(MAX_REWARD_ABS, value))


def _adjust_reward_by_selection(detail: Dict[str, Any]) -> None:
    reward_val = float(detail.get("reward", 0.0) or 0.0)
    raw_delta = float(detail.get("raw_delta", 0.0) or 0.0)
    if detail.get("selected", False):


        if raw_delta <= 0.0:
            detail["reward"] = min(0.0, reward_val)
        else:
            detail["reward"] = max(0.0, reward_val)
    else:
        detail["reward"] = min(0.0, reward_val, raw_delta)


def _refresh_edge_logits(
        actor: Actor,
        edge_details: List[Dict[str, Any]],
        spatial_edge_map: Dict[Tuple[str, str], int],
        temporal_edge_map: Dict[Tuple[str, str], int],
) -> None:

    for detail in edge_details:
        logit_tensor = _get_edge_logit(
            actor,
            detail.get("type", "spatial"),
            detail.get("out_node_id", ""),
            detail.get("in_node_id", ""),
            spatial_edge_map,
            temporal_edge_map,
        )
        if logit_tensor is None:
            detail["logit"] = None
            detail["prob"] = None
        else:
            detail["logit"] = float(logit_tensor.detach().cpu())
            detail["prob"] = float(torch.sigmoid(logit_tensor).detach().cpu())


def _sample_roundwise_selected_edges(
        actor: Actor,
        spatial_probs: torch.Tensor,
        temporal_probs: Optional[torch.Tensor],
        num_rounds: int,
        max_selected_edges: int,
        include_output: bool = False,
        include_prompt: bool = True,
) -> List[Dict[str, Any]]:

    selected_edges: List[Dict[str, Any]] = []
    temporal_available = actor.optimized_temporal and temporal_probs is not None

    for round_idx in range(num_rounds):
        candidate_edges: List[Tuple[str, str, str, float]] = []

        for edge_idx, edge in enumerate(actor.potential_spatial_edges):
            if edge[0] == edge[1]:
                continue
            prob = spatial_probs[edge_idx].item()
            if torch.rand(1).item() < prob:
                candidate_edges.append((edge[0], edge[1], "spatial", prob))

        if temporal_available and round_idx > 0:
            for edge_idx, edge in enumerate(actor.potential_temporal_edges):
                prob = temporal_probs[edge_idx].item()
                if torch.rand(1).item() < prob:
                    candidate_edges.append((edge[0], edge[1], "temporal", prob))

        if len(candidate_edges) > max_selected_edges:
            candidate_edges.sort(key=lambda x: x[3], reverse=True)
            candidate_edges = candidate_edges[:max_selected_edges]

        for out_id, in_id, edge_type, _ in candidate_edges:
            node_parts = actor.get_edge_node_info_parts(
                out_id,
                in_id,
                include_output=include_output,
                max_output_len=MAX_EDGE_OUTPUT_LEN,
                include_prompt=include_prompt,
                max_prompt_len=MAX_EDGE_PROMPT_LEN,
            )
            selected_edges.append(
                {
                    "out_node_id": out_id,
                    "in_node_id": in_id,
                    "type": edge_type,
                    "round": round_idx,
                    "selected": True,
                    "node_parts": node_parts,
                }
            )

    return selected_edges


def _build_edge_details(
        actor: Actor,
        edge_records: List[Dict[str, Any]],
        final_answer: str,
) -> List[Dict[str, Any]]:

    selected_map: Dict[Tuple[str, str, str, int], bool] = {}
    for record in edge_records:
        out_id = record.get("out_node_id", "")
        in_id = record.get("in_node_id", "")
        edge_type = record.get("type", "spatial")
        round_idx = int(record.get("round", 0))
        key = (out_id, in_id, edge_type, round_idx)
        selected_map[key] = bool(record.get("selected", True))

    prepared_edges = prepare_edge_inputs(
        summary=final_answer,
        edge_records=edge_records,
        actor=actor,
        include_output=True,
        max_output_len=MAX_EDGE_OUTPUT_LEN,
    )

    edge_details: List[Dict[str, Any]] = []
    for prepared in prepared_edges:
        out_id = prepared["out_node_id"]
        in_id = prepared["in_node_id"]
        edge_type = prepared.get("type", "spatial")
        round_idx = int(prepared.get("round", 0))
        key = (out_id, in_id, edge_type, round_idx)
        raw_selected = selected_map.get(key, True)

        node_parts = actor.get_edge_node_info_parts(
            out_id,
            in_id,
            include_output=True,
            max_output_len=MAX_EDGE_OUTPUT_LEN,
            include_prompt=True,
            max_prompt_len=MAX_EDGE_PROMPT_LEN,
        )
        edge_details.append(
            {
                "edge_input": prepared,
                "out_node_id": out_id,
                "in_node_id": in_id,
                "type": edge_type,
                "round": round_idx,
                "node_parts": node_parts,
                "selected": raw_selected,
            }
        )
    return edge_details


def _critic_build_edge_batch(
    expanded_subset: List[Dict[str, Any]],
    actor: Actor,
    task_text: str,
    device: torch.device,
) -> Tuple[
    List[str],
    List[str],
    List[str],
    List[str],
    List[str],
    torch.Tensor,
]:
    """为 Critic 构造一批边的输入张量与 Δ 目标（expanded_subset 为一条批次内的边列表）。"""
    in_node_description_list: List[str] = []
    in_node_history_list: List[str] = []
    query_list: List[str] = []
    out_node_description_list: List[str] = []
    out_node_history_list: List[str] = []

    for d in expanded_subset:
        out_node_id = d.get("out_node_id", "")
        in_node_id = d.get("in_node_id", "")
        node_info = actor.get_edge_node_info_with_history(
            out_node_id=out_node_id,
            in_node_id=in_node_id,
            include_output=False,
            include_prompt=True,
            max_history_len=MAX_EDGE_HISTORY_LEN,
        )
        in_node_description_list.append(node_info["in_node"]["description"])
        in_node_history_list.append(node_info["in_node"]["history"])
        query_list.append(task_text)
        out_node_description_list.append(node_info["out_node"]["description"])
        out_node_history_list.append(node_info["out_node"]["history"])

    critic_targets = torch.tensor(
        [[float(d.get("delta", 0.0) or 0.0)] for d in expanded_subset],
        dtype=torch.float32,
        device=device,
    )
    return (
        in_node_description_list,
        in_node_history_list,
        query_list,
        out_node_description_list,
        out_node_history_list,
        critic_targets,
    )


def _compute_ranking_loss(
    predictions: "torch.Tensor",
    targets: "torch.Tensor",
    edge_details: "List[Dict[str, Any]]",
    margin: float = 0.3,
) -> "torch.Tensor":
    """Compute pairwise ranking loss for critic predictions.
    
    This is a placeholder that computes a simple margin-based ranking loss
    between selected and unselected edges within each sample group.
    """
    import torch
    if predictions.numel() < 2:
        return torch.tensor(0.0, device=predictions.device)
    # Simple pairwise ranking: correct edges should score higher than incorrect ones
    pred_flat = predictions.view(-1)
    tgt_flat = targets.view(-1)
    pos_mask = tgt_flat > 0.5
    neg_mask = ~pos_mask
    if pos_mask.any() and neg_mask.any():
        pos_mean = pred_flat[pos_mask].mean()
        neg_mean = pred_flat[neg_mask].mean()
        return torch.clamp(neg_mean - pos_mean + margin, min=0.0)
    return torch.tensor(0.0, device=predictions.device)


def _critic_loss_from_batch(
    critic_predictions: torch.Tensor,
    critic_targets: torch.Tensor,
    expanded_subset: List[Dict[str, Any]],
    sparsity_weight: float = 0.1,
) -> torch.Tensor:
    device = critic_predictions.device
    delta_flat = critic_targets.view(-1)
    pred_flat = critic_predictions.view(-1)
    sq_error_flat = (pred_flat - delta_flat) ** 2
    tau_pos = 0.0
    class_weight = torch.ones_like(delta_flat)
    neg_mask = delta_flat < -tau_pos
    pos_mask = delta_flat > tau_pos
    neutral_mask = (~neg_mask) & (~pos_mask)
    w_pos, w_neu, w_neg = 1.0, 1.0, 1.0
    class_weight[neg_mask] = w_neg
    class_weight[neutral_mask] = w_neu
    class_weight[pos_mask] = w_pos

    abs_delta = delta_flat.abs()
    mag_weight = torch.ones_like(delta_flat)
    strong_threshold = 0.1
    strong_neg_mask = delta_flat < -strong_threshold
    strong_pos_mask = delta_flat > strong_threshold
    weak_mask = abs_delta <= strong_threshold
    mag_weight[weak_mask] = 0.5
    mag_weight[strong_pos_mask] = 1.5
    mag_weight[strong_neg_mask] = 3.0
    class_weight = class_weight * mag_weight

    beta_mis_neg = 3.0
    neg_mis_margin = 0.05
    misclassified_neg_mask = neg_mask & (pred_flat > neg_mis_margin)
    if misclassified_neg_mask.any():
        class_weight[misclassified_neg_mask] = class_weight[misclassified_neg_mask] * beta_mis_neg

    mis_neg_hinge = torch.tensor(0.0, device=device)
    if neg_mask.any():
        pred_neg = pred_flat[neg_mask]
        mis_neg_hinge = torch.clamp(pred_neg - neg_mis_margin, min=0.0).mean()

    mse_loss = (sq_error_flat * class_weight).mean()

    ranking_loss = _compute_ranking_loss(
        predictions=critic_predictions,
        targets=critic_targets,
        edge_details=expanded_subset,
        margin=0.3,
    )

    margin_pos = 0.05
    margin_neutral = 0.02
    pos_mask = delta_flat > tau_pos
    neg_mask = delta_flat < -tau_pos
    neutral_mask = (~pos_mask) & (~neg_mask)

    bucket_terms = []
    if pos_mask.any():
        pos_violation = torch.clamp(margin_pos - pred_flat[pos_mask], min=0.0) * 2.0
        bucket_terms.append(pos_violation)
    if neg_mask.any():
        neg_violation = torch.clamp(margin_pos + pred_flat[neg_mask], min=0.0) * 2.0
        bucket_terms.append(neg_violation)
    if neutral_mask.any():
        neu_violation = torch.clamp(pred_flat[neutral_mask].abs() - margin_neutral, min=0.0)
        bucket_terms.append(neu_violation)
    # Elastic net regularization: L1 encourages sparsity, L2 prevents overfitting
    lambda_l1 = sparsity_weight * 0.5
    lambda_l2 = sparsity_weight * 0.1
    l1_term = lambda_l1 * pred_flat.abs().mean()
    l2_term = lambda_l2 * (pred_flat ** 2).mean()
    critic_loss = (
        mse_loss + 0.1 * ranking_loss + l1_term + l2_term
    )
    return critic_loss


STAGE3_MSE_LOW_THRESHOLD = 0.015


def _stage3_mse_low_count(
    mse_list: List[float], threshold: float = STAGE3_MSE_LOW_THRESHOLD
) -> Tuple[int, int]:
    """返回 (MSE<=threshold 的样本数, 总样本数)。"""
    if not mse_list:
        return 0, 0
    n_ok = sum(1 for x in mse_list if float(x) <= threshold)
    return n_ok, len(mse_list)


def _epn_edge_correlation_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    n = int(y_true.size)
    out: Dict[str, float] = {
        "n": float(n),
        "pearson_r": 0.0,
        "spearman_r": 0.0,
        "spearman_pvalue": float("nan"),
        "r2": 0.0,
        "mse": 0.0,
    }
    if n < 2 or y_true.shape != y_pred.shape:
        return out
    out["mse"] = float(np.mean((y_pred - y_true) ** 2))
    if np.std(y_true) > 1e-8 and np.std(y_pred) > 1e-8:
        out["pearson_r"] = float(np.corrcoef(y_true, y_pred)[0, 1])
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
        out["r2"] = float(1.0 - ss_res / (ss_tot + 1e-12))
    try:
        from scipy import stats as scipy_stats

        sp = scipy_stats.spearmanr(y_true, y_pred, nan_policy="omit")
        if sp.correlation is not None and not np.isnan(sp.correlation):
            out["spearman_r"] = float(sp.correlation)
        if sp.pvalue is not None and not np.isnan(sp.pvalue):
            out["spearman_pvalue"] = float(sp.pvalue)
    except Exception:
        pass
    return out


def _compute_real_deltas(
        edge_details: List[Dict[str, Any]],
        edge_judge: EdgeJudge,
        encoder: Encoder,
        task_text: str,
        feedback_summary: str,
        pass_ratio: float,
        unit_tests: List[str],
        test_state: Tuple[bool, ...],
        total_rounds: int,
        verbose: bool = False,
) -> Dict[int, float]:

    if not edge_details:
        return {}, {}

    incoming_groups: Dict[Tuple[int, str], List[int]] = defaultdict(list)
    round_groups: Dict[int, List[int]] = defaultdict(list)

    for idx, detail in enumerate(edge_details):
        round_idx = int(detail["round"])
        round_groups[round_idx].append(idx)
        incoming_groups[(round_idx, detail["in_node_id"])].append(idx)

        score = edge_judge.score_edge(
            encoder=encoder,
            edge_input=detail["edge_input"],
            task_text=task_text,
            feedback_summary=feedback_summary,
            pass_ratio=pass_ratio,
            unit_tests=unit_tests,
            test_state=test_state,
        )
        detail["soft_score"] = float(score)

    round_total_delta: Dict[int, float] = defaultdict(float)
    edge_reward_map: Dict[Tuple[str, str, str], float] = defaultdict(float)

    rounds_denominator = max(1, total_rounds)

    round_reward_records: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for round_idx, indices in round_groups.items():
        for idx in indices:
            detail = edge_details[idx]


            # avg_incoming 应该基于“同一个接收节点(in_node_id)”的其他入边计算
            incoming_idx_to_source = incoming_groups[(round_idx, detail["in_node_id"])]
            if len(incoming_idx_to_source) > 1:
                total = sum(edge_details[i]["soft_score"] for i in incoming_idx_to_source) - detail["soft_score"]
                avg_incoming = total / max(1, len(incoming_idx_to_source) - 1)
            else:
                avg_incoming = 0.0
            delta = detail["soft_score"] - avg_incoming
            if verbose:
                print(detail["soft_score"], "->", delta)
            detail["delta"] = delta
            round_total_delta[round_idx] += delta

        round_total = round_total_delta[round_idx]
        round_size = len(indices)

        if round_total < -EPS:
            print()
        for idx in indices:
            detail = edge_details[idx]
            delta = detail.get("delta", 0.0)
            reward = 0.0
            if abs(round_total) > EPS:
                reward = delta / (abs(round_total) + EPS)
            detail["reward"] = reward
            round_reward_records[round_idx].append((idx, reward))

    for round_idx, reward_pairs in round_reward_records.items():
        for idx, _ in reward_pairs:
            edge_details[idx]["raw_delta"] = edge_details[idx].get("delta", 0.0)

    for detail in edge_details:
        _adjust_reward_by_selection(detail)

    return round_total_delta


def _compute_reward_from_deltas(
        edge_details: List[Dict[str, Any]],
        total_rounds: int,
) -> Dict[int, float]:

    if not edge_details:
        return {}, {}

    round_groups: Dict[int, List[int]] = defaultdict(list)
    for idx, detail in enumerate(edge_details):
        round_groups[int(detail["round"])].append(idx)

    round_total_delta: Dict[int, float] = defaultdict(float)
    for round_idx, indices in round_groups.items():
        round_total_delta[round_idx] = sum(edge_details[idx].get("delta", 0.0) for idx in indices)

    rounds_denominator = max(1, total_rounds)

    round_reward_records: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for round_idx, indices in round_groups.items():
        total = round_total_delta[round_idx]
        round_size = len(indices)
        for idx in indices:
            delta = edge_details[idx].get("delta", 0.0)
            reward = 0.0
            if abs(total) > EPS:

                reward = delta / (abs(total) + EPS)
            edge_details[idx]["reward"] = reward
            round_reward_records[round_idx].append((idx, reward))

    for round_idx, reward_pairs in round_reward_records.items():
        for idx, _ in reward_pairs:
            edge_details[idx]["raw_delta"] = edge_details[idx].get("delta", 0.0)

    for detail in edge_details:
        _adjust_reward_by_selection(detail)

    return round_total_delta

def _policy_loss_from_rewards(
        actor: Actor,
        edge_details: List[Dict[str, Any]],
        selected_keys: set,
        is_correct: bool,
        spatial_edge_map: Dict[Tuple[str, str], int],
        temporal_edge_map: Dict[Tuple[str, str], int],
        num_rounds: int,
        *,
        force_unselected_penalty: bool = False,
        penalty_scale_override: Optional[float] = None,
        penalize_unselected_edges: bool = True,
        penalize_all_potential_edges: bool = True,
) -> Tuple[torch.Tensor, List[Dict[str, Any]], Set[Tuple[str, str, str]]]:

    if not edge_details and not (is_correct and selected_keys is not None):
        return torch.tensor(0.0, device=device), [], set()

    edge_reward_map: Dict[Tuple[str, str, str], float] = defaultdict(float)
    edge_raw_delta_map: Dict[Tuple[str, str, str], float] = defaultdict(float)
    selected_keys_set: Set[Tuple[str, str, str]] = set()
    unselected_keys_set: Set[Tuple[str, str, str]] = set()

    for detail in edge_details:
        key = _edge_key(
            detail.get("out_node_id", ""),
            detail.get("in_node_id", ""),
            detail.get("type", ""),
        )
        is_selected = detail.get("selected", False)

        if is_selected:
            selected_keys_set.add(key)
            reward_val = _clamp_reward(float(detail.get("reward", 0.0)))
            if abs(reward_val) < EPS:
                reward_val = 0.0
        else:
            unselected_keys_set.add(key)
            reward_val = -UNSELECTED_PENALTY

        detail["reward_clamped"] = reward_val
        edge_reward_map[key] += reward_val


        raw_delta = float(detail.get("raw_delta", 0.0) or 0.0)
        if key not in edge_raw_delta_map or abs(raw_delta) > abs(edge_raw_delta_map[key]):
            edge_raw_delta_map[key] = raw_delta

    log_terms: List[torch.Tensor] = []
    edge_logs: List[Dict[str, Any]] = []

    handled_keys: Set[Tuple[str, str, str]] = set()
    penalty_keys: Set[Tuple[str, str, str]] = set()
    negative_utility_keys: Set[Tuple[str, str, str]] = set()


    for (out_id, in_id, edge_type), reward in edge_reward_map.items():
        if (out_id, in_id, edge_type) not in selected_keys_set:
            continue


        raw_delta = edge_raw_delta_map.get((out_id, in_id, edge_type), 0.0)
        if raw_delta <= 0.0:

            negative_utility_keys.add((out_id, in_id, edge_type))

        if abs(reward) < EPS:
            handled_keys.add((out_id, in_id, edge_type))
            continue

        logit = _get_edge_logit(actor, edge_type, out_id, in_id, spatial_edge_map, temporal_edge_map)
        if logit is None or not logit.requires_grad:
            continue
        prob = torch.sigmoid(logit)
        log_prob = torch.log(prob + 1e-8)
        reward_tensor = torch.tensor(float(reward), device=log_prob.device, requires_grad=False)

        log_terms.append(log_prob * reward_tensor)
        edge_logs.append(
            {
                "out_node_id": out_id,
                "in_node_id": in_id,
                "type": edge_type,
                "reward": float(reward),
                "logit": float(logit.detach().cpu()),
                "prob": float(prob.detach().cpu()),
            }
        )
        handled_keys.add((out_id, in_id, edge_type))

    penalty_scale = (
        penalty_scale_override
        if penalty_scale_override is not None
        else UNSELECTED_PENALTY * max(1, num_rounds)
    )

    if penalize_unselected_edges:
        penalty_keys.update(unselected_keys_set)

    if penalize_all_potential_edges:
        for edge in actor.potential_spatial_edges:
            key = _edge_key(edge[0], edge[1], "spatial")
            if key not in handled_keys and key not in penalty_keys:
                penalty_keys.add(key)

        for edge in actor.potential_temporal_edges:
            key = _edge_key(edge[0], edge[1], "temporal")
            if key not in handled_keys and key not in penalty_keys:
                penalty_keys.add(key)

    penalty_keys.update(negative_utility_keys)

    for key in penalty_keys:
        out_id, in_id, edge_type = key
        logit = _get_edge_logit(actor, edge_type, out_id, in_id, spatial_edge_map, temporal_edge_map)
        if logit is None or not logit.requires_grad:
            continue
        prob = torch.sigmoid(logit)
        penalty_term = torch.log(1 - prob + 1e-8)
        penalty = torch.tensor(penalty_scale, device=prob.device)
        log_terms.append(-penalty * penalty_term)
        edge_logs.append(
            {
                "out_node_id": out_id,
                "in_node_id": in_id,
                "type": edge_type,
                "reward": -float(penalty.detach().cpu()),
                "logit": float(logit.detach().cpu()),
                "prob": float(prob.detach().cpu()),
            }
        )

    if not log_terms:

        return torch.tensor(0.0, device=device), edge_logs, set(edge_reward_map.keys())

    log_terms_tensor = torch.stack(log_terms)
    policy_loss = -log_terms_tensor.mean()

    selected_log_terms = []
    penalty_log_terms = []
    for i, log in enumerate(edge_logs):
        if log.get("reward", 0) != -float(penalty_scale):
            selected_log_terms.append(log_terms_tensor[i].item())
        else:
            penalty_log_terms.append(log_terms_tensor[i].item())

    return policy_loss, edge_logs, set(edge_reward_map.keys())


def _print_actor_edge_info(
        edge_details: List[Dict[str, Any]],
        actor: Actor,
) -> None:
    print("\nActor_edge_info")

    combined_edges: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for detail in edge_details:
        out_id = detail.get("out_node_id", "")
        in_id = detail.get("in_node_id", "")
        edge_type = detail.get("type", "spatial")
        key = (out_id, in_id, edge_type)

        reward_clamped = float(detail.get("reward_clamped", detail.get("reward", 0.0)) or 0.0)
        raw_delta = float(detail.get("raw_delta", 0.0) or 0.0)
        selected_flag = bool(detail.get("selected", False))
        logit_val = detail.get("logit")
        prob_val = detail.get("prob")
        if key not in combined_edges:
            combined_edges[key] = {
                "out_id": out_id,
                "in_id": in_id,
                "type": edge_type,
                "reward": reward_clamped,
                "reward_clamped": reward_clamped,
                "raw_delta": raw_delta,
                "selected": selected_flag,
                "logit": logit_val,
                "prob": prob_val,
            }
        else:
            combined_edges[key]["reward"] += reward_clamped

            combined_edges[key]["reward_clamped"] = combined_edges[key].get("reward_clamped", 0.0) + reward_clamped

            if abs(raw_delta) > abs(combined_edges[key].get("raw_delta", 0.0)):
                combined_edges[key]["raw_delta"] = raw_delta
            combined_edges[key]["selected"] = selected_flag
            if logit_val is not None:
                combined_edges[key]["logit"] = logit_val
            if prob_val is not None:
                combined_edges[key]["prob"] = prob_val


    for edge in actor.potential_spatial_edges:
        key = (edge[0], edge[1], "spatial")
        if key not in combined_edges:
            combined_edges[key] = {
                "out_id": edge[0],
                "in_id": edge[1],
                "type": "spatial",
                "reward": 0.0,
                "logit": None,
                "prob": None,
            }
    for edge in actor.potential_temporal_edges:
        key = (edge[0], edge[1], "temporal")
        if key not in combined_edges:
            combined_edges[key] = {
                "out_id": edge[0],
                "in_id": edge[1],
                "type": "temporal",
                "reward": 0.0,
                "logit": None,
                "prob": None,
            }

    unique_edges: List[Dict[str, Any]] = []
    for key, record in combined_edges.items():
        out_id, in_id, edge_type = key
        logit_val = record.get("logit")
        prob_val = record.get("prob")
        idx = _find_edge_index(actor, out_id, in_id, edge_type)
        if logit_val is None or prob_val is None:
            if edge_type == "spatial" and idx >= 0:
                logit_tensor = actor.spatial_logits[idx]
                logit_val = float(logit_tensor.detach().cpu())
                prob_val = float(torch.sigmoid(logit_tensor).detach().cpu())
            elif edge_type == "temporal" and idx >= 0 and actor.optimized_temporal:
                logit_tensor = actor.temporal_logits[idx]
                logit_val = float(logit_tensor.detach().cpu())
                prob_val = float(torch.sigmoid(logit_tensor).detach().cpu())
        unique_edges.append(
            {
                "index": idx,
                "type": edge_type,
                "out_id": out_id,
                "in_id": in_id,
                "logit": logit_val,
                "prob": prob_val,
                "reward": record.get("reward", 0.0),
                "selected": record.get("selected", False),
            }
        )

    sorted_edges = sorted(
        unique_edges,
        key=lambda d: (
            d.get("logit", float("-inf")) if d.get("logit") is not None else float("-inf"),
            d.get("prob", float("-inf")) if d.get("prob") is not None else float("-inf"),
        ),
        reverse=True,
    )

    header = (
        f"{'index':<8}{'type':<10}{'in_node_role':<22}{'out_node_role':<22}"
        f"{'logit':>12}{'prob':>12}{'selected':>10}{'reward':>12}{'raw_delta':>12}"
    )
    print(header)
    print("=" * len(header))
    for record in sorted_edges:
        in_role = _get_role(actor, record["in_id"])
        out_role = _get_role(actor, record["out_id"])
        logit_val = record.get("logit")
        prob_val = record.get("prob")

        key = (record["out_id"], record["in_id"], record["type"])
        edge_info = combined_edges.get(key, {})
        selected_info = edge_info.get("selected", False)
        raw_delta_val = edge_info.get("raw_delta", 0.0)


        reward_val = edge_info.get("reward_clamped", record.get("reward", 0.0))
        if reward_val is None:
            reward_val = record.get("reward", 0.0)
        selected_str = "✓" if selected_info else "✗"
        logit_str = f"{logit_val:>12.6f}" if isinstance(logit_val, (int, float)) else f"{'N/A':>12}"
        prob_str = f"{prob_val:>12.6f}" if isinstance(prob_val, (int, float)) else f"{'N/A':>12}"
        reward_str = f"{reward_val:>12.4f}" if isinstance(reward_val, (int, float)) else f"{'N/A':>12}"
        raw_delta_str = f"{raw_delta_val:>12.6f}" if isinstance(raw_delta_val, (int, float)) else f"{'N/A':>12}"
        print(
            f"{record['index']:<8}{record['type']:<10}{in_role:<22}{out_role:<22}"
            f"{logit_str}{prob_str}{selected_str:>10}{reward_str}{raw_delta_str}"
        )


def _print_edge_utilities(edge_details: List[Dict[str, Any]], actor: Actor) -> None:
    print("\nEdge utility summary")
    if not edge_details:
        print("No edge records.")
        return

    utility_map: Dict[Tuple[str, str, str], float] = defaultdict(float)
    for detail in edge_details:
        key = (
            detail.get("out_node_id", ""),
            detail.get("in_node_id", ""),
            detail.get("type", ""),
        )

        reward_val = float(detail.get("reward_clamped", detail.get("reward", 0.0)) or 0.0)
        utility_map[key] += reward_val

    sorted_items = sorted(utility_map.items(), key=lambda kv: kv[1], reverse=True)
    header = f"{'out_node_role':<22}{'in_node_role':<22}{'type':<10}{'utility':>12}"
    print(header)
    print("=" * len(header))
    for (out_id, in_id, edge_type), utility in sorted_items:
        out_role = _get_role(actor, out_id)
        in_role = _get_role(actor, in_id)
        print(f"{out_role:<22}{in_role:<22}{edge_type:<10}{utility:>12.4f}")


def _print_critic_round_info(
        edge_details: List[Dict[str, Any]],
        actor: Actor,
        real_round_totals: Optional[Dict[int, float]] = None,
        critic_round_totals: Optional[Dict[int, float]] = None,
) -> None:
    print("\nCritic:")
    if not edge_details:
        print("No edge records.")
        return

    round_to_edges: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for detail in edge_details:
        round_to_edges[int(detail.get("round", 0))].append(detail)

    for round_idx in sorted(round_to_edges.keys()):
        print(f"round{round_idx}:")
        header = (
            f"{'in_node_role':<22}{'out_node_role':<22}"
            f"{'real(Δ/Δall)':>20}{'Critic(Δ/critic(Δ)_all)':>28}{'real_Δ':>12}{'critic_Δ':>12}"
        )
        print(header)
        print("=" * len(header))
        real_total = real_round_totals.get(round_idx) if real_round_totals else None
        critic_total = critic_round_totals.get(round_idx) if critic_round_totals else None
        real_sum = 0.0 if real_total is not None and abs(real_total) > EPS else None
        critic_sum = 0.0 if critic_total is not None and abs(critic_total) > EPS else None
        for detail in round_to_edges[round_idx]:
            in_role = _get_role(actor, detail.get("in_node_id", ""))
            out_role = _get_role(actor, detail.get("out_node_id", ""))
            real_ratio = detail.get("real_ratio") if real_total is not None else None
            critic_ratio = detail.get("critic_ratio") if critic_total is not None else None
            real_delta = detail.get("delta", 0.0)
            critic_delta = detail.get("critic_pred", 0.0)
            real_str = f"{real_ratio:>20.4f}" if isinstance(real_ratio, (int, float)) else f"{'--':>20}"
            critic_str = f"{critic_ratio:>28.4f}" if isinstance(critic_ratio, (int, float)) else f"{'--':>28}"
            real_delta_str = f"{real_delta:>12.6f}" if isinstance(real_delta, (int, float)) else f"{'--':>12}"
            critic_delta_str = f"{critic_delta:>12.6f}" if isinstance(critic_delta, (int, float)) else f"{'--':>12}"
            print(f"{in_role:<22}{out_role:<22}{real_str}{critic_str}{real_delta_str}{critic_delta_str}")
            if real_sum is not None and isinstance(real_ratio, (int, float)):
                real_sum += real_ratio
            if critic_sum is not None and isinstance(critic_ratio, (int, float)):
                critic_sum += critic_ratio
        summary_parts = []
        if real_total is not None:
            summary_parts.append(f"real_total_Δ={real_total:.6f}")
        if critic_total is not None:
            summary_parts.append(f"critic_total_Δ={critic_total:.6f}")
        if real_sum is not None:
            summary_parts.append(f"real_sum={real_sum:.6f}")
        if critic_sum is not None:
            summary_parts.append(f"critic_sum={critic_sum:.6f}")
        if summary_parts:
            print("  -> " + " | ".join(summary_parts))


def _log_sample(
        label: str,
        record_name: str,
        is_passing: bool,
        pass_ratio: float,
        actor_loss: float,
        critic_loss: float,
        round_totals: Dict[int, float],
        edge_details: List[Dict[str, Any]],
        training_state: TrainingState,
        actor: Actor,
        critic_round_totals: Optional[Dict[int, float]] = None,
        verbose: bool = True,
) -> None:
    # Console policy: detailed sample logs only when verbose=True (training loops use verbose=False).
    if not verbose:
        return
    status = "✅" if is_passing else "❌"
    prompt_tokens = int(PromptTokens.instance().value)
    completion_tokens = int(CompletionTokens.instance().value)
    print(f"{status} PromptTokens={prompt_tokens} CompletionTokens={completion_tokens}")


def _print_question_progress(
        stage: str,
        idx: int,
        total: int,
        is_passing: bool,
        baseline_prompt: int,
        baseline_completion: int,
        training_state: Optional[TrainingState] = None,
) -> None:
    """Print running mean tokens **within this stage** (delta since stage baseline / samples done)."""
    status = "✅" if is_passing else "❌"
    cur_p = int(PromptTokens.instance().value)
    cur_c = int(CompletionTokens.instance().value)
    d_p = max(0, cur_p - int(baseline_prompt))
    d_c = max(0, cur_c - int(baseline_completion))
    n = max(1, int(idx))
    mean_p = d_p / n
    mean_c = d_c / n
    mean_t = (d_p + d_c) / n
    if training_state is not None:
        c_correct = int(training_state.cumulative_correct)
        c_total = int(training_state.cumulative_total)
        c_acc = float(training_state.accuracy)
        acc_part = f" cumulative_acc={c_acc:.2%} ({c_correct}/{c_total})"
    else:
        acc_part = ""
    print(
        f"[{stage} {idx}/{total}] {status} "
        f"mean_prompt={mean_p:.2f} mean_completion={mean_c:.2f} mean_total={mean_t:.2f}"
        f"{acc_part}"
    )


async def real_execution(
        record: Dict[str, Any],
        actor: Actor,
        critics: Critics,
        encoder: Encoder,
        dataset: HumanEvalDataset,
        edge_judge: EdgeJudge,
        actor_optimizer: optim.Optimizer,
        critic_optimizer: Optional[optim.Optimizer],
        spatial_edge_map: Dict[Tuple[str, str], int],
        temporal_edge_map: Dict[Tuple[str, str], int],
        num_rounds: int,
        sparsity_weight: float,
        training_state: TrainingState,
        lambda3: Optional[float] = None,
        verbose: bool = True,
        eval_only: bool = False,
        edge_sample_threshold: Optional[float] = None,
) -> Dict[str, Any]:

    import time
    sample_start_time = time.time()

    if lambda3 is not None:

        effective_lr = max(lambda3 * training_state.accuracy, MIN_VIRTUAL_LR)
        _set_optimizer_lr(actor_optimizer, effective_lr)
    else:

        _set_optimizer_lr(actor_optimizer, actor_optimizer.defaults.get("lr", 1e-4))

    task_input = dataset.record_to_input(record)
    task_text = task_input.get("task", "")

    answers, _, edge_records = await actor.arun(
        task_input,
        num_rounds=num_rounds,
        aggregate_mode="all connected",
        edge_sample_threshold=edge_sample_threshold,
    )

    final_answer = dataset.postprocess_answer(answers[-1] if answers else "")
    pass_ratio, is_passing, feedback, test_state, unit_tests = dataset.evaluate_candidate(
        final_answer,
        record,
    )
    feedback_summary = dataset.format_feedback_summary(pass_ratio, test_state, feedback)

    edge_details = _build_edge_details(actor, edge_records, final_answer)
    round_totals = _compute_real_deltas(
        edge_details=edge_details,
        edge_judge=edge_judge,
        encoder=encoder,
        task_text=task_text,
        feedback_summary=feedback_summary,
        pass_ratio=pass_ratio,
        unit_tests=unit_tests,
        test_state=test_state,
        total_rounds=num_rounds,
        verbose=verbose,
    )

    if not edge_details:
        training_state.update(is_passing)
        if verbose:
            _log_sample(
                label="Real",
                record_name=record.get("name", ""),
                is_passing=is_passing,
                pass_ratio=pass_ratio,
                actor_loss=0.0,
                critic_loss=0.0,
                round_totals=round_totals,
                edge_details=edge_details,
                training_state=training_state,
                actor=actor,
                verbose=verbose,
            )
        return {
            "is_passing": is_passing,
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "num_selected_edges": 0,
            "pass_ratio": pass_ratio,
        }

    if eval_only:
        training_state.update(is_passing)
        sample_end_time = time.time()
        if verbose:
            _log_sample(
                label="Real-eval",
                record_name=record.get("name", ""),
                is_passing=is_passing,
                pass_ratio=pass_ratio,
                actor_loss=0.0,
                critic_loss=0.0,
                round_totals=round_totals,
                edge_details=edge_details,
                training_state=training_state,
                actor=actor,
                verbose=verbose,
            )
        return {
            "is_passing": is_passing,
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "num_selected_edges": len(edge_details),
            "pass_ratio": pass_ratio,
            "execution_time": sample_end_time - sample_start_time,
            "edge_details": [dict(d) for d in edge_details],
            "round_totals": dict(round_totals),
        }


    selected_keys = {_edge_key(d["out_node_id"], d["in_node_id"], d["type"]) for d in edge_details}

    critic_edge_details: List[Dict[str, Any]] = list(edge_details)

    def _make_extra_detail(out_id: str, in_id: str, edge_type: str) -> Dict[str, Any]:
        extra_delta = -0.1 if (random.random() < 0.7) else 0.0
        return {
            "out_node_id": out_id,
            "in_node_id": in_id,
            "type": edge_type,
            "round": 0,
            "delta": extra_delta,
            "selected": False,
        }

    for out_id, in_id in actor.potential_spatial_edges:
        key = _edge_key(out_id, in_id, "spatial")
        if key not in selected_keys:
            critic_edge_details.append(_make_extra_detail(out_id, in_id, "spatial"))
    for out_id, in_id in actor.potential_temporal_edges:
        key = _edge_key(out_id, in_id, "temporal")
        if key not in selected_keys:
            critic_edge_details.append(_make_extra_detail(out_id, in_id, "temporal"))

    extra_start_idx = len(edge_details)
    total_extra = len(critic_edge_details) - extra_start_idx
    if total_extra > 0:
        keep_ratio = 0.3
        max_extra = min(total_extra, max(1, int(len(edge_details) * keep_ratio)))
        if total_extra > max_extra:
            extra_indices = list(range(extra_start_idx, len(critic_edge_details)))
            keep_extra = set(random.sample(extra_indices, max_extra))
            filtered: List[Dict[str, Any]] = []
            for idx, d in enumerate(critic_edge_details):
                if idx < extra_start_idx or idx in keep_extra:
                    filtered.append(d)
            critic_edge_details = filtered
    expanded_edge_details: List[Dict[str, Any]] = []
    orig_to_expanded: List[List[int]] = [[] for _ in range(len(edge_details))]
    for i, d in enumerate(critic_edge_details):
        idx = len(expanded_edge_details)
        expanded_edge_details.append(dict(d))
        if i < len(edge_details):
            orig_to_expanded[i].append(idx)
    round_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, d in enumerate(expanded_edge_details):
        round_to_indices[int(d.get("round", 0))].append(idx)
    per_round_loss_div_edges: List[float] = []
    for round_idx in sorted(round_to_indices.keys()):
        idxs = round_to_indices[round_idx]
        subset = [expanded_edge_details[i] for i in idxs]
        if not subset:
            continue
        in_d, in_h, q_l, out_d, out_h, tgt = _critic_build_edge_batch(
            subset, actor, task_text, device
        )
        pred = critics.run_batch_differentiated(
            in_node_description_list=in_d,
            in_node_history_list=in_h,
            query_list=q_l,
            out_node_description_list=out_d,
            out_node_history_list=out_h,
            use_locked=False,
        ).to(device)
        critic_loss_r = _critic_loss_from_batch(pred, tgt, subset, sparsity_weight=sparsity_weight)
        n_edges_r = max(1, len(subset))
        per_round_loss_div_edges.append(float(critic_loss_r.detach().cpu()) / float(n_edges_r))

        critic_optimizer.zero_grad()
        critic_loss_r.backward()
        torch.nn.utils.clip_grad_norm_(critics.epn.parameters(), max_norm=1.0)
        critic_optimizer.step()

    # Record critic loss for logging
    critic_loss_recorded = float(sum(per_round_loss_div_edges)) / max(1, len(per_round_loss_div_edges)) if per_round_loss_div_edges else 0.0

    with torch.no_grad():
        in_d, in_h, q_l, out_d, out_h, critic_targets = _critic_build_edge_batch(
            expanded_edge_details, actor, task_text, device
        )
        critic_predictions = critics.run_batch_differentiated(
            in_node_description_list=in_d,
            in_node_history_list=in_h,
            query_list=q_l,
            out_node_description_list=out_d,
            out_node_history_list=out_h,
            use_locked=False,
        ).to(device)

    critic_preds_expanded = critic_predictions.view(-1)
    for i, detail in enumerate(edge_details):
        idx_list = orig_to_expanded[i]
        if idx_list:
            vals = critic_preds_expanded[idx_list]
            agg_pred = float(vals.mean().detach().cpu())
        else:
            agg_pred = 0.0
        detail["critic_pred"] = agg_pred
        logit_tensor = _get_edge_logit(
            actor,
            detail["type"],
            detail["out_node_id"],
            detail["in_node_id"],
            spatial_edge_map,
            temporal_edge_map,
        )
        if logit_tensor is None:
            detail["logit"] = 0.0
            prob_tensor = torch.tensor(0.0, device=device)
        else:
            detail["logit"] = float(logit_tensor.detach().cpu())
            prob_tensor = torch.sigmoid(logit_tensor)
        detail["prob"] = float(prob_tensor.detach().cpu())

    critic_round_totals: Dict[int, float] = defaultdict(float)
    for detail in edge_details:
        round_idx = int(detail["round"])
        critic_round_totals[round_idx] += detail["critic_pred"]

    for detail in edge_details:
        round_idx = int(detail["round"])
        real_total = round_totals.get(round_idx, 0.0)
        critic_total = critic_round_totals.get(round_idx, 0.0)
        detail["real_ratio"] = detail.get("delta", 0.0) / (real_total + EPS) if abs(real_total) > EPS else 0.0
        detail["critic_ratio"] = (
            detail.get("critic_pred", 0.0) / (critic_total + EPS) if abs(critic_total) > EPS else 0.0
        )

    policy_loss, log_edges, affected_edge_keys = _policy_loss_from_rewards(
        actor=actor,
        edge_details=edge_details,
        selected_keys=selected_keys,
        is_correct=is_passing,
        spatial_edge_map=spatial_edge_map,
        temporal_edge_map=temporal_edge_map,
        num_rounds=num_rounds,
    )

    sparsity_term = sparsity_weight * (
            torch.abs(actor.spatial_logits).mean()
            + (torch.abs(actor.temporal_logits).mean() if actor.optimized_temporal else 0.0)
    )
    actor_loss = policy_loss + sparsity_term
    actor_loss_value = float(actor_loss.detach().cpu())

    actor_optimizer.zero_grad()
    if is_passing:
        actor_loss.backward()
        actor_optimizer.step()

    _refresh_edge_logits(
        actor=actor,
        edge_details=edge_details,
        spatial_edge_map=spatial_edge_map,
        temporal_edge_map=temporal_edge_map,
    )

    training_state.update(is_passing)

    if verbose:
        _log_sample(
            label="Real",
            record_name=record.get("name", ""),
            is_passing=is_passing,
            pass_ratio=pass_ratio,
            actor_loss=actor_loss_value,
            critic_loss=critic_loss_recorded,
            round_totals=round_totals,
            edge_details=edge_details,
            training_state=training_state,
            critic_round_totals=critic_round_totals,
            actor=actor,
            verbose=verbose,
        )

    sample_end_time = time.time()
    sample_execution_time = sample_end_time - sample_start_time

    return {
        "is_passing": is_passing,
        "actor_loss": actor_loss_value,
        "critic_loss": critic_loss_recorded,
        "num_selected_edges": len(edge_details),
        "pass_ratio": pass_ratio,
        "execution_time": sample_execution_time,
        "edge_logs": log_edges,
        "critic_pred_values": critic_predictions.detach().view(-1).cpu().tolist(),
        "critic_target_values": critic_targets.detach().view(-1).cpu().tolist(),
    }

async def virtual_execution(
        record: Dict[str, Any],
        actor: Actor,
        critics: Critics,
        dataset: HumanEvalDataset,
        actor_optimizer: optim.Optimizer,
        spatial_edge_map: Dict[Tuple[str, str], int],
        temporal_edge_map: Dict[Tuple[str, str], int],
        num_rounds: int,
        sparsity_weight: float,
        training_state: TrainingState,
        lambda2: float,
        override_lr: Optional[float] = None,
        verbose: bool = True,
) -> Dict[str, Any]:

    import time
    virtual_start_time = time.time()

    task_input = dataset.record_to_input(record)
    task_text = task_input.get("task", "")

    spatial_probs = torch.sigmoid(actor.spatial_logits.detach())
    temporal_probs = (
        torch.sigmoid(actor.temporal_logits.detach())
        if actor.optimized_temporal
        else torch.zeros_like(actor.temporal_logits)
    )
    sampled_edges = _sample_roundwise_selected_edges(
        actor=actor,
        spatial_probs=spatial_probs,
        temporal_probs=temporal_probs,
        num_rounds=num_rounds,
        max_selected_edges=15,
        include_output=False,
        include_prompt=True,
    )

    if not sampled_edges:
        return {"actor_loss": 0.0, "num_selected_edges": 0}

    edge_details: List[Dict[str, Any]] = []
    for edge in sampled_edges:
        out_id = edge["out_node_id"]
        in_id = edge["in_node_id"]
        node_parts = actor.get_edge_node_info_parts(
            out_id,
            in_id,
            include_output=False,
            max_output_len=MAX_EDGE_OUTPUT_LEN,
            include_prompt=True,
            max_prompt_len=MAX_EDGE_PROMPT_LEN,
        )
        edge_details.append(
            {
                "out_node_id": out_id,
                "in_node_id": in_id,
                "type": edge["type"],
                "round": edge["round"],
                "node_parts": node_parts,
                "selected": True,
            }
        )


    in_node_description_list = []
    in_node_history_list = []
    query_list = []
    out_node_description_list = []
    out_node_history_list = []

    for d in edge_details:
        out_node_id = d.get("out_node_id", "")
        in_node_id = d.get("in_node_id", "")

        node_info = actor.get_edge_node_info_with_history(
            out_node_id=out_node_id,
            in_node_id=in_node_id,
            include_output=False,
            include_prompt=True,
            max_history_len=MAX_EDGE_HISTORY_LEN,
        )
        in_node_description_list.append(node_info["in_node"]["description"])
        in_node_history_list.append(node_info["in_node"]["history"])
        query_list.append(task_text)
        out_node_description_list.append(node_info["out_node"]["description"])
        out_node_history_list.append(node_info["out_node"]["history"])

    critic_predictions = critics.run_batch_differentiated(
        in_node_description_list=in_node_description_list,
        in_node_history_list=in_node_history_list,
        query_list=query_list,
        out_node_description_list=out_node_description_list,
        out_node_history_list=out_node_history_list,
        use_locked=False,
    ).detach().cpu().numpy()

    for idx, detail in enumerate(edge_details):
        detail["delta"] = float(critic_predictions[idx])
        detail["real_ratio"] = None

    round_totals = _compute_reward_from_deltas(edge_details, total_rounds=num_rounds)
    selected_keys = {_edge_key(d["out_node_id"], d["in_node_id"], d["type"]) for d in edge_details}

    if override_lr is not None:
        effective_lr = max(override_lr, MIN_VIRTUAL_LR)
    else:
        effective_lr = max(lambda2 * training_state.accuracy, MIN_VIRTUAL_LR)
    _set_optimizer_lr(actor_optimizer, effective_lr)

    virtual_penalty_scale = UNSELECTED_PENALTY * max(1, num_rounds)

    policy_loss, log_edges, _ = _policy_loss_from_rewards(
        actor=actor,
        edge_details=edge_details,
        selected_keys=selected_keys,
        is_correct=False,
        spatial_edge_map=spatial_edge_map,
        temporal_edge_map=temporal_edge_map,
        num_rounds=num_rounds,
        force_unselected_penalty=True,
        penalty_scale_override=virtual_penalty_scale,
        penalize_unselected_edges=False,
        penalize_all_potential_edges=False,
    )

    sparsity_term = sparsity_weight * (
            torch.abs(actor.spatial_logits).mean()
            + (torch.abs(actor.temporal_logits).mean() if actor.optimized_temporal else 0.0)
    )
    actor_loss = policy_loss + sparsity_term

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    _refresh_edge_logits(
        actor=actor,
        edge_details=edge_details,
        spatial_edge_map=spatial_edge_map,
        temporal_edge_map=temporal_edge_map,
    )

    for detail in edge_details:
        round_idx = int(detail["round"])
        critic_ratio = (
            detail["delta"] / (round_totals[round_idx] + EPS) if abs(round_totals[round_idx]) > EPS else 0.0
        )
        detail["critic_ratio"] = critic_ratio

    # keep console output minimal; suppress verbose virtual-execution details

    virtual_end_time = time.time()
    virtual_execution_time = virtual_end_time - virtual_start_time

    return {
        "actor_loss": float(actor_loss.detach().cpu()),
        "num_selected_edges": len(edge_details),
        "execution_time": virtual_execution_time,
    }

async def stage1_training(
        critics: Critics,
        actor: Actor,
        encoder: Encoder,
        dataset: HumanEvalDataset,
        records: List[Dict[str, Any]],
        actor_optimizer: optim.Optimizer,
        critic_optimizer: optim.Optimizer,
        edge_judge: EdgeJudge,
        num_rounds: int,
        sparsity_weight: float,
        training_state: TrainingState,
        show_timing: bool = True,
        show_progress: bool = True,
) -> Dict[str, List[float]]:
    stats = {"actor_loss": [], "critic_loss": [], "accuracy": []}
    spatial_edge_map = {(e[0], e[1]): idx for idx, e in enumerate(actor.potential_spatial_edges)}
    temporal_edge_map = {(e[0], e[1]): idx for idx, e in enumerate(actor.potential_temporal_edges)}

    stage1_baseline_prompt = int(PromptTokens.instance().value)
    stage1_baseline_completion = int(CompletionTokens.instance().value)

    total = len(records)
    for sample_idx, record in enumerate(records):
        result = await real_execution(
            record=record,
            actor=actor,
            critics=critics,
            encoder=encoder,
            dataset=dataset,
            edge_judge=edge_judge,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            spatial_edge_map=spatial_edge_map,
            temporal_edge_map=temporal_edge_map,
            num_rounds=num_rounds,
            sparsity_weight=sparsity_weight,
            training_state=training_state,
            verbose=False,
        )
        stats["actor_loss"].append(result["actor_loss"])
        stats["critic_loss"].append(result["critic_loss"])
        stats["accuracy"].append(1.0 if result["is_passing"] else 0.0)
        _print_question_progress(
            "Stage1",
            sample_idx + 1,
            total,
            bool(result["is_passing"]),
            stage1_baseline_prompt,
            stage1_baseline_completion,
            training_state=training_state,
        )

    return stats


async def stage2_training(
        critics: Critics,
        actor: Actor,
        encoder: Encoder,
        dataset: HumanEvalDataset,
        records: List[Dict[str, Any]],
        actor_optimizer: optim.Optimizer,
        critic_optimizer: optim.Optimizer,
        edge_judge: EdgeJudge,
        num_rounds: int,
        sparsity_weight: float,
        virtual_steps_per_sample: int,
        lambda2: float,
        training_state: TrainingState,
        show_timing: bool = True,
        show_progress: bool = True,
) -> Dict[str, List[float]]:
    stats = {
        "virtual_actor_loss": [],
        "real_actor_loss": [],
        "real_critic_loss": [],
        "accuracy": [],
        "train_pred_values": [],
        "train_target_values": [],
    }
    spatial_edge_map = {(e[0], e[1]): idx for idx, e in enumerate(actor.potential_spatial_edges)}
    temporal_edge_map = {(e[0], e[1]): idx for idx, e in enumerate(actor.potential_temporal_edges)}

    stage2_baseline_prompt = int(PromptTokens.instance().value)
    stage2_baseline_completion = int(CompletionTokens.instance().value)

    total = len(records)
    for sample_idx, record in enumerate(records):
        for _ in range(virtual_steps_per_sample):
            v_result = await virtual_execution(
                record=record,
                actor=actor,
                critics=critics,
                dataset=dataset,
                actor_optimizer=actor_optimizer,
                spatial_edge_map=spatial_edge_map,
                temporal_edge_map=temporal_edge_map,
                num_rounds=num_rounds,
                sparsity_weight=sparsity_weight,
                training_state=training_state,
                lambda2=lambda2,
                verbose=False,
            )
            stats["virtual_actor_loss"].append(v_result["actor_loss"])

        real_result = await real_execution(
            record=record,
            actor=actor,
            critics=critics,
            encoder=encoder,
            dataset=dataset,
            edge_judge=edge_judge,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            spatial_edge_map=spatial_edge_map,
            temporal_edge_map=temporal_edge_map,
            num_rounds=num_rounds,
            sparsity_weight=sparsity_weight,
            training_state=training_state,
            verbose=False,
        )
        stats["real_actor_loss"].append(real_result["actor_loss"])
        stats["real_critic_loss"].append(real_result["critic_loss"])
        stats["accuracy"].append(1.0 if real_result["is_passing"] else 0.0)
        stats["train_pred_values"].extend(real_result.get("critic_pred_values", []))
        stats["train_target_values"].extend(real_result.get("critic_target_values", []))
        _print_question_progress(
            "Stage2",
            sample_idx + 1,
            total,
            bool(real_result["is_passing"]),
            stage2_baseline_prompt,
            stage2_baseline_completion,
            training_state=training_state,
        )

    return stats


def _prune_bottom_edges(
        actor: Actor,
        prune_ratio: float,
        spatial_edge_map: Dict[Tuple[str, str], int],
        temporal_edge_map: Dict[Tuple[str, str], int],
) -> None:

    if prune_ratio <= 0.0 or prune_ratio >= 1.0:
        return

    try:

        if actor.optimized_spatial:

            available_indices = torch.where(actor.spatial_masks > 0)[0]
            if len(available_indices) == 0:
                return


            max_valid_index = len(actor.spatial_logits) - 1
            available_indices = available_indices[available_indices <= max_valid_index]
            if len(available_indices) == 0:
                return


            max_prune = len(available_indices) - 1
            if max_prune <= 0:
                return


            probs = torch.sigmoid(actor.spatial_logits[available_indices])
            sorted_indices = available_indices[torch.argsort(probs)]


            num_to_prune = min(max(1, int(len(sorted_indices) * prune_ratio)), max_prune)
            prune_indices = sorted_indices[:num_to_prune]


            with torch.no_grad():
                actor.spatial_masks[prune_indices] = 0

            remaining = len(available_indices) - num_to_prune
            _ = remaining


        if actor.optimized_temporal:
            available_indices = torch.where(actor.temporal_masks > 0)[0]
            if len(available_indices) == 0:
                return


            max_valid_index = len(actor.temporal_logits) - 1
            available_indices = available_indices[available_indices <= max_valid_index]
            if len(available_indices) == 0:
                return

            max_prune = len(available_indices) - 1
            if max_prune <= 0:
                return

            probs = torch.sigmoid(actor.temporal_logits[available_indices])
            sorted_indices = available_indices[torch.argsort(probs)]

            num_to_prune = min(max(1, int(len(sorted_indices) * prune_ratio)), max_prune)
            prune_indices = sorted_indices[:num_to_prune]

            with torch.no_grad():
                actor.temporal_masks[prune_indices] = 0

            remaining = len(available_indices) - num_to_prune
            _ = remaining
    except Exception as e:
        # keep console output minimal
        _ = e


async def stage3_training(
        critics: Critics,
        actor: Actor,
        encoder: Encoder,
        dataset: HumanEvalDataset,
        records: List[Dict[str, Any]],
        actor_optimizer: optim.Optimizer,
        critic_optimizer: Optional[optim.Optimizer],
        edge_judge: EdgeJudge,
        num_rounds: int,
        sparsity_weight: float,
        training_state: Optional[TrainingState] = None,
        lambda3: Optional[float] = None,
        stage3_virtual_steps: Optional[int] = None,
        stage3_prune_ratio: Optional[float] = None,
        lambda2: Optional[float] = None,
        stage3_spatial_masks_snapshot: Optional[torch.Tensor] = None,
        stage3_temporal_masks_snapshot: Optional[torch.Tensor] = None,
        stage3_spatial_logits_snapshot: Optional[torch.Tensor] = None,
        stage3_temporal_logits_snapshot: Optional[torch.Tensor] = None,
        show_timing: bool = True,
        show_progress: bool = True,
) -> Dict[str, List[float]]:
    if stage3_virtual_steps is None:
        stage3_virtual_steps = 0
    if stage3_prune_ratio is None:
        stage3_prune_ratio = 0.0
    # keep console output minimal (per-question progress only)

    stats = {
        "virtual_actor_loss": [],
        "real_actor_loss": [],
        "real_critic_loss": [],
        "accuracy": [],
        "stage3_mse": [],
        "stage3_critic_loss": [],
        "stage3_pred_values": [],
        "stage3_target_values": [],
    }
    spatial_edge_map = {(e[0], e[1]): idx for idx, e in enumerate(actor.potential_spatial_edges)}
    temporal_edge_map = {(e[0], e[1]): idx for idx, e in enumerate(actor.potential_temporal_edges)}
    stage_state = training_state or TrainingState()

    total = len(records)
    if total > 0:
        PromptTokens.instance().reset()
        CompletionTokens.instance().reset()
        Cost.instance().reset()
        ApiCalls.instance().reset()
        stage_state.reset_token_baseline()

    stage3_token_baseline_prompt = int(PromptTokens.instance().value)
    stage3_token_baseline_completion = int(CompletionTokens.instance().value)
    for sample_idx, record in enumerate(records):

        with torch.no_grad():
            if stage3_spatial_masks_snapshot is not None:
                actor.spatial_masks.copy_(stage3_spatial_masks_snapshot.to(actor.spatial_masks.device))
            if stage3_temporal_masks_snapshot is not None and actor.optimized_temporal:
                actor.temporal_masks.copy_(stage3_temporal_masks_snapshot.to(actor.temporal_masks.device))

            if stage3_spatial_logits_snapshot is not None:
                actor.spatial_logits.copy_(stage3_spatial_logits_snapshot.to(actor.spatial_logits.device))
            if stage3_temporal_logits_snapshot is not None:
                temporal_param = getattr(actor, "temporal_logits", None)
                if temporal_param is not None:
                    temporal_param.copy_(stage3_temporal_logits_snapshot.to(temporal_param.device))

        try:
            result = await real_execution(
                record=record,
                actor=actor,
                critics=critics,
                encoder=encoder,
                dataset=dataset,
                edge_judge=edge_judge,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                spatial_edge_map=spatial_edge_map,
                temporal_edge_map=temporal_edge_map,
                num_rounds=num_rounds,
                sparsity_weight=sparsity_weight,
                training_state=stage_state,
                lambda3=lambda3,
                verbose=False,
                eval_only=True,
            )
            stats["real_actor_loss"].append(result["actor_loss"])
            stats["real_critic_loss"].append(result["critic_loss"])
            stats["accuracy"].append(1.0 if result["is_passing"] else 0.0)

            # --- Per-sample MSE on Stage3 (test/val) ---
            # Use edge-level lists returned by real_execution.
            y_true = np.asarray([], dtype=np.float32)
            y_pred = np.asarray([], dtype=np.float32)
            try:
                y_true = np.asarray(result.get("critic_target_values", []), dtype=np.float32)
                y_pred = np.asarray(result.get("critic_pred_values", []), dtype=np.float32)
                if y_true.size > 0 and y_true.shape == y_pred.shape:
                    mse_val = float(np.mean((y_pred - y_true) ** 2))
                else:
                    mse_val = 0.0
            except Exception:
                mse_val = 0.0
                y_true = np.asarray([], dtype=np.float32)
                y_pred = np.asarray([], dtype=np.float32)
            stats["stage3_mse"].append(mse_val)
            cl_val = float(result.get("critic_loss", 0.0))
            stats["stage3_critic_loss"].append(cl_val)
            if y_true.size > 0 and y_true.shape == y_pred.shape:
                stats["stage3_pred_values"].extend(y_pred.tolist())
                stats["stage3_target_values"].extend(y_true.tolist())

            _print_question_progress(
                "Stage3",
                sample_idx + 1,
                total,
                bool(result["is_passing"]),
                stage3_token_baseline_prompt,
                stage3_token_baseline_completion,
                training_state=stage_state,
            )
        except Exception as e:
            stage_state.update(False)
            _print_question_progress(
                "Stage3",
                sample_idx + 1,
                total,
                False,
                stage3_token_baseline_prompt,
                stage3_token_baseline_completion,
                training_state=stage_state,
            )
            _ = e

            stats["real_actor_loss"].append(0.0)
            stats["real_critic_loss"].append(0.0)
            stats["accuracy"].append(0.0)
            stats["stage3_mse"].append(0.0)
            stats["stage3_critic_loss"].append(0.0)

        # suppress periodic progress logs

    try:
        # Keep stats computation but suppress printing.
        if stats.get("stage3_mse"):
            mse_vals = stats["stage3_mse"]
            n_le, n_tot = _stage3_mse_low_count(mse_vals)
            stats["stage3_mse_le_threshold"] = float(STAGE3_MSE_LOW_THRESHOLD)
            stats["stage3_mse_count_le_threshold"] = n_le
            stats["stage3_mse_total"] = n_tot
            if n_tot > 0:
                stats["stage3_mse_fraction_le_threshold"] = n_le / float(n_tot)
        s3_y_t = np.asarray(stats.get("stage3_target_values", []), dtype=np.float32)
        s3_y_p = np.asarray(stats.get("stage3_pred_values", []), dtype=np.float32)
        m3 = _epn_edge_correlation_metrics(s3_y_t, s3_y_p)
        stats["stage3_pearson_r"] = float(m3["pearson_r"])
        stats["stage3_spearman_r"] = float(m3["spearman_r"])
        stats["stage3_spearman_pvalue"] = float(m3["spearman_pvalue"])
        stats["stage3_r2"] = float(m3["r2"])
        stats["stage3_mse_pooled"] = float(m3["mse"])
        stats["stage3_n_edges_pooled"] = int(m3["n"])

        stage3_prompt_tokens = int(PromptTokens.instance().value) - stage3_token_baseline_prompt
        stage3_completion_tokens = int(CompletionTokens.instance().value) - stage3_token_baseline_completion
        stage3_total_tokens = stage3_prompt_tokens + stage3_completion_tokens

        acc_list = stats.get("accuracy", [])
        stage3_correct = sum(1 for x in acc_list if float(x) >= 0.5)
        stage3_total = len(acc_list)

        stats["stage3_prompt_tokens"] = stage3_prompt_tokens
        stats["stage3_completion_tokens"] = stage3_completion_tokens
        stats["stage3_total_tokens"] = stage3_total_tokens
        stats["stage3_correct"] = stage3_correct
        stats["stage3_total"] = stage3_total
    except Exception as e:
        _ = e

        stats["stage3_prompt_tokens"] = 0
        stats["stage3_completion_tokens"] = 0
        stats["stage3_total_tokens"] = 0
        stats["stage3_correct"] = 0
        stats["stage3_total"] = 0

    return stats


def _close_blueshirt_http_after(coro_fn):
    """在 train_all 正常返回或抛错后关闭 BlueShirt 全局 aiohttp 会话，避免 Unclosed client session / SSL transport 噪声。"""

    @functools.wraps(coro_fn)
    async def _wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await coro_fn(*args, **kwargs)
        finally:
            try:
                from AgentTailor.llm.blueshirt_chat import close_blueshirt_session

                await close_blueshirt_session()
            except Exception:
                pass

    return _wrapper


@_close_blueshirt_http_after
async def train_all(
        agent_names: Optional[List[str]] = None,
        llm_name: str = "DeepSeekChat",
        decision_method: str = "FinalRefer",
        optimized_spatial: bool = True,
        optimized_temporal: bool = True,
        domain: str = "humaneval",
        num_rounds: int = 3,
        lr_actor: Optional[float] = None,
        lr_critic: Optional[float] = None,
        sparsity_weight: Optional[float] = None,
        stage1_sample_count: Optional[int] = None,
        stage2_sample_count: Optional[int] = None,
        stage2_virtual_steps: Optional[int] = None,
        lambda2: Optional[float] = None,
        lambda3: Optional[float] = None,
        edge_judge: Optional[EdgeJudge] = None,
        stage3_virtual_steps: Optional[int] = None,
        stage3_prune_ratio: Optional[float] = None,
        node_kwargs: Optional[List[Dict[str, Any]]] = None,
        show_timing: bool = True,
        show_progress: bool = True,
        epn_dims: Optional[List[int]] = None,
        lock_threshold: Optional[float] = None,
        temperature: Optional[float] = None,
        epn_dropout: Optional[float] = None,
        critic_weight_decay: Optional[float] = None,
        max_train_split_samples: Optional[int] = None,
        max_test_split_samples: Optional[int] = None,
        gsm8k_shuffle_seed: Optional[int] = None,
) -> Dict[str, Any]:
    # All training hyperparameters must be passed from experiments/train4*.py::_build_config().
    # They are never read from environment variables.
    _required_floats: Tuple[Tuple[str, Optional[float]], ...] = (
        ("lr_actor", lr_actor),
        ("lr_critic", lr_critic),
        ("sparsity_weight", sparsity_weight),
        ("lambda2", lambda2),
        ("stage3_prune_ratio", stage3_prune_ratio),
        ("lock_threshold", lock_threshold),
        ("temperature", temperature),
        ("epn_dropout", epn_dropout),
        ("critic_weight_decay", critic_weight_decay),
    )
    _required_ints: Tuple[Tuple[str, Optional[int]], ...] = (
        ("stage1_sample_count", stage1_sample_count),
        ("stage2_sample_count", stage2_sample_count),
        ("stage2_virtual_steps", stage2_virtual_steps),
        ("stage3_virtual_steps", stage3_virtual_steps),
    )
    _missing = [n for n, v in _required_floats + _required_ints if v is None]
    if _missing:
        raise ValueError(
            "train_all: the following hyperparameters are None: "
            + ", ".join(_missing)
            + ". Set them explicitly in experiments/train4<dataset>.py inside _build_config()."
        )

    lr_actor = float(lr_actor)
    lr_critic = float(lr_critic)
    sparsity_weight = float(sparsity_weight)
    stage1_sample_count = int(stage1_sample_count)
    stage2_sample_count = int(stage2_sample_count)
    stage2_virtual_steps = int(stage2_virtual_steps)
    lambda2 = float(lambda2)
    stage3_virtual_steps = int(stage3_virtual_steps)
    stage3_prune_ratio = float(stage3_prune_ratio)
    lock_threshold = float(lock_threshold)
    temperature = float(temperature)
    epn_dropout = float(epn_dropout)
    critic_weight_decay = float(critic_weight_decay)

    SEED = 888
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    # ===== Pipeline-level statistics (API calls / wall-clock / cost) =====
    import time
    start_wall_clock = time.time()
    ApiCalls.instance().reset()
    Cost.instance().reset()
    PromptTokens.instance().reset()
    CompletionTokens.instance().reset()

    if not llm_name or not str(llm_name).strip():
        raise ValueError("train_all: llm_name is empty; set it in train4 _build_config().")
    if agent_names is None:
        agent_names = ["AnalyzeAgent"] * 5

    actor = Actor(
        domain=domain,
        llm_name=llm_name,
        agent_names=agent_names,
        decision_method=decision_method,
        optimized_spatial=optimized_spatial,
        optimized_temporal=optimized_temporal,
        initial_spatial_probability=0.5,
        node_kwargs=node_kwargs,
    )
    if epn_dims is None:
        epn_dims = [epn_concat_input_dim()] + epn_head_hidden_sizes()
    
    critics = Critics(
        epn_dims=epn_dims,
        lock_threshold=lock_threshold,  # 降低阈值，更难锁定，需要更充分训练
        temperature=temperature,  # 使用参数传入的temperature，确保训练和评估一致
        dropout=epn_dropout,  # 添加dropout防止过拟合
    )
    encoder = Encoder()

    def _max_samples_kw(n: Optional[int]) -> Dict[str, Any]:
        return {} if n is None else {"max_samples": n}

    def _gsm8k_seed_kw() -> Dict[str, Any]:
        return {} if gsm8k_shuffle_seed is None else {"seed": int(gsm8k_shuffle_seed)}

    if domain == "aqua":
        from dataset.aqua_dataset import AQuADataset
        # val -> dev.jsonl, test -> test.jsonl
        train_dataset = AQuADataset(split="val", **_max_samples_kw(max_train_split_samples))
        test_dataset = AQuADataset(split="test", **_max_samples_kw(max_test_split_samples))
        test_max_samples = 129
    elif domain == "mmlu":
        from experiments.train4mmlu import MMLUDataset
        train_dataset = MMLUDataset(split="dev", **_max_samples_kw(max_train_split_samples))
        test_dataset = MMLUDataset(split="val", **_max_samples_kw(max_test_split_samples))
        test_max_samples = 153
    elif domain == "svamp":
        from dataset.svamp_dataset import SvampDataset
        train_dataset = SvampDataset(split="train", **_max_samples_kw(max_train_split_samples))
        test_dataset = SvampDataset(split="test", **_max_samples_kw(max_test_split_samples))
        test_max_samples = 121
    elif domain == "multiarith":
        from dataset.multiarith_dataset import MultiArithDataset
        train_dataset = MultiArithDataset(split="train", **_max_samples_kw(max_train_split_samples))
        test_dataset = MultiArithDataset(split="test", **_max_samples_kw(max_test_split_samples))
        test_max_samples = 121
    elif domain == "gsm8k":
        from experiments.train4gms8k import GSM8KDataset
        _gkw = _gsm8k_seed_kw()
        train_dataset = GSM8KDataset(split="train", **_max_samples_kw(max_train_split_samples), **_gkw)
        test_dataset = GSM8KDataset(split="test", **_max_samples_kw(max_test_split_samples), **_gkw)
        test_max_samples = 157
    else:
        # Default (e.g., HumanEval): strict split separation to avoid leakage.
        train_dataset = HumanEvalDataset(split="train")
        test_dataset = HumanEvalDataset(split="test")
        test_max_samples = 121
        if max_train_split_samples is not None:
            train_dataset.records = train_dataset.records[: int(max_train_split_samples)]
        if max_test_split_samples is not None:
            test_dataset.records = test_dataset.records[: int(max_test_split_samples)]
    if edge_judge is None:
        if domain == "mmlu":
            edge_judge = TextQEdgeJudge()
        elif domain in {"aqua", "svamp", "multiarith", "gsm8k"}:
            edge_judge = TextQEdgeJudge()
        else:
            edge_judge = Train4SoftJudge()

    train_total_records = len(train_dataset.records)
    test_total_records = len(test_dataset.records)

    if train_total_records == 0:
        raise ValueError("HumanEval training split is empty; cannot start training.")

    if stage1_sample_count + stage2_sample_count > train_total_records:
        raise ValueError(
            f"Stage1 ({stage1_sample_count}) + Stage2 ({stage2_sample_count}) samples "
            f"({stage1_sample_count + stage2_sample_count}) exceed available train split size ({train_total_records})."
        )

    stage1_records = train_dataset.records[:stage1_sample_count]
    stage2_records = train_dataset.records[stage1_sample_count: stage1_sample_count + stage2_sample_count]

    stage3_records = test_dataset.records[: min(test_max_samples, test_total_records)]

    stage1_count = len(stage1_records)
    stage2_count = len(stage2_records)
    stage3_count = len(stage3_records)

    dataset_name = domain.upper() if domain else "HumanEval"
    if stage3_virtual_steps == 0:
        print(
            f"{dataset_name} split: Stage1={stage1_count} | Stage2={stage2_count} | Stage3=SKIPPED "
            f"(train={stage1_sample_count + stage2_sample_count}/{train_total_records}, "
            f"test={stage3_count}/{test_total_records})"
        )
    else:
        print(
            f"{dataset_name} split: Stage1={stage1_count} | Stage2={stage2_count} | Stage3={stage3_count} "
            f"(train={stage1_sample_count + stage2_sample_count}/{train_total_records}, "
            f"test={stage3_count}/{test_total_records})"
        )

    actor_params = [actor.spatial_logits]
    actor_temporal_param = getattr(actor, "temporal_logits", None)
    if actor_temporal_param is not None:
        actor_params.append(actor_temporal_param)
    actor_optimizer = optim.Adam(actor_params, lr=lr_actor)
    # 添加weight_decay防止过拟合
    critic_optimizer = optim.Adam(critics.epn.parameters(), lr=lr_critic, weight_decay=critic_weight_decay)

    training_state = TrainingState()
    training_state.reset_token_baseline()

    stage1_stats = await stage1_training(
        critics=critics,
        actor=actor,
        encoder=encoder,
        dataset=train_dataset,
        records=stage1_records,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        edge_judge=edge_judge,
        num_rounds=num_rounds,
        sparsity_weight=sparsity_weight,
        training_state=training_state,
        show_timing=show_timing,
        show_progress=show_progress,
    )

    # Suppress stage summaries to keep output minimal.
    stage2_stats = await stage2_training(
        critics=critics,
        actor=actor,
        encoder=encoder,
        dataset=train_dataset,
        records=stage2_records,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        edge_judge=edge_judge,
        num_rounds=num_rounds,
        sparsity_weight=sparsity_weight,
        virtual_steps_per_sample=stage2_virtual_steps,
        lambda2=lambda2,
        training_state=training_state,
        show_timing=show_timing,
        show_progress=show_progress,
    )

    if stage2_count > 0:
        print("\n  Stage2 edge-weight snapshot (printed once):")
        _print_actor_edge_info([], actor)

    # Stage2 结束后即锁定 CEPN（冻结快照 + 不再训练主 EPN）；后续虚拟步与 Stage3 均使用锁定副本。
    if not critics.is_locked:
        critics.lock_critic()

    spatial_snapshot = actor.spatial_logits.detach().clone().cpu()
    temporal_snapshot = (
        actor_temporal_param.detach().clone().cpu() if actor_temporal_param is not None else None
    )

    # Fully skip Stage 3 when stage3_virtual_steps=0.
    if stage3_virtual_steps == 0:
        print("\n" + "=" * 80)
        print("Stage 3: Skipped")
        print("=" * 80)
        stage3_actor = None
        stage3_critics = None
        stage3_stats = {
            "virtual_actor_loss": [],
            "real_actor_loss": [],
            "real_critic_loss": [],
            "accuracy": [],
            "stage3_mse": [],
            "stage3_critic_loss": [],
            "stage3_prompt_tokens": 0,
            "stage3_completion_tokens": 0,
            "stage3_total_tokens": 0,
            "stage3_correct": 0,
            "stage3_total": 0,
        }
    else:
        # 运行完整的 Stage 3
        stage3_actor = copy.deepcopy(actor)
        with torch.no_grad():
            stage3_actor.spatial_logits.copy_(spatial_snapshot.to(stage3_actor.spatial_logits.device))
            stage3_temporal_param = getattr(stage3_actor, "temporal_logits", None)
            if stage3_temporal_param is not None and temporal_snapshot is not None:
                stage3_temporal_param.copy_(temporal_snapshot.to(stage3_temporal_param.device))

        stage3_spatial_masks_snapshot = stage3_actor.spatial_masks.detach().clone()
        stage3_temporal_masks_snapshot = (
            stage3_actor.temporal_masks.detach().clone() if stage3_actor.optimized_temporal else None
        )

        stage3_spatial_logits_snapshot = stage3_actor.spatial_logits.detach().clone()
        stage3_temporal_logits_snapshot = (
            stage3_temporal_param.detach().clone() if stage3_temporal_param is not None else None
        )
        stage3_critics = copy.deepcopy(critics)

        stage3_actor_params = [stage3_actor.spatial_logits]
        if stage3_temporal_param is not None:
            stage3_actor_params.append(stage3_temporal_param)
        stage3_actor_optimizer = optim.Adam(stage3_actor_params, lr=lr_actor)

        stage3_stats = await stage3_training(
            critics=stage3_critics,
            actor=stage3_actor,
            encoder=encoder,
            dataset=test_dataset,
            records=stage3_records,
            actor_optimizer=stage3_actor_optimizer,
            critic_optimizer=None,
            edge_judge=edge_judge,
            num_rounds=num_rounds,
            sparsity_weight=sparsity_weight,
            training_state=training_state,
            lambda3=lambda3,
            stage3_virtual_steps=stage3_virtual_steps,
            stage3_prune_ratio=stage3_prune_ratio,
            lambda2=lambda2,
            stage3_spatial_masks_snapshot=stage3_spatial_masks_snapshot,
            stage3_temporal_masks_snapshot=stage3_temporal_masks_snapshot,
            stage3_spatial_logits_snapshot=stage3_spatial_logits_snapshot,
            stage3_temporal_logits_snapshot=stage3_temporal_logits_snapshot,
            show_timing=show_timing,
            show_progress=show_progress,
        )

    prompt_tokens, completion_tokens, total_tokens = training_state.get_token_stats()
    return {
        "stage1": stage1_stats,
        "stage2": stage2_stats,
        "stage3": stage3_stats,
        "actor": actor,
        "critics": critics,
        "stage3_actor": stage3_actor,
        "stage3_critics": stage3_critics,
        "cumulative_accuracy": training_state.accuracy,
        "cumulative_correct": training_state.cumulative_correct,
        "cumulative_total": training_state.cumulative_total,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "wall_clock_seconds": time.time() - start_wall_clock,
        "api_calls": int(ApiCalls.instance().value),
        "estimated_cost_usd": float(Cost.instance().value),
    }
