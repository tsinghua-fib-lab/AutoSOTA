# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import numpy as np
import torch
from collections import defaultdict

import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(config):
    if config.critic.kl_ctrl.type == 'fixed':
        kl_ctrl = FixedKLController(kl_coef=config.critic.kl_ctrl.kl_coef)
    elif config.critic.kl_ctrl.type == 'adaptive':
        assert config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
        kl_ctrl = AdaptiveKLController(init_kl_coef=config.critic.kl_ctrl.kl_coef,
                                       target_kl=config.critic.kl_ctrl.target_kl,
                                       horizon=config.critic.kl_ctrl.horizon)
    else:
        raise ValueError('Unknown kl_ctrl type')

    return kl_ctrl


def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_drgrpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                     eos_mask: torch.Tensor,
                                     index,
                                     advantage_scale: float = 2.0):
    """Group-center outcome rewards without the difficulty-dependent std.

    Dr.GRPO identifies the per-prompt standard deviation in GRPO as a source
    of question-level difficulty bias.  ``advantage_scale`` is a constant,
    prompt-independent calibration; 2.0 preserves the familiar +/-1 scale for
    a balanced binary-reward group while retaining the de-biasing property.
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)
    id_to_rows = defaultdict(list)
    for row in range(scores.shape[0]):
        id_to_rows[index[row]].append(row)

    centered = torch.empty_like(scores)
    with torch.no_grad():
        for rows in id_to_rows.values():
            row_tensor = torch.as_tensor(rows, device=scores.device)
            group_scores = scores.index_select(0, row_tensor)
            centered.index_copy_(
                0,
                row_tensor,
                (group_scores - group_scores.mean()) * float(advantage_scale),
            )
        advantages = centered.unsqueeze(-1).expand(-1, response_length) * eos_mask
    return advantages, advantages


def build_protocol_projection(response_ids: torch.Tensor,
                              eos_mask: torch.Tensor,
                              format_valid: torch.Tensor,
                              protocol_token_sequences,
                              projection: str = 'local',
                              invalid_tail_tokens: int = 16,
                              protocol_mass: float = 16.0):
    """Project a sequence-level format contrast onto protocol-relevant tokens.

    For a well-formed response, the local projection selects exact protocol-tag
    token sequences and the final valid token.  For a malformed response, it
    selects only the final ``invalid_tail_tokens`` positions, where a missing or
    broken closing contract can still be corrected.  Every non-empty projection
    is L1-normalized to ``protocol_mass`` so format credit is independent of
    reasoning length and of tokenizer-specific tag length.

    ``projection='global'`` is an ablation with the same mass but spread over
    every valid response token.  It isolates locality from reward reweighting.
    """
    if projection not in ('local', 'global'):
        raise ValueError("pda.projection must be 'local' or 'global'")
    if invalid_tail_tokens <= 0:
        raise ValueError('pda.invalid_tail_tokens must be positive')
    if protocol_mass <= 0.0:
        raise ValueError('pda.protocol_mass must be positive')
    if response_ids.shape != eos_mask.shape:
        raise ValueError('response_ids and eos_mask must have the same shape')
    if format_valid.shape[0] != response_ids.shape[0]:
        raise ValueError('format_valid must have one value per response')

    projection_weights = torch.zeros_like(eos_mask, dtype=torch.float32)
    selected_mask = torch.zeros_like(eos_mask, dtype=torch.bool)
    valid_mask = eos_mask.to(dtype=torch.bool)
    sequences = [
        [int(token_id) for token_id in sequence]
        for sequence in protocol_token_sequences
        if len(sequence) > 0
    ]

    with torch.no_grad():
        for row in range(response_ids.shape[0]):
            valid_positions = torch.nonzero(valid_mask[row], as_tuple=False).flatten()
            if valid_positions.numel() == 0:
                continue

            if projection == 'global':
                selected = valid_positions
            elif not bool(format_valid[row].item()):
                selected = valid_positions[-int(invalid_tail_tokens):]
            else:
                compact_ids = response_ids[row].index_select(0, valid_positions).tolist()
                compact_selected = torch.zeros(
                    len(compact_ids), device=response_ids.device, dtype=torch.bool
                )
                for sequence in sequences:
                    width = len(sequence)
                    for start in range(0, len(compact_ids) - width + 1):
                        if compact_ids[start:start + width] == sequence:
                            compact_selected[start:start + width] = True
                # Include EOS/the final generated token. If tokenization or a
                # custom formatter does not expose exact tags, use the same
                # conservative tail fallback as a malformed response.
                compact_selected[-1] = True
                if int(compact_selected.sum().item()) == 1:
                    compact_selected[-int(invalid_tail_tokens):] = True
                selected = valid_positions[compact_selected]

            selected_mask[row, selected] = True
            projection_weights[row, selected] = (
                float(protocol_mass) / float(selected.numel())
            )

    metrics = {
        'pda/protocol_token_fraction': float(
            selected_mask.sum().to(torch.float32).div(
                valid_mask.sum().clamp_min(1).to(torch.float32)
            ).detach().cpu().item()
        ),
        'pda/protocol_projection_mass_mean': float(
            projection_weights.sum(dim=-1).mean().detach().cpu().item()
        ),
        'pda/invalid_response_fraction': float(
            (~format_valid.to(torch.bool)).to(torch.float32).mean().detach().cpu().item()
        ),
    }
    return projection_weights, selected_mask, metrics


def compute_pda_drgrpo_advantage(token_level_scores: torch.Tensor,
                                 eos_mask: torch.Tensor,
                                 index,
                                 response_ids: torch.Tensor,
                                 protocol_token_sequences,
                                 correctness_score: float = 1.0,
                                 format_score: float = 0.1,
                                 correctness_scale: float = 1.8,
                                 format_scale: float = 0.8,
                                 projection: str = 'local',
                                 invalid_tail_tokens: int = 16,
                                 protocol_mass: float = 16.0,
                                 view_labels=None,
                                 view_contrast_scale: float = 0.0,
                                 protocol_guard_coef: float = 0.0,
                                 protocol_guard_target: float = 0.98,
                                 protocol_guard_mask: torch.Tensor = None,
                                 protocol_guard_scope: str = 'all',
                                 protocol_guard_valid_credit: float = 1.0,
                                 correct_length_coef: float = 0.0,
                                 response_lengths: torch.Tensor = None,
                                 epsilon: float = 1e-6):
    """Protocol-Decoupled Advantage (PDA) on top of Dr.GRPO.

    The scalar verifier reward is decomposed into binary semantic correctness
    and protocol validity.  Each signal is centered independently within its
    prompt group without standard-deviation normalization.  Semantic credit is
    broadcast over the reasoning trajectory, while protocol credit uses a
    fixed-mass token projection produced by :func:`build_protocol_projection`.

    On Countdown, the usual reward ``1.0 / 0.1 / 0.0`` can be written as
    ``0.9 * correctness + 0.1 * format``.  Therefore correctness_scale=1.8 and
    format_scale=0.2 exactly recover Dr.GRPO's sequence-level coefficients;
    other format scales alter only the explicitly separated protocol channel.

    With paired semantic views, ``view_contrast_scale`` adds a zero-mean
    within-view correctness contrast.  This keeps easy-view outcomes from
    setting the baseline for the harder view: a rare correct rollout is
    compared against siblings generated from the same semantic view.  The
    ordinary group residual remains as a fallback when a view is uniformly
    correct or wrong.

    ``protocol_guard_coef`` implements a deployment-oriented inequality
    constraint using training-only unassisted samples.  Its asymmetric barrier
    ``format_valid - protocol_guard_target`` gives malformed responses a strong
    local negative correction while giving valid responses only a small
    positive correction. The dual coefficient is managed by the trainer and is
    zero whenever the observed no-hint format rate satisfies the constraint.

    ``protocol_guard_scope='all_wrong'`` routes that auxiliary signal only to
    prompt groups whose semantic correctness residual is identically zero.
    This empirical null-space routing prevents the guard from competing with a
    mixed group's already-informative correctness ranking. Setting
    ``protocol_guard_valid_credit=0`` additionally turns the barrier into a
    malformed-tail-only penalty: formatted-but-incorrect trajectories never
    receive positive guard credit.

    ``correct_length_coef`` adds an orthogonal, zero-mean tie-breaker only
    among two or more verifier-correct responses from the same prompt. Shorter
    failures can never outrank correct responses, while redundant reasoning in
    already-correct trajectories receives a controlled negative residual.
    """
    if correctness_scale < 0.0 or format_scale < 0.0:
        raise ValueError('pda correctness and format scales must be non-negative')
    if view_contrast_scale < 0.0:
        raise ValueError('pda.view_contrast_scale must be non-negative')
    if view_contrast_scale > 0.0 and view_labels is None:
        raise ValueError('pda view contrast requires paired view labels')
    if view_labels is not None and len(view_labels) != token_level_scores.shape[0]:
        raise ValueError('view_labels must have one value per response')
    if protocol_guard_coef < 0.0:
        raise ValueError('pda protocol guard coefficient must be non-negative')
    if not 0.0 <= protocol_guard_target <= 1.0:
        raise ValueError('pda protocol guard target must be in [0, 1]')
    if protocol_guard_coef > 0.0 and protocol_guard_mask is None:
        raise ValueError('pda protocol guard requires an application mask')
    if protocol_guard_scope not in ('all', 'all_wrong'):
        raise ValueError("pda.protocol_guard_scope must be 'all' or 'all_wrong'")
    if not 0.0 <= float(protocol_guard_valid_credit) <= 1.0:
        raise ValueError('pda.protocol_guard_valid_credit must be in [0, 1]')
    if correct_length_coef < 0.0:
        raise ValueError('pda correct length coefficient must be non-negative')
    if correct_length_coef > 0.0 and response_lengths is None:
        raise ValueError('pda correct length frontier requires response lengths')
    if (response_lengths is not None
            and len(response_lengths) != token_level_scores.shape[0]):
        raise ValueError('response_lengths must have one value per response')
    if not 0.0 < float(format_score) < float(correctness_score):
        raise ValueError('pda requires 0 < format_score < correctness_score')

    sequence_scores = token_level_scores.sum(dim=-1)
    correctness = torch.isclose(
        sequence_scores,
        torch.as_tensor(
            correctness_score,
            device=sequence_scores.device,
            dtype=sequence_scores.dtype,
        ),
        atol=1e-6,
    )
    format_valid = sequence_scores >= float(format_score) - 1e-6
    correctness_values = correctness.to(dtype=sequence_scores.dtype)
    format_values = format_valid.to(dtype=sequence_scores.dtype)

    id_to_rows = defaultdict(list)
    for row in range(sequence_scores.shape[0]):
        id_to_rows[index[row]].append(row)

    correctness_residual = torch.empty_like(sequence_scores)
    format_residual = torch.empty_like(sequence_scores)
    view_residual = torch.zeros_like(sequence_scores)
    length_residual = torch.zeros_like(sequence_scores)
    guard_scope_rows = torch.zeros_like(format_valid, dtype=torch.bool)
    correctness_active_groups = 0
    format_active_groups = 0
    view_active_strata = 0
    view_strata = 0
    length_active_groups = 0
    length_active_responses = 0
    all_wrong_groups = 0
    with torch.no_grad():
        for rows in id_to_rows.values():
            row_tensor = torch.as_tensor(rows, device=sequence_scores.device)
            group_correctness = correctness_values.index_select(0, row_tensor)
            group_format = format_values.index_select(0, row_tensor)
            correctness_residual.index_copy_(
                0, row_tensor, group_correctness - group_correctness.mean()
            )
            format_residual.index_copy_(
                0, row_tensor, group_format - group_format.mean()
            )
            correctness_active_groups += int(
                float(group_correctness.min().item())
                != float(group_correctness.max().item())
            )
            group_all_wrong = float(group_correctness.max().item()) == 0.0
            all_wrong_groups += int(group_all_wrong)
            if protocol_guard_scope == 'all' or group_all_wrong:
                guard_scope_rows.index_fill_(0, row_tensor, True)
            format_active_groups += int(
                float(group_format.min().item()) != float(group_format.max().item())
            )
            if view_labels is not None and view_contrast_scale > 0.0:
                view_to_rows = defaultdict(list)
                for row in rows:
                    view_to_rows[view_labels[row]].append(row)
                for view_rows in view_to_rows.values():
                    view_row_tensor = torch.as_tensor(
                        view_rows, device=sequence_scores.device
                    )
                    view_correctness = correctness_values.index_select(
                        0, view_row_tensor
                    )
                    view_residual.index_copy_(
                        0,
                        view_row_tensor,
                        view_correctness - view_correctness.mean(),
                    )
                    view_strata += 1
                    view_active_strata += int(
                        float(view_correctness.min().item())
                        != float(view_correctness.max().item())
                    )

            if response_lengths is not None and correct_length_coef > 0.0:
                correct_rows = [
                    row for row in rows if bool(correctness[row].item())
                ]
                if len(correct_rows) >= 2:
                    correct_row_tensor = torch.as_tensor(
                        correct_rows, device=sequence_scores.device
                    )
                    correct_lengths = response_lengths.index_select(
                        0, correct_row_tensor
                    ).to(torch.float32)
                    std = torch.std(correct_lengths, unbiased=False)
                    if float(std.item()) > float(epsilon):
                        residual = (
                            correct_lengths.mean() - correct_lengths
                        ) / (std + float(epsilon))
                        length_residual.index_copy_(
                            0,
                            correct_row_tensor,
                            residual.to(length_residual.dtype),
                        )
                        length_active_groups += 1
                        length_active_responses += len(correct_rows)

        protocol_weights, _, projection_metrics = build_protocol_projection(
            response_ids=response_ids,
            eos_mask=eos_mask,
            format_valid=format_valid,
            protocol_token_sequences=protocol_token_sequences,
            projection=projection,
            invalid_tail_tokens=invalid_tail_tokens,
            protocol_mass=protocol_mass,
        )
        group_semantic_component = (
            float(correctness_scale)
            * correctness_residual.unsqueeze(-1)
            * eos_mask
        )
        view_semantic_component = (
            float(correctness_scale)
            * float(view_contrast_scale)
            * view_residual.unsqueeze(-1)
            * eos_mask
        )
        semantic_component = group_semantic_component + view_semantic_component
        protocol_component = (
            float(format_scale)
            * format_residual.unsqueeze(-1)
            * protocol_weights
        )
        if protocol_guard_mask is None:
            configured_guard_rows = torch.zeros_like(format_valid, dtype=torch.bool)
        else:
            configured_guard_rows = protocol_guard_mask.to(
                device=format_valid.device, dtype=torch.bool
            )
            if configured_guard_rows.shape != format_valid.shape:
                raise ValueError(
                    'protocol_guard_mask must have one value per response'
                )
        guard_rows = configured_guard_rows & guard_scope_rows
        valid_guard_value = (
            (1.0 - float(protocol_guard_target))
            * float(protocol_guard_valid_credit)
        )
        invalid_guard_value = -float(protocol_guard_target)
        guard_residual = torch.where(
            format_valid,
            torch.full_like(format_values, valid_guard_value),
            torch.full_like(format_values, invalid_guard_value),
        ) * guard_rows.to(format_values.dtype)
        protocol_guard_component = (
            float(protocol_guard_coef)
            * guard_residual.unsqueeze(-1)
            * protocol_weights
        )
        correct_length_component = (
            float(correct_length_coef)
            * length_residual.unsqueeze(-1)
            * eos_mask
        )
        advantages = (
            semantic_component
            + protocol_component
            + protocol_guard_component
            + correct_length_component
        )

    valid_token_count = eos_mask.sum().clamp_min(1).to(torch.float32)
    num_groups = max(len(id_to_rows), 1)
    metrics = {
        **projection_metrics,
        'pda/correctness_active_group_fraction': (
            correctness_active_groups / num_groups
        ),
        'pda/format_active_group_fraction': format_active_groups / num_groups,
        'pda/view_contrast_active_stratum_fraction': (
            view_active_strata / max(view_strata, 1)
        ),
        'pda/correctness_residual_abs_mean': float(
            correctness_residual.abs().mean().detach().cpu().item()
        ),
        'pda/format_residual_abs_mean': float(
            format_residual.abs().mean().detach().cpu().item()
        ),
        'pda/semantic_component_abs_token_mean': float(
            semantic_component.abs().sum().div(valid_token_count).detach().cpu().item()
        ),
        'pda/view_contrast_component_abs_token_mean': float(
            view_semantic_component.abs().sum().div(
                valid_token_count
            ).detach().cpu().item()
        ),
        'pda/protocol_guard_coef': float(protocol_guard_coef),
        'pda/protocol_guard_application_fraction': float(
            guard_rows.to(torch.float32).mean().detach().cpu().item()
        ),
        'pda/protocol_guard_scope_response_fraction': float(
            guard_scope_rows.to(torch.float32).mean().detach().cpu().item()
        ),
        'pda/protocol_guard_all_wrong_group_fraction': (
            all_wrong_groups / num_groups
        ),
        'pda/protocol_guard_negative_response_fraction': float(
            (guard_rows & ~format_valid).to(torch.float32).mean()
            .detach().cpu().item()
        ),
        'pda/protocol_guard_positive_response_fraction': float(
            (guard_rows & format_valid).to(torch.float32).mean()
            .detach().cpu().item()
            if float(protocol_guard_valid_credit) > 0.0 else 0.0
        ),
        'pda/protocol_guard_valid_credit': float(
            protocol_guard_valid_credit
        ),
        'pda/protocol_guard_component_abs_token_mean': float(
            protocol_guard_component.abs().sum().div(
                valid_token_count
            ).detach().cpu().item()
        ),
        'pda/protocol_component_abs_token_mean': float(
            protocol_component.abs().sum().div(valid_token_count).detach().cpu().item()
        ),
        'pda/correct_length_coef': float(correct_length_coef),
        'pda/correct_length_active_group_fraction': (
            length_active_groups / num_groups
        ),
        'pda/correct_length_active_response_fraction': (
            length_active_responses / max(len(sequence_scores), 1)
        ),
        'pda/correct_length_residual_abs_mean': float(
            length_residual.abs().mean().detach().cpu().item()
        ),
        'pda/correct_length_component_abs_token_mean': float(
            correct_length_component.abs().sum().div(
                valid_token_count
            ).detach().cpu().item()
        ),
    }
    return advantages, advantages, metrics


def compute_hcaa_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index,
                                   correctness: torch.Tensor,
                                   advantage_scale: float = 2.0,
                                   amplification: float = 1.0,
                                   hardness_power: float = 1.0):
    """Difficulty-asymmetric credit for rare correct responses.

    HCAA starts from Dr.GRPO's unstandardized group-centered outcome signal.
    In mixed-success groups, positive residuals are amplified in proportion to
    the prompt's empirical failure rate.  The result is re-centered per group,
    so the method reallocates credit within a prompt rather than changing its
    average weight.  Uniform all-correct/all-wrong groups are left untouched.
    """
    if amplification < 0.0:
        raise ValueError('hcaa.amplification must be non-negative')
    if hardness_power <= 0.0:
        raise ValueError('hcaa.hardness_power must be positive')

    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)
    correctness = correctness.to(device=scores.device, dtype=torch.float32)
    id_to_rows = defaultdict(list)
    for row in range(scores.shape[0]):
        id_to_rows[index[row]].append(row)

    calibrated = torch.empty_like(scores)
    active_groups = 0
    boost_values = []
    with torch.no_grad():
        for rows in id_to_rows.values():
            row_tensor = torch.as_tensor(rows, device=scores.device)
            group_scores = scores.index_select(0, row_tensor)
            centered = group_scores - group_scores.mean()
            success_rate = correctness.index_select(0, row_tensor).mean()
            mixed = 0.0 < float(success_rate.item()) < 1.0
            boost = 1.0
            if mixed:
                boost = 1.0 + float(amplification) * (
                    (1.0 - float(success_rate.item())) ** float(hardness_power)
                )
                positive = centered > 0
                centered = torch.where(positive, centered * boost, centered)
                centered = centered - centered.mean()
                active_groups += 1
            calibrated.index_copy_(0, row_tensor, centered * float(advantage_scale))
            boost_values.append(boost)
        advantages = calibrated.unsqueeze(-1).expand(-1, response_length) * eos_mask

    metrics = {
        'hcaa/active_group_fraction': active_groups / max(len(id_to_rows), 1),
        'hcaa/mean_positive_boost': sum(boost_values) / max(len(boost_values), 1),
        'hcaa/sequence_advantage_abs_mean': float(
            calibrated.abs().mean().detach().cpu().item()
        ),
    }
    return advantages, advantages, metrics


def compute_bifrontier_grpo_advantage(
        token_level_rewards: torch.Tensor,
        eos_mask: torch.Tensor,
        index,
        correctness: torch.Tensor,
        response_lengths: torch.Tensor,
        length_coef: float,
        epsilon: float = 1e-6):
    """GRPO outcome advantage plus a lexicographic within-correct tie-breaker.

    Outcome advantages are identical to ordinary GRPO.  Separately within each
    prompt group, two or more correct responses are ranked by length; shorter
    correct responses receive positive residual advantage and longer correct
    responses receive negative residual advantage.  Incorrect responses never
    receive a length bonus, so brevity cannot compensate for a wrong answer.

    Applying the length coefficient *after* standardization is important:
    putting a small length bonus into scalar rewards before GRPO normalization
    would turn any non-zero bonus into a full-strength update in all-correct
    groups.
    """
    if length_coef < 0:
        raise ValueError("length_coef must be non-negative")
    base_advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        eos_mask=eos_mask,
        index=index,
        epsilon=epsilon,
    )
    length_residual = torch.zeros_like(response_lengths, dtype=torch.float32)

    groups = defaultdict(list)
    for position, group_id in enumerate(index):
        groups[group_id].append(position)
    active_groups = 0
    active_responses = 0
    for positions in groups.values():
        correct_positions = [p for p in positions if bool(correctness[p].item())]
        if len(correct_positions) < 2:
            continue
        group_lengths = response_lengths[correct_positions].to(torch.float32)
        std = torch.std(group_lengths, unbiased=False)
        if std <= epsilon:
            continue
        residual = (torch.mean(group_lengths) - group_lengths) / (std + epsilon)
        length_residual[correct_positions] = residual
        active_groups += 1
        active_responses += len(correct_positions)

    # Add the residual directly to the original tensor.  In particular, when
    # length_coef is zero this is bitwise identical to ordinary GRPO; an
    # earlier mean-then-expand implementation was mathematically equivalent
    # but introduced avoidable floating-point perturbations before activation.
    advantages = base_advantages + float(length_coef) * length_residual.unsqueeze(-1) * eos_mask
    metrics = {
        "bifrontier/active_group_fraction": float(active_groups / max(len(groups), 1)),
        "bifrontier/active_response_fraction": float(
            active_responses / max(len(response_lengths), 1)
        ),
        "bifrontier/length_residual_abs_mean": float(
            length_residual.abs().mean().detach().cpu().item()
        ),
    }
    return advantages, advantages, metrics


def compute_lexiguard_grpo_advantage(
        token_level_rewards: torch.Tensor,
        eos_mask: torch.Tensor,
        index,
        correctness: torch.Tensor,
        format_valid: torch.Tensor,
        response_lengths: torch.Tensor,
        length_coef: float = 0.0,
        format_coef: float = 0.5,
        incorrect_tail_tokens: int = 128,
        incorrect_tail_coef: float = 0.0,
        epsilon: float = 1e-6):
    """Lexicographic GRPO with a token-local guard against wrong long tails.

    The primary GRPO advantage is computed from binary correctness only.
    Format then breaks ties *only among incorrect responses from the same
    prompt*, and response length breaks ties only among correct responses.
    Consequently neither good formatting nor brevity can compensate for an
    incorrect answer.

    Incorrect trajectories receive no brevity bonus.  Instead, only tokens
    after ``incorrect_tail_tokens`` receive a small negative continuation
    advantage.  This localizes the intervention to runaway suffixes instead
    of suppressing a potentially useful prefix or rewarding short failures.
    """
    if length_coef < 0:
        raise ValueError("length_coef must be non-negative")
    if format_coef < 0:
        raise ValueError("format_coef must be non-negative")
    if incorrect_tail_tokens < 0:
        raise ValueError("incorrect_tail_tokens must be non-negative")
    if incorrect_tail_coef < 0:
        raise ValueError("incorrect_tail_coef must be non-negative")

    correctness_rewards = torch.zeros_like(token_level_rewards)
    correctness_rewards[:, -1] = correctness.to(token_level_rewards.dtype)
    base_advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards=correctness_rewards,
        eos_mask=eos_mask,
        index=index,
        epsilon=epsilon,
    )

    format_residual = torch.zeros_like(response_lengths, dtype=torch.float32)
    length_residual = torch.zeros_like(response_lengths, dtype=torch.float32)
    groups = defaultdict(list)
    for position, group_id in enumerate(index):
        groups[group_id].append(position)

    format_active_groups = 0
    length_active_groups = 0
    for positions in groups.values():
        incorrect_positions = [p for p in positions if not bool(correctness[p].item())]
        if len(incorrect_positions) >= 2:
            values = format_valid[incorrect_positions].to(torch.float32)
            std = torch.std(values, unbiased=False)
            if std > epsilon:
                format_residual[incorrect_positions] = (
                    values - torch.mean(values)
                ) / (std + epsilon)
                format_active_groups += 1

        correct_positions = [p for p in positions if bool(correctness[p].item())]
        if len(correct_positions) >= 2:
            lengths = response_lengths[correct_positions].to(torch.float32)
            std = torch.std(lengths, unbiased=False)
            if std > epsilon:
                length_residual[correct_positions] = (
                    torch.mean(lengths) - lengths
                ) / (std + epsilon)
                length_active_groups += 1

    advantages = base_advantages
    if format_coef:
        advantages = advantages + (
            float(format_coef) * format_residual.unsqueeze(-1) * eos_mask
        )
    if length_coef:
        advantages = advantages + (
            float(length_coef) * length_residual.unsqueeze(-1) * eos_mask
        )

    token_positions = torch.arange(
        eos_mask.size(-1), device=eos_mask.device
    ).unsqueeze(0)
    incorrect_tail_mask = (
        (token_positions >= int(incorrect_tail_tokens))
        & eos_mask.to(torch.bool)
        & (~correctness.to(torch.bool)).unsqueeze(-1)
    )
    if incorrect_tail_coef:
        advantages = advantages - (
            float(incorrect_tail_coef) * incorrect_tail_mask.to(advantages.dtype)
        )

    metrics = {
        "lexiguard/format_active_group_fraction": float(
            format_active_groups / max(len(groups), 1)
        ),
        "lexiguard/length_active_group_fraction": float(
            length_active_groups / max(len(groups), 1)
        ),
        "lexiguard/format_residual_abs_mean": float(
            format_residual.abs().mean().detach().cpu().item()
        ),
        "lexiguard/length_residual_abs_mean": float(
            length_residual.abs().mean().detach().cpu().item()
        ),
        "lexiguard/incorrect_tail_response_fraction": float(
            incorrect_tail_mask.any(dim=-1).to(torch.float32).mean().detach().cpu().item()
        ),
        "lexiguard/incorrect_tail_token_fraction": float(
            incorrect_tail_mask.to(torch.float32).sum().detach().cpu().item()
            / max(float(eos_mask.sum().detach().cpu().item()), 1.0)
        ),
    }
    return advantages, advantages, metrics


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange,
                        loss_normalization='token_mean',
                        fixed_loss_normalizer=64.0,
                        cliprange_high=None):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    upper_clip = cliprange if cliprange_high is None else float(cliprange_high)
    if upper_clip < 0.0:
        raise ValueError('cliprange_high must be non-negative')
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1.0 - cliprange, 1.0 + upper_clip
    )

    clipped_losses = torch.max(pg_losses, pg_losses2)
    if loss_normalization == 'token_mean':
        pg_loss = verl_F.masked_mean(clipped_losses, eos_mask)
    elif loss_normalization == 'fixed':
        if float(fixed_loss_normalizer) <= 0.0:
            raise ValueError('fixed_loss_normalizer must be positive')
        pg_loss = (
            (clipped_losses * eos_mask).sum()
            / (clipped_losses.shape[0] * float(fixed_loss_normalizer))
        )
    else:
        raise ValueError('loss_normalization must be token_mean or fixed')
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl


def compute_entropy_loss(logits, eos_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns)**2
    vf_losses2 = (vpredclipped - returns)**2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == 'low_var_kl':
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
