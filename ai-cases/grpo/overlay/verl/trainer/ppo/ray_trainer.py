# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.bifrontier import ResponseBudgetConfig, ResponseBudgetController
from verl.trainer.ppo.posterior_frontier import (
    PosteriorFrontierConfig,
    PosteriorFrontierController,
)
from verl.trainer.ppo.frontier_scaling import (
    FrontierScalingConfig,
    FrontierScalingController,
)
from verl.trainer.ppo.repetition_quarantine import (
    RepetitionQuarantineConfig,
    apply_repetition_quarantine,
)
from verl.trainer.ppo.strategy_archive import (
    StrategyArchiveConfig,
    StrategyArchiveController,
)
from verl.trainer.ppo.format_constraint import (
    FormatConstraintConfig,
    FormatConstraintController,
    apply_format_constraint,
    build_format_application_mask,
)
from verl.trainer.ppo.deployment_alignment import (
    DeploymentProbeConfig,
    DeploymentProbeController,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
import wandb

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0,
                      num_repeat=1, response_budget_controller=None,
                      correctness_score=1.0, format_score=0.1,
                      advantage_variant='standard', lexiguard_config=None,
                      drgrpo_advantage_scale=2.0,
                      pda_config=None, protocol_token_sequences=None,
                      protocol_guard_coef=0.0,
                      protocol_guard_target=0.98,
                      protocol_guard_mask=None,
                      hcaa_config=None,
                      posterior_frontier_controller=None,
                      frontier_scaling_controller=None,
                      repetition_quarantine_config=None,
                      strategy_archive_controller=None,
                      response_texts=None,
                      global_step=0, metrics=None):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        if advantage_variant == 'standard':
            advantages, returns = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=token_level_rewards,
                eos_mask=response_mask,
                index=index,
            )
        elif advantage_variant == 'drgrpo':
            advantages, returns = core_algos.compute_drgrpo_outcome_advantage(
                token_level_rewards=token_level_rewards,
                eos_mask=response_mask,
                index=index,
                advantage_scale=drgrpo_advantage_scale,
            )
        elif advantage_variant == 'pda_drgrpo':
            config = pda_config or {}
            correct_length_coef = 0.0
            response_lengths = None
            if bool(config.get('correct_length_frontier_enable', False)):
                if response_budget_controller is None:
                    raise ValueError(
                        'pda correct length frontier requires response_budget.enable=True'
                    )
                sequence_scores = data.batch['token_level_scores'].sum(dim=-1)
                correctness = torch.isclose(
                    sequence_scores,
                    torch.as_tensor(
                        correctness_score,
                        device=sequence_scores.device,
                        dtype=sequence_scores.dtype,
                    ),
                    atol=1e-6,
                )
                response_lengths = response_mask.sum(dim=-1).to(torch.float32)
                correct_lengths = response_lengths[correctness]
                mean_correct_tokens = (
                    float(correct_lengths.mean().detach().cpu().item())
                    if correct_lengths.numel() else None
                )
                budget_metrics = response_budget_controller.observe(
                    mean_correct_tokens=mean_correct_tokens,
                    step=global_step,
                )
                correct_length_coef = response_budget_controller.coef
                if metrics is not None:
                    metrics.update(budget_metrics)
            advantages, returns, pda_metrics = \
                core_algos.compute_pda_drgrpo_advantage(
                    token_level_scores=data.batch['token_level_scores'],
                    eos_mask=response_mask,
                    index=index,
                    response_ids=responses,
                    protocol_token_sequences=protocol_token_sequences or (),
                    correctness_score=correctness_score,
                    format_score=format_score,
                    correctness_scale=float(config.get('correctness_scale', 1.8)),
                    format_scale=float(config.get('format_scale', 0.8)),
                    projection=str(config.get('projection', 'local')),
                    invalid_tail_tokens=int(config.get('invalid_tail_tokens', 16)),
                    protocol_mass=float(config.get('protocol_mass', 16.0)),
                    view_labels=data.non_tensor_batch.get('mirror_view'),
                    view_contrast_scale=float(
                        config.get('view_contrast_scale', 0.0)
                    ),
                    protocol_guard_coef=protocol_guard_coef,
                    protocol_guard_target=protocol_guard_target,
                    protocol_guard_mask=protocol_guard_mask,
                    protocol_guard_scope=str(
                        config.get('protocol_guard_scope', 'all')
                    ),
                    protocol_guard_valid_credit=float(
                        config.get('protocol_guard_valid_credit', 1.0)
                    ),
                    correct_length_coef=correct_length_coef,
                    response_lengths=response_lengths,
                )
            if metrics is not None:
                metrics.update(pda_metrics)
        elif advantage_variant == 'hcaa':
            sequence_scores = data.batch['token_level_scores'].sum(dim=-1)
            correctness = torch.isclose(
                sequence_scores,
                torch.as_tensor(
                    correctness_score,
                    device=sequence_scores.device,
                    dtype=sequence_scores.dtype,
                ),
                atol=1e-6,
            )
            config = hcaa_config or {}
            advantages, returns, hcaa_metrics = \
                core_algos.compute_hcaa_outcome_advantage(
                    token_level_rewards=token_level_rewards,
                    eos_mask=response_mask,
                    index=index,
                    correctness=correctness,
                    advantage_scale=float(config.get('advantage_scale', 2.0)),
                    amplification=float(config.get('amplification', 1.0)),
                    hardness_power=float(config.get('hardness_power', 1.0)),
                )
            if metrics is not None:
                metrics.update(hcaa_metrics)
        elif advantage_variant == 'bifrontier':
            if response_budget_controller is None:
                raise ValueError('bifrontier requires response_budget.enable=True')
            sequence_scores = data.batch['token_level_scores'].sum(dim=-1)
            correctness = torch.isclose(
                sequence_scores,
                torch.as_tensor(
                    correctness_score,
                    device=sequence_scores.device,
                    dtype=sequence_scores.dtype,
                ),
                atol=1e-6,
            )
            response_lengths = response_mask.sum(dim=-1).to(torch.float32)
            correct_lengths = response_lengths[correctness]
            mean_correct_tokens = (
                float(correct_lengths.mean().detach().cpu().item())
                if correct_lengths.numel() else None
            )
            controller_metrics = response_budget_controller.observe(
                mean_correct_tokens=mean_correct_tokens,
                step=global_step,
            )
            advantages, returns, advantage_metrics = \
                core_algos.compute_bifrontier_grpo_advantage(
                    token_level_rewards=token_level_rewards,
                    eos_mask=response_mask,
                    index=index,
                    correctness=correctness,
                    response_lengths=response_lengths,
                    length_coef=response_budget_controller.coef,
                )
            if metrics is not None:
                metrics.update(controller_metrics)
                metrics.update(advantage_metrics)
        elif advantage_variant in ('lexi', 'lexiguard'):
            sequence_scores = data.batch['token_level_scores'].sum(dim=-1)
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
            response_lengths = response_mask.sum(dim=-1).to(torch.float32)
            length_coef = 0.0
            if advantage_variant == 'lexiguard':
                if response_budget_controller is None:
                    raise ValueError('lexiguard requires response_budget.enable=True')
                correct_lengths = response_lengths[correctness]
                mean_correct_tokens = (
                    float(correct_lengths.mean().detach().cpu().item())
                    if correct_lengths.numel() else None
                )
                controller_metrics = response_budget_controller.observe(
                    mean_correct_tokens=mean_correct_tokens,
                    step=global_step,
                )
                length_coef = response_budget_controller.coef
                if metrics is not None:
                    metrics.update(controller_metrics)

            config = lexiguard_config or {}
            tail_coef = (
                float(config.get('incorrect_tail_coef', 0.05))
                if advantage_variant == 'lexiguard' else 0.0
            )
            advantages, returns, advantage_metrics = \
                core_algos.compute_lexiguard_grpo_advantage(
                    token_level_rewards=token_level_rewards,
                    eos_mask=response_mask,
                    index=index,
                    correctness=correctness,
                    format_valid=format_valid,
                    response_lengths=response_lengths,
                    length_coef=length_coef,
                    format_coef=float(config.get('format_coef', 0.5)),
                    incorrect_tail_tokens=int(config.get('incorrect_tail_tokens', 128)),
                    incorrect_tail_coef=tail_coef,
                )
            if metrics is not None:
                metrics.update(advantage_metrics)
        else:
            raise ValueError(f'unknown advantage_variant: {advantage_variant}')
        if posterior_frontier_controller is not None:
            sequence_scores = data.batch['token_level_scores'].sum(dim=-1)
            correctness = torch.isclose(
                sequence_scores,
                torch.as_tensor(
                    correctness_score,
                    device=sequence_scores.device,
                    dtype=sequence_scores.dtype,
                ),
                atol=1e-6,
            )
            residual, posterior_metrics = posterior_frontier_controller.observe(
                sample_indices=data.non_tensor_batch['index'],
                correctness=correctness,
            )
            coefficient = posterior_frontier_controller.config.coefficient
            if coefficient:
                advantages = advantages + (
                    float(coefficient) * residual.unsqueeze(-1) * response_mask
                )
                returns = advantages
            if metrics is not None:
                metrics.update(posterior_metrics)
        if frontier_scaling_controller is not None:
            sequence_scores = data.batch['token_level_scores'].sum(dim=-1)
            correctness = torch.isclose(
                sequence_scores,
                torch.as_tensor(
                    correctness_score,
                    device=sequence_scores.device,
                    dtype=sequence_scores.dtype,
                ),
                atol=1e-6,
            )
            weights, scaling_metrics = frontier_scaling_controller.observe(
                sample_indices=data.non_tensor_batch['index'],
                correctness=correctness,
            )
            advantages = advantages * weights.unsqueeze(-1) * response_mask
            returns = advantages
            if metrics is not None:
                metrics.update(scaling_metrics)
        if repetition_quarantine_config is not None:
            sequence_scores = data.batch['token_level_scores'].sum(dim=-1)
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
            advantages, quarantine_metrics = apply_repetition_quarantine(
                advantages=advantages,
                response_ids=responses,
                response_mask=response_mask,
                correctness=correctness,
                format_valid=format_valid,
                config=repetition_quarantine_config,
            )
            returns = advantages
            if metrics is not None:
                metrics.update(quarantine_metrics)
        if strategy_archive_controller is not None:
            if response_texts is None:
                raise ValueError('strategy_archive requires decoded response_texts')
            sequence_scores = data.batch['token_level_scores'].sum(dim=-1)
            correctness = torch.isclose(
                sequence_scores,
                torch.as_tensor(
                    correctness_score,
                    device=sequence_scores.device,
                    dtype=sequence_scores.dtype,
                ),
                atol=1e-6,
            )
            residual, archive_metrics = strategy_archive_controller.observe(
                sample_indices=data.non_tensor_batch['index'],
                correctness=correctness,
                response_texts=response_texts,
            )
            coefficient = strategy_archive_controller.config.coefficient
            if coefficient:
                advantages = advantages + (
                    float(coefficient) * residual.unsqueeze(-1) * response_mask
                )
                returns = advantages
            if metrics is not None:
                metrics.update(archive_metrics)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]
    hint_slice_prop = batch.non_tensor_batch['hint_slice_prop']

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
        hint_slice_prop=hint_slice_prop,
    )

def compute_acc(sequence_score, format_score=0.1, score=1.0):
    format_acc = (np.sum(np.array(sequence_score) >= format_score - 1e-6)) / len(sequence_score)
    correctness_acc = (np.sum(np.abs(np.array(sequence_score) - score) < 1e-6)) / len(sequence_score)
    return format_acc, correctness_acc

def compute_data_metrics(batch, use_critic=True, format_score=0.1, score=1.0):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    format_acc, correctness_acc = compute_acc(sequence_score.detach().cpu().numpy(), format_score=format_score, score=score)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']
    hint_slice_prop = response_info['hint_slice_prop'].astype(float)

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        'critic/format_accuracy':
            format_acc,
        'critic/correctness_accuracy':
            correctness_acc,
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
        
        # hint slice length
        'hint_slice_prop/mean':
            np.mean(hint_slice_prop),
        'hint_slice_prop/max':
            np.max(hint_slice_prop),
        'hint_slice_prop/min':
            np.min(hint_slice_prop),
    }
    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.last_saved_checkpoint_idx = None

        response_budget_config = config.trainer.get('response_budget', {})
        self.advantage_variant = str(config.trainer.get('advantage_variant', 'standard'))
        if self.advantage_variant not in (
                'standard', 'drgrpo', 'pda_drgrpo', 'hcaa', 'bifrontier',
                'lexi', 'lexiguard'):
            raise ValueError(
                'trainer.advantage_variant must be standard, drgrpo, '
                'pda_drgrpo, hcaa, bifrontier, lexi, or lexiguard'
            )
        self.drgrpo_advantage_scale = float(
            config.trainer.get('drgrpo', {}).get('advantage_scale', 2.0)
        )
        self.pda_config = config.trainer.get('pda', {})
        protocol_tags = self.pda_config.get(
            'protocol_tags', ['</think>', '<answer>', '</answer>']
        )
        self.pda_protocol_token_sequences = [
            self.tokenizer.encode(str(tag), add_special_tokens=False)
            for tag in protocol_tags
        ]
        if self.advantage_variant == 'pda_drgrpo' and not any(
                self.pda_protocol_token_sequences):
            raise ValueError('pda.protocol_tags must tokenize to non-empty sequences')
        self.hcaa_config = config.trainer.get('hcaa', {})
        self.lexiguard_config = config.trainer.get('lexiguard', {})
        paired_hint_config = config.trainer.get('paired_hint', {})
        self.paired_hint_enabled = bool(paired_hint_config.get('enable', False))
        if self.paired_hint_enabled and int(config.actor_rollout_ref.rollout.n) != 2:
            raise ValueError('paired_hint requires actor_rollout_ref.rollout.n=2')
        mirror_config = config.trainer.get('mirror_view', {})
        self.mirror_view_enabled = bool(mirror_config.get('enable', False))
        self.mirror_view_paired = bool(mirror_config.get('paired', True))
        if (self.mirror_view_enabled and self.mirror_view_paired
                and int(config.actor_rollout_ref.rollout.n) != 2):
            raise ValueError('paired mirror_view requires actor_rollout_ref.rollout.n=2')
        if (float(self.pda_config.get('view_contrast_scale', 0.0)) > 0.0
                and not (self.mirror_view_enabled and self.mirror_view_paired)):
            raise ValueError(
                'pda.view_contrast_scale requires paired mirror_view'
            )
        if self.mirror_view_enabled and self.paired_hint_enabled:
            raise ValueError('mirror_view and paired_hint cannot be enabled together')
        format_constraint_config = config.trainer.get('format_constraint', {})
        self.format_constraint_controller = None
        if bool(format_constraint_config.get('enable', False)):
            self.format_constraint_controller = FormatConstraintController(
                FormatConstraintConfig(
                    target_rate=float(format_constraint_config.get('target_rate', 0.995)),
                    dual_lr=float(format_constraint_config.get('dual_lr', 2.0)),
                    max_bonus=float(format_constraint_config.get('max_bonus', 0.20)),
                    min_bonus=float(format_constraint_config.get('min_bonus', 0.0)),
                    release_lr_scale=float(format_constraint_config.get(
                        'release_lr_scale', 1.0
                    )),
                    lcb_z=float(format_constraint_config.get('lcb_z', 0.0)),
                    ema_alpha=float(format_constraint_config.get('ema_alpha', 0.10)),
                    warmup_steps=int(format_constraint_config.get('warmup_steps', 40)),
                    feedback_source=str(format_constraint_config.get(
                        'feedback_source', 'train'
                    )),
                    apply_scope=str(format_constraint_config.get('apply_scope', 'all')),
                    apply_hint_threshold=float(format_constraint_config.get(
                        'apply_hint_threshold', 0.0
                    )),
                )
            )
        self.anchor_probe_config = config.trainer.get('anchor_probe', {})
        self.anchor_probe_enabled = bool(
            self.anchor_probe_config.get('enable', False)
        )
        if (self.anchor_probe_enabled
                and self.format_constraint_controller is not None
                and self.format_constraint_controller.config.feedback_source
                != 'anchor'):
            raise ValueError(
                'anchor_probe with a format controller requires '
                'format_constraint feedback_source=anchor'
            )
        if self.anchor_probe_enabled:
            if int(self.anchor_probe_config.get('size', 128)) <= 0:
                raise ValueError('anchor_probe.size must be positive')
            if int(self.anchor_probe_config.get('frequency', 10)) <= 0:
                raise ValueError('anchor_probe.frequency must be positive')
        if bool(self.pda_config.get('protocol_guard_enable', False)):
            if self.advantage_variant != 'pda_drgrpo':
                raise ValueError('pda protocol guard requires pda_drgrpo')
            if self.format_constraint_controller is None:
                raise ValueError(
                    'pda protocol guard requires format_constraint.enable=True'
                )
            guard_format_config = self.format_constraint_controller.config
            if (guard_format_config.feedback_source not in ('no_hint', 'anchor')
                    or guard_format_config.apply_scope not in (
                        'no_hint', 'low_hint'
                    )):
                raise ValueError(
                    'pda protocol guard requires training-only no_hint/anchor '
                    'feedback and no_hint/low_hint application scope'
                )
            if str(self.pda_config.get(
                    'protocol_guard_scope', 'all')) not in ('all', 'all_wrong'):
                raise ValueError(
                    "pda.protocol_guard_scope must be 'all' or 'all_wrong'"
                )
            valid_credit = float(self.pda_config.get(
                'protocol_guard_valid_credit', 1.0
            ))
            if not 0.0 <= valid_credit <= 1.0:
                raise ValueError(
                    'pda.protocol_guard_valid_credit must be in [0, 1]'
                )
        deployment_probe_config = config.trainer.get('deployment_probe', {})
        self.deployment_probe_controller = None
        if bool(deployment_probe_config.get('enable', False)):
            if not self.mirror_view_enabled:
                raise ValueError('deployment_probe requires mirror_view.enable=True')
            self.deployment_probe_controller = DeploymentProbeController(
                DeploymentProbeConfig(
                    feedback_source=str(deployment_probe_config.get(
                        'feedback_source', 'no_hint'
                    )),
                    target_format_rate=float(deployment_probe_config.get(
                        'target_format_rate', 0.98
                    )),
                    breach_patience=int(deployment_probe_config.get(
                        'breach_patience', 2
                    )),
                    decay_factor=float(deployment_probe_config.get(
                        'decay_factor', 0.25
                    )),
                    recovery_step=float(deployment_probe_config.get(
                        'recovery_step', 0.20
                    )),
                    min_scale=float(deployment_probe_config.get('min_scale', 0.10)),
                    warmup_steps=int(deployment_probe_config.get('warmup_steps', 50)),
                )
            )
        posterior_config = config.trainer.get('posterior_frontier', {})
        self.posterior_frontier_controller = None
        if bool(posterior_config.get('enable', False)):
            self.posterior_frontier_controller = PosteriorFrontierController(
                PosteriorFrontierConfig(
                    coefficient=float(posterior_config.get('coefficient', 0.5)),
                    prior_alpha=float(posterior_config.get('prior_alpha', 1.0)),
                    prior_beta=float(posterior_config.get('prior_beta', 1.0)),
                    min_observations=int(posterior_config.get('min_observations', 1)),
                )
            )
        scaling_config = config.trainer.get('frontier_scaling', {})
        self.frontier_scaling_controller = None
        if bool(scaling_config.get('enable', False)):
            self.frontier_scaling_controller = FrontierScalingController(
                FrontierScalingConfig(
                    min_weight=float(scaling_config.get('min_weight', 0.25)),
                    max_weight=float(scaling_config.get('max_weight', 1.75)),
                    prior_alpha=float(scaling_config.get('prior_alpha', 1.0)),
                    prior_beta=float(scaling_config.get('prior_beta', 1.0)),
                    min_observations=int(scaling_config.get('min_observations', 1)),
                )
            )
        quarantine_config = config.trainer.get('repetition_quarantine', {})
        self.repetition_quarantine_config = None
        if bool(quarantine_config.get('enable', False)):
            self.repetition_quarantine_config = RepetitionQuarantineConfig(
                coefficient=float(quarantine_config.get('coefficient', 0.10)),
                ngram_size=int(quarantine_config.get('ngram_size', 8)),
                min_repeats=int(quarantine_config.get('min_repeats', 3)),
                invalid_tail_tokens=int(
                    quarantine_config.get('invalid_tail_tokens', 256)
                ),
            )
        archive_config = config.trainer.get('strategy_archive', {})
        self.strategy_archive_controller = None
        if bool(archive_config.get('enable', False)):
            self.strategy_archive_controller = StrategyArchiveController(
                StrategyArchiveConfig(
                    coefficient=float(archive_config.get('coefficient', 0.15))
                )
            )
        self.response_budget_controller = None
        if bool(response_budget_config.get('enable', False)):
            self.response_budget_controller = ResponseBudgetController(
                ResponseBudgetConfig(
                    target_tokens=float(response_budget_config.get('target_tokens', 192)),
                    dual_lr=float(response_budget_config.get('dual_lr', 0.05)),
                    max_coef=float(response_budget_config.get('max_coef', 0.35)),
                    ema_alpha=float(response_budget_config.get('ema_alpha', 0.10)),
                    warmup_steps=int(response_budget_config.get('warmup_steps', 40)),
                )
            )

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        self._create_dataloader()

    def _create_dataloader(self):
        from torch.utils.data import DataLoader, Subset
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='right', # right for problems too long
                                         is_eval=False,
                                         config=self.config)
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           shuffle=True,
                                           drop_last=True,
                                           collate_fn=collate_fn)
        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='right', # right for problems too long
                                       is_eval=True,
                                       config=self.config)
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=len(self.val_dataset),
                                         shuffle=True,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        self.anchor_dataloader = None
        if self.anchor_probe_enabled:
            anchor_dataset = RLHFDataset(
                parquet_files=self.config.data.train_files,
                tokenizer=self.tokenizer,
                prompt_key=self.config.data.prompt_key,
                max_prompt_length=self.config.data.max_prompt_length,
                filter_prompts=True,
                return_raw_chat=self.config.data.get('return_raw_chat', False),
                truncation='right',
                is_eval=True,
                config=self.config,
            )
            requested_size = int(self.anchor_probe_config.get('size', 128))
            anchor_size = min(requested_size, len(anchor_dataset))
            anchor_seed = int(self.anchor_probe_config.get('seed', 20260908))
            generator = np.random.RandomState(anchor_seed)
            indices = np.sort(generator.choice(
                len(anchor_dataset), size=anchor_size, replace=False
            )).tolist()
            self.anchor_dataloader = DataLoader(
                dataset=Subset(anchor_dataset, indices),
                batch_size=anchor_size,
                shuffle=False,
                drop_last=False,
                collate_fn=collate_fn,
            )
            print(
                f'Anchor probe size: {anchor_size}; seed: {anchor_seed}; '
                f'frequency: {int(self.anchor_probe_config.get("frequency", 10))}'
            )

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []
        response_length_lst = []
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            # test_batch = test_batch.to('cuda')

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            # for certain reward function (e.g. sandbox), the generation can overlap with reward
            reward_tensor = self.val_reward_fn(test_batch)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
            response_length = test_batch.batch['responses'].size(-1)
            response_length_lst.append(
                test_batch.batch['attention_mask'][:, -response_length:]
                .sum(dim=-1).detach().cpu().to(torch.float32)
            )

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        response_lengths = torch.cat(response_length_lst, dim=0).numpy()
        # evaluate test_score based on data source
        data_source_reward = {}
        data_source_lengths = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())
            data_source_lengths.setdefault(data_source, []).append(
                float(response_lengths[i])
            )

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)
            format_acc, correctness_acc = compute_acc(rewards, self.config.trainer.format_score, self.config.trainer.correctness_score)
            metric_dict[f'val/test_score/{data_source}/format_accuracy'] = format_acc
            metric_dict[f'val/test_score/{data_source}/correctness_accuracy'] = correctness_acc
            lengths = np.asarray(data_source_lengths[data_source], dtype=float)
            metric_dict[f'val/test_score/{data_source}/response_length_mean'] = float(
                np.mean(lengths)
            )
            metric_dict[f'val/test_score/{data_source}/response_clip_ratio'] = float(
                np.mean(lengths >= float(self.config.data.max_response_length))
            )

        return metric_dict

    def _probe_training_anchor(self):
        """Greedy no-hint format probe on a fixed subset of training data."""
        if self.anchor_dataloader is None:
            return {}
        sequence_scores = []
        response_lengths = []
        for anchor_data in self.anchor_dataloader:
            anchor_batch = DataProto.from_single_dict(anchor_data)
            anchor_gen_batch = anchor_batch.pop([
                'input_ids', 'attention_mask', 'position_ids'
            ])
            anchor_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }
            padded, pad_size = pad_dataproto_to_divisor(
                anchor_gen_batch, self.actor_rollout_wg.world_size
            )
            generated_padded = self.actor_rollout_wg.generate_sequences(padded)
            generated = unpad_dataproto(generated_padded, pad_size=pad_size)
            anchor_batch = anchor_batch.union(generated)
            rewards = self.val_reward_fn(anchor_batch).sum(-1).detach().cpu()
            sequence_scores.append(rewards)
            response_length = anchor_batch.batch['responses'].size(-1)
            response_lengths.append(
                anchor_batch.batch['attention_mask'][:, -response_length:]
                .sum(dim=-1).detach().cpu().to(torch.float32)
            )
        scores = torch.cat(sequence_scores)
        lengths = torch.cat(response_lengths)
        format_valid = scores >= float(self.config.trainer.format_score) - 1e-6
        correctness = torch.isclose(
            scores,
            torch.as_tensor(
                self.config.trainer.correctness_score, dtype=scores.dtype
            ),
            atol=1e-6,
        )
        return {
            'anchor_probe/count': float(scores.numel()),
            'anchor_probe/score': float(scores.mean().item()),
            'anchor_probe/format_rate': float(
                format_valid.to(torch.float32).mean().item()
            ),
            'anchor_probe/correctness_rate': float(
                correctness.to(torch.float32).mean().item()
            ),
            'anchor_probe/response_length_mean': float(lengths.mean().item()),
        }

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        local_dir = self.config.trainer.default_local_dir + '-' + str(wandb.run.id)
        actor_local_path = os.path.join(local_dir, 'actor',
                                        f'global_step_{self.global_steps}')
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

        if self.use_critic:
            critic_local_path = os.path.join(local_dir, 'critic',
                                             f'global_step_{self.global_steps}')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, 'critic')
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def _expand_paired_hint_batch(self, batch: DataProto) -> DataProto:
        """Interleave one unhinted and one hinted prompt per logical problem."""
        pair_keys = {
            'unhinted_input_ids', 'unhinted_attention_mask',
            'unhinted_position_ids', 'unhinted_hint_mask',
        }
        missing = pair_keys - set(batch.batch.keys())
        if missing:
            raise ValueError(f'paired_hint batch is missing tensors: {sorted(missing)}')

        def interleave(first, second):
            return torch.stack((first, second), dim=1).reshape(
                -1, *first.shape[1:]
            )

        tensors = {}
        for key, value in batch.batch.items():
            if key in pair_keys:
                continue
            if key == 'input_ids':
                tensors[key] = interleave(batch.batch['unhinted_input_ids'], value)
            elif key == 'attention_mask':
                tensors[key] = interleave(batch.batch['unhinted_attention_mask'], value)
            elif key == 'position_ids':
                tensors[key] = interleave(batch.batch['unhinted_position_ids'], value)
            elif key == 'hint_mask':
                tensors[key] = interleave(batch.batch['unhinted_hint_mask'], value)
            else:
                tensors[key] = value.repeat_interleave(2, dim=0)

        non_tensors = {
            key: np.repeat(value, 2, axis=0)
            for key, value in batch.non_tensor_batch.items()
        }
        size = len(batch)
        treatment = np.tile(np.asarray([0, 1], dtype=object), size)
        non_tensors['paired_hint_treatment'] = treatment
        unhinted = np.arange(0, size * 2, 2)
        for key, value in (
            ('hint_slice_prop', 0.0),
            ('hint_arm', 0),
            ('hint_arm_probability', 1.0),
            ('hint_budget_multiplier', 0.0),
            ('hint_expected_fraction', 0.0),
        ):
            non_tensors[key][unhinted] = value
        return DataProto.from_dict(
            tensors=tensors,
            non_tensors=non_tensors,
            meta_info=batch.meta_info,
        )

    def _expand_mirror_view_batch(self, batch: DataProto) -> DataProto:
        """Interleave the original and a number-order permutation per problem."""
        mirror_keys = {
            'mirror_input_ids', 'mirror_attention_mask',
            'mirror_position_ids', 'mirror_hint_mask',
        }
        missing = mirror_keys - set(batch.batch.keys())
        if missing:
            raise ValueError(f'mirror_view batch is missing tensors: {sorted(missing)}')

        def interleave(first, second):
            return torch.stack((first, second), dim=1).reshape(
                -1, *first.shape[1:]
            )

        tensors = {}
        key_map = {
            'input_ids': 'mirror_input_ids',
            'attention_mask': 'mirror_attention_mask',
            'position_ids': 'mirror_position_ids',
            'hint_mask': 'mirror_hint_mask',
        }
        for key, value in batch.batch.items():
            if key in mirror_keys:
                continue
            if key in key_map:
                tensors[key] = interleave(value, batch.batch[key_map[key]])
            else:
                tensors[key] = value.repeat_interleave(2, dim=0)

        non_tensors = {
            key: np.repeat(value, 2, axis=0)
            for key, value in batch.non_tensor_batch.items()
        }
        non_tensors['mirror_view'] = np.tile(
            np.asarray([0, 1], dtype=object), len(batch)
        )
        return DataProto.from_dict(
            tensors=tensors,
            non_tensors=non_tensors,
            meta_info=batch.meta_info,
        )

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        def observe_deployment_validation(val_metrics, target_metrics):
            format_key = 'val/test_score/countdown/format_accuracy'
            if format_key not in val_metrics:
                return
            format_rate = float(val_metrics[format_key])
            if (self.format_constraint_controller is not None
                    and self.format_constraint_controller.config.feedback_source == 'validation'):
                target_metrics.update(self.format_constraint_controller.observe(
                    format_rate, step=self.global_steps, update=True
                ))
            if (self.deployment_probe_controller is not None
                    and self.deployment_probe_controller.config.feedback_source == 'validation'):
                probe_metrics = self.deployment_probe_controller.observe(
                    format_rate, step=self.global_steps
                )
                self.train_dataset.mirror_probability_scale = (
                    self.deployment_probe_controller.augmentation_scale
                )
                target_metrics.update(probe_metrics)

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            observe_deployment_validation(val_metrics, val_metrics)
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                print(f'epoch {epoch}, step {self.global_steps}')
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                if 'extra_info' in batch.non_tensor_batch:
                    metrics['data/forge_fraction'] = float(np.mean([
                        info.get('source') == 'forge'
                        for info in batch.non_tensor_batch['extra_info']
                    ]))
                if 'forge_anneal_multiplier' in batch.non_tensor_batch:
                    metrics['forge_anneal/target_multiplier'] = float(np.mean(
                        batch.non_tensor_batch['forge_anneal_multiplier'].astype(float)
                    ))
                    metrics['forge_anneal/replacement_fraction'] = float(np.mean(
                        batch.non_tensor_batch['forge_replaced'].astype(float)
                    ))

                # Assign logical group IDs before optionally expanding each
                # problem into paired unhinted/hinted prompt views.
                batch.non_tensor_batch['uid'] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )
                if self.paired_hint_enabled:
                    batch = self._expand_paired_hint_batch(batch)
                elif self.mirror_view_enabled and self.mirror_view_paired:
                    batch = self._expand_mirror_view_batch(batch)

                # pop those keys for generation
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_tensor = self.reward_fn(batch)
                        batch.batch['token_level_scores'] = reward_tensor
                        sequence_scores = reward_tensor.sum(-1).detach().cpu().numpy()
                        correctness = np.isclose(
                            sequence_scores,
                            self.config.trainer.correctness_score,
                            atol=1e-6,
                        ).astype(np.float64)

                        if self.config.trainer.adaptive_hint.enable:
                            feedback = dict(
                                indices=batch.non_tensor_batch['index'],
                                hint_arms=batch.non_tensor_batch['hint_arm'],
                                hint_contexts=batch.non_tensor_batch['hint_context'],
                                hint_chunk_counts=batch.non_tensor_batch['hint_chunk_count'],
                                hint_arm_probabilities=batch.non_tensor_batch['hint_arm_probability'],
                                hint_budget_multipliers=batch.non_tensor_batch['hint_budget_multiplier'],
                                hint_expected_fractions=batch.non_tensor_batch['hint_expected_fraction'],
                                correctness=correctness,
                                hint_slice_props=batch.non_tensor_batch['hint_slice_prop'],
                            )
                            if self.paired_hint_enabled:
                                adaptive_metrics = self.train_dataset.update_paired_hint_feedback(
                                    treatments=batch.non_tensor_batch['paired_hint_treatment'],
                                    hint_forced=batch.non_tensor_batch['paired_hint_forced'],
                                    **feedback,
                                )
                            else:
                                adaptive_metrics = self.train_dataset.update_adaptive_hint_feedback(
                                    **feedback
                                )
                            metrics.update(adaptive_metrics)
                        if self.mirror_view_enabled:
                            view_key = (
                                'mirror_view' if self.mirror_view_paired
                                else 'mirror_applied'
                            )
                            views = np.asarray(
                                batch.non_tensor_batch[view_key], dtype=int
                            )
                            metrics['mirror/view_fraction'] = float(np.mean(views))
                            if 'mirror_probability' in batch.non_tensor_batch:
                                metrics['mirror/target_probability'] = float(np.mean(
                                    batch.non_tensor_batch['mirror_probability'].astype(float)
                                ))
                            original = correctness[views == 0]
                            mirrored = correctness[views == 1]
                            if original.size and mirrored.size:
                                metrics.update({
                                    'mirror/original_success_mean': float(np.mean(original)),
                                    'mirror/permuted_success_mean': float(np.mean(mirrored)),
                                    'mirror/view_gap_abs': float(abs(
                                        np.mean(original) - np.mean(mirrored)
                                    )),
                                })

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.use_kl_loss:
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        protocol_guard_coef = 0.0
                        protocol_guard_target = 0.98
                        protocol_guard_mask = None
                        if self.format_constraint_controller is not None:
                            response_length = batch.batch['responses'].size(-1)
                            response_mask = batch.batch['attention_mask'][:, -response_length:]
                            hint_props = torch.as_tensor(
                                batch.non_tensor_batch['hint_slice_prop'].astype(float),
                                device=batch.batch['token_level_scores'].device,
                            )
                            no_hint_mask = hint_props <= 1e-9
                            format_config = self.format_constraint_controller.config
                            observation_mask = (
                                no_hint_mask if format_config.feedback_source == 'no_hint'
                                else None
                            )
                            apply_mask = build_format_application_mask(
                                hint_props,
                                format_config.apply_scope,
                                format_config.apply_hint_threshold,
                            )
                            constrained_rewards, constraint_metrics = apply_format_constraint(
                                token_level_rewards=batch.batch['token_level_rewards'],
                                token_level_scores=batch.batch['token_level_scores'],
                                response_mask=response_mask,
                                controller=self.format_constraint_controller,
                                correctness_score=self.config.trainer.correctness_score,
                                format_score=self.config.trainer.format_score,
                                global_step=self.global_steps,
                                observation_mask=observation_mask,
                                apply_mask=apply_mask,
                                update_controller=(
                                    format_config.feedback_source in ('train', 'no_hint')
                                ),
                            )
                            batch.batch['token_level_rewards'] = constrained_rewards
                            metrics.update(constraint_metrics)
                            if bool(self.pda_config.get(
                                    'protocol_guard_enable', False)):
                                protocol_guard_coef = (
                                    self.format_constraint_controller.coef
                                    * float(self.pda_config.get(
                                        'protocol_guard_multiplier', 1.0
                                    ))
                                )
                                protocol_guard_target = format_config.target_rate
                                protocol_guard_mask = apply_mask
                            if (self.deployment_probe_controller is not None
                                    and self.deployment_probe_controller.config.feedback_source == 'no_hint'):
                                probe_metrics = self.deployment_probe_controller.observe(
                                    constraint_metrics['format_constraint/batch_rate'],
                                    step=self.global_steps,
                                )
                                self.train_dataset.mirror_probability_scale = (
                                    self.deployment_probe_controller.augmentation_scale
                                )
                                metrics.update(probe_metrics)

                        # compute advantages, executed on the driver process
                        response_texts = None
                        if self.strategy_archive_controller is not None:
                            response_texts = self.tokenizer.batch_decode(
                                batch.batch['responses'].detach().cpu().tolist(),
                                skip_special_tokens=True,
                            )
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n,
                                                  response_budget_controller=self.response_budget_controller,
                                                  correctness_score=self.config.trainer.correctness_score,
                                                  format_score=self.config.trainer.format_score,
                                                  advantage_variant=self.advantage_variant,
                                                  lexiguard_config=self.lexiguard_config,
                                                  drgrpo_advantage_scale=self.drgrpo_advantage_scale,
                                                  pda_config=self.pda_config,
                                                  protocol_token_sequences=self.pda_protocol_token_sequences,
                                                  protocol_guard_coef=protocol_guard_coef,
                                                  protocol_guard_target=protocol_guard_target,
                                                  protocol_guard_mask=protocol_guard_mask,
                                                  hcaa_config=self.hcaa_config,
                                                  posterior_frontier_controller=self.posterior_frontier_controller,
                                                  frontier_scaling_controller=self.frontier_scaling_controller,
                                                  repetition_quarantine_config=self.repetition_quarantine_config,
                                                  strategy_archive_controller=self.strategy_archive_controller,
                                                  response_texts=response_texts,
                                                  global_step=self.global_steps,
                                                  metrics=metrics)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # AnchorGuard observes only a fixed subset of train data,
                    # generated greedily without hints or semantic MIRROR.
                    # It is intentionally separate from the reported test set.
                    anchor_frequency = int(
                        self.anchor_probe_config.get('frequency', 10)
                    )
                    if (self.anchor_probe_enabled
                            and self.global_steps % anchor_frequency == 0):
                        with _timer('anchor_probe', timing_raw):
                            anchor_metrics = self._probe_training_anchor()
                        metrics.update(anchor_metrics)
                        if self.format_constraint_controller is not None:
                            controller_metrics = \
                                self.format_constraint_controller.observe(
                                    anchor_metrics['anchor_probe/format_rate'],
                                    step=self.global_steps,
                                    update=True,
                                    sample_size=int(
                                        anchor_metrics['anchor_probe/count']
                                    ),
                                )
                            metrics.update(controller_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        observe_deployment_validation(val_metrics, metrics)
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()
                            self.last_saved_checkpoint_idx = self.global_steps

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic, 
                                                    format_score=self.config.trainer.format_score, score=self.config.trainer.correctness_score))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        observe_deployment_validation(val_metrics, val_metrics)
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    return
                
                self.global_steps += 1
                self.train_dataloader.dataset.timestep = self.global_steps
                self.val_dataloader.dataset.timestep = self.global_steps # synchronize the timestep
