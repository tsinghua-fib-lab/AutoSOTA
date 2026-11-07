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

from omegaconf import ListConfig
import os
import re
from typing import List, Union

import pandas as pd

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
from verl.utils.fs import copy_local_path_from_hdfs

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from verl.utils.dataset.adaptive_hint import (
    BanditConfig,
    BudgetedHintBandit,
    LeastAssistedMasteryTracker,
    MasteryConfig,
    group_learning_utility,
)


def forge_anneal_multiplier(step: int, start_step: int, end_step: int) -> float:
    """Linearly remove forged examples while preserving the batch size."""
    if start_step < 0 or end_step <= start_step:
        raise ValueError('forge anneal requires 0 <= start_step < end_step')
    if step <= start_step:
        return 1.0
    if step >= end_step:
        return 0.0
    return float(end_step - step) / float(end_step - start_step)


def mirror_probability(
    step: int,
    schedule: str,
    max_probability: float,
    ramp_start_step: int,
    ramp_end_step: int,
) -> float:
    """Return the active MIRROR probability for a constant or ramp schedule."""
    if not 0.0 <= max_probability <= 1.0:
        raise ValueError('mirror_view.random_probability must be in [0, 1]')
    if schedule == 'constant':
        return float(max_probability)
    if schedule != 'ramp':
        raise ValueError('mirror_view.schedule must be constant or ramp')
    if ramp_start_step < 0 or ramp_end_step <= ramp_start_step:
        raise ValueError('mirror_view ramp requires 0 <= start_step < end_step')
    if step <= ramp_start_step:
        return 0.0
    if step >= ramp_end_step:
        return float(max_probability)
    progress = float(step - ramp_start_step) / float(
        ramp_end_step - ramp_start_step
    )
    return float(max_probability) * progress


def deterministic_mirror_decision(
    item: int,
    epoch: int,
    probability: float,
) -> bool:
    """Stateless per-sample view assignment, balanced at probability 0.5."""
    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    if abs(probability - 0.5) <= 1e-12:
        return bool((int(item) + int(epoch)) % 2)
    # Knuth-style integer mixing gives reproducible assignments without
    # coupling augmentation choices to NumPy's hint-sampling RNG stream.
    mixed = (
        int(item) * 2654435761 + int(epoch) * 1013904223 + 0x9E3779B9
    ) & 0xFFFFFFFF
    return (mixed / float(2 ** 32)) < float(probability)


def collate_fn(data_list: list[dict]) -> dict:
    tensors = {}
    non_tensors = {}

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if key not in tensors:
                    tensors[key] = []
                tensors[key].append(val)
            else:
                if key not in non_tensors:
                    non_tensors[key] = []
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    output = {}
    output.update(tensors)
    output.update(non_tensors)
    return output


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 prompt_key='prompt',
                 max_prompt_length=1024,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error',
                 is_eval=False,
                 config=None):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer

        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation

        self.is_eval = is_eval
        self.timestep = 1 # start from 1
        self.config = config

        adaptive_config = self.config.trainer.get('adaptive_hint', {}) if self.config is not None else {}
        self.adaptive_hint_enabled = bool(adaptive_config.get('enable', False)) and not self.is_eval
        paired_config = self.config.trainer.get('paired_hint', {}) if self.config is not None else {}
        self.paired_hint_enabled = bool(paired_config.get('enable', False)) and not self.is_eval
        mirror_config = self.config.trainer.get('mirror_view', {}) if self.config is not None else {}
        self.mirror_view_enabled = bool(mirror_config.get('enable', False)) and not self.is_eval
        self.mirror_view_paired = bool(mirror_config.get('paired', True))
        self.mirror_view_probability = float(mirror_config.get('random_probability', 0.5))
        self.mirror_view_schedule = str(mirror_config.get('schedule', 'constant'))
        self.mirror_view_ramp_start_step = int(mirror_config.get('ramp_start_step', 50))
        self.mirror_view_ramp_end_step = int(mirror_config.get('ramp_end_step', 100))
        self.mirror_view_balanced = bool(mirror_config.get('balanced', False))
        self.mirror_probability_scale = 1.0
        mirror_probability(
            self.timestep,
            self.mirror_view_schedule,
            self.mirror_view_probability,
            self.mirror_view_ramp_start_step,
            self.mirror_view_ramp_end_step,
        )
        if self.paired_hint_enabled and self.mirror_view_enabled:
            raise ValueError('paired_hint and mirror_view cannot be enabled together')
        forge_config = self.config.trainer.get('forge_anneal', {}) if self.config is not None else {}
        self.forge_anneal_enabled = bool(forge_config.get('enable', False)) and not self.is_eval
        self.forge_anneal_start_step = int(forge_config.get('start_step', 75))
        self.forge_anneal_end_step = int(forge_config.get('end_step', 100))
        if self.forge_anneal_enabled:
            forge_anneal_multiplier(
                self.timestep,
                self.forge_anneal_start_step,
                self.forge_anneal_end_step,
            )
        self.paired_min_hint_chunks = int(paired_config.get('min_hint_chunks', 1))
        self.paired_ate_bonus = float(paired_config.get('ate_bonus', 0.5))
        if self.paired_hint_enabled and not self.adaptive_hint_enabled:
            raise ValueError('paired_hint requires adaptive_hint.enable=True')
        if self.paired_min_hint_chunks < 1:
            raise ValueError('paired_hint.min_hint_chunks must be positive')
        if self.paired_ate_bonus < 0:
            raise ValueError('paired_hint.ate_bonus must be non-negative')
        self.adaptive_hint_warmup_steps = int(adaptive_config.get('warmup_steps', 40))
        self.sample_ema_alpha = float(adaptive_config.get('sample_ema_alpha', 0.30))
        self.hard_threshold = float(adaptive_config.get('hard_threshold', 0.25))
        self.easy_threshold = float(adaptive_config.get('easy_threshold', 0.75))
        self.frontier_uncertainty_threshold = float(
            adaptive_config.get('frontier_uncertainty_threshold', 0.50)
        )
        self.bandit_success_bonus = float(adaptive_config.get('success_bonus', 0.25))
        self.adaptive_difficulty_mode = str(adaptive_config.get('difficulty_mode', 'observed'))
        self.mastery_hint_tolerance = float(adaptive_config.get('mastery_hint_tolerance', 0.05))
        self.easy_max_hint_fraction = float(adaptive_config.get('easy_max_hint_fraction', 0.25))
        if self.adaptive_difficulty_mode not in ('observed', 'least_assisted'):
            raise ValueError('adaptive_hint.difficulty_mode must be observed or least_assisted')
        if not 0 < self.sample_ema_alpha <= 1:
            raise ValueError('trainer.adaptive_hint.sample_ema_alpha must be in (0, 1]')
        if not 0 <= self.hard_threshold < self.easy_threshold <= 1:
            raise ValueError('adaptive hint thresholds must satisfy 0 <= hard < easy <= 1')
        if not 0 <= self.frontier_uncertainty_threshold <= 1:
            raise ValueError('frontier uncertainty threshold must be in [0, 1]')
        if self.bandit_success_bonus < 0:
            raise ValueError('adaptive hint success bonus must be non-negative')
        self.hint_bandit = BudgetedHintBandit(BanditConfig(
            reward_ema_alpha=float(adaptive_config.get('reward_ema_alpha', 0.25)),
            exploration_coef=float(adaptive_config.get('exploration_coef', 0.10)),
            temperature=float(adaptive_config.get('temperature', 0.15)),
            prior_utility=float(adaptive_config.get('prior_utility', 0.50)),
        )) if self.adaptive_hint_enabled else None

        self._download()
        self._read_files_and_tokenize()
        train_batch_size = int(
            self.config.data.train_batch_size if self.config is not None else len(self)
        )
        self.steps_per_epoch = max(1, len(self) // max(1, train_batch_size))

    def _download(self):
        from verl.utils.fs import copy_local_path_from_hdfs
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        print(f'original dataset len: {len(self.dataframe)}')

        # filter out too long prompts
        tokenizer = self.tokenizer
        prompt_key = self.prompt_key

        # nvm if prompt is too long
        # self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(
        #     tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)) <= self.max_prompt_length,
        #                                                      axis=1)]

        print(f'filter dataset len: {len(self.dataframe)}')

        if self.forge_anneal_enabled:
            self.original_positions = np.asarray([
                position
                for position, info in enumerate(self.dataframe['extra_info'])
                if info.get('source') != 'forge'
            ], dtype=np.int64)
            forge_count = len(self.dataframe) - len(self.original_positions)
            if not len(self.original_positions) or forge_count <= 0:
                raise ValueError(
                    'forge anneal requires both original and source=forge rows'
                )

        if self.adaptive_hint_enabled:
            if 'hint' not in self.dataframe.columns:
                raise ValueError('adaptive hint scheduling requires a hint column')
            self.sample_success_ema = np.full(len(self.dataframe), np.nan, dtype=np.float64)
            self.sample_uncertainty_ema = np.full(len(self.dataframe), np.nan, dtype=np.float64)
            self.sample_visits = np.zeros(len(self.dataframe), dtype=np.int64)
            self.hint_chunk_counts = np.asarray([
                min(5, len(hint)) for hint in self.dataframe['hint']
            ], dtype=np.int64)
            sample_ids = [
                extra_info.get('index', position)
                for position, extra_info in enumerate(self.dataframe['extra_info'])
            ]
            if len(set(sample_ids)) != len(sample_ids):
                raise ValueError('adaptive hint scheduling requires unique sample indices')
            self.sample_id_to_position = {
                int(sample_id): position for position, sample_id in enumerate(sample_ids)
            }
            self.mastery_tracker = None
            if self.adaptive_difficulty_mode == 'least_assisted':
                self.mastery_tracker = LeastAssistedMasteryTracker(
                    len(self.dataframe),
                    MasteryConfig(
                        ema_alpha=self.sample_ema_alpha,
                        hint_tolerance=self.mastery_hint_tolerance,
                        hard_threshold=self.hard_threshold,
                        easy_threshold=self.easy_threshold,
                        frontier_uncertainty_threshold=self.frontier_uncertainty_threshold,
                        easy_max_hint_fraction=self.easy_max_hint_fraction,
                    ),
                )
            self._context_weights_cache = {}

    def __len__(self):
        return len(self.dataframe)

    def _base_hint_probability(self) -> float:
        if self.timestep > self.config.trainer.total_training_steps_hint:
            return 0.0
        current_timestep = self.timestep
        return float(
            self.config.trainer.lower_prob
            + 0.5 * (self.config.trainer.upper_prob - self.config.trainer.lower_prob)
            * (1 + np.cos(
                np.pi * current_timestep / self.config.trainer.total_training_steps_hint
            ))
        )

    def _sample_context(self, item: int) -> int:
        if self.mastery_tracker is not None:
            return self.mastery_tracker.context(item)
        success = self.sample_success_ema[item]
        if np.isnan(success):
            return 1
        uncertainty = self.sample_uncertainty_ema[item]
        if uncertainty >= self.frontier_uncertainty_threshold:
            return 1
        if success <= self.hard_threshold:
            return 0
        if success >= self.easy_threshold:
            return 2
        return 1

    def _context_weights(self, max_chunks: int) -> np.ndarray:
        cached = self._context_weights_cache.get(max_chunks)
        if cached is not None:
            return cached
        selected = np.flatnonzero(self.hint_chunk_counts == max_chunks)
        contexts = np.fromiter(
            (self._sample_context(int(item)) for item in selected),
            dtype=np.int64,
            count=len(selected),
        )
        weights = np.bincount(contexts, minlength=3).astype(np.float64)
        self._context_weights_cache[max_chunks] = weights
        return weights

    def update_adaptive_hint_feedback(
        self,
        indices,
        hint_arms,
        hint_contexts,
        hint_chunk_counts,
        hint_arm_probabilities,
        hint_budget_multipliers,
        hint_expected_fractions,
        correctness,
        hint_slice_props,
    ) -> dict:
        """Update sample mastery and bandit rewards from repeated rollouts."""
        if not self.adaptive_hint_enabled:
            return {}
        grouped = {}
        for index, arm, context, max_chunks, arm_probability, budget_multiplier, \
                expected_fraction, correct, hint_prop in zip(
            indices,
            hint_arms,
            hint_contexts,
            hint_chunk_counts,
            hint_arm_probabilities,
            hint_budget_multipliers,
            hint_expected_fractions,
            correctness,
            hint_slice_props,
        ):
            key = int(index)
            group = grouped.setdefault(key, {
                'arm': int(arm),
                'context': int(context),
                'max_chunks': int(max_chunks),
                'arm_probability': float(arm_probability),
                'budget_multiplier': float(budget_multiplier),
                'expected_fraction': float(expected_fraction),
                'correctness': [],
                'hint_prop': float(hint_prop),
            })
            if (group['arm'], group['context'], group['max_chunks']) != (
                int(arm), int(context), int(max_chunks)
            ):
                raise ValueError(f'inconsistent adaptive hint metadata for sample {key}')
            group['correctness'].append(float(correct))

        successes = []
        uncertainties = []
        utilities = []
        hint_props = []
        arm_probabilities = []
        budget_multipliers = []
        expected_fractions = []
        zero_actions = []
        full_actions = []
        context_hint_props = [[] for _ in range(3)]
        context_counts = np.zeros(3, dtype=np.int64)
        mastery_updates = []
        for index, group in grouped.items():
            if index not in self.sample_id_to_position:
                raise KeyError(f'unknown adaptive hint sample index: {index}')
            position = self.sample_id_to_position[index]
            success = float(np.mean(group['correctness']))
            uncertainty = 4.0 * success * (1.0 - success)
            utility = group_learning_utility(success, self.bandit_success_bonus)
            if self.mastery_tracker is not None:
                mastery_updates.append(self.mastery_tracker.observe(
                    position,
                    success_rate=success,
                    hint_fraction=group['hint_prop'],
                ))
            else:
                if self.sample_visits[position] == 0:
                    self.sample_success_ema[position] = success
                    self.sample_uncertainty_ema[position] = uncertainty
                else:
                    alpha = self.sample_ema_alpha
                    self.sample_success_ema[position] = (
                        (1.0 - alpha) * self.sample_success_ema[position] + alpha * success
                    )
                    self.sample_uncertainty_ema[position] = (
                        (1.0 - alpha) * self.sample_uncertainty_ema[position] + alpha * uncertainty
                    )
            self.sample_visits[position] += 1
            self.hint_bandit.observe(
                max_chunks=group['max_chunks'],
                context=group['context'],
                arm=group['arm'],
                group_utility=utility,
            )
            successes.append(success)
            uncertainties.append(uncertainty)
            utilities.append(utility)
            hint_props.append(group['hint_prop'])
            arm_probabilities.append(group['arm_probability'])
            budget_multipliers.append(group['budget_multiplier'])
            expected_fractions.append(group['expected_fraction'])
            zero_actions.append(group['arm'] == 0)
            full_actions.append(group['arm'] == group['max_chunks'])
            context_hint_props[group['context']].append(group['hint_prop'])
            context_counts[group['context']] += 1

        self._context_weights_cache.clear()

        metrics = {
            'adaptive_hint/group_success_mean': float(np.mean(successes)),
            'adaptive_hint/group_uncertainty_mean': float(np.mean(uncertainties)),
            'adaptive_hint/group_utility_mean': float(np.mean(utilities)),
            'adaptive_hint/actual_fraction': float(np.mean(hint_props)),
            'adaptive_hint/target_fraction': self._base_hint_probability(),
            'adaptive_hint/calibrated_expected_fraction': float(np.mean(expected_fractions)),
            'adaptive_hint/selected_arm_probability_mean': float(np.mean(arm_probabilities)),
            'adaptive_hint/budget_multiplier_mean': float(np.mean(budget_multipliers)),
            'adaptive_hint/zero_action_fraction': float(np.mean(zero_actions)),
            'adaptive_hint/full_action_fraction': float(np.mean(full_actions)),
            'adaptive_hint/context_hard_fraction': float(context_counts[0] / len(grouped)),
            'adaptive_hint/context_frontier_fraction': float(context_counts[1] / len(grouped)),
            'adaptive_hint/context_easy_fraction': float(context_counts[2] / len(grouped)),
            'adaptive_hint/hard_hint_fraction': float(np.mean(context_hint_props[0]))
                if context_hint_props[0] else 0.0,
            'adaptive_hint/frontier_hint_fraction': float(np.mean(context_hint_props[1]))
                if context_hint_props[1] else 0.0,
            'adaptive_hint/easy_hint_fraction': float(np.mean(context_hint_props[2]))
                if context_hint_props[2] else 0.0,
        }
        if self.mastery_tracker is not None:
            observed = self.mastery_tracker.update_counts > 0
            metrics.update({
                'adaptive_hint/mastery_update_fraction': float(np.mean(mastery_updates)),
                'adaptive_hint/mastery_low_assistance_coverage':
                    self.mastery_tracker.low_assistance_coverage(),
                'adaptive_hint/mastery_hint_floor_mean': float(np.mean(
                    self.mastery_tracker.hint_floor[observed]
                )) if np.any(observed) else 1.0,
            })
        return metrics

    def update_paired_hint_feedback(
        self,
        indices,
        treatments,
        hint_arms,
        hint_contexts,
        hint_chunk_counts,
        hint_arm_probabilities,
        hint_budget_multipliers,
        hint_expected_fractions,
        correctness,
        hint_slice_props,
        hint_forced,
    ) -> dict:
        """Update mastery from unhinted rollouts and the allocator from paired ATE."""
        if not self.paired_hint_enabled:
            return {}
        grouped = {}
        for index, treatment, arm, context, max_chunks, arm_probability, \
                budget_multiplier, expected_fraction, correct, hint_prop, forced in zip(
            indices, treatments, hint_arms, hint_contexts, hint_chunk_counts,
            hint_arm_probabilities, hint_budget_multipliers, hint_expected_fractions,
            correctness, hint_slice_props, hint_forced,
        ):
            key = int(index)
            group = grouped.setdefault(key, {
                'correctness': [[], []],
                'arm': None,
                'context': None,
                'max_chunks': None,
                'arm_probability': 1.0,
                'budget_multiplier': 0.0,
                'expected_fraction': 0.0,
                'hint_prop': 0.0,
                'forced': False,
            })
            treatment = int(treatment)
            if treatment not in (0, 1):
                raise ValueError(f'invalid paired treatment: {treatment}')
            group['correctness'][treatment].append(float(correct))
            if treatment == 1:
                metadata = (int(arm), int(context), int(max_chunks))
                previous = (group['arm'], group['context'], group['max_chunks'])
                if group['arm'] is not None and previous != metadata:
                    raise ValueError(f'inconsistent paired hint metadata for sample {key}')
                group.update({
                    'arm': metadata[0],
                    'context': metadata[1],
                    'max_chunks': metadata[2],
                    'arm_probability': float(arm_probability),
                    'budget_multiplier': float(budget_multiplier),
                    'expected_fraction': float(expected_fraction),
                    'hint_prop': float(hint_prop),
                    'forced': bool(forced),
                })

        unhinted_successes = []
        hinted_successes = []
        effects = []
        utilities = []
        active_groups = []
        forced_groups = []
        hint_props = []
        mastery_updates = []
        for index, group in grouped.items():
            if not group['correctness'][0] or not group['correctness'][1]:
                raise ValueError(f'paired sample {index} is missing a treatment arm')
            position = self.sample_id_to_position[index]
            unhinted = float(np.mean(group['correctness'][0]))
            hinted = float(np.mean(group['correctness'][1]))
            effect = hinted - unhinted
            uncertainty = 4.0 * unhinted * (1.0 - unhinted)
            base_utility = group_learning_utility(unhinted, self.bandit_success_bonus)
            utility = float(np.clip(
                base_utility + self.paired_ate_bonus * max(effect, 0.0), 0.0, 1.0
            ))
            mastery_updates.append(self.mastery_tracker.observe(
                position, success_rate=unhinted, hint_fraction=0.0
            ))
            self.sample_visits[position] += 1
            self.hint_bandit.observe(
                max_chunks=group['max_chunks'],
                context=group['context'],
                arm=group['arm'],
                group_utility=utility,
            )
            unhinted_successes.append(unhinted)
            hinted_successes.append(hinted)
            effects.append(effect)
            utilities.append(utility)
            active_groups.append(len(set(
                group['correctness'][0] + group['correctness'][1]
            )) > 1)
            forced_groups.append(group['forced'])
            hint_props.append(group['hint_prop'])

        self._context_weights_cache.clear()
        return {
            'paired_hint/unhinted_success_mean': float(np.mean(unhinted_successes)),
            'paired_hint/hinted_success_mean': float(np.mean(hinted_successes)),
            'paired_hint/ate_mean': float(np.mean(effects)),
            'paired_hint/positive_ate_fraction': float(np.mean(np.asarray(effects) > 0)),
            'paired_hint/negative_ate_fraction': float(np.mean(np.asarray(effects) < 0)),
            'paired_hint/effective_group_fraction': float(np.mean(active_groups)),
            'paired_hint/forced_nonzero_fraction': float(np.mean(forced_groups)),
            'paired_hint/response_hint_fraction': float(0.5 * np.mean(hint_props)),
            'adaptive_hint/group_success_mean': float(np.mean(unhinted_successes)),
            'adaptive_hint/group_uncertainty_mean': float(np.mean(
                4.0 * np.asarray(unhinted_successes) * (1.0 - np.asarray(unhinted_successes))
            )),
            'adaptive_hint/group_utility_mean': float(np.mean(utilities)),
            'adaptive_hint/actual_fraction': float(0.5 * np.mean(hint_props)),
            'adaptive_hint/target_fraction': float(0.5 * self._base_hint_probability()),
            'adaptive_hint/mastery_update_fraction': float(np.mean(mastery_updates)),
            'adaptive_hint/mastery_low_assistance_coverage':
                self.mastery_tracker.low_assistance_coverage(),
        }

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        forge_multiplier = 0.0
        forge_replaced = False
        if self.forge_anneal_enabled:
            forge_multiplier = forge_anneal_multiplier(
                self.timestep,
                self.forge_anneal_start_step,
                self.forge_anneal_end_step,
            )
            source = self.dataframe.iloc[item]['extra_info'].get('source')
            if source == 'forge' and np.random.random() >= forge_multiplier:
                item = int(np.random.choice(self.original_positions))
                forge_replaced = True

        row_dict = self.dataframe.iloc[item].to_dict()
        if self.forge_anneal_enabled:
            row_dict['forge_anneal_multiplier'] = forge_multiplier
            row_dict['forge_replaced'] = forge_replaced

        chat = row_dict.pop(self.prompt_key)

        hint_slice_prop = 0.0
        sliced_hint = ""
        hint_arm = 0
        hint_arm_probability = 1.0
        hint_context = 1
        hint_chunk_count = 0
        hint_budget_multiplier = 0.0
        hint_expected_fraction = 0.0
        paired_hint_forced = False
        if "hint" in row_dict.keys() and not self.is_eval:
            hint = row_dict.pop("hint")
            num = min(5, len(hint))
            hint_chunk_count = num
            separation = [round(_ / num * len(hint)) for _ in range(num + 1)]
            separation[-1] = len(hint)

            if self.config.trainer.uniform_sampling:
                slice_length = np.random.choice(separation)
            elif self.config.trainer.stage:
                if self.timestep > self.config.trainer.total_training_steps_hint:
                    slice_length = 0
                else:
                    current_timestep = self.timestep
                    slice_length = num - current_timestep // (self.config.trainer.total_training_steps_hint // (num + 1))
                    slice_length = np.clip(slice_length, 0, num)
                    slice_length = separation[slice_length]
            else:
                if self.timestep > self.config.trainer.total_training_steps_hint:
                    slice_length = 0
                else:
                    prob = self._base_hint_probability()
                    if self.adaptive_hint_enabled and self.timestep > self.adaptive_hint_warmup_steps:
                        hint_context = self._sample_context(item)
                        hint_arm, hint_arm_probability, hint_budget_multiplier, hint_expected_fraction = \
                            self.hint_bandit.sample(
                                max_chunks=num,
                                context=hint_context,
                                target_fraction=prob,
                                context_weights=self._context_weights(num),
                            )
                    else:
                        hint_arm = int(np.random.choice(
                            [0, 1], p=[1.0 - prob, prob], size=len(separation)-1
                        ).sum())
                        hint_expected_fraction = prob
                    if self.paired_hint_enabled and hint_arm < self.paired_min_hint_chunks:
                        hint_arm = min(self.paired_min_hint_chunks, num)
                        paired_hint_forced = True
                    slice_length = separation[hint_arm]
            sliced_hint = "".join(hint[:slice_length])

            hint_slice_prop = slice_length / len(hint)

        prompt_text = chat[0]['content']
        mirror_applied = False
        active_mirror_probability = mirror_probability(
            self.timestep,
            self.mirror_view_schedule,
            self.mirror_view_probability,
            self.mirror_view_ramp_start_step,
            self.mirror_view_ramp_end_step,
        ) * self.mirror_probability_scale
        mirror_selected = False
        if self.mirror_view_enabled and not self.mirror_view_paired:
            if self.mirror_view_balanced:
                epoch = (self.timestep - 1) // self.steps_per_epoch
                mirror_selected = deterministic_mirror_decision(
                    item, epoch, active_mirror_probability
                )
            else:
                mirror_selected = np.random.random() < active_mirror_probability
        if mirror_selected:
            numbers = [int(value) for value in row_dict['nums']]
            permutation = list(np.random.permutation(numbers))
            if len(permutation) > 1 and permutation == numbers:
                permutation = permutation[1:] + permutation[:1]
            replacement = "Using the numbers [" + ", ".join(map(str, permutation)) + "]"
            prompt_text, replacements = re.subn(
                r"Using the numbers \[[^\]]+\]", replacement, prompt_text, count=1
            )
            if replacements != 1:
                raise ValueError('mirror_view could not locate the Countdown number list')
            mirror_applied = True

        input_ids, attention_mask, hint_ids, hint_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_text,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation,
                                                                         hint_prompt = sliced_hint)

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]
        row_dict['hint_slice_prop'] = hint_slice_prop
        row_dict['hint_mask'] = hint_mask[0]
        row_dict['hint_arm'] = hint_arm
        row_dict['hint_arm_probability'] = hint_arm_probability
        row_dict['hint_context'] = hint_context
        row_dict['hint_chunk_count'] = hint_chunk_count
        row_dict['hint_budget_multiplier'] = hint_budget_multiplier
        row_dict['hint_expected_fraction'] = hint_expected_fraction
        row_dict['paired_hint_forced'] = paired_hint_forced
        row_dict['mirror_applied'] = mirror_applied
        row_dict['mirror_probability'] = active_mirror_probability

        if self.mirror_view_enabled and self.mirror_view_paired:
            numbers = [int(value) for value in row_dict['nums']]
            permutation = list(np.random.permutation(numbers))
            if len(permutation) > 1 and permutation == numbers:
                permutation = permutation[1:] + permutation[:1]
            replacement = "Using the numbers [" + ", ".join(map(str, permutation)) + "]"
            mirror_prompt, replacements = re.subn(
                r"Using the numbers \[[^\]]+\]",
                replacement,
                chat[0]['content'],
                count=1,
            )
            if replacements != 1:
                raise ValueError('mirror_view could not locate the Countdown number list')
            mirror_input_ids, mirror_attention_mask, _, mirror_hint_mask = \
                verl_F.tokenize_and_postprocess_data(
                    prompt=mirror_prompt,
                    tokenizer=self.tokenizer,
                    max_length=self.max_prompt_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    left_pad=True,
                    truncation=self.truncation,
                    hint_prompt=sliced_hint,
                )
            row_dict['mirror_input_ids'] = mirror_input_ids[0]
            row_dict['mirror_attention_mask'] = mirror_attention_mask[0]
            row_dict['mirror_position_ids'] = compute_position_id_with_mask(
                mirror_attention_mask
            )[0]
            row_dict['mirror_hint_mask'] = mirror_hint_mask[0]

        if self.paired_hint_enabled:
            unhinted_input_ids, unhinted_attention_mask, _, unhinted_hint_mask = \
                verl_F.tokenize_and_postprocess_data(
                    prompt=chat[0]['content'],
                    tokenizer=self.tokenizer,
                    max_length=self.max_prompt_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    left_pad=True,
                    truncation=self.truncation,
                    hint_prompt="",
                )
            row_dict['unhinted_input_ids'] = unhinted_input_ids[0]
            row_dict['unhinted_attention_mask'] = unhinted_attention_mask[0]
            row_dict['unhinted_position_ids'] = compute_position_id_with_mask(
                unhinted_attention_mask
            )[0]
            row_dict['unhinted_hint_mask'] = unhinted_hint_mask[0]

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat.tolist()

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict
