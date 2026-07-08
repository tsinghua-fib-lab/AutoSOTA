# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Meta class for handling the synthetic data generation."""

from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import pandas as pd

from synthetic.policy import (
    BaseEarlyStagePolicy,
    BaseJointPolicy,
    BaseLateStagePolicy,
    BaseSingleStagePolicy,
)

from .base import (
    BaseContextSampler,
    BaseDataGenerator,
    BaseLatentSampler,
    BaseRewardModel,
)
from .dataset import LoggedDataset
from .vectorial import (
    VectorialActionSet,
    VectorialContextSampler,
    VectorialLatentSampler,
    VectorialRewardModel,
)


@dataclass
class SyntheticDataGenerator(BaseDataGenerator):
    """Class for the synthetic data generation.

    Input
    ------
    action_set: ActionSet
        The action set object.

    context_sampler: ContextSampler
        The context sampler object.

    latent_sampler: LatentSampler
        The latent sampler object.

    reward_model: RewardModel
        The reward model object.

    """

    action_set: Optional[VectorialActionSet] = None
    context_sampler: Optional[BaseContextSampler] = None
    latent_sampler: Optional[BaseLatentSampler] = None
    reward_model: Optional[BaseRewardModel] = None

    def __post_init__(self):
        if (
            self.action_set is None
            and self.context_sampler is None
            and self.latent_sampler is None
            and self.reward_model is None
        ):
            action_set = VectorialActionSet(
                n_action=1000,
                dim_action_emb=10,
                device="cuda" if torch.cuda.is_available() else "cpu",
                random_seed=12345,
            )
            self.context_sampler = VectorialContextSampler(
                is_discrete=True,
                n_discrete_context=1000,
                dim_context=10,
                device="cuda" if torch.cuda.is_available() else "cpu",
                random_seed=12345,
            )
            self.latent_sampler = VectorialLatentSampler(
                is_discrete=True,
                n_discrete_latent=10,
                dim_context=self.context_sampler.dim_context,
                dim_action_emb=action_set.dim_action_emb,
                device="cuda" if torch.cuda.is_available() else "cpu",
                random_seed=12345,
            )
            self.reward_model = VectorialRewardModel(
                context_sampler=self.context_sampler,
                action_set=action_set,
                device="cuda" if torch.cuda.is_available() else "cpu",
                random_seed=12345,
            )
        elif (
            isinstance(self.action_set, VectorialActionSet)
            and isinstance(self.context_sampler, VectorialContextSampler)
            and isinstance(self.latent_sampler, VectorialLatentSampler)
            and isinstance(self.reward_model, VectorialRewardModel)
        ):
            assert (
                self.context_sampler.dim_context
                == self.latent_sampler.dim_context
                == self.reward_model.dim_context
            )
            assert (
                self.action_set.dim_action_emb
                == self.latent_sampler.dim_action_emb
                == self.reward_model.action_set.dim_action_emb
            )
        else:
            if self.action_set is None:
                raise ValueError("action_set is not given.")
            if self.context_sampler is None:
                raise ValueError("context_sampler is not given.")
            if self.latent_sampler is None:
                raise ValueError("latent_sampler is not given.")
            if self.reward_model is None:
                raise ValueError("reward_model is not given.")

        self.n_output_action = self.reward_model.n_output_action

    def sample_dataset(
        self,
        policy: Optional[Union[BaseJointPolicy, BaseSingleStagePolicy]] = None,
        early_stage_policy: Optional[BaseEarlyStagePolicy] = None,
        late_stage_policy: Optional[BaseLateStagePolicy] = None,
        n_candidate_action: int = 100,
        n_candidate_per_model: Optional[List[int]] = None,
        is_deterministic_policy: bool = False,
        is_deterministic_early_stage: bool = False,
        is_deterministic_late_stage: bool = False,
        n_samples: int = 10000,
    ) -> LoggedDataset:
        """Function to sample the dataset.

        Input
        ------
        policy: Optional[Union[BaseJointPolicy, BaseSingleStagePolicy]]
            The policy to evaluate. Either policy or (early_stage_policy, late_stage_policy) should be given.

        early_stage_policy: EarlyStagePolicy
            The early stage policy object. Either policy or (early_stage_policy, late_stage_policy) should be given.

        late_stage_policy: LateStagePolicy
            The late stage policy object. Either policy or (early_stage_policy, late_stage_policy) should be given.

        n_candidate_action: int, default=100
            The number of candidate actions to sample.

        n_candidate_per_model: List[int], default=None
            The number of candidate actions to sample per model.

        is_deterministic_policy: bool, default=False
            Whether the policy is deterministic.

        is_deterministic_early_stage: bool, default=False
            Whether the early stage policy is deterministic.

        is_deterministic_late_stage: bool, default=False
            Whether the late stage policy is deterministic.

        n_samples: int, default=10000
            The number of samples to generate.

        Output
        ------
        logged_dataset: LoggedDataset
            The logged dataset object.

        """
        if policy is None and (
            early_stage_policy is None
            or late_stage_policy is None
            or n_candidate_action is None
        ):
            raise ValueError(
                "policy or (early_stage_policy, late_stage_policy, n_candidate_action) should be given."
            )

        context, context_id = self.context_sampler.sample(n_samples)
        latent, latent_id = self.latent_sampler.sample(n_samples)

        if context_id is not None and latent_id is not None:
            factual_rewards = self.retrieve_factual_rewards(
                context_id=torch.arange(
                    self.context_sampler.n_discrete_context,
                    device=context.device,
                ),
            )
            factual_rewards = factual_rewards[context_id, latent_id, :]
        else:
            factual_rewards = None

        if policy is not None:
            output_actions = policy.sample(
                context=context,
                context_id=context_id,
                latent=latent,
                latent_id=latent_id,
                n_candidate_action=n_candidate_action,
                n_candidate_per_model=n_candidate_per_model,
                n_output_action=self.n_output_action,
                factual_rewards=factual_rewards,  # only when using the optimal policy
                is_deterministic=is_deterministic_policy,
            )

        else:  # early_stage_policy and late_stage_policy are given
            candidate_actions = early_stage_policy.sample(
                context=context,
                context_id=context_id,
                n_candidate_action=n_candidate_action,
                n_candidate_per_model=n_candidate_per_model,
                n_output_action=self.n_output_action,  # only when using the greedy algorithm
                factual_rewards=factual_rewards,  # only when using the optimal policy
                is_deterministic=is_deterministic_early_stage,
            )
            output_actions = late_stage_policy.sample(
                context=context,
                context_id=context_id,
                latent=latent,
                latent_id=latent_id,
                candidate_actions=candidate_actions,
                n_output_action=self.n_output_action,
                factual_rewards=factual_rewards,  # only when using the optimal policy
                is_deterministic=is_deterministic_late_stage,
            )

        reward, agg_reward = self.reward_model.sample(
            context=context,
            latent=latent,
            actions=output_actions,
        )

        logged_dataset = LoggedDataset(
            context=context,
            latent=latent,
            action=output_actions,
            reward=reward,
            agg_reward=agg_reward,
            factual_rewards=factual_rewards,
            context_id=context_id,
            latent_id=latent_id,
        )
        return logged_dataset

    def evaluate_policy_online(
        self,
        policy: Optional[Union[BaseJointPolicy, BaseSingleStagePolicy]] = None,
        early_stage_policy: Optional[BaseEarlyStagePolicy] = None,
        late_stage_policy: Optional[BaseLateStagePolicy] = None,
        n_candidate_action: int = 100,
        n_candidate_per_model: Optional[List[int]] = None,
        is_deterministic_policy: bool = False,
        is_deterministic_early_stage: bool = False,
        is_deterministic_late_stage: bool = False,
        n_samples: int = 10000,
    ) -> torch.Tensor:
        """Function to evaluate the policy via online rollouts.

        Input
        ------
        policy: Optional[Union[BaseJointPolicy, BaseSingleStagePolicy]]
            The policy to evaluate. Either policy or (early_stage_policy, late_stage_policy) should be given.

        early_stage_policy: Optional[EarlyStagePolicy]
            The early stage policy object. Either policy or (early_stage_policy, late_stage_policy) should be given.

        late_stage_policy: LateStagePolicy
            The late stage policy object. Either policy or (early_stage_policy, late_stage_policy) should be given.

        n_candidate_action: int, default=100
            The number of candidate actions to sample.

        n_candidate_per_model: List[int], default=None
            The number of candidate actions to sample per model.

        is_deterministic_policy: bool, default=False
            Whether the policy is deterministic.

        is_deterministic_early_stage: bool, default=False
            Whether the early stage policy is deterministic.

        is_deterministic_late_stage: bool, default=False
            Whether the late stage policy is deterministic.

        n_samples: int, default=10000
            The number of samples to generate.

        Output
        ------
        logged_dataset: LoggedDataset
            The logged dataset object.

        """
        logged_dataset = self.sample_dataset(
            policy=policy,
            early_stage_policy=early_stage_policy,
            late_stage_policy=late_stage_policy,
            n_candidate_action=n_candidate_action,
            n_candidate_per_model=n_candidate_per_model,
            is_deterministic_policy=is_deterministic_policy,
            is_deterministic_early_stage=is_deterministic_early_stage,
            is_deterministic_late_stage=is_deterministic_late_stage,
            n_samples=n_samples,
        )
        return logged_dataset.agg_reward.mean()

    def retrieve_factual_rewards(
        self,
        context_id: torch.Tensor,
        latent_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Retrieve factual rewards for all latents and actions (when using discrete latents).

        Input
        ------
        context_id: torch.Tensor (n_samples, )
            The context id tensor.

        latent_id: torch.Tensor (n_samples, ), default=None
            The latent id tensor. If None, all latents will be considered.

        Output
        ------
        factual_rewards: torch.Tensor, (n_sample, n_action) or (n_samples, n_latent, n_action)

        """
        n_samples = context_id.shape[0]
        n_latent = self.latent_sampler.n_discrete_latent
        n_action = self.reward_model.action_set.n_action

        context = self.context_sampler.retrieve_embeddings(
            context_id=context_id
        )

        if latent_id is None:
            factual_rewards = torch.zeros(
                (n_samples, n_latent, n_action), device=context_id.device
            )
            for latent_id_ in range(n_latent):
                latent_id = torch.full(
                    (n_samples,), latent_id_, device=context_id.device
                )

                latent = self.latent_sampler.retrieve_embeddings(
                    latent_id=latent_id
                )
                factual_rewards[:, latent_id_, :] = (
                    self.reward_model.all_action_expected_reward(
                        context=context,
                        latent=latent,
                    )
                )
        else:
            latent = self.latent_sampler.retrieve_embeddings(latent_id=latent_id)
            factual_rewards = self.reward_model.all_action_expected_reward(
                context=context,
                latent=latent,
            )

        return factual_rewards


@dataclass
class KuaiRecDataGenerator(BaseDataGenerator):
    """Class for the semi-synthetic data generation with KuaiRec (small) dataset.

    Input
    ------
    dataset_path: str
        Path to the KuaiRec dataset.

    reward_std: float, default=0.1
        Noise level of the reward.

    n_output_action: int
        The number of output actions.

    ranking_weight: Optional[torch.Tensor], shape (n_output_action, )
        The ranking weight for each action.

    random_seed: int, default=None
        Random seed

    """

    dataset_path: str
    reward_std: float = 0.1
    n_output_action: int = 1
    ranking_weight: Optional[torch.Tensor] = None 
    random_seed: Optional[int] = None

    def __post_init__(self):
        torch.manual_seed(self.random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)

        if self.ranking_weight is not None and len(self.ranking_weight) != self.n_output_action:
            raise ValueError("The length of the ranking weight must be the same with n_output_action.")
        
        # load KuaiRec data
        print("Loading small (dense) matrix...")
        small_matrix = pd.read_csv(self.dataset_path)
        print("done!")

        # clip values
        small_matrix["watch_ratio"] = small_matrix["watch_ratio"].clip(lower=0, upper=10)

        # re-index user and content ids
        small_matrix["user_id"] = pd.factorize(small_matrix["user_id"])[0]
        small_matrix["video_id"] = pd.factorize(small_matrix["video_id"])[0]

        # from dataframe format to matrix
        small_matrix = small_matrix.pivot(index="user_id", columns="video_id", values="watch_ratio").fillna(0)
        self.expected_rewards = torch.tensor(small_matrix.values, dtype=torch.float32)

        self.n_users, self.n_items = self.expected_rewards.shape

    def sample_reward(
        self,
        context_id: torch.Tensor,
        action_ids: torch.Tensor,
    ):
        """Calculate expected reward.
        
        context_id: torch.Tensor, shape (n_samples, )
            Context (user) id.

        action_ids: torch.Tensor, shape (n_samples, n_output_actions)
            Action (item) id.

        Output
        ------
        reward: torch.Tensor, shape (n_samples, n_output_actions)
            Sampled reward.

        agg_reward: torch.Tensor, shape (n_samples, )
            Sampled reward aggregated within the ranking.
        
        """
        expected_reward = self.expected_rewards[context_id, action_ids]  # (n_samples, n_output_actions)
        reward_noise = torch.normal(torch.zeros_like(expected_reward), torch.full_like(expected_reward, self.reward_std))
        reward = expected_reward + reward_noise
        
        if self.ranking_weight is not None:
            agg_reward = (
                expected_reward * self.ranking_weight.unsqueeze(0)
            ).sum(keepdim=True)
        else:
            agg_reward = expected_reward.mean(dim=-1, keepdim=True)

        return reward, agg_reward

    def sample_dataset(
        self,
        policy: Optional[Union[BaseJointPolicy, BaseSingleStagePolicy]] = None,
        early_stage_policy: Optional[BaseEarlyStagePolicy] = None,
        late_stage_policy: Optional[BaseLateStagePolicy] = None,
        n_candidate_action: int = 100,
        n_candidate_per_model: Optional[List[int]] = None,
        is_deterministic_policy: bool = False,
        is_deterministic_early_stage: bool = False,
        is_deterministic_late_stage: bool = False,
        n_samples: int = 10000,
    ) -> LoggedDataset:
        """Function to sample the dataset.

        Input
        ------
        policy: Optional[Union[BaseJointPolicy, BaseSingleStagePolicy]]
            The policy to evaluate. Either policy or (early_stage_policy, late_stage_policy) should be given.

        early_stage_policy: EarlyStagePolicy
            The early stage policy object. Either policy or (early_stage_policy, late_stage_policy) should be given.

        late_stage_policy: LateStagePolicy
            The late stage policy object. Either policy or (early_stage_policy, late_stage_policy) should be given.

        n_candidate_action: int, default=100
            The number of candidate actions to sample.

        n_candidate_per_model: List[int], default=None
            The number of candidate actions to sample per model.

        is_deterministic_policy: bool, default=False
            Whether the policy is deterministic.

        is_deterministic_early_stage: bool, default=False
            Whether the early stage policy is deterministic.

        is_deterministic_late_stage: bool, default=False
            Whether the late stage policy is deterministic.

        n_samples: int, default=10000
            The number of samples to generate.

        Output
        ------
        logged_dataset: LoggedDataset
            The logged dataset object.

        """
        if policy is None and (
            early_stage_policy is None
            or late_stage_policy is None
            or n_candidate_action is None
        ):
            raise ValueError(
                "policy or (early_stage_policy, late_stage_policy, n_candidate_action) should be given."
            )

        context_id = torch.randint(self.n_users, (n_samples, ))
        latent_id = torch.zeros((n_samples, ))
        factual_rewards = self.expected_rewards[context_id]

        if policy is not None:
            output_actions = policy.sample(
                context_id=context_id,
                latent_id=latent_id,
                n_candidate_action=n_candidate_action,
                n_candidate_per_model=n_candidate_per_model,
                n_output_action=self.n_output_action,
                factual_rewards=factual_rewards,  # only when using the optimal policy
                is_deterministic=is_deterministic_policy,
            )

        else:  # early_stage_policy and late_stage_policy are given
            candidate_actions = early_stage_policy.sample(
                context_id=context_id,
                n_candidate_action=n_candidate_action,
                n_candidate_per_model=n_candidate_per_model,
                n_output_action=self.n_output_action,  # only when using the greedy algorithm
                factual_rewards=factual_rewards,  # only when using the optimal policy
                is_deterministic=is_deterministic_early_stage,
            )
            output_actions = late_stage_policy.sample(
                context_id=context_id,
                latent_id=latent_id,
                candidate_actions=candidate_actions,
                n_output_action=self.n_output_action,
                factual_rewards=factual_rewards,  # only when using the optimal policy
                is_deterministic=is_deterministic_late_stage,
            )

        reward, agg_reward = self.sample_reward(
            context_id=context_id,
            action_ids=output_actions,
        )

        logged_dataset = LoggedDataset(
            context=context_id,
            latent=latent_id,
            action=output_actions,
            reward=reward,
            agg_reward=agg_reward,
            factual_rewards=factual_rewards,
            context_id=context_id,
            latent_id=latent_id,
        )
        return logged_dataset

    def evaluate_policy_online(
        self,
        policy: Optional[Union[BaseJointPolicy, BaseSingleStagePolicy]] = None,
        early_stage_policy: Optional[BaseEarlyStagePolicy] = None,
        late_stage_policy: Optional[BaseLateStagePolicy] = None,
        n_candidate_action: int = 100,
        n_candidate_per_model: Optional[List[int]] = None,
        is_deterministic_policy: bool = False,
        is_deterministic_early_stage: bool = False,
        is_deterministic_late_stage: bool = False,
        n_samples: int = 10000,
    ) -> torch.Tensor:
        """Function to evaluate the policy via online rollouts.

        Input
        ------
        policy: Optional[Union[BaseJointPolicy, BaseSingleStagePolicy]]
            The policy to evaluate. Either policy or (early_stage_policy, late_stage_policy) should be given.

        early_stage_policy: Optional[EarlyStagePolicy]
            The early stage policy object. Either policy or (early_stage_policy, late_stage_policy) should be given.

        late_stage_policy: LateStagePolicy
            The late stage policy object. Either policy or (early_stage_policy, late_stage_policy) should be given.

        n_candidate_action: int, default=100
            The number of candidate actions to sample.

        n_candidate_per_model: List[int], default=None
            The number of candidate actions to sample per model.

        is_deterministic_policy: bool, default=False
            Whether the policy is deterministic.

        is_deterministic_early_stage: bool, default=False
            Whether the early stage policy is deterministic.

        is_deterministic_late_stage: bool, default=False
            Whether the late stage policy is deterministic.

        n_samples: int, default=10000
            The number of samples to generate.

        Output
        ------
        logged_dataset: LoggedDataset
            The logged dataset object.

        """
        logged_dataset = self.sample_dataset(
            policy=policy,
            early_stage_policy=early_stage_policy,
            late_stage_policy=late_stage_policy,
            n_candidate_action=n_candidate_action,
            n_candidate_per_model=n_candidate_per_model,
            is_deterministic_policy=is_deterministic_policy,
            is_deterministic_early_stage=is_deterministic_early_stage,
            is_deterministic_late_stage=is_deterministic_late_stage,
            n_samples=n_samples,
        )
        return logged_dataset.agg_reward.mean()
