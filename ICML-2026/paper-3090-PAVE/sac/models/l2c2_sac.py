import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import SAC
from custom_sac import CustomSAC
from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from torch import nn
from gymnasium import spaces
from torch.nn import functional as F
import math
import copy
from torch.func import functional_call

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic, BaseModel
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, PyTorchObs
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.sac.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.preprocessing import get_action_dim

LOG_STD_MAX = 2
LOG_STD_MIN = -20

SelfSAC = TypeVar("SelfSAC", bound="SAC")

class EntBeta(BaseModel):
    features_extractor: BaseFeaturesExtractor

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: list[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        self.share_features_extractor = share_features_extractor
        beta_list = create_mlp(features_dim, 1, net_arch, activation_fn, squash_output=True)
        self.latent_beta = nn.Sequential(*beta_list)
        self.latent_beta_target = nn.Sequential(*beta_list)
        self.target_distance = 1
        self.sync_online()

    def forward(self, obs: th.Tensor) -> th.Tensor:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        return self.latent_beta(features)
    
    def forward_target(self, obs: th.Tensor) -> th.Tensor:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        return self.latent_beta_target(features)
    
    def target_update(self, tau: float = 0.02) -> None:
        with th.no_grad():
            # 온라인 β 파라미터와 타깃 β 파라미터를 한 쌍씩 순회
            for param, target_param in zip(
                self.latent_beta.parameters(),
                self.latent_beta_target.parameters()
            ):
                # target ← (1 - τ) * target + τ * online
                target_param.data.mul_(1.0 - tau)
                target_param.data.add_(tau * param.data)
            self.target_distance = (1-tau) * self.target_distance

    def sync_online(self) -> None:
        with th.no_grad():
            # 온라인 β 파라미터와 타깃 β 파라미터를 한 쌍씩 순회
            for param, target_param in zip(
                self.latent_beta.parameters(),
                self.latent_beta_target.parameters()
            ):
                param.data.copy_(target_param.data)

    def reset_target_distance(self):
        self.target_distance = 1



class EntPolicy(SACPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            use_sde=use_sde,
            log_std_init=log_std_init,
            use_expln=use_expln,
            clip_mean=clip_mean,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            n_critics=n_critics,
            share_features_extractor=share_features_extractor,
        )

        self.beta_kwargs = self.net_args.copy()
        self.beta_kwargs.update({
            "share_features_extractor": share_features_extractor,
        })
        self._build_beta(lr_schedule)


    def _build_beta(self, lr_schedule: Schedule):
        if self.share_features_extractor:
            self.beta = self.make_beta(features_extractor=self.actor.features_extractor)
        else :
            self.beta = self.make_beta(features_extractor=None)

        self.beta.optimizer = self.optimizer_class(
            self.beta.parameters(),
            lr = lr_schedule(1),
            **self.optimizer_kwargs
        )
    
    def make_beta(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> EntBeta:
        beta_kwargs = self._update_features_extractor(self.beta_kwargs, features_extractor)
        return EntBeta(**beta_kwargs).to(self.device)
        

class L2C2SAC(CustomSAC):
    def __init__(
        self,
        policy: Union[str, type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        l2c2_sigma = 1.0,
        l2c2_lamD = 0.01,
        l2c2_lamU = 1.0,
        l2c2_beta = 0.1
    ):
        self.l2c2_sigma = l2c2_sigma
        self.l2c2_lamD = l2c2_lamD 
        self.l2c2_lamU = l2c2_lamU
        self.l2c2_beta = l2c2_beta
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            ent_coef=ent_coef,
            target_update_interval=target_update_interval,
            target_entropy=target_entropy,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # L2C2 Loss
            observations = replay_data.observations.clone()
            next_observations = replay_data.next_observations.clone()

            sigma = self.l2c2_sigma
            ulam = self.l2c2_lamU
            dlam = self.l2c2_lamD
            beta = self.l2c2_beta
            epsilon = sigma * dlam / (ulam - sigma * dlam)
            width = sigma + (sigma - 1) * epsilon
            u = th.rand_like(observations) * 2 * width - width
            s_bar = observations + (next_observations - observations) * u

            diff = (s_bar - observations) / (next_observations - observations + 1e-8)
            d_us = th.norm(diff, p=float("inf"), dim=-1).detach() + epsilon

            # 정책 및 가치 함수의 출력 계산
            num_mc = 4
            hellinger_terms = []

            self.policy.observation_space

            for _ in range(num_mc):
                a_s, log_p_s = self.policy.actor.action_log_prob(observations)
                a_s_bar, log_q_sbar = self.policy.actor.action_log_prob(s_bar)

                sqrt_ratio = th.exp(0.5 * (log_q_sbar - log_p_s).clamp(min=-10, max=10))
                hellinger_terms.append(sqrt_ratio)
            
            
            v_s_c = th.cat(self.critic(observations, replay_data.actions), dim=1)
            v_s_bar_c = th.cat(self.critic(s_bar, replay_data.actions), dim=1)
            v_s, _ = th.min(v_s_c, dim=1, keepdim=True)
            v_s_bar, _ = th.min(v_s_bar_c, dim=1, keepdim=True)
            value_distance = th.norm(v_s-v_s_bar, 2.0, dim=-1)/2
            
            hellinger_terms_tensor = th.stack(hellinger_terms, dim=1)
            mean_sqrt_ratio = hellinger_terms_tensor.mean(dim=1)
            d_pi_per_sample = (1.0 - mean_sqrt_ratio).clamp(min=0.0) 

            # L2C2 정규화 손실
            lambda_pi = epsilon * ulam / d_us  # 정책 정규화 가중치
            lambda_v = beta * lambda_pi  / d_us  # 가치 정규화 가중치


            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values) + th.mean(lambda_v * value_distance)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks

            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean() + th.mean(lambda_pi * d_pi_per_sample)
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
