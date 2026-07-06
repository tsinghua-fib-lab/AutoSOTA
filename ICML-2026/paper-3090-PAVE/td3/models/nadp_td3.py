import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import TD3
from custom_td3 import CustomTD3
from stable_baselines3.td3.policies import TD3Policy, Actor
from typing import Any, ClassVar, Optional, TypeVar, Union, Tuple

import numpy as np
import torch as th
from torch import nn
from gymnasium import spaces
from torch.nn import functional as F
import math
from copy import deepcopy

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, PyTorchObs
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)


class DAActor(Actor):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: list[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            net_arch=net_arch,
            features_extractor=features_extractor,
            features_dim=features_dim,
            activation_fn=activation_fn,
            normalize_images=normalize_images,
        )

        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn

        action_dim = get_action_dim(self.action_space)
        latent_pi_net = create_mlp(features_dim, -1, net_arch, activation_fn)
        self.latent_pi = nn.Sequential(*latent_pi_net)
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim
        # actor_net = create_mlp(features_dim, action_dim, net_arch, activation_fn, squash_output=True)
        # Deterministic action
        self.mu = nn.Sequential(
            nn.Linear(last_layer_dim, action_dim),
            nn.Tanh()
        )
        self.mu_next = nn.Sequential(
            nn.Linear(last_layer_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        # assert deterministic, 'The TD3 actor only outputs deterministic actions'
        features = self.extract_features(obs, self.features_extractor)
        latent_pi = self.latent_pi(features)
        return self.mu(latent_pi)
    
    def predict_next(self, obs: PyTorchObs, deterministic: bool = True) -> th.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        latent_pi = self.latent_pi(features)
        return self.mu_next(latent_pi)

class DAPolicy(TD3Policy):
    actor: DAActor

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule=lr_schedule)

        self.next_actor_target = deepcopy(self.actor)
        for p in self.next_actor_target.parameters():
            p.requires_grad = False

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> DAActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return DAActor(**actor_kwargs).to(self.device)
  
    def _predict_next(self, observation: PyTorchObs, deterministic: bool = True) -> th.Tensor:
        return self.actor.predict_next(observation, deterministic)
    
    def _predict_next_target(self, observation: PyTorchObs, deterministic: bool = True) -> th.Tensor:
        with th.no_grad():
            return self.next_actor_target.predict_next(observation, deterministic)
        
    def _polyak_update_targets(self, tau: float):
        # Polyak averaging: θ_target ← (1−τ)·θ_target + τ·θ
        with th.no_grad():
            # next actor
            for p, p_targ in zip(self.actor.parameters(),
                                  self.next_actor_target.parameters()):
                p_targ.data.mul_(1.0 - tau)
                p_targ.data.add_(tau * p.data)


class DATD3(CustomTD3):
    policy: DAPolicy
    def __init__(
        self,
        policy: Union[str, type[DAPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
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
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        lam_smooth = 0.5,
        lam_predict = 1.0,
        da_tau = 0.01,
        da_target_update_interval : int = 2,
        da_update_delay : int = 0,
        l1_beta = 0.1
    ):
        self.da_lamL = lam_smooth # lambda Lipschitz
        self.da_lamP = lam_predict # lambda predict
        self.da_tau = da_tau
        self.da_target_update_interval = da_target_update_interval
        self._p_updates = 0
        self.da_update_delay = da_update_delay
        self.l1_beta = l1_beta
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
            policy_delay=policy_delay,
            target_policy_noise=target_policy_noise,
            target_noise_clip=target_noise_clip,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        next_action_predict_losses = []
        for gradient_step in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute NADP loss 
                d_observations = replay_data.observations
                d_next_observations = replay_data.next_observations

                d_predict_next_actions = self.policy._predict_next_target(d_observations, True).detach()
                d_next_actions = self.policy._predict(d_next_observations, True)

                predict_loss = 0.5 * F.mse_loss(d_next_actions, d_predict_next_actions)
                # predict_loss = F.l1_loss(d_next_actions, d_predict_next_actions)
                next_action_predict_losses.append(predict_loss.item())

                d_predict_next_actions_train = self.policy._predict_next(d_observations, True)
                d_next_actions_target = self.policy._predict(d_next_observations, True).detach()

                next_loss = F.mse_loss(d_predict_next_actions_train, d_next_actions_target)
                # next_loss = F.l1_loss(d_predict_next_actions_train, d_next_actions_target)


                # Compute actor loss
                actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean() + predict_loss * self.da_lamL + next_loss * self.da_lamP
                if self._n_updates < self.da_update_delay:
                    actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean() + next_loss * self.da_lamP
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

            # Update next actor target networks
            if self._n_updates % self.da_target_update_interval == 0:
                self.policy._polyak_update_targets(self.da_tau)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/next_action_predict_loss", np.mean(next_action_predict_losses))


class DAL1TD3(CustomTD3):
    policy: DAPolicy
    def __init__(
        self,
        policy: Union[str, type[DAPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
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
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        lam_smooth = 0.5,
        lam_predict = 1.0,
        da_tau = 0.01,
        da_target_update_interval : int = 2,
        da_update_delay : int = 0,
        l1_beta = 0.05
    ):
        self.da_lamL = lam_smooth # lambda Lipschitz
        self.da_lamP = lam_predict # lambda predict
        self.da_tau = da_tau
        self.da_target_update_interval = da_target_update_interval
        self._p_updates = 0
        self.da_update_delay = da_update_delay
        self.l1_beta = l1_beta
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
            policy_delay=policy_delay,
            target_policy_noise=target_policy_noise,
            target_noise_clip=target_noise_clip,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        next_action_predict_losses = []
        for gradient_step in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute NADP loss 
                d_observations = replay_data.observations
                d_next_observations = replay_data.next_observations

                d_predict_next_actions = self.policy._predict_next_target(d_observations, True).detach()
                d_next_actions = self.policy._predict(d_next_observations, True)

                # predict_loss = 0.5 * F.mse_loss(d_next_actions, d_predict_next_actions)
                predict_loss = F.l1_loss(d_next_actions, d_predict_next_actions)
                next_action_predict_losses.append(predict_loss.item())

                d_predict_next_actions_train = self.policy._predict_next(d_observations, True)
                d_next_actions_target = self.actor_target(d_next_observations).detach()

                # next_loss = F.mse_loss(d_predict_next_actions_train, d_next_actions_target)
                next_loss = F.l1_loss(d_predict_next_actions_train, d_next_actions_target)


                # Compute actor loss
                actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean() + predict_loss * self.da_lamL + next_loss * self.da_lamP
                if self._n_updates < self.da_update_delay:
                    actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean() + next_loss * self.da_lamP
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

            # Update next actor target networks
            if self._n_updates % self.da_target_update_interval == 0:
                self.policy._polyak_update_targets(self.da_tau)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/next_action_predict_loss", np.mean(next_action_predict_losses))