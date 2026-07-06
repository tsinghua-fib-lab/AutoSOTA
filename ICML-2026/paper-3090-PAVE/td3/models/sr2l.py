import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import TD3
from stable_baselines3 import DDPG
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

from torch.autograd.functional import jacobian

"""
"sr2l" : dict(
    ant = dict(
        adv_lambda = 0.3,
        adv_epsilon = 0.05,
        adv_steps = 10,
        adv_alpha = 0.01),      # 0.2 * 0.05
    hopper = dict(
        adv_lambda = 0.3,
        adv_epsilon = 0.05,
        adv_steps = 10,
        adv_alpha = 0.01),
    humanoid = dict(
        adv_lambda = 0.3,
        adv_epsilon = 0.05,
        adv_steps = 10,
        adv_alpha = 0.01),
    lunar = dict(
        adv_lambda = 0.03,
        adv_epsilon = 0.01,
        adv_steps = 10,
        adv_alpha = 0.002),     # 0.2 * 0.01
    reacher = dict(
        adv_lambda = 0.1,
        adv_epsilon = 0.02,
        adv_steps = 10,
        adv_alpha = 0.004),     # 0.2 * 0.02
    pendulum = dict(
        adv_lambda = 0.03,
        adv_epsilon = 0.01,
        adv_steps = 10,
        adv_alpha = 0.002),
    walker = dict(
        adv_lambda = 0.3,
        adv_epsilon = 0.05,
        adv_steps = 10,
        adv_alpha = 0.01),
)
"""

class SR2L_A(CustomTD3):

    def __init__(
        self,
        policy: Union[str, type[TD3Policy]],
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
        # S2RL parameters
        adv_lambda: float = 0.3,
        adv_epsilon: float = 0.05,
        adv_steps: int = 3,
        adv_alpha: float = 0.005,
    ):
        # S2RL hyperparameters
        self.adv_lambda = adv_lambda
        self.adv_epsilon = adv_epsilon
        self.adv_steps = adv_steps
        self.adv_alpha = adv_alpha
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
        adv_reg_losses = []
        avg_deltas = []
        for _ in range(gradient_steps):
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
                # Compute actor loss
                actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()

                # --- SR2L-A START ---
                if self.adv_lambda > 0:
                    obs = replay_data.observations

                    # PGD on δ to maximize D_J
                    delta = th.zeros_like(obs).uniform_(-self.adv_epsilon, self.adv_epsilon)
                    mu_anchor = self.actor(obs).detach()  # <== 루프 밖
                    for _k in range(self.adv_steps):
                        delta.requires_grad_(True)
                        mu_adv_tmp = self.actor(obs + delta)
                        inner = F.mse_loss(mu_adv_tmp, mu_anchor, reduction="mean")
                        (g_delta,) = th.autograd.grad(inner, delta, create_graph=False, retain_graph=False)
                        delta = (delta + self.adv_alpha * g_delta.sign()).clamp(-self.adv_epsilon, self.adv_epsilon).detach()

                    delta_adv = delta
                    avg_deltas.append(delta_adv.abs().mean().item())

                    # final
                    mu_clean = self.actor(obs)
                    mu_adv = self.actor(obs + delta_adv)
                    adv_reg_loss = self.adv_lambda * F.mse_loss(mu_adv, mu_clean, reduction="mean")
                    actor_loss = actor_loss + adv_reg_loss
                    adv_reg_losses.append(adv_reg_loss.item())
                # --- SR2L-A END ---

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

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(adv_reg_losses) > 0:
            self.logger.record("train/adv_reg_loss", np.mean(adv_reg_losses))
        self.logger.record("train/delta_size", np.mean(avg_deltas))


class SR2L_C(CustomTD3):

    def __init__(
        self,
        policy: Union[str, type[TD3Policy]],
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
        # S2RL parameters
        adv_lambda: float = 0.1,
        adv_epsilon: float = 0.01,
        adv_steps: int = 3,
        adv_alpha: float = 0.005,
    ):
        # S2RL hyperparameters
        self.adv_lambda = adv_lambda
        self.adv_epsilon = adv_epsilon
        self.adv_steps = adv_steps
        self.adv_alpha = adv_alpha
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
        adv_reg_losses = []
        avg_deltas = []
        for _ in range(gradient_steps):
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

            # --- SR2L-C START (critic smoothness penalty inside train) ---
            if self.adv_lambda > 0:
                s = replay_data.observations
                a_fixed = replay_data.actions.detach()  # inner PGD에서 고정

                # PGD로 δ_adv = argmax || Q(s,a) - Q(s+δ,a) ||^2 찾기 (||δ||_∞ ≤ ε)
                delta = th.zeros_like(s).uniform_(-self.adv_epsilon, self.adv_epsilon)
                for _k in range(self.adv_steps):
                    delta.requires_grad_(True)
                    q1_base = self.critic(s, a_fixed)[0]
                    q1_adv  = self.critic(s + delta, a_fixed)[0]
                    inner = F.mse_loss(q1_adv, q1_base.detach(), reduction="mean")
                    (g_delta,) = th.autograd.grad(inner, delta, create_graph=False, retain_graph=False)
                    delta = (delta + self.adv_alpha * g_delta.sign()).clamp(-self.adv_epsilon, self.adv_epsilon).detach()

                delta_adv = delta
                avg_deltas.append(delta_adv.abs().mean().item())

                # 최종 패널티(critic 파라미터로만 그래디언트 흐름)
                q1_b = self.critic(s, replay_data.actions)[0]
                q1_a = self.critic(s + delta_adv, replay_data.actions)[0]
                adv_reg_loss = self.adv_lambda * (
                    F.mse_loss(q1_a, q1_b, reduction="mean")
                )
                critic_loss = critic_loss + adv_reg_loss
                adv_reg_losses.append(adv_reg_loss.item())
            # --- SR2L-C END ---

            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()
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

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(adv_reg_losses) > 0:
            self.logger.record("train/adv_reg_loss", np.mean(adv_reg_losses))
        self.logger.record("train/delta_size", np.mean(avg_deltas))