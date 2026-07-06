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

from torch.autograd.functional import jacobian


class AQFR_FL_TD3(CustomTD3):
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
        # AQFR parameters
        adv_lambda: float = 0.1,
        adv_epsilon: float = 0.01,
        adv_steps: int = 10,
        adv_alpha: float = 0.005,
    ):
        # AQFR hyperparameters
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
        critic_diffs =[]
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

            # --- AQFR Implementation Start ---
            # 1) Adversary finds the worst-case perturbation delta_adv
            k_max, eps, alpha_init = self.adv_steps, self.adv_epsilon, self.adv_alpha
            s_t = replay_data.observations.detach()
            delta_1 = th.zeros_like(s_t).requires_grad_(True)
            delta_2 = th.zeros_like(s_t).requires_grad_(True)
            a_t = replay_data.actions.detach()

            # (Optional but recommended) Pre-compute the reference ∂Q/∂a at the clean state once
            a_t_ref = a_t.clone().requires_grad_(True)
            q1_orig_ref, _ = self.critic(s_t + delta_1, a_t_ref)
            _, q2_orig_ref = self.critic(s_t + delta_2, a_t_ref)

            grad_q1_k = th.autograd.grad(q1_orig_ref.clone().sum(), delta_1, retain_graph=True)[0].detach()
            grad_q2_k = th.autograd.grad(q2_orig_ref.clone().sum(), delta_2, retain_graph=True)[0].detach()         

            delta_1 = th.clamp(delta_1 + alpha_init * grad_q1_k, -eps, eps)
            delta_2 = th.clamp(delta_2 + alpha_init * grad_q2_k, -eps, eps)
            for k in range(k_max - 1):
                q1_adv, _ = self.critic(s_t + delta_1, a_t_ref)
                _, q2_adv = self.critic(s_t + delta_2, a_t_ref)
                grad_q1_k = th.autograd.grad(q1_adv.sum(), delta_1, retain_graph=False)[0].detach()
                grad_q2_k = th.autograd.grad(q2_adv.sum(), delta_2, retain_graph=False)[0].detach()
                # critic 1
                delta_1 = th.clamp(delta_1 + alpha_init * grad_q1_k, -eps, eps)
                # critic 2
                delta_2 = th.clamp(delta_2 + alpha_init * grad_q2_k, -eps, eps)


            grad_q1_orig = th.autograd.grad(q1_orig_ref.sum(), a_t_ref, retain_graph=True, create_graph=True)[0]
            grad_q2_orig = th.autograd.grad(q2_orig_ref.sum(), a_t_ref, retain_graph=True, create_graph=True)[0]

            q1_delta, _ = self.critic(s_t + delta_1, a_t_ref)
            _, q2_delta = self.critic(s_t + delta_2, a_t_ref)

            grad_q1_delta = th.autograd.grad(q1_delta.sum(), a_t_ref, retain_graph=True)[0].detach()
            grad_q2_delta = th.autograd.grad(q2_delta.sum(), a_t_ref, retain_graph=True)[0].detach()

            adv_reg_loss1 = F.mse_loss(grad_q1_delta, grad_q1_orig)
            adv_reg_loss2 = F.mse_loss(grad_q2_delta, grad_q2_orig)

            adv_reg_loss  = adv_reg_loss1 + adv_reg_loss2

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)

            critic_loss += self.adv_lambda * adv_reg_loss
            adv_reg_losses.append(adv_reg_loss.item())
            critic_diffs.append(th.mean(q1_delta - q1_orig_ref).item())

            # --- AQFR Implementation End ---

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
        self.logger.record("train/critic_diff", np.mean(critic_diffs))