import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import TD3
from custom_td3 import CustomTD3
from stable_baselines3.td3.policies import TD3Policy, Actor

from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from torch import nn
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, PyTorchObs
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update

# Reuse LipsTD3Policy from lips_td3 (LipsNet actor with standard critic)
from lips_td3 import LipsTD3Policy


class PaveLipsTD3(CustomTD3):
    """
    PAVE (Q-Flow) + LipsNet TD3

    - Actor: LipsNet with MGN (Lipschitz-constrained actor, ReLU internally)
    - Critic: PAVE Q-Flow losses (MPR, VFC, Curvature) with SiLU activation
    - Buffer: Standard ReplayBuffer (PAVE uses replay_data.next_observations)
    """

    def __init__(
        self,
        policy: Union[str, type[LipsTD3Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 1_000_000,
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
        # === PAVE Q-Flow Hyperparameters ===
        grad_lamT: float = 0.1,    # VFC (Temporal) weight
        grad_lamS: float = 0.1,    # MPR (Spatial) weight
        grad_lamC: float = 0.01,   # Curvature weight
        grad_sigma: float = 0.01,  # MPR noise std
        grad_delta: float = 1.0,   # Curvature margin
        # === LipsNet Hyperparameters ===
        lips_lam: float = 1e-5,
        lips_eps: float = 1e-4,
        lips_k_init: float = 50.0,
        lips_f_size: list = [64, 64],
        lips_k_size: list = [32],
        lips_global: bool = False,
    ):
        # Save PAVE parameters
        self.grad_lamT = grad_lamT
        self.grad_lamS = grad_lamS
        self.grad_lamC = grad_lamC
        self.grad_sigma = grad_sigma
        self.grad_delta = grad_delta

        if policy_kwargs is None:
            policy_kwargs = {}

        # SiLU for critic (PAVE needs 2nd-order derivatives)
        # LipsNet actor uses its own ReLU internally (hardcoded in LipsNet f_hid_nonliear=nn.ReLU)
        if "activation_fn" not in policy_kwargs:
            policy_kwargs["activation_fn"] = nn.SiLU

        # LipsNet kwargs
        policy_kwargs.update({
            "lips_kwargs": {
                "lips_lam": lips_lam,
                "lips_eps": lips_eps,
                "lips_k_init": lips_k_init,
                "lips_f_size": lips_f_size,
                "lips_k_size": lips_k_size,
                "lips_global": lips_global,
            }
        })

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

        # Verify activations
        critic_activations = [m for m in self.critic.modules() if isinstance(m, nn.SiLU)]
        print(f"[PAVE+LipsNet TD3] Critic Activation: {'SiLU' if critic_activations else 'Other'}")

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        mpr_losses, vfc_losses, curv_losses = [], [], []

        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # -----------------------------------------------------------
            # [PAVE Implementation] Critic Gradient Preparation
            # -----------------------------------------------------------

            # Prepare inputs — requires_grad on action for gradient/Hessian computation
            a_input = replay_data.actions.clone().detach().requires_grad_(True)
            obs_input = replay_data.observations

            # Forward pass for PAVE losses (current critic, not target)
            q1_pred, q2_pred = self.critic(obs_input, a_input)

            # Gradients nabla_a Q(s, a)
            grad_q1 = th.autograd.grad(q1_pred.sum(), a_input, create_graph=True)[0]
            grad_q2 = th.autograd.grad(q2_pred.sum(), a_input, create_graph=True)[0]

            # --- [Loss 1: MPR (Mixed-Partial Regularization)] ---
            noise = th.randn_like(obs_input) * self.grad_sigma
            q1_noisy, q2_noisy = self.critic(obs_input + noise, a_input)

            grad_q1_noisy = th.autograd.grad(q1_noisy.sum(), a_input, create_graph=True)[0]
            grad_q2_noisy = th.autograd.grad(q2_noisy.sum(), a_input, create_graph=True)[0]

            mpr_loss = F.mse_loss(grad_q1, grad_q1_noisy) + F.mse_loss(grad_q2, grad_q2_noisy)

            # --- [Loss 2: VFC (Vector Field Consistency)] ---
            obs_next = replay_data.next_observations
            q1_next, q2_next = self.critic(obs_next, a_input)

            grad_q1_next = th.autograd.grad(q1_next.sum(), a_input, create_graph=True)[0]
            grad_q2_next = th.autograd.grad(q2_next.sum(), a_input, create_graph=True)[0]

            vfc_loss = F.mse_loss(grad_q1, grad_q1_next) + F.mse_loss(grad_q2, grad_q2_next)

            # --- [Loss 3: Curvature Preservation] ---
            v = (th.randint_like(a_input, high=2) * 2 - 1).to(dtype=a_input.dtype)

            # For Q1
            grad_q1_v_product = (grad_q1 * v).sum()
            hessian_vec_prod1 = th.autograd.grad(grad_q1_v_product, a_input, create_graph=True)[0]
            trace_approx1 = (hessian_vec_prod1 * v).sum(dim=1)
            curv_loss1 = th.mean(th.relu(trace_approx1 + self.grad_delta))

            # For Q2
            grad_q2_v_product = (grad_q2 * v).sum()
            hessian_vec_prod2 = th.autograd.grad(grad_q2_v_product, a_input, create_graph=True)[0]
            trace_approx2 = (hessian_vec_prod2 * v).sum(dim=1)
            curv_loss2 = th.mean(th.relu(trace_approx2 + self.grad_delta))

            curv_loss = curv_loss1 + curv_loss2

            # Weighted Sum of PAVE Losses
            q_flow_loss = (self.grad_lamS * mpr_loss) + \
                          (self.grad_lamT * vfc_loss) + \
                          (self.grad_lamC * curv_loss)

            # -----------------------------------------------------------
            # [Standard TD3 Critic Update]
            # -----------------------------------------------------------

            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise_act = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise_act = noise_act.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise_act).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # TD3 Original Loss
            critic_td_loss = F.mse_loss(q1_pred, target_q_values) + F.mse_loss(q2_pred, target_q_values)

            # Final Critic Loss
            critic_loss = critic_td_loss + q_flow_loss

            critic_losses.append(critic_loss.item())
            mpr_losses.append(mpr_loss.item())
            vfc_losses.append(vfc_loss.item())
            curv_losses.append(curv_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # -----------------------------------------------------------
            # [Standard TD3 Actor Update (Delayed)] + LipsNet K(x) L2 reg
            # -----------------------------------------------------------

            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                actor_action = self.actor(replay_data.observations)
                actor_loss = -self.critic.q1_forward(replay_data.observations, actor_action).mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                # LipsNet K(x) L2 regularization gradient
                self.actor.forward(replay_data.observations)
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

        # PAVE Metrics Logging
        if len(mpr_losses) > 0:
            self.logger.record("train/qflow_mpr_loss", np.mean(mpr_losses))
        if len(vfc_losses) > 0:
            self.logger.record("train/qflow_vfc_loss", np.mean(vfc_losses))
        if len(curv_losses) > 0:
            self.logger.record("train/qflow_curv_loss", np.mean(curv_losses))
