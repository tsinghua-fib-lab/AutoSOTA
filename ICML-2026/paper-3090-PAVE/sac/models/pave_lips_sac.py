import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import SAC
from custom_sac import CustomSAC
from typing import Any, ClassVar, Optional, TypeVar, Union, NamedTuple

import numpy as np
import torch as th
from torch import nn
from gymnasium import spaces
from torch.nn import functional as F
import warnings

from stable_baselines3.common.buffers import ReplayBuffer, BaseBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.vec_env import VecNormalize
from copy import deepcopy

# Reuse LipsNet actor components from lips_sac
from lips_sac import LipsSACPolicy

# Reuse GradBuffer from pave_sac
from pave_sac import GradBuffer

try:
    import psutil
except ImportError:
    psutil = None

SelfSAC = TypeVar("SelfSAC", bound="SAC")


class PAVE_LIPS_SAC(CustomSAC):
    """
    PAVE (Q-Flow) + LipsNet SAC

    - Actor: LipsNet with MGN (Lipschitz-constrained actor)
    - Critic: PAVE Q-Flow losses (MPR, VFC, Curvature) with SiLU activation
    """
    def __init__(
        self,
        policy: Union[str, type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[type[GradBuffer]] = None,
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
        # === PAVE Q-Flow Parameters ===
        grad_lamT: float = 0.1,    # VFC weight (Temporal)
        grad_lamS: float = 0.1,    # MPR weight (Spatial)
        grad_lamC: float = 0.01,   # Curvature weight
        grad_sigma: float = 0.01,  # MPR noise std
        grad_delta: float = 1.0,   # Curvature delta
        # === LipsNet Parameters ===
        lips_lam: float = 1e-5,
        lips_eps: float = 1e-4,
        lips_k_init: float = 50.0,
        lips_f_size: list = [256, 256],
        lips_k_size: list = [32],
    ):
        # Save Q-Flow parameters
        self.grad_lamT = grad_lamT
        self.grad_lamS = grad_lamS
        self.grad_lamC = grad_lamC
        self.grad_sigma = grad_sigma
        self.grad_delta = grad_delta

        if policy_kwargs is None:
            policy_kwargs = {}

        # SiLU for critic (PAVE needs 2nd-order derivatives)
        # LipsNet actor uses its own ReLU internally
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

        # Verify activations
        critic_activations = [m for m in self.critic.modules() if isinstance(m, nn.SiLU)]
        print(f"[PAVE+LipsNet] Critic Activation: {'SiLU' if critic_activations else 'Other'}")

    def _setup_model(self):
        self.replay_buffer_class = GradBuffer
        return super()._setup_model()

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)

        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        mpr_losses, vfc_losses, curv_losses = [], [], []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            if self.use_sde:
                self.actor.reset_noise()

            # Actor action for current observations
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            # Entropy coefficient
            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            # Target Q-values
            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # ===== PAVE Q-Flow Critic Losses =====
            a_input = replay_data.actions.clone().detach().requires_grad_(True)
            obs_input = replay_data.observations

            q1_pred, q2_pred = self.critic(obs_input, a_input)

            grad_q1 = th.autograd.grad(q1_pred.sum(), a_input, create_graph=True)[0]
            grad_q2 = th.autograd.grad(q2_pred.sum(), a_input, create_graph=True)[0]

            # [Loss 1: MPR]
            noise = th.randn_like(obs_input) * self.grad_sigma
            q1_noisy, q2_noisy = self.critic(obs_input + noise, a_input)

            grad_q1_noisy = th.autograd.grad(q1_noisy.sum(), a_input, create_graph=True)[0]
            grad_q2_noisy = th.autograd.grad(q2_noisy.sum(), a_input, create_graph=True)[0]

            mpr_loss = F.mse_loss(grad_q1, grad_q1_noisy) + F.mse_loss(grad_q2, grad_q2_noisy)

            # [Loss 2: VFC]
            obs_next = replay_data.next_observations
            q1_next, q2_next = self.critic(obs_next, a_input)

            grad_q1_next = th.autograd.grad(q1_next.sum(), a_input, create_graph=True)[0]
            grad_q2_next = th.autograd.grad(q2_next.sum(), a_input, create_graph=True)[0]

            vfc_loss = F.mse_loss(grad_q1, grad_q1_next) + F.mse_loss(grad_q2, grad_q2_next)

            # [Loss 3: Curvature Preservation]
            v = th.randint_like(a_input, high=2) * 2 - 1  # Rademacher

            grad_q1_v_product = (grad_q1 * v).sum()
            hessian_vec_prod1 = th.autograd.grad(grad_q1_v_product, a_input, create_graph=True)[0]
            trace_approx1 = (hessian_vec_prod1 * v).sum(dim=1)
            curv_loss1 = th.mean(th.relu(trace_approx1 + self.grad_delta))

            grad_q2_v_product = (grad_q2 * v).sum()
            hessian_vec_prod2 = th.autograd.grad(grad_q2_v_product, a_input, create_graph=True)[0]
            trace_approx2 = (hessian_vec_prod2 * v).sum(dim=1)
            curv_loss2 = th.mean(th.relu(trace_approx2 + self.grad_delta))

            curv_loss = curv_loss1 + curv_loss2

            q_flow_loss = (self.grad_lamS * mpr_loss) + \
                          (self.grad_lamT * vfc_loss) + \
                          (self.grad_lamC * curv_loss)

            # Standard SAC TD Loss
            critic_td_loss = 0.5 * (F.mse_loss(q1_pred, target_q_values) + F.mse_loss(q2_pred, target_q_values))

            critic_loss = critic_td_loss + q_flow_loss

            critic_losses.append(critic_loss.item())
            mpr_losses.append(mpr_loss.item())
            vfc_losses.append(vfc_loss.item())
            curv_losses.append(curv_loss.item())

            # Optimize critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # ===== Actor Update (standard SAC + LipsNet MGN) =====
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            # LipsNet forward for K(x) L2 regularization gradient
            self.actor.forward(replay_data.observations)
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/qflow_mpr_loss", np.mean(mpr_losses))
        self.logger.record("train/qflow_vfc_loss", np.mean(vfc_losses))
        self.logger.record("train/qflow_curv_loss", np.mean(curv_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def _store_transition(
        self,
        replay_buffer: GradBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            self._last2_original_obs = self._last2_obs
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        next_obs = deepcopy(new_obs_)
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])

        replay_buffer.add(
            self._last_original_obs,
            self._last2_original_obs,
            next_obs,
            buffer_action,
            reward_,
            dones,
            infos,
        )
        self._last2_obs = self._last_obs
        self._last_obs = new_obs
        if self._vec_normalize_env is not None:
            self._last2_original_obs = self._last_original_obs
            self._last_original_obs = new_obs_

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ) -> tuple[int, BaseCallback]:
        total_timesteps, callback = super()._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )
        if reset_num_timesteps or self._last2_obs is None:
            assert self.env is not None
            if self._last_obs is not None:
                self._last2_obs = deepcopy(self._last_obs)
            if self._last_original_obs is not None:
                self._last2_original_obs = deepcopy(self._last_original_obs)

        return total_timesteps, callback
