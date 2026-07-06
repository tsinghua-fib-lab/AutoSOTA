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
import math
import copy
from torch.func import functional_call
import warnings

from stable_baselines3.common.buffers import ReplayBuffer, BaseBuffer, DictReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.callbacks import BaseCallback
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
from stable_baselines3.common.vec_env import VecNormalize
from copy import deepcopy

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

LOG_STD_MAX = 2
LOG_STD_MIN = -20

SelfSAC = TypeVar("SelfSAC", bound="SAC")

class GradBuffer_Samples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    prev_observations: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor

class GradBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    observations: np.ndarray
    prev_observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    timeouts: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        # there is a bug if both optimize_memory_usage and handle_timeout_termination are true
        # see https://github.com/DLR-RM/stable-baselines3/issues/934
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        if not optimize_memory_usage:
            # When optimizing memory, `observations` contains also the next observation
            self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)
            self.prev_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            total_memory_usage: float = (
                self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            )

            if not optimize_memory_usage:
                total_memory_usage += self.next_observations.nbytes
                total_memory_usage += self.prev_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: np.ndarray,
        prev_obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))
            prev_obs = prev_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs)

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
            self.observations[(self.pos - 1) % self.buffer_size] = np.array(prev_obs)
        else:
            self.next_observations[self.pos] = np.array(next_obs)
            self.prev_observations[self.pos] = np.array(prev_obs)

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> GradBuffer_Samples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> GradBuffer_Samples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
            prev_obs = self._normalize_obs(self.observations[(batch_inds - 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)
            prev_obs = self._normalize_obs(self.prev_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            prev_obs,
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return GradBuffer_Samples(*tuple(map(self.to_torch, data)))

    @staticmethod
    def _maybe_cast_dtype(dtype: np.typing.DTypeLike) -> np.typing.DTypeLike:
        """
        Cast `np.float64` action datatype to `np.float32`,
        keep the others dtype unchanged.
        See GH#1572 for more information.

        :param dtype: The original action space dtype
        :return: ``np.float32`` if the dtype was float64,
            the original dtype otherwise.
        """
        if dtype == np.float64:
            return np.float32
        return dtype


class PAVE_SAC(CustomSAC):
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
        # Q-Flow Parameters
        grad_lamT = 0.1,    # lambda_3: VFC weight (Temporal)
        grad_lamS = 0.1,    # lambda_1: MPR weight (Spatial)
        grad_lamC = 0.01,   # lambda_2: Curvature weight
        grad_sigma = 0.01,  # MPR noise std
        grad_delta = 1.0,   # Curvature delta
    ):
        # Save Q-Flow parameters
        self.grad_lamT = grad_lamT
        self.grad_lamS = grad_lamS
        self.grad_lamC = grad_lamC
        self.grad_sigma = grad_sigma
        self.grad_delta = grad_delta
    
        if policy_kwargs is None:
            policy_kwargs = {}
        
        if "activation_fn" not in policy_kwargs:
            policy_kwargs["activation_fn"] = nn.SiLU

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

        # Actor(Policy)의 활성화 함수 확인
        actor_activations = [m for m in self.actor.modules() if isinstance(m, nn.SiLU)]
        # Critic의 활성화 함수 확인
        critic_activations = [m for m in self.critic.modules() if isinstance(m, nn.SiLU)]

        print(f"[*] Actor Activation: {'SiLU' if actor_activations else 'Other'}")
        print(f"[*] Critic Activation: {'SiLU' if critic_activations else 'Other'}")

    def _setup_model(self):
        self.replay_buffer_class = GradBuffer
        return super()._setup_model()

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
        # Lists for Q-Flow logging
        mpr_losses, vfc_losses, curv_losses = [], [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

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

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # --- Q-Flow Implementation Starts Here ---
            
            # [Step 1] Prepare Inputs for Critic Gradient Calculation
            a_input = replay_data.actions.clone().detach().requires_grad_(True)
            obs_input = replay_data.observations
            
            # [Step 2] Compute Gradients for Q1 and Q2
            q1_pred, q2_pred = self.critic(obs_input, a_input)
            
            # Gradients \nabla_a Q(s, a)
            grad_q1 = th.autograd.grad(q1_pred.sum(), a_input, create_graph=True)[0]
            grad_q2 = th.autograd.grad(q2_pred.sum(), a_input, create_graph=True)[0]
            
            # --- [Loss 1: MPR] ---
            noise = th.randn_like(obs_input) * self.grad_sigma
            q1_noisy, q2_noisy = self.critic(obs_input + noise, a_input)
            
            grad_q1_noisy = th.autograd.grad(q1_noisy.sum(), a_input, create_graph=True)[0]
            grad_q2_noisy = th.autograd.grad(q2_noisy.sum(), a_input, create_graph=True)[0]
            
            mpr_loss = F.mse_loss(grad_q1, grad_q1_noisy) + F.mse_loss(grad_q2, grad_q2_noisy)

            # --- [Loss 2: VFC] ---
            obs_next = replay_data.next_observations
            q1_next, q2_next = self.critic(obs_next, a_input)
            
            grad_q1_next = th.autograd.grad(q1_next.sum(), a_input, create_graph=True)[0]
            grad_q2_next = th.autograd.grad(q2_next.sum(), a_input, create_graph=True)[0]
            
            vfc_loss = F.mse_loss(grad_q1, grad_q1_next) + F.mse_loss(grad_q2, grad_q2_next)
            
            # --- [Loss 3: Curvature Preservation] ---
            v = th.randint_like(a_input, high=2) * 2 - 1 # Rademacher (+1 or -1)
            
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

            # ================= [DEBUGGING BLOCK START] =================
            # 100 �]�� Gradient@ Hessiant 0x� D�� �lX� �%
            if self._n_updates % 100 == 0:
                grad_mag = grad_q1.abs().mean().item()
                hess_mag = hessian_vec_prod1.abs().mean().item()
                
                # print(f"\n[DEBUG Step {self._n_updates}] Geometry Check:")
                # print(f"  > Gradient Mag (1st Derivative): {grad_mag:.8f}")
                # print(f"  > Hessian-Vec Mag (2nd Derivative): {hess_mag:.8f}")
                
                # if grad_mag == 0:
                #     print("  =� CRITICAL ALERT: Gradient is ZERO. Something is wrong.")
                # if hess_mag == 0:
                #     print("  =� CRITICAL ALERT: Hessian is ZERO. SiLU activation might not be working.")
                # else:
                #     print("   Status OK: Non-zero Gradient & Hessian detected.")
                
                # P���� 0]
                self.logger.record("debug/grad_magnitude", grad_mag)
                self.logger.record("debug/hessian_magnitude", hess_mag)
            # ================= [DEBUGGING BLOCK END] ===================

            # --- Final Critic Loss Combination ---
            q_flow_loss = (self.grad_lamS * mpr_loss) + \
                          (self.grad_lamT * vfc_loss) + \
                          (self.grad_lamC * curv_loss)

            # Calculate standard SAC TD Loss.
            # Note: q1_pred, q2_pred have attached graphs for grad calculation, so using them directly is fine.
            critic_td_loss = 0.5 * (F.mse_loss(q1_pred, target_q_values) + F.mse_loss(q2_pred, target_q_values))
            
            critic_loss = critic_td_loss + q_flow_loss
            
            critic_losses.append(critic_loss.item())
            mpr_losses.append(mpr_loss.item())
            vfc_losses.append(vfc_loss.item())
            curv_losses.append(curv_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # --- Actor Update ---
            # Compute actor loss
            # Q-Flow does not modify the Actor Update formula, so CAPS logic 
            # (grad_prev_actions, etc.) is not needed for strict Q-Flow theory.
            # However, we keep it commented out or disable it (lam=0) to preserve structure.
            
            # [Original CAPS Logic Disabled for Q-Flow Compliance]
            # grad_prev_actions = self.policy._predict(replay_data.prev_observations, deterministic=True).type(th.float32)
            # grad_now_actions = self.policy._predict(replay_data.observations, deterministic=True).type(th.float32)
            # grad_next_actions = self.policy._predict(replay_data.next_observations, deterministic=True).type(th.float32)
            # derv_t = 0.5 * ((2*grad_now_actions - grad_next_actions - grad_prev_actions)**2)
            # delta = grad_next_actions - grad_prev_actions + 1e-4
            # hdelta = F.tanh((1/delta)**2).detach()
            # loss_t = th.mean(derv_t*hdelta)

            # Standard SAC Actor Loss
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            
            # Removed 'loss_t * self.grad_lamT' as Q-Flow only affects the Critic.
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
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
        
        # Q-Flow Metrics Logging
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
        """
        Store transition in the replay buffer.
        We store the normalized action and the unnormalized observation.
        It also handles terminal observations (because VecEnv resets automatically).

        :param replay_buffer: Replay buffer object where to store the transition.
        :param buffer_action: normalized action
        :param new_obs: next observation in the current episode
            or first observation of the episode (when dones is True)
        :param reward: reward for the current transition
        :param dones: Termination signal
        :param infos: List of additional information about the transition.
            It may contain the terminal observations and information about timeout.
        """
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last2_original_obs = self._last2_obs
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])  # type: ignore[assignment]

        replay_buffer.add(
            self._last_original_obs,  # type: ignore[arg-type]
            self._last2_original_obs,
            next_obs,  # type: ignore[arg-type]
            buffer_action,
            reward_,
            dones,
            infos,
        )
        self._last2_obs = self._last_obs
        self._last_obs = new_obs
        # Save the unnormalized observation
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
            # Retrieve unnormalized observation for saving into the buffer
            if self._last_original_obs is not None:
                self._last2_original_obs = deepcopy(self._last_original_obs)

        return total_timesteps, callback