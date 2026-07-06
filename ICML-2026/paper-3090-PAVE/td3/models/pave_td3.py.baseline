from stable_baselines3 import TD3
from typing import Any, ClassVar, Optional, TypeVar, Union, List, Tuple
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule

import numpy as np
import torch as th
from torch.nn import functional as F # train에서 F.mse_loss 사용을 위해 필요
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps, polyak_update, get_parameters_by_name
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.td3.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, TD3Policy


SelfTD3 = TypeVar("SelfTD3", bound="TD3")

class PaveTD3(TD3):

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
        # === [PAVE: Q-Flow Hyperparameters] ===
        grad_lamT: float = 0.1,    # VFC (Temporal) weight
        grad_lamS: float = 0.1,    # MPR (Spatial) weight
        grad_lamC: float = 0.01,   # Curvature weight
        grad_sigma: float = 0.01,  # MPR noise std
        grad_delta: float = 1.0,   # Curvature margin
    ):
        # 파라미터 저장
        self.grad_lamT = grad_lamT
        self.grad_lamS = grad_lamS
        self.grad_lamC = grad_lamC
        self.grad_sigma = grad_sigma
        self.grad_delta = grad_delta

        if policy_kwargs is None:
            policy_kwargs = {}
        
        # [중요] 2계 미분(Curvature) 계산을 위해 ReLU 대신 SiLU(Swish) 강제
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

        # Actor(Policy)의 활성화 함수 확인
        actor_activations = [m for m in self.actor.modules() if isinstance(m, nn.SiLU)]
        # Critic의 활성화 함수 확인
        critic_activations = [m for m in self.critic.modules() if isinstance(m, nn.SiLU)]

        print(f"[*] Actor Activation: {'SiLU' if actor_activations else 'Other'}")
        print(f"[*] Critic Activation: {'SiLU' if critic_activations else 'Other'}")
        
    def learn(
        self: SelfTD3,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "TD3",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfTD3:
        self.pure_actions = [ [] for _ in range(self.env.num_envs)]
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
    
    def _sample_action_with_pure(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            assert self._last_obs is not None, "self._last_obs was not set"
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)
            pure_action = unscaled_action

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
            pure_action = unscaled_action
        return action, buffer_action, pure_action

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True

        pure_actions = self.pure_actions

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions, pure_action = self._sample_action_with_pure(learning_starts, action_noise, env.num_envs)
            
            for env_idx in range(env.num_envs):
                pure_actions[env_idx].append(pure_action[env_idx])

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if not callback.on_step():
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)  # type: ignore[arg-type]

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()

                    ### 진동량 계산
                    # === Done 발생한 env의 pure_actions buffer로 진동량 계산 ===
                    pure_actions_array = np.stack(pure_actions[idx], axis=0)  # (steps, action_dim)
                    
                    if pure_actions_array.shape[0] > 1:  # step 2개 이상이어야 diff 가능
                        action_diffs = pure_actions_array[1:, :] - pure_actions_array[:-1, :]
                        diff_norms = np.linalg.norm(action_diffs, axis=-1)
                        mean_oscillation = np.mean(diff_norms)
                        # Log smoothness per episode
                        self.logger.record(f"train/oscillation", mean_oscillation)

                    # === Done 처리: buffer 초기화 ===
                    pure_actions[idx] = []
            
            # [Fix] Critical: Update last obs for the next step prediction
            self._last_obs = new_obs

        callback.on_rollout_end()
        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        
        # PAVE Logging lists
        mpr_losses, vfc_losses, curv_losses = [], [], []

        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # -----------------------------------------------------------
            # [PAVE Implementation Start] Critic Gradient Preparation
            # -----------------------------------------------------------
            
            # 1. Prepare Inputs calculates gradients w.r.t Action
            # TD3의 action은 replay buffer에서 가져온 것을 사용 (buffer action은 이미 noise가 포함됨)
            a_input = replay_data.actions.clone().detach().requires_grad_(True)
            obs_input = replay_data.observations

            # 2. Forward pass for PAVE losses
            # PAVE는 현재 Critic의 기하학적 구조를 제어하므로, target이 아닌 current critic 사용
            q1_pred, q2_pred = self.critic(obs_input, a_input)
            
            # Gradients \nabla_a Q(s, a)
            # create_graph=True is essential for higher-order derivatives (MPR, Curv)
            grad_q1 = th.autograd.grad(q1_pred.sum(), a_input, create_graph=True)[0]
            grad_q2 = th.autograd.grad(q2_pred.sum(), a_input, create_graph=True)[0]

            # --- [Loss 1: MPR (Mixed-Partial Regularization)] ---
            # Eq 11 in paper: penalize sensitivity to state perturbation
            noise = th.randn_like(obs_input) * self.grad_sigma
            q1_noisy, q2_noisy = self.critic(obs_input + noise, a_input)
            
            grad_q1_noisy = th.autograd.grad(q1_noisy.sum(), a_input, create_graph=True)[0]
            grad_q2_noisy = th.autograd.grad(q2_noisy.sum(), a_input, create_graph=True)[0]
            
            mpr_loss = F.mse_loss(grad_q1, grad_q1_noisy) + F.mse_loss(grad_q2, grad_q2_noisy)

            # --- [Loss 2: VFC (Vector Field Consistency)] ---
            # Eq 12 in paper: align gradients across time steps
            obs_next = replay_data.next_observations
            q1_next, q2_next = self.critic(obs_next, a_input)
            
            grad_q1_next = th.autograd.grad(q1_next.sum(), a_input, create_graph=True)[0]
            grad_q2_next = th.autograd.grad(q2_next.sum(), a_input, create_graph=True)[0]
            
            vfc_loss = F.mse_loss(grad_q1, grad_q1_next) + F.mse_loss(grad_q2, grad_q2_next)

            # --- [Loss 3: Curvature Preservation] ---
            # Eq 13 in paper: Ensure trace of Hessian is sufficiently negative (concave)
            # Hutchinson's estimator: v^T H v approx Tr(H)
            v = (th.randint_like(a_input, high=2) * 2 - 1).to(dtype=a_input.dtype)
            
            # For Q1
            grad_q1_v_product = (grad_q1 * v).sum()
            hessian_vec_prod1 = th.autograd.grad(grad_q1_v_product, a_input, create_graph=True)[0]
            trace_approx1 = (hessian_vec_prod1 * v).sum(dim=1)
            # If trace + delta > 0, it means it's not "sharp" enough (or convex), so we penalize.
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

            # Get current Q-values estimates (Re-using computation graph from PAVE is risky due to detach, 
            # so we re-compute or use q1_pred/q2_pred carefully. 
            # q1_pred, q2_pred above have `create_graph=True`. 
            # For TD3 loss, standard graph is enough, but using the existing one is fine.)
            
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
            # [Standard TD3 Actor Update (Delayed)]
            # -----------------------------------------------------------
            
            if self._n_updates % self.policy_delay == 0:
                print(f"[LOSS] q_flow={q_flow_loss.item():.10f}, mpr={mpr_loss.item():.10f}, vfc={vfc_loss.item():.10f}, curv={curv_loss.item():.10f}")
                # Compute actor loss
                actor_action = self.actor(replay_data.observations)
                actor_loss = -self.critic.q1_forward(replay_data.observations, actor_action).mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
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

    def _log_hessian_stats(self, obs: th.Tensor) -> None:
        """
        Compute and log the Hessian max eigenvalue and trace of Q(s, a) w.r.t a.
        """
        # Enable gradient tracking for this calculation
        with th.set_grad_enabled(True):
            # We need to detach to avoid messing up the main computation graph
            # but we need 'requires_grad' on action to compute Hessian w.r.t it.
            obs_temp = obs.detach()
            
            # Predict action
            action = self.actor(obs_temp)
            
            # Calculate Q value
            q_val = self.critic.q1_forward(obs_temp, action)
            q_sum = q_val.sum()
            
            # First derivative (Gradient)
            grads = th.autograd.grad(q_sum, action, create_graph=True)[0]
            
            # Hessian Trace and Max Eigenvalue
            hessian_traces = []
            max_eigenvalues = []

            # Loop over batch elements
            for i in range(action.shape[0]):
                hessian_matrix = []
                action_dim = action.shape[1]
                
                # Construct Hessian Matrix
                for j in range(action_dim):
                    # Grad of the j-th component of the gradient
                    grad_2 = th.autograd.grad(grads[i, j], action, retain_graph=True)[0]
                    hessian_matrix.append(grad_2[i].detach().cpu().numpy())
                
                hessian_matrix = np.array(hessian_matrix)
                eigenvalues = np.linalg.eigvalsh(hessian_matrix)
                
                hessian_traces.append(np.sum(eigenvalues))
                max_eigenvalues.append(np.max(eigenvalues))
            
            self.logger.record("train/hessian_max_eigen", np.mean(max_eigenvalues))
            self.logger.record("train/hessian_trace", np.mean(hessian_traces))