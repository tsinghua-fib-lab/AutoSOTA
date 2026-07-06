import sys
import os
import warnings

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import SAC
from custom_sac import CustomSAC
from typing import Any, ClassVar, Optional, TypeVar, Union, Tuple

import numpy as np
import torch as th
from torch import nn
from gymnasium import spaces
from torch.nn import functional as F
import math
from copy import deepcopy
from torch.func import functional_call

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic, BaseModel
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, PyTorchObs, TrainFreq, RolloutReturn, TrainFrequencyUnit, ReplayBufferSamples
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
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
import pickle
import gzip

LOG_STD_MAX = 2
LOG_STD_MIN = -20

SelfSAC = TypeVar("SelfSAC", bound="ASAPSAC_feasibility")

class ASAPActor(Actor):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: list[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            net_arch=net_arch,
            features_extractor=features_extractor,
            features_dim=features_dim,
            activation_fn=activation_fn,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            clip_mean=clip_mean,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim

        if self.use_sde:
            self.mu_next = nn.Linear(last_layer_dim, action_dim)
            self.mu_next_log_std = nn.Linear(last_layer_dim, action_dim)
            if clip_mean > 0.0:
                self.mu_next = nn.Sequential(self.mu_next, nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean))
        else:
            self.mu_next = nn.Linear(last_layer_dim, action_dim)
            self.mu_next_log_std = nn.Linear(last_layer_dim, action_dim)

    def predict_next(self, obs: PyTorchObs, deterministic: bool = True) -> th.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        latent_pi = self.latent_pi(features)
        next_mean_actions = self.mu_next(latent_pi)
        return next_mean_actions
    
    def predict_next_std(self, obs: PyTorchObs, deterministic: bool = True) -> th.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        latent_pi = self.latent_pi(features)
        log_std = self.mu_next_log_std(latent_pi).clamp(-20, 2)
        std = th.exp(log_std)
        return std, log_std


class ASAPPolicy_soft(SACPolicy):
    actor: ASAPActor

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule=lr_schedule)

        self.next_actor_target = deepcopy(self.actor)
        for p in self.next_actor_target.parameters():
            p.requires_grad = False

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ASAPActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return ASAPActor(**actor_kwargs).to(self.device)
  
    def _predict_next(self, observation: PyTorchObs, deterministic: bool = True) -> th.Tensor:
        return self.actor.predict_next(observation, deterministic)
    
    def _predict_next_target(self, observation: PyTorchObs, deterministic: bool = True) -> th.Tensor:
        with th.no_grad():
            return self.next_actor_target.predict_next(observation, deterministic)
        
    def _predict_next_std(self, observation: PyTorchObs, deterministic: bool = True) -> Tuple[th.Tensor, th.Tensor]:
        """
        std, log_std
        """
        return self.actor.predict_next_std(observation, deterministic)
    
    def _predict_next_std_target(self, observation: PyTorchObs, deterministic: bool = True) -> Tuple[th.Tensor, th.Tensor]:
        """
        std, log_std
        """
        with th.no_grad():
            return self.next_actor_target.predict_next_std(observation, deterministic)
    
    def _polyak_update_targets(self, tau: float):
        # Polyak averaging: θ_target ← (1−τ)·θ_target + τ·θ
        with th.no_grad():
            # next actor
            for p, p_targ in zip(self.actor.parameters(),
                                  self.next_actor_target.parameters()):
                p_targ.data.mul_(1.0 - tau)
                p_targ.data.add_(tau * p.data)
    

class ASAPPolicy(SACPolicy):
    actor: ASAPActor

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ASAPActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return ASAPActor(**actor_kwargs).to(self.device)
  
    def _predict_next(self, observation: PyTorchObs, deterministic: bool = True) -> th.Tensor:
        return self.actor.predict_next(observation, deterministic)
        

class ASAPSAC(CustomSAC):
    policy: ASAPPolicy
    def __init__(
        self,
        policy: Union[str, type[ASAPPolicy]],
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
        lam_s = 0.05,
        lam_p = 0.2,
    ):
        self.lam_s = lam_s # lambda Lipschitz
        self.lam_p = lam_p # lambda predict
        self._p_updates = 0
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

    def _setup_model(self) -> None:
        super()._setup_model()

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
        asap_losses = []

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

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks

            # ASAP Loss
            d_observations = replay_data.observations
            d_next_observations = replay_data.next_observations

            d_predict_next_actions = self.policy._predict_next(d_observations, True).detach()
            d_next_actions = self.policy._predict(d_next_observations, True)

            spatial_loss = F.mse_loss(d_next_actions, d_predict_next_actions)
            asap_losses.append(spatial_loss.item())

            d_predict_next_actions_train = self.policy._predict_next(d_observations, True)
            d_next_actions_target = self.policy._predict(d_next_observations, True).detach()

            predictor_loss = F.mse_loss(d_predict_next_actions_train, d_next_actions_target)

            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean() + spatial_loss * self.lam_s + predictor_loss * self.lam_p
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
        self.logger.record("train/asap_loss", np.mean(asap_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))





class ASAPPolicy_nonshare(SACPolicy):
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

        self.next_actor_kwargs = self.actor_kwargs.copy()
        self._build_next_actor(lr_schedule)


    def _build_next_actor(self, lr_schedule: Schedule):

        self.next_actor = self.make_next_actor(features_extractor=None)

        self.next_actor.optimizer = self.optimizer_class(
            self.next_actor.parameters(),
            lr = lr_schedule(1),
            **self.optimizer_kwargs
        )
    
    def make_next_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        next_actor_kwargs = self._update_features_extractor(self.next_actor_kwargs, features_extractor)
        return Actor(**next_actor_kwargs).to(self.device)
    
    def _predict_next(self, observation: PyTorchObs, deterministic: bool = True) -> th.Tensor:
        return self.next_actor(observation, deterministic)


class ASAPSAC_feasibility(CustomSAC):
    policy: ASAPPolicy_nonshare
    def __init__(
        self,
        policy: Union[str, type[ASAPPolicy]],
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
        lam_smooth = 0.5,
        lam_predict = 1.0,
        asap_tau = 0.01,
        asap_target_update_interval : int = 1
    ):
        self.lam_s = lam_smooth # lambda Lipschitz
        self.lam_p = lam_predict # lambda predict
        self.asap_tau = asap_tau
        self.asap_target_update_interval = asap_target_update_interval
        self._p_updates = 0
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

    def _setup_model(self) -> None:
        super()._setup_model()

    def reset_buffer(self, num_samples:int = 1000000):
        self.replay_buffer.reset()

        self.policy.set_training_mode(False)
        
        env = self.env
        action_noise=self.action_noise
        learning_starts=self.learning_starts
        replay_buffer=self.replay_buffer

        self._last_obs = env.reset()

        # 3) num_samples만큼 경험 수집
        for num_sample in range(num_samples):
            if num_sample % 50000 == 0 :
                print(f"수집 : {num_sample}/{num_samples}")
            # if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
            #     # Sample a new noise matrix
            #     self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action_b(learning_starts, action_noise, env.num_envs)

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)  # type: ignore[arg-type]

            # self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

    def train_predictor(self, total_steps: int) -> None:
        batch_size = self.batch_size
        self.policy.set_training_mode(True)
        asap_losses_target = []
        asap_losses = []
        for step_num in range(total_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # ASAP Loss
            d_observations = replay_data.observations
            d_next_observations = replay_data.next_observations

            d_predict_next_actions_train = self.policy._predict_next(d_observations, True)
            d_next_actions_target = self.policy._predict(d_next_observations, True).detach()

            predictor_loss = F.mse_loss(d_predict_next_actions_train, d_next_actions_target)
            asap_losses.append(predictor_loss.item())

            actor_loss = predictor_loss * self.lam_p
            self.policy.next_actor.optimizer.zero_grad()
            actor_loss.backward()
            self.policy.next_actor.optimizer.step()

            if step_num % 10000 == 0 :
                print(f"timestep : {step_num}, spatial_loss : {predictor_loss}")

        return

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
        asap_losses_target = []
        asap_losses = []

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

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks

            # ASAP Loss
            # d_observations = replay_data.observations
            # d_next_observations = replay_data.next_observations

            # d_predict_next_actions = self.policy._predict_next_target(d_observations, True).detach()
            # d_next_actions = self.policy._predict(d_next_observations, True)

            # spatial_loss = 0.5*F.mse_loss(d_next_actions, d_predict_next_actions)
            # asap_losses_target.append(spatial_loss.item())

            # d_predict_next_actions_train = self.policy._predict_next(d_observations, True)
            # d_next_actions_target = self.policy._predict(d_next_observations, True).detach()

            # predictor_loss = F.mse_loss(d_predict_next_actions_train, d_next_actions_target)
            # asap_losses.append(predictor_loss.item())

            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean() #+ spatial_loss * self.lam_s + predictor_loss * self.lam_p
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
        # self.logger.record("train/asap_loss", np.mean(asap_losses))
        # self.logger.record("train/asap_loss_target", np.mean(asap_losses_target))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def train_with_asap(self, gradient_steps: int, batch_size: int = 64) -> None:
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
        asap_losses_target = []
        asap_losses = []

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

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks

            # ASAP Loss
            d_observations = replay_data.observations
            d_next_observations = replay_data.next_observations

            d_predict_next_actions = self.policy._predict_next(d_observations, True).detach()
            d_next_actions = self.policy._predict(d_next_observations, True)

            spatial_loss = 0.5*F.mse_loss(d_next_actions, d_predict_next_actions)
            asap_losses_target.append(spatial_loss.item())

            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean() + spatial_loss * self.lam_s # + predictor_loss * self.lam_p
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
        # self.logger.record("train/asap_loss", np.mean(asap_losses))
        self.logger.record("train/asap_loss_target", np.mean(asap_losses_target))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def learn_with_asap(
        self: SelfSAC,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "SAC",
        reset_num_timesteps: bool = False,
        progress_bar: bool = False,
    ) -> SelfSAC:
        # total_timesteps=total_timesteps,
        # callback=callback,
        # log_interval=log_interval,
        # tb_log_name=tb_log_name,
        # reset_num_timesteps=reset_num_timesteps,
        # progress_bar=progress_bar,

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None, "You must set the environment before calling learn()"
        assert isinstance(self.train_freq, TrainFreq)  # check done in _setup_learn()

        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if not rollout.continue_training:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train_with_asap(batch_size=self.batch_size, gradient_steps=gradient_steps)

        callback.on_training_end()

        return self
    
    def save_replay_buffer(self, path: str, compress: bool = False) -> None:
        """
        현재 모델의 리플레이 버퍼를 파일로 저장합니다.

        :param path: 저장할 파일 경로 (예: "buffer.pkl" 또는 "buffer.pkl.gz")
        :param compress: True 이면 gzip 압축(.gz)으로 저장
        """
        if compress:
            # .gz 확장자를 권장
            with gzip.open(path, "wb") as f:
                pickle.dump(self.replay_buffer, f)
        else:
            with open(path, "wb") as f:
                pickle.dump(self.replay_buffer, f)
        print(f"Replay buffer saved to {path}")

    def load_replay_buffer(self, path: str, compress: bool = False) -> None:
        """
        파일에 저장된 리플레이 버퍼를 불러와서 현재 모델에 할당합니다.

        :param path: 불러올 파일 경로
        :param compress: True 이면 gzip 압축(.gz) 파일로 간주
        """
        if compress:
            with gzip.open(path, "rb") as f:
                loaded_buffer = pickle.load(f)
        else:
            with open(path, "rb") as f:
                loaded_buffer = pickle.load(f)

        # 모델에 로드한 버퍼 할당
        self.replay_buffer = loaded_buffer
        print(f"Replay buffer loaded from {path}, contains {self.replay_buffer.size()} samples")

    def _sample_action_b(
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
        unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

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
        return action, buffer_action

