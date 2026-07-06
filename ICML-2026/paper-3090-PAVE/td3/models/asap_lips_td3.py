import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import TD3
from custom_td3 import CustomTD3
from stable_baselines3.td3.policies import TD3Policy, Actor

from typing import Any, ClassVar, Optional, TypeVar, Union, NamedTuple
from functorch import jacrev, vmap

import numpy as np
import torch as th
from torch import nn
from gymnasium import spaces
from torch.nn import functional as F
from copy import deepcopy
import math
import warnings

from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, PyTorchObs
from stable_baselines3.common.buffers import ReplayBuffer, BaseBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import BasePolicy
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
from stable_baselines3.common.vec_env import VecNormalize

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

# ============================================================
# Reuse GradBuffer from asap_td3
# ============================================================
from asap_td3 import GradBuffer, GradBuffer_Samples


# ============================================================
# LipsNet utilities (from lips_td3)
# ============================================================

def mlp(sizes, hid_nonliear, out_nonliear):
    # declare layers
    layers = []
    for j in range(len(sizes) - 1):
        nonliear = hid_nonliear if j < len(sizes) - 2 else out_nonliear
        layers += [nn.Linear(sizes[j], sizes[j + 1]), nonliear()]
    # init weight
    for i in range(len(layers) - 1):
        if isinstance(layers[i], nn.Linear):
            if isinstance(layers[i+1], nn.ReLU):
                nn.init.kaiming_normal_(layers[i].weight, nonlinearity='relu')
            elif isinstance(layers[i+1], nn.LeakyReLU):
                nn.init.kaiming_normal_(layers[i].weight, nonlinearity='leaky_relu')
            else:
                nn.init.xavier_normal_(layers[i].weight)
    return nn.Sequential(*layers)


def mlp_f(sizes, hid_nonliear):
    # declare layers (no output nonlinearity — used for shared backbone)
    layers = []
    for j in range(len(sizes) - 1):
        nonliear = hid_nonliear
        layers += [nn.Linear(sizes[j], sizes[j + 1]), nonliear()]
    # init weight
    for i in range(len(layers) - 1):
        if isinstance(layers[i], nn.Linear):
            if isinstance(layers[i+1], nn.ReLU):
                nn.init.kaiming_normal_(layers[i].weight, nonlinearity='relu')
            elif isinstance(layers[i+1], nn.LeakyReLU):
                nn.init.kaiming_normal_(layers[i].weight, nonlinearity='leaky_relu')
            else:
                nn.init.xavier_normal_(layers[i].weight)
    return nn.Sequential(*layers)


class K_net(nn.Module):
    def __init__(self, global_lips, k_init, sizes, hid_nonliear, out_nonliear) -> None:
        super().__init__()
        self.global_lips = global_lips
        if global_lips:
            # declare global Lipschitz constant
            self.k = th.nn.Parameter(th.tensor(k_init, dtype=th.float), requires_grad=True)
        else:
            # declare network
            self.k = mlp(sizes, hid_nonliear, out_nonliear)
            # set K_init
            self.k[-2].bias.data += th.tensor(k_init, dtype=th.float).data

    def forward(self, x):
        if self.global_lips:
            return F.softplus(self.k).repeat(x.shape[0]).unsqueeze(1)
        else:
            return self.k(x)


# ============================================================
# ASAPLipsNet: LipsNet with ASAP prediction head
# Shared f_net backbone, act_head (MGN) + pred_head (no MGN)
# ============================================================

class ASAPLipsNet(nn.Module):
    def __init__(self, f_sizes, f_hid_nonliear=nn.ReLU, f_out_nonliear=nn.Identity,
                 global_lips=True, k_init=100, k_sizes=None, k_hid_act=nn.Tanh, k_out_act=nn.Identity,
                 loss_lambda=0.1, eps=1e-4, squash_action=True) -> None:
        super().__init__()
        # declare network: shared backbone (all layers except last)
        self.f_net = mlp_f(f_sizes[:-1], f_hid_nonliear)
        self.act_head = nn.Sequential(nn.Linear(f_sizes[-2], f_sizes[-1]), f_out_nonliear())
        self.pred_head = nn.Sequential(nn.Linear(f_sizes[-2], f_sizes[-1]), f_out_nonliear())
        self.k_net = K_net(global_lips, k_init, k_sizes, k_hid_act, k_out_act)
        # declare hyperparameters
        self.loss_lambda = loss_lambda
        self.eps = eps
        self.squash_action = squash_action
        # initialize heads
        for head in [self.act_head[0], self.pred_head[0]]:
            nn.init.xavier_normal_(head.weight)
        # initialize as eval mode
        self.eval()

    def forward(self, x):
        # K(x) forward
        k_out = self.k_net(x)
        # L2 regularization backward
        if self.training and k_out.requires_grad:
            lips_loss = self.loss_lambda * (k_out ** 2).mean()
            lips_loss.backward(retain_graph=True)
        # f(x) forward
        f_out = self.f_net(x)
        f_out = self.act_head(f_out)
        # calcute jac matrix
        if k_out.requires_grad:
            jacobi = vmap(jacrev(self.f_net))(x)
        else:
            with th.no_grad():
                jacobi = vmap(jacrev(self.f_net))(x)
        # calcute jac norm
        jac_norm = th.norm(jacobi, 2, dim=(1,2)).unsqueeze(1)
        # multi-dimensional gradient normalization (MGN)
        action = k_out * f_out / (jac_norm + self.eps)
        # squash action
        if self.squash_action:
            action = th.tanh(action)
        return action

    def pred_next(self, x):
        # predict(x) forward — no MGN, just pred_head
        f_out = self.f_net(x)
        action = self.pred_head(f_out)
        if self.squash_action:
            action = th.tanh(action)
        return action

    def forward_latent(self, x):
        """
        Return latent features from shared backbone (f_net output).
        """
        f_out = self.f_net(x)
        return f_out


# ============================================================
# ASAPLipsActor: TD3 Actor using ASAPLipsNet
# ============================================================

class ASAPLipsActor(Actor):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: list[int],
        lips_arch: list[int],
        lips_kwargs: dict,
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
        self.mu = ASAPLipsNet(
            f_sizes=[features_dim, *lips_kwargs["lips_f_size"], action_dim],
            f_hid_nonliear=nn.ReLU,
            f_out_nonliear=nn.Identity,
            global_lips=lips_kwargs["lips_global"],
            k_init=lips_kwargs["lips_k_init"],
            k_sizes=[features_dim, *lips_kwargs["lips_k_size"], 1],
            k_hid_act=nn.Tanh,
            k_out_act=nn.Softplus,
            loss_lambda=lips_kwargs["lips_lam"],
            eps=lips_kwargs["lips_eps"],
            squash_action=True,
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        return self.mu(features)

    def predict_next(self, obs: PyTorchObs, deterministic: bool = True) -> th.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        return self.mu.pred_next(features)

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self(observation)


# ============================================================
# ASAPLipsTD3Policy
# ============================================================

class ASAPLipsTD3Policy(TD3Policy):
    actor: ASAPLipsActor

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        lips_kwargs: Optional[dict[str, Any]] = dict(
            {
                "lips_lam": 1e-5,
                "lips_eps": 1e-4,
                "lips_k_init": [32],
                "lips_f_size": [64, 64],
                "lips_k_size": 1,
                "lips_global": False,
            }
        ),
    ):
        BasePolicy.__init__(
            self,
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
            normalize_images=normalize_images,
        )

        # Default network architecture
        if net_arch is None:
            net_arch = dict(pi=[], qf=[64, 64])

        net_arch.update(
            lips=dict(f=lips_kwargs["lips_f_size"], k=lips_kwargs["lips_k_size"])
        )

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)
        lips_arch = net_arch["lips"]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "lips_arch": lips_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
            "lips_kwargs": lips_kwargs,
        }
        self.actor_kwargs = self.net_args.copy()

        critic_base_kwargs = {
            k: v for k, v in self.net_args.items() if k not in ["lips_arch", "lips_kwargs"]
        }

        self.critic_kwargs = critic_base_kwargs
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
            }
        )

        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        # Create actor and target
        self.actor = self.make_actor(features_extractor=None)
        self.actor_target = self.make_actor(features_extractor=None)
        # Initialize the target to have the same weights as the actor
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
            self.critic_target = self.make_critic(features_extractor=self.actor_target.features_extractor)
        else:
            self.critic = self.make_critic(features_extractor=None)
            self.critic_target = self.make_critic(features_extractor=None)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.optimizer = self.optimizer_class(
            self.critic.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

        # Target networks should always be in eval mode
        self.actor_target.set_training_mode(False)
        self.critic_target.set_training_mode(False)

        # ASAP: next_actor_target for prediction head target
        self.next_actor_target = deepcopy(self.actor)
        for p in self.next_actor_target.parameters():
            p.requires_grad = False

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ASAPLipsActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return ASAPLipsActor(**actor_kwargs).to(self.device)

    def _predict_next(self, observation: PyTorchObs, deterministic: bool = True) -> th.Tensor:
        return self.actor.predict_next(observation, deterministic)

    def _predict_next_target(self, observation: PyTorchObs, deterministic: bool = True) -> th.Tensor:
        with th.no_grad():
            return self.next_actor_target.predict_next(observation, deterministic)

    def _polyak_update_targets(self, tau: float):
        # Polyak averaging for next_actor_target
        with th.no_grad():
            for p, p_targ in zip(self.actor.parameters(),
                                  self.next_actor_target.parameters()):
                p_targ.data.mul_(1.0 - tau)
                p_targ.data.add_(tau * p.data)


# ============================================================
# ASAP + LipsNet TD3 Algorithm
# ============================================================

class ASAPLipsTD3(CustomTD3):
    policy: ASAPLipsTD3Policy

    def __init__(
        self,
        policy: Union[str, type[ASAPLipsTD3Policy]],
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
        # === ASAP Hyperparameters ===
        asap_lamT: float = 0.1,
        asap_lamS: float = 0.5,
        asap_lamP: float = 0.1,
        asap_tau: float = 0.01,
        asap_target_update_interval: int = 1,
        # === LipsNet Hyperparameters ===
        lips_lam: float = 1e-5,
        lips_eps: float = 1e-4,
        lips_k_init: float = 50.0,
        lips_f_size: list = [64, 64],
        lips_k_size: list = [32],
        lips_global: bool = False,
    ):
        self.asap_lamT = asap_lamT
        self.asap_lamS = asap_lamS
        self.asap_lamP = asap_lamP
        self.asap_tau = asap_tau
        self.asap_target_update_interval = asap_target_update_interval

        if policy_kwargs is None:
            policy_kwargs = dict()
        policy_kwargs.update(
            {
                "lips_kwargs": dict(
                    {
                        "lips_lam": lips_lam,
                        "lips_eps": lips_eps,
                        "lips_k_init": lips_k_init,
                        "lips_f_size": lips_f_size,
                        "lips_k_size": lips_k_size,
                        "lips_global": lips_global,
                    }
                )
            }
        )
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

    def _setup_model(self):
        self.replay_buffer_class = GradBuffer
        return super()._setup_model()

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        asap_losses = []

        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

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
                # --- ASAP temporal loss ---
                grad_prev_actions = self.policy._predict(replay_data.prev_observations, deterministic=True).type(th.float32)
                grad_now_actions = self.policy._predict(replay_data.observations, deterministic=True).type(th.float32)
                grad_next_actions = self.policy._predict(replay_data.next_observations, deterministic=True).type(th.float32)

                derv_t = 0.5 * ((2 * grad_now_actions - grad_next_actions - grad_prev_actions) ** 2)

                delta = grad_next_actions - grad_prev_actions + 1e-4
                hdelta = F.tanh((1 / delta) ** 2).detach()

                loss_t = th.mean(derv_t * hdelta)

                # --- ASAP spatial loss ---
                d_observations = replay_data.observations
                d_next_observations = replay_data.next_observations

                d_predict_next_actions = self.policy._predict_next_target(d_observations, True).detach()
                d_next_actions = self.policy._predict(d_next_observations, True)

                spatial_loss = 0.5 * F.mse_loss(d_next_actions, d_predict_next_actions)
                asap_losses.append(spatial_loss.item())

                # --- ASAP predictor loss ---
                d_predict_next_actions_train = self.policy._predict_next(d_observations, True)
                d_next_actions_target = self.policy._predict(d_next_observations, True).detach()

                predictor_loss = 0.5 * F.mse_loss(d_predict_next_actions_train, d_next_actions_target)

                # Compute actor loss (Q-value + ASAP losses)
                actor_loss = (
                    -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()
                    + loss_t * self.asap_lamT
                    + spatial_loss * self.asap_lamS
                    + predictor_loss * self.asap_lamP
                )
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

            # ASAP target update for next_actor_target
            if self._n_updates % self.asap_target_update_interval == 0:
                self.policy._polyak_update_targets(self.asap_tau)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(asap_losses) > 0:
            self.logger.record("train/asap_loss", np.mean(asap_losses))

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
