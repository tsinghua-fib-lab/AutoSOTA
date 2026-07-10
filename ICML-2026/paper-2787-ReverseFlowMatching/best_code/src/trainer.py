import functools

import gymnasium as gym
import numpy as onp
import tqdm

from src.components.buffer import ReplayBuffer


def _extract_obs(observation):
    if isinstance(observation, dict) and "observation" in observation:
        return observation["observation"]
    return observation


class Trainer:
    def __init__(
        self,
        env_fn,
        agent,
        logger,
        num_train_envs=1,
        num_eval_envs=10,
        episode_length=None,
        max_steps=int(1e6),
        replay_buffer_size=int(1e6),
        start_training=int(1e4),
        eval_interval=int(1e5),
        log_interval=int(1000),
        batch_size=512,
        reward_scale=1.0,
        num_updates_per_step=1,
        seed=0,
    ):
        self.env_fn = env_fn
        self.agent = agent
        self.logger = logger

        self.num_train_envs = num_train_envs  # currently only supports 1
        self.num_eval_envs = num_eval_envs
        self.max_steps = max_steps
        self.start_training = start_training
        self.eval_interval = eval_interval
        self.log_interval = log_interval
        self.batch_size = batch_size
        self.num_updates_per_step = num_updates_per_step
        self.reward_scale = reward_scale
        self.seed = seed

        self.env = self.env_fn(seed=self.seed)
        obs0, _ = self.env.reset(seed=self.seed)
        obs0_extracted = _extract_obs(obs0)

        observation_space = getattr(self.env, "observation_space", None)
        if (
            isinstance(obs0, dict)
            and hasattr(observation_space, "__getitem__")
            and "observation" in observation_space.spaces
        ):
            buffer_obs_space = observation_space["observation"]
        else:
            buffer_obs_space = observation_space

        self.replay_buffer = ReplayBuffer(
            buffer_obs_space, self.env.action_space, replay_buffer_size
        )

        if episode_length is None:
            raise ValueError(
                "episode_length must be provided; set env.episode_length in the config (or ensure main.py resolves it)."
            )
        if int(episode_length) <= 0:
            raise ValueError(f"episode_length must be positive; got {episode_length}")
        self.episode_length = int(episode_length)

        self.step = 0
        self._last_obs = obs0_extracted
        self._episode_steps = 0

    def _sample_actions(self, observation, deterministic=False):
        out = self.agent.sample_actions(observation, deterministic=deterministic)
        if isinstance(out, tuple):
            return out[0]
        return out

    def _mask_function(self, terminated, truncated):
        return 1.0 if (not terminated or truncated) else 0.0

    def train(self):
        observation = self._last_obs
        self.step = 0
        self._episode_steps = 0

        for i in tqdm.tqdm(
            range(0, self.max_steps, self.num_train_envs),
            smoothing=0.1,
        ):
            self.step = i

            if i < self.start_training:
                action = self.env.action_space.sample()
            else:
                action = self._sample_actions(observation)

            next_obs_raw, reward, terminated, truncated, info = self.env.step(action)
            reward = self.reward_scale * reward

            self._episode_steps += 1
            if (
                (not terminated)
                and (not truncated)
                and (self._episode_steps >= self.episode_length)
            ):
                truncated = True

            mask = self._mask_function(terminated, truncated)

            next_observation = _extract_obs(next_obs_raw)
            self.replay_buffer.insert(
                observation,
                action,
                reward,
                mask,
                onp.asarray(terminated or truncated, dtype=onp.float32),
                next_observation,
            )
            observation = next_observation

            if terminated or truncated:
                observation, _ = self.env.reset()
                observation = _extract_obs(observation)
                self._episode_steps = 0

            if i >= self.start_training:
                for _ in range(self.num_updates_per_step):
                    batch = self.replay_buffer.sample(self.batch_size)
                    train_metrics = self.agent.update(*batch)

                if i % self.log_interval == 0:
                    metrics = {"step": self.step}
                    metrics.update(train_metrics)
                    self.logger.log(metrics, "train")

            if i % self.eval_interval == 0:
                metrics = {"step": self.step}
                eval_metrics = self.evaluate()
                metrics.update(eval_metrics)
                self.logger.log(metrics, "eval")
                self.logger.save_agent(agent=self.agent, step=self.step)

        self.step += self.num_train_envs
        metrics = {"step": self.step}
        eval_metrics = self.evaluate()
        metrics.update(eval_metrics)
        self.logger.log(metrics, "eval")
        self.logger.save_agent(agent=self.agent, step=self.step)
        self.logger.finish(agent=self.agent, step=self.step)

    def evaluate(self):
        envs = gym.vector.SyncVectorEnv(
            [functools.partial(self.env_fn, seed=i) for i in range(self.num_eval_envs)]
        )
        try:
            obs, _ = envs.reset()
            obs = _extract_obs(obs)
            ep_rewards = onp.zeros(self.num_eval_envs)
            alive = onp.ones(self.num_eval_envs, dtype=onp.bool_)
            ep_lengths = onp.zeros(self.num_eval_envs, dtype=onp.int32)

            for t in range(self.episode_length):
                action = self._sample_actions(obs, deterministic=True)
                next_obs_raw, reward, terminated, truncated, info = envs.step(action)
                ep_rewards += onp.array(reward) * alive
                ep_lengths += alive.astype(onp.int32)
                done = onp.logical_or(terminated, truncated)
                alive = onp.logical_and(alive, ~done)
                obs = _extract_obs(next_obs_raw)
                if not onp.any(alive):
                    break

            return {
                "episode_reward": onp.nanmean(ep_rewards),
                "episode_length": float(onp.mean(ep_lengths)),
            }
        finally:
            try:
                envs.close()
            except Exception:
                pass
