import os
import sys
import jax
import time
import yaml
import wandb
import jraph
import optax
import distrax
import argparse
import numpy as np
import pandas as pd
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from pathlib import Path
from typing import Tuple
from ppo import make_train
from wrappers import LogWrapper
from collections import deque, Counter
from dfax import list2batch, batch2graph
from dfa_gym import TokenEnv, DFAWrapper
import flax.serialization as serialization
from flax.traverse_util import flatten_dict
from flax.training.train_state import TrainState
from rad_embeddings import Encoder, EncoderModule
from flax.linen.initializers import constant, orthogonal
from dfax.samplers import ReachSampler, ReachAvoidSampler, RADSampler


@struct.dataclass
class Transition():
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    hstate: jnp.ndarray | Tuple[jnp.ndarray, jnp.ndarray]

def make_train(config, env, network, batchify):
    config["NUM_AGENTS"] = env.num_agents
    config["NUM_ACTORS"] = config["NUM_AGENTS"] * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def exponential_schedule(count):
        updates_done = count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])
        return config["LR"] * jnp.exp(-config["EXP_DECAY_RATE"] * updates_done)

    def cosine_schedule(count):
        updates_done = count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])
        cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * updates_done / config["NUM_UPDATES"]))
        return config["LR"] * cosine_decay

    def warmup_schedule(count):
        updates_done = count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])
        updates_done = jnp.minimum(updates_done, config["NUM_UPDATES"])

        warmup_updates = (config["LR_ANNEAL_WARMUP_PARAM"] * config["NUM_UPDATES"])
        min_frac = config.get("LR_ANNEAL_MIN_FRAC", 0.0)

        warmup_lr = config["LR"] * (updates_done / jnp.maximum(1, warmup_updates))

        progress = (updates_done - warmup_updates) / jnp.maximum(1, config["NUM_UPDATES"] - warmup_updates)
        progress = jnp.clip(progress, 0.0, 1.0)

        cosine_part = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
        post_warmup_lr = config["LR"] * (min_frac + (1.0 - min_frac) * cosine_part)

        return jnp.where(updates_done < warmup_updates, warmup_lr, post_warmup_lr)


    def train(rng):
        # INIT NETWORK
        rng, _rng = jax.random.split(rng)
        init_x = jax.tree.map(lambda x: jnp.stack([x] * config["NUM_ACTORS"], axis=0), env.observation_space(env.agents[0]).sample(_rng))
        rng, _rng = jax.random.split(rng)
        if config["LSTM"]:
            init_hstate = nn.OptimizedLSTMCell(features=128).initialize_carry(_rng, (config["NUM_ACTORS"], 128))
        elif config["GRU"]:
            init_hstate = nn.GRUCell(features=128).initialize_carry(_rng, (config["NUM_ACTORS"], 128))
        rng, _rng = jax.random.split(rng)
        network_params = network.init(_rng, init_hstate, init_x)
        if config.get("LR_ANNEAL_LINEAR"):
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        elif config.get("LR_ANNEAL_EXP"):
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=exponential_schedule, eps=1e-5),
            )
        elif config.get("LR_ANNEAL_COS"):
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=cosine_schedule, eps=1e-5),
            )
        elif config.get("LR_ANNEAL_WARMUP"):
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=warmup_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_rng)

        # TRAIN LOOP
        def _update_step(runner_state, step_idx):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, hstate, rng = runner_state

                obs_batch = batchify(last_obs, env.agents)

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                new_hstate, pi, value = network.apply(train_state.params, hstate, obs_batch)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                _action = action.reshape((-1, config["NUM_ENVS"]))
                env_act = {agent: _action[i] for i, agent in enumerate(env.agents)}

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step)(rng_step, env_state, env_act)
                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                done = jnp.concatenate([done[agent] for agent in env.agents])
                transition = Transition(
                    done=done,
                    action=action,
                    value=value,
                    reward=jnp.concatenate([reward[agent] for agent in env.agents]),
                    log_prob=log_prob,
                    obs=obs_batch,
                    info=info,
                    hstate=hstate
                )
                mask = done[:, None]
                if config["LSTM"]:
                    new_c = (1.0 - mask) * new_hstate[0] + mask * init_hstate[0]
                    new_h = (1.0 - mask) * new_hstate[1] + mask * init_hstate[1]
                    new_hstate = (new_c, new_h)
                else:
                    new_hstate = (1.0 - mask) * new_hstate + mask * init_hstate
                runner_state = (train_state, env_state, obsv, new_hstate, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, hstate, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents)
            _, _, last_val = network.apply(train_state.params, hstate, last_obs_batch)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # PREPARE MASK
                        # mask = jnp.zeros(
                        #     (config["NUM_ENVS"], config["NUM_AGENTS"])
                        # ).at[:, step_idx % config["NUM_AGENTS"]].set(1).astype(jnp.float32).flatten()
                        # mask = jnp.tile(mask, config["NUM_STEPS"] // config["NUM_MINIBATCHES"])

                        # RERUN NETWORK
                        _, pi, value = network.apply(params, traj_batch.hstate, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        # value_loss = jnp.maximum(value_losses, value_losses_clipped)
                        # value_loss = 0.5 * (value_loss * mask).sum() / mask.sum()
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        # loss_actor = (loss_actor * mask).sum() / mask.sum()
                        loss_actor = loss_actor.mean()
                        # entropy = (pi.entropy() * mask).sum() / mask.sum()
                        entropy = pi.entropy().mean()

                        ent_coef = config["ENT_COEF"] * (1.0 - (step_idx * config["ENT_COEF_DECAY"]) / config["NUM_UPDATES"])

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - ent_coef * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                # Batching and Shuffling
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                # Mini-batch Updates
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss
            # Updating Training State and Metrics:
            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            metric["ent_coef"] = config["ENT_COEF"] * (1.0 - (step_idx * config["ENT_COEF_DECAY"]) / config["NUM_UPDATES"])

            steps_per_update = config["NUM_ENVS"] * config["NUM_STEPS"]

            if config.get("LOG"):
                ep_len_buffer_log = deque(maxlen=steps_per_update)
                return_buffer_log = deque(maxlen=steps_per_update)
                last_return_buffer_log = deque(maxlen=steps_per_update)
                disc_return_buffer_log = deque(maxlen=steps_per_update)
                start_time_log = time.time()

                def callback(info, loss_info):
                    nonlocal start_time_log

                    elapsed = time.time() - start_time_log

                    log = {}

                    timesteps = info["timestep"][-1, :]
                    timestep = int(np.sum(timesteps) / config["NUM_AGENTS"])
                    log["timestep"] = timestep

                    fps = (steps_per_update / elapsed) if elapsed > 0 else 0.0
                    log["fps"] = np.mean(fps)

                    ep_len_values = info["returned_episode_lengths"][info["returned_episode"]]
                    ep_len_buffer_log.extend(ep_len_values)

                    return_values = info["returned_episode_returns"][info["returned_episode"]]
                    return_buffer_log.extend(return_values)

                    last_return_values = info["returned_episode_last_returns"][info["returned_episode"]]
                    last_return_buffer_log.extend(last_return_values)

                    disc_return_values = info["returned_episode_disc_returns"][info["returned_episode"]]
                    disc_return_buffer_log.extend(disc_return_values)

                    log["ep_len_min"] = np.min(ep_len_buffer_log)
                    log["ep_len_mean"] = np.mean(ep_len_buffer_log)
                    log["ep_len_max"] = np.max(ep_len_buffer_log)
                    log["ep_len_std"] = np.std(ep_len_buffer_log)

                    log["return_min"] = np.min(return_buffer_log)
                    log["return_mean"] = np.mean(return_buffer_log)
                    log["return_max"] = np.max(return_buffer_log)
                    log["return_std"] = np.std(return_buffer_log)

                    log["disc_return_mean"] = np.mean(disc_return_buffer_log)

                    total_loss, (value_loss, actor_loss, entropy) = loss_info

                    log["total_loss"] = np.mean(total_loss)
                    log["value_loss"] = np.mean(value_loss)
                    log["actor_loss"] = np.mean(actor_loss)
                    log["entropy"] = np.mean(entropy)

                    n = len(last_return_buffer_log)
                    log["prob_fail"] = sum(r <= 0 for r in last_return_buffer_log) / n
                    log["prob_success"] = sum(r > 0 for r in last_return_buffer_log) / n

                    log["ent_coef"] = np.mean(info["ent_coef"])

                    log_file = Path(config.get("LOG"))
                    df = pd.DataFrame([log])
                    df.to_csv(
                        log_file,
                        mode="a",
                        header=not log_file.exists(),
                        index=False
                    )

                    start_time_log = time.time()
                jax.experimental.io_callback(callback, None, metric, loss_info)

            if config.get("WANDB"):
                ep_len_buffer_wandb = deque(maxlen=steps_per_update)
                return_buffer_wandb = deque(maxlen=steps_per_update)
                disc_return_buffer_wandb = deque(maxlen=steps_per_update)
                start_time_wandb = time.time()

                def callback(info, loss_info):
                    nonlocal start_time_wandb

                    elapsed = time.time() - start_time_wandb
                    fps = (steps_per_update / elapsed) if elapsed > 0 else 0.0

                    log = {
                        "fps": np.mean(fps),
                    }

                    ep_len_values = info["returned_episode_lengths"][info["returned_episode"]]
                    ep_len_buffer_wandb.extend(ep_len_values)

                    return_values = info["returned_episode_returns"][info["returned_episode"]]
                    return_buffer_wandb.extend(return_values)

                    disc_return_values = info["returned_episode_disc_returns"][info["returned_episode"]]
                    disc_return_buffer_wandb.extend(disc_return_values)

                    log["ep_len_min"] = np.min(ep_len_buffer_wandb)
                    log["ep_len_mean"] = np.mean(ep_len_buffer_wandb)
                    log["ep_len_max"] = np.max(ep_len_buffer_wandb)
                    log["ep_len_std"] = np.std(ep_len_buffer_wandb)

                    log["return_min"] = np.min(return_buffer_wandb)
                    log["return_mean"] = np.mean(return_buffer_wandb)
                    log["return_max"] = np.max(return_buffer_wandb)
                    log["return_std"] = np.std(return_buffer_wandb)

                    log["disc_return_mean"] = np.mean(disc_return_buffer_wandb)

                    total_loss, (value_loss, actor_loss, entropy) = loss_info

                    log["total_loss"] = np.mean(total_loss)
                    log["value_loss"] = np.mean(value_loss)
                    log["actor_loss"] = np.mean(actor_loss)
                    log["entropy"] = np.mean(entropy)

                    n = len(last_return_buffer_log)
                    log["prob_fail"] = sum(r <= 0 for r in last_return_buffer_log) / n
                    log["prob_success"] = sum(r > 0 for r in last_return_buffer_log) / n

                    log["ent_coef"] = np.mean(info["ent_coef"])

                    timesteps = info["timestep"][-1, :]
                    timestep = int(np.sum(timesteps) / config["NUM_AGENTS"])

                    wandb.log(log, step=timestep)

                    start_time_wandb = time.time()
                jax.experimental.io_callback(callback, None, metric, loss_info)
            
            # Debugging mode
            if config.get("DEBUG"):
                ep_len_buffer_debug = deque(maxlen=steps_per_update)
                return_buffer_debug = deque(maxlen=steps_per_update)
                disc_return_buffer_debug = deque(maxlen=steps_per_update)
                start_time_debug = time.time()

                def callback(info, loss_info):
                    nonlocal start_time_debug

                    elapsed = time.time() - start_time_debug
                    fps = (steps_per_update / elapsed) if elapsed > 0 else 0.0

                    log = {
                        "fps": np.mean(fps),
                    }

                    ep_len_values = info["returned_episode_lengths"][info["returned_episode"]]
                    ep_len_buffer_debug.extend(ep_len_values)

                    return_values = info["returned_episode_returns"][info["returned_episode"]]
                    return_buffer_debug.extend(return_values)

                    disc_return_values = info["returned_episode_disc_returns"][info["returned_episode"]]
                    disc_return_buffer_debug.extend(disc_return_values)

                    log["ep_len_min"] = np.min(ep_len_buffer_debug)
                    log["ep_len_mean"] = np.mean(ep_len_buffer_debug)
                    log["ep_len_max"] = np.max(ep_len_buffer_debug)
                    log["ep_len_std"] = np.std(ep_len_buffer_debug)

                    log["return_min"] = np.min(return_buffer_debug)
                    log["return_mean"] = np.mean(return_buffer_debug)
                    log["return_max"] = np.max(return_buffer_debug)
                    log["return_std"] = np.std(return_buffer_debug)

                    log["disc_return_mean"] = np.mean(disc_return_buffer_debug)

                    total_loss, (value_loss, actor_loss, entropy) = loss_info

                    log["total_loss"] = np.mean(total_loss)
                    log["value_loss"] = np.mean(value_loss)
                    log["actor_loss"] = np.mean(actor_loss)
                    log["entropy"] = np.mean(entropy)

                    timesteps = info["timestep"][-1, :]
                    log["timestep"] = int(np.sum(timesteps) / config["NUM_AGENTS"])

                    n = len(last_return_buffer_log)
                    log["prob_fail"] = sum(r <= 0 for r in last_return_buffer_log) / n
                    log["prob_success"] = sum(r > 0 for r in last_return_buffer_log) / n

                    log["ent_coef"] = np.mean(info["ent_coef"])

                    jax.debug.print(
                        """
timestep         = {timestep}
prob_success     = {prob_success}
prob_fail        = {prob_fail}
disc_return_mean = {disc_return_mean}
return_mean      = {return_mean}
return_std       = {return_std}
return_min       = {return_min}
return_max       = {return_max}
ep_len_min       = {ep_len_min}
ep_len_mean      = {ep_len_mean}
ep_len_max       = {ep_len_max}
ep_len_std       = {ep_len_std}
total_loss       = {total_loss}
value_loss       = {value_loss}
actor_loss       = {actor_loss}
entropy          = {entropy}
ent_coef         = {ent_coef}
fps              = {fps}
                        """,
                        timestep=log["timestep"],
                        prob_success=log["prob_success"],
                        prob_fail=log["prob_fail"],
                        disc_return_mean=log["disc_return_mean"],
                        return_mean=log["return_mean"],
                        return_std=log["return_std"],
                        return_min=log["return_min"],
                        return_max=log["return_max"],
                        ep_len_min=log["ep_len_min"],
                        ep_len_mean=log["ep_len_mean"],
                        ep_len_max=log["ep_len_max"],
                        ep_len_std=log["ep_len_std"],
                        total_loss=log["total_loss"],
                        value_loss=log["value_loss"],
                        actor_loss=log["actor_loss"],
                        entropy=log["entropy"],
                        ent_coef=log["ent_coef"],
                        fps=log["fps"],
                        ordered=True)

                    start_time_debug = time.time()

                jax.debug.callback(callback, metric, loss_info)

            runner_state = (train_state, env_state, last_obs, hstate, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, init_hstate, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, jnp.arange(config["NUM_UPDATES"])
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


class ActorCritic(nn.Module):
    action_dim: int
    encoder: Encoder
    n_agents: int
    deterministic: bool = False
    padding: str = "VALID"
    kind: str = "lstm"  # or "gru"

    @nn.compact
    def __call__(self, hstate, batch):

        _id_batch = batch["_id"]
        if _id_batch.ndim == 0:
            _id_batch = _id_batch[None, ...] # -> (1,)
        elif _id_batch.ndim != 1:
            raise ValueError(f"Expected () or (B,), got {_id_batch.shape} for agent_id")
        _id_embd = nn.Embed(self.n_agents, 32)(_id_batch)

        obs_batch = batch["obs"]
        if obs_batch.ndim == 3: # (C, H, W)
            obs_batch = obs_batch[None, ...] # -> (1, C, H, W)
        elif obs_batch.ndim != 4:
            raise ValueError(f"Expected (C, H, W) or (B, C, H, W), got {obs_batch.shape} for obs")
        obs_batch = jnp.transpose(obs_batch, (0, 2, 3, 1)) # -> (B, H, W, C)

        obs_feat = nn.Sequential([
            nn.Conv(16, (2, 2), padding=self.padding, kernel_init=orthogonal(np.sqrt(2))),
            nn.relu,
            nn.Conv(32, (2, 2), padding=self.padding, kernel_init=orthogonal(np.sqrt(2))),
            nn.relu,
            nn.Conv(64, (2, 2), padding=self.padding, kernel_init=orthogonal(np.sqrt(2))),
            nn.relu,
            lambda x: x.reshape((x.shape[0], -1)),
        ])(obs_batch)

        dfa_batch = batch["dfa"]
        dfa_graph = batch2graph(dfa_batch)
        dfa_feat = self.encoder(dfa_graph)

        batch_size = _id_batch.shape[0]
        rad_size = dfa_feat.shape[-1]
        dfa_feat = dfa_feat.reshape(batch_size, self.n_agents, rad_size)

        def move_to_front(feat_batch: jnp.ndarray, id_batch: jnp.ndarray):
            base = jnp.arange(self.n_agents, dtype=id_batch.dtype)[None, :]
            sentinel = jnp.array(self.n_agents, dtype=id_batch.dtype)
            rem = jnp.where(base == id_batch[:, None], sentinel, base)
            remaining = jnp.sort(rem, axis=1)[:, : self.n_agents - 1]
            perm = jnp.concatenate([id_batch[:, None], remaining], axis=1)
            batch_idx = jnp.arange(batch_size)[:, None]
            return feat_batch[batch_idx, perm, :]

        dfa_feat = move_to_front(dfa_feat, _id_batch)
        dfa_feat = dfa_feat.reshape(batch_size, -1)

        task_feat = jnp.concatenate([_id_embd, obs_feat, dfa_feat], axis=-1)

        if self.kind == "lstm":
            new_hstate, task_embd = nn.OptimizedLSTMCell(features=128)(hstate, task_feat)
        elif self.kind == "gru":
            new_hstate, task_embd = nn.GRUCell(features=128)(hstate, task_feat)
        else:
            raise ValueError(f"Unknown RNN kind {self.kind}")

        feat = jnp.concatenate([_id_embd, obs_feat, task_embd], axis=-1)

        value = nn.Sequential([
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            nn.relu,
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            nn.relu,
            nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
        ])(feat)

        logits = nn.Sequential([
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            nn.relu,
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            nn.relu,
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            nn.relu,
            nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))
        ])(feat)

        if self.deterministic:
            action = jnp.argmax(logits, axis=-1)
            return new_hstate, action, jnp.squeeze(value, axis=-1)
        else:
            pi = distrax.Categorical(logits=logits)
            return new_hstate, pi, jnp.squeeze(value, axis=-1)


def _batchify(obss: dict, agents):

    _id_batch = jnp.stack([obss[agent]["_id"] for agent in agents], axis=0)
    _id_batch = _id_batch if _id_batch.ndim == 1 else jnp.concatenate(_id_batch, axis=0)

    obs_batch = jnp.stack([obss[agent]["obs"] for agent in agents], axis=0)
    obs_batch = obs_batch if obs_batch.ndim == 4 else jnp.concatenate(obs_batch, axis=0)

    dfa_batch = list2batch([obss[agent]["dfa"] for agent in agents])

    batch = {
        "_id": _id_batch,
        "obs": obs_batch,
        "dfa": dfa_batch
    }

    return batch


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train TokenEnv policy")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for PRNGKey (default: 42)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config file"
    )
    parser.add_argument(
        "--no-rad",
        action="store_true",
        help="Don't use pretrained RAD embeddings"
    )
    parser.add_argument(
        "--no-pbrs",
        action="store_true",
        help="Don't use PBRS"
    )
    parser.add_argument(
        "--lstm",
        action="store_true",
        help="Use OptimizedLSTMCell"
    )
    parser.add_argument(
        "--gru",
        action="store_true",
        help="Use GRUCell"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    assert config is not None

    assert args.lstm != args.gru

    config["LSTM"] = args.lstm
    config["GRU"] = args.gru

    if config["LSTM"]:
        recurrent_str = "lstm"
    else:
        recurrent_str = "gru"

    if config["WANDB"]:
        wandb.init(
            entity=config["WANDB_ENTITY"],
            project=config["WANDB_PROJECT"],
            config=config
        )

    token_env = TokenEnv(
        layout=config["LAYOUT"],
        max_steps_in_episode=config["MAX_EP_LEN"]
    )

    if config["DFA_SAMPLER"] == "Reach":
        sampler = ReachSampler(
            p=config["DFA_SIZE_P"],
            max_size=config["DFA_MAX_SIZE"],
            prob_stutter=config["DFA_PROB_STUTTER"],
            n_tokens=token_env.n_tokens
        )
    elif config["DFA_SAMPLER"] == "ReachAvoid":
        sampler = ReachAvoidSampler(
            p=config["DFA_SIZE_P"],
            max_size=config["DFA_MAX_SIZE"],
            prob_stutter=config["DFA_PROB_STUTTER"],
            n_tokens=token_env.n_tokens
        )
    elif config["DFA_SAMPLER"] == "RAD":
        sampler = RADSampler(
            p=config["DFA_SIZE_P"],
            max_size=config["DFA_MAX_SIZE"],
            prob_stutter=config["DFA_PROB_STUTTER"],
            n_tokens=token_env.n_tokens
        )
    else:
        raise ValueError

    if args.no_pbrs:
        pbrs_str = "no_pbrs"
        gamma = None
    else:
        pbrs_str = "pbrs"
        gamma = config["GAMMA"]

    env = DFAWrapper(
        env=token_env,
        gamma=gamma,
        sampler=sampler,
        progress=False
    )
    env = LogWrapper(env=env, config=config)

    if args.no_rad:
        rad_str = "no_rad"
        encoder = EncoderModule(
            max_size=env.sampler.max_size
        )
    else:
        rad_str = "rad"
        encoder = Encoder(
            max_size=env.sampler.max_size,
            n_tokens=token_env.n_tokens,
            seed=args.seed
        )

    config["LOG"] = f"""{config["LOG_FILE_PREFIX"]}_{rad_str}_{pbrs_str}_{recurrent_str}_{args.seed}.csv"""

    network = ActorCritic(
        action_dim=env.action_space(env.agents[0]).n,
        encoder=encoder,
        n_agents=env.num_agents,
        kind="lstm" if config["LSTM"] else "gru"
    )

    key = jax.random.PRNGKey(args.seed)

    if config["DEBUG"]:
        key, subkey = jax.random.split(key)
        init_x = jax.tree.map(lambda x: jnp.stack([x] * env.num_agents * config["NUM_ENVS"], axis=0), env.observation_space(env.agents[0]).sample(subkey))
        key, subkey = jax.random.split(key)
        if config["LSTM"]:
            init_hstate = nn.OptimizedLSTMCell(features=128).initialize_carry(subkey, (env.num_agents * config["NUM_ENVS"], 128))
        elif config["GRU"]:
            init_hstate = nn.GRUCell(features=128).initialize_carry(subkey, (env.num_agents * config["NUM_ENVS"], 128))
        key, subkey = jax.random.split(key)
        params = network.init(subkey, init_hstate, init_x)
        flat = flatten_dict(params, sep="/")
        total = 0
        for k, v in flat.items():
            count = v.size
            total += count
            print(f"{k:60} {v.shape} {v.dtype} ({count:,} params)")
        print(f"\nTotal parameters: {total:,}")

    train_jit = jax.jit(make_train(config, env, network, _batchify))
    out = train_jit(key)

    trained_params = out["runner_state"][0].params
    with open(f"""{config["SAVE_FILE_PREFIX"]}_{rad_str}_{pbrs_str}_{recurrent_str}_{args.seed}""", "wb") as f:
        f.write(serialization.to_bytes(trained_params))

    if config["WANDB"]:
        wandb.finish()

