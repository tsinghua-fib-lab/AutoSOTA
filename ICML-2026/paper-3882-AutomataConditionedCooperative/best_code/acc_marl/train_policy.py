import os
import sys
import jax
import yaml
import wandb
import jraph
import distrax
import argparse
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from ppo import make_train
from wrappers import LogWrapper
from dfax import list2batch, batch2graph
from dfa_gym import TokenEnv, DFAWrapper
from rad_embeddings import Encoder, EncoderModule
import flax.serialization as serialization
from flax.traverse_util import flatten_dict
from flax.linen.initializers import constant, orthogonal
from dfax.samplers import ReachSampler, ReachAvoidSampler, RADSampler


class ActorCritic(nn.Module):
    action_dim: int
    encoder: Encoder
    n_agents: int
    deterministic: bool = False
    padding: str = "VALID"

    @nn.compact
    def __call__(self, batch):

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

        task_embd = nn.Sequential([
            nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            nn.tanh,
            nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            nn.tanh,
            nn.Dense(32, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        ])(task_feat)

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
            return action, jnp.squeeze(value, axis=-1)
        else:
            pi = distrax.Categorical(logits=logits)
            return pi, jnp.squeeze(value, axis=-1)


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
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    assert config is not None

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
        cross_pbrs_alpha=config.get("CROSS_PBRS_ALPHA", 0.0)
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

    config["LOG"] = f"""{config["LOG_FILE_PREFIX"]}_{rad_str}_{pbrs_str}_{args.seed}.csv"""

    network = ActorCritic(
        action_dim=env.action_space(env.agents[0]).n,
        encoder=encoder,
        n_agents=env.num_agents
    )

    key = jax.random.PRNGKey(args.seed)

    if config["DEBUG"]:
        key, subkey = jax.random.split(key)
        init_x = env.observation_space(env.agents[0]).sample(subkey)
        key, subkey = jax.random.split(key)
        params = network.init(subkey, init_x)
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
    with open(f"""{config["SAVE_FILE_PREFIX"]}_{rad_str}_{pbrs_str}_{args.seed}""", "wb") as f:
        f.write(serialization.to_bytes(trained_params))

    if config["WANDB"]:
        wandb.finish()

