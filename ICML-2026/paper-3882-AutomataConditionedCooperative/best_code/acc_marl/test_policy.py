import jax
import yaml
import argparse
import itertools
import jax.numpy as jnp
from functools import partial
from rad_embeddings import Encoder, EncoderModule
from flax.traverse_util import flatten_dict
from dfa_gym import TokenEnv, DFAWrapper
import flax.serialization as serialization
from dfax.samplers import ReachSampler, ReachAvoidSampler, RADSampler
from train_policy import ActorCritic, _batchify
from collections import Counter


if __name__ == "__main__":

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description="Train TokenEnv policy")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="List of seeds for testing (must match trained checkpoints)"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of samples (default: 10)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=-1,
        help="Batch size (default: n)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config file"
    )
    parser.add_argument(
        "--rad",
        type=str2bool,
        required=True,
        help="Whether to use pretrained RAD embeddings"
    )
    parser.add_argument(
        "--pbrs",
        type=str2bool,
        required=True,
        help="Whether to use PBRS"
    )
    parser.add_argument(
        "--assign",
        type=str2bool,
        required=True,
        help="Whether to optimally assign DFAs to agents at the beginning of the episode"
    )
    parser.add_argument(
        "--sampler",
        type=str,
        required=True,
        help="DFA sampler for testing (default: RAD)"
    )
    parser.add_argument(
        "--ood",
        type=str2bool,
        required=True,
        help="Whether to sample out of distribution DFAs -- doubles max DFA size and max episode length given in config"
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Output results on a comma seperated line"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    assert config is not None

    batch_size = args.batch_size
    if batch_size == -1:
        batch_size = args.n

    k = 1
    if args.ood:
        k = 2

    token_env = TokenEnv(
        layout=config["LAYOUT"],
        max_steps_in_episode=config["MAX_EP_LEN"]*k
    )

    if args.sampler in ["R", "Reach"]:
        sampler = ReachSampler(
            p=config["DFA_SIZE_P"],
            max_size=config["DFA_MAX_SIZE"]*k,
            prob_stutter=config["DFA_PROB_STUTTER"],
            n_tokens=token_env.n_tokens
        )
    elif args.sampler in ["RA", "ReachAvoid"]:
        sampler = ReachAvoidSampler(
            p=config["DFA_SIZE_P"],
            max_size=config["DFA_MAX_SIZE"]*k,
            prob_stutter=config["DFA_PROB_STUTTER"],
            n_tokens=token_env.n_tokens
        )
    elif args.sampler in ["RAD", "ReachAvoidDerived"]:
        sampler = RADSampler(
            p=config["DFA_SIZE_P"],
            max_size=config["DFA_MAX_SIZE"]*k,
            prob_stutter=config["DFA_PROB_STUTTER"],
            n_tokens=token_env.n_tokens
        )
    else:
        raise ValueError

    if args.pbrs:
        pbrs_str = "pbrs"
        gamma = config["GAMMA"]
    else:
        pbrs_str = "no_pbrs"
        gamma = None

    env = DFAWrapper(
        env=token_env,
        gamma=gamma,
        sampler=sampler
    )

    n_seeds = len(args.seeds)
    success_rate_list = jnp.zeros((n_seeds,))
    avg_len_list = jnp.zeros((n_seeds,))
    avg_reward_list = jnp.zeros((n_seeds,))
    avg_disc_return_list = jnp.zeros((n_seeds,))

    for i, seed in enumerate(args.seeds):

        if args.rad:
            rad_str = "rad"
            encoder = Encoder(
                max_size=env.sampler.max_size,
                n_tokens=token_env.n_tokens,
                seed=seed
            )
        else:
            rad_str = "no_rad"
            encoder = EncoderModule(
                max_size=env.sampler.max_size
            )

        network = ActorCritic(
            action_dim=env.action_space(env.agents[0]).n,
            encoder=encoder,
            n_agents=env.num_agents
        )

        key = jax.random.PRNGKey(seed + 100) # Use a different seed to avoid correlations between train and test

        key, subkey = jax.random.split(key)
        init_x = env.observation_space(env.agents[0]).sample(subkey)
        key, subkey = jax.random.split(key)
        params = network.init(subkey, init_x)

        with open(f"""{config["SAVE_FILE_PREFIX"]}_{rad_str}_{pbrs_str}_{seed}""", "rb") as f:
            params = serialization.from_bytes(params, f.read())

        @partial(jax.jit, static_argnums=(0, 1))
        def reorder_dfas_by_value(env, network, params, state):
            n_agents = len(env.agents)
            dfas_list = [state.dfas[agent] for agent in env.agents]
            best_dfas = {agent: state.dfas[agent]  for agent in env.agents}
            best_value = -jnp.inf
            for perm in itertools.permutations(range(n_agents)):
                permuted_dfas = {agent: dfas_list[perm[i]] for i, agent in enumerate(env.agents)}
                new_state = state.replace(init_dfas=permuted_dfas).replace(dfas=permuted_dfas)
                new_obs = env.get_obs(new_state)
                _, values = network.apply(params, _batchify(new_obs, env.agents))
                value = jnp.sum(values, axis=-1)
                better = value > best_value
                best_value = jnp.where(better, value, best_value)
                for agent in env.agents:
                    best_dfas[agent] = jax.tree_util.tree_map(
                        lambda new_leaf, old_leaf, b=better: jnp.where(b, new_leaf, old_leaf),
                        permuted_dfas[agent],
                        best_dfas[agent]
                    )
            return state.replace(init_dfas=best_dfas).replace(dfas=best_dfas)

        @partial(jax.jit, static_argnums=(0, 1))
        def run_episode(env, network, params, key):
            key, subkey = jax.random.split(key)
            obs, state = env.reset(subkey)
            if args.assign:
                state = reorder_dfas_by_value(env, network, params, state)
                obs = env.get_obs(state)

            carry = {
                "key": key,
                "obs": obs,
                "state": state,
                "done": False,
                "success": 0,
                "ep_len": 0,
                "ep_reward": 0.0,
                "ep_discounted_return": 0.0,
                "discount": 1.0,
            }

            def cond_fn(carry):
                return ~carry["done"]

            def body_fn(carry):
                key, subkey = jax.random.split(carry["key"])
                pi, value = network.apply(params, _batchify(carry["obs"], env.agents))
                actions = pi.sample(seed=subkey)
                actions = {agent: actions[i] for i, agent in enumerate(env.agents)}
                key, subkey = jax.random.split(key)
                obs, state, rewards, dones, infos = env.step(subkey, carry["state"], actions)

                done = dones["__all__"]
                rewards_arr = jnp.array([rewards[a] for a in env.agents])
                success = carry["success"] + jnp.all(rewards_arr > 0) * done
                ep_reward = carry["ep_reward"] + jnp.mean(rewards_arr)
                ep_discounted_return = carry["ep_discounted_return"] + carry["discount"] * jnp.mean(rewards_arr)
                discount = carry["discount"] * (env.gamma if env.gamma is not None else 1.0)
                ep_len = carry["ep_len"] + 1

                return {
                    "key": key,
                    "obs": obs,
                    "state": state,
                    "done": done,
                    "success": success,
                    "ep_len": ep_len,
                    "ep_reward": ep_reward,
                    "ep_discounted_return": ep_discounted_return,
                    "discount": discount,
                }

            final_carry = jax.lax.while_loop(cond_fn, body_fn, carry)
            return (
                final_carry["success"],
                final_carry["ep_len"],
                final_carry["ep_reward"],
                final_carry["ep_discounted_return"]
            )

        @partial(jax.jit, static_argnums=(0, 1))
        def run_episodes(env, network, params, keys):
            return jax.vmap(run_episode, (None, None, None, 0))(env, network, params, keys)

        def batched_vmap_run(env, network, params, keys, batch_size):
            n = keys.shape[0]
            results = []
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_keys = keys[start:end]
                batch_results = run_episodes(env, network, params, batch_keys)
                results.append(batch_results)
            results = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *results)
            return results

        keys = jax.random.split(key, args.n)
        results = batched_vmap_run(env, network, params, keys, batch_size=batch_size)

        success_counts, ep_lens, ep_rewards, ep_disc_returns = results

        success_rate = jnp.mean(success_counts)
        avg_len = jnp.mean(ep_lens)
        avg_reward = jnp.mean(ep_rewards)
        avg_disc_return = jnp.mean(ep_disc_returns)

        success_rate_list = success_rate_list.at[i].set(success_rate)
        avg_len_list = avg_len_list.at[i].set(avg_len)
        avg_reward_list = avg_reward_list.at[i].set(avg_reward)
        avg_disc_return_list = avg_disc_return_list.at[i].set(avg_disc_return)

    success_rate_mean = jnp.mean(success_rate_list)
    success_rate_std = jnp.std(success_rate_list)

    avg_len_mean = jnp.mean(avg_len_list)
    avg_len_std = jnp.std(avg_len_list)

    avg_reward_mean = jnp.mean(avg_reward_list)
    avg_reward_std = jnp.std(avg_reward_list)

    avg_disc_return_mean = jnp.mean(avg_disc_return_list)
    avg_disc_return_std = jnp.std(avg_disc_return_list)

    if args.csv:
        print(f"{args.config}, {rad_str}_{pbrs_str}, {args.sampler}, {args.ood}, {args.assign}, {success_rate_mean} +/- {success_rate_std}, {avg_len_mean} +/- {avg_len_std}, {avg_reward_mean} +/- {avg_reward_std}, {avg_disc_return_mean} +/- {avg_disc_return_std}")
    else:
        print(f"Test completed for {n_seeds} seeds.")
        print(f"Success rate: {success_rate_mean:.2f} +/- {success_rate_std:.2f}")
        print(f"Average episode length: {avg_len_mean:.2f} +/- {avg_len_std:.2f}")
        print(f"Average episode reward: {avg_reward_mean:.2f} +/- {avg_reward_std:.2f}")
        print(f"Average episode discounted return: {avg_disc_return_mean:.2f} +/- {avg_disc_return_std:.2f}")

