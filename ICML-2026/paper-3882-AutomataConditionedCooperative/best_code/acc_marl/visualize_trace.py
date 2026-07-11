import jax
import dfax
import yaml
import dfa_gym
import argparse
import itertools
import jax.numpy as jnp
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
        "--seed",
        type=int,
        required=True,
        help="Seed for testing (must match trained checkpoints)"
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
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    assert config is not None

    k = 1
    if args.ood:
        k = 2

    token_env = TokenEnv(
        layout=config["LAYOUT"],
        max_steps_in_episode=config["MAX_EP_LEN"]*k
    )

    if args.sampler == "R" or args.sampler == "Reach":
        sampler = ReachSampler(
            p=config["DFA_SIZE_P"],
            max_size=config["DFA_MAX_SIZE"]*k,
            prob_stutter=config["DFA_PROB_STUTTER"],
            n_tokens=token_env.n_tokens
        )
    elif args.sampler == "RA" or args.sampler == "ReachAvoid":
        sampler = ReachAvoidSampler(
            p=config["DFA_SIZE_P"],
            max_size=config["DFA_MAX_SIZE"]*k,
            prob_stutter=config["DFA_PROB_STUTTER"],
            n_tokens=token_env.n_tokens
        )
    elif args.sampler == "RAD" or args.sampler == "ReachAvoidDerived":
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

    if args.rad:
        rad_str = "rad"
        encoder = Encoder(
            max_size=env.sampler.max_size,
            n_tokens=token_env.n_tokens,
            seed=args.seed
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

    key = jax.random.PRNGKey(args.seed + 100) # Use a different seed to avoid correlations between train and test

    key, subkey = jax.random.split(key)
    init_x = env.observation_space(env.agents[0]).sample(subkey)
    key, subkey = jax.random.split(key)
    params = network.init(subkey, init_x)

    with open(f"""{config["SAVE_FILE_PREFIX"]}_{rad_str}_{pbrs_str}_{args.seed}""", "rb") as f:
        params = serialization.from_bytes(params, f.read())

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

    def run_episode(env, network, params, key):
        key, subkey = jax.random.split(key)
        obs, state = env.reset(subkey)
        if args.assign:
            state = reorder_dfas_by_value(env, network, params, state)
            obs = env.get_obs(state)

        from dfax import DFAx
        # dfa_1 = DFAx(
        #     start=0,
        #     transitions=jnp.array([
        #         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
        #         [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        #         [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        #     ]),
        #     labels=jnp.array([False, False, True, False])
        # ).minimize()
        # dfa_2 = DFAx(
        #     start=0,
        #     transitions=jnp.array([
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #         [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        #         [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        #     ]),
        #     labels=jnp.array([False, True, False, False])
        # ).minimize()
        # # 2buttons_2agents hold_the_door
        # dfa_1 = DFAx(
        #     start=0,
        #     transitions=jnp.array([
        #         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [1, 1, 1, 1, 2, 1, 1, 1, 3, 1],
        #         [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        #         [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        #         [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        #     ]),
        #     labels=jnp.array([False, False, True, False, False])
        # ).minimize()
        # dfa_2 = DFAx(
        #     start=0,
        #     transitions=jnp.array([
        #         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        #         [1, 1, 1, 3, 2, 1, 1, 1, 1, 1],
        #         [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        #         [3, 1, 3, 3, 3, 3, 3, 3, 3, 3],
        #         [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        #     ]),
        #     labels=jnp.array([False, False, True, False, False])
        # ).minimize()
        # # 2buttons_2agents helper_agent
        # dfa_1 = DFAx(
        #     start=0,
        #     transitions=jnp.array([
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #         [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        #         [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        #         [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        #     ]),
        #     labels=jnp.array([True, False, False, False, False])
        # ).minimize()
        # dfa_2 = DFAx(
        #     start=0,
        #     transitions=jnp.array([
        #         [0, 0, 0, 3, 0, 0, 3, 0, 1, 0],
        #         [1, 1, 1, 3, 1, 2, 1, 1, 1, 3],
        #         [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        #         [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        #         [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        #     ]),
        #     labels=jnp.array([False, False, True, False, False])
        # ).minimize()
        # # 2rooms_2agents short_circuit
        # dfa_1 = DFAx(
        #     start=0,
        #     transitions=jnp.array([
        #         [1, 0, 0, 4, 0, 0, 0, 0, 0, 0],
        #         [1, 1, 2, 1, 1, 1, 1, 1, 1, 1],
        #         [2, 2, 2, 2, 3, 2, 2, 2, 2, 2],
        #         [3, 3, 3, 3, 3, 3, 4, 3, 3, 3],
        #         [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        #     ]),
        #     labels=jnp.array([False, False, False, False, True])
        # ).minimize()
        # dfa_2 = DFAx(
        #     start=0,
        #     transitions=jnp.array([
        #         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [1, 2, 1, 1, 1, 1, 1, 1, 1, 1],
        #         [2, 2, 3, 2, 2, 2, 2, 2, 2, 2],
        #         [3, 3, 3, 4, 3, 3, 3, 3, 3, 3],
        #         [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        #     ]),
        #     labels=jnp.array([False, False, False, False, True])
        # ).minimize()
        # 2rooms_2agents helper_agent
        dfa_1 = DFAx(
            start=0,
            transitions=jnp.array([
                [0, 0, 0, 1, 0, 0, 0, 0, 3, 3],
                [2, 1, 1, 1, 1, 1, 1, 1, 3, 3],
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            ]),
            labels=jnp.array([False, False, True, False, False])
        ).minimize()
        dfa_2 = DFAx(
            start=0,
            transitions=jnp.array([
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            ]),
            labels=jnp.array([True, False, False, False, False])
        ).minimize()
        my_dfas = [dfa_1, dfa_2]
        dfax.visualize(dfa_1, save_path="gifs/2rooms_2agents/helper_agent/agent_1_dfa.pdf")
        dfax.visualize(dfa_2, save_path="gifs/2rooms_2agents/helper_agent/agent_2_dfa.pdf")

        dfas = {agent: my_dfas[i] for i, agent in enumerate(env.agents)}
        state = state.replace(init_dfas=dfas).replace(dfas=dfas)

        done = False
        success = False
        trace = [state]
        while not done:
            key, subkey = jax.random.split(key)
            pi, value = network.apply(params, _batchify(obs, env.agents))
            actions = pi.sample(seed=subkey)
            actions = {agent: actions[i] for i, agent in enumerate(env.agents)}
            key, subkey = jax.random.split(key)
            obs, state, rewards, dones, infos = env.step_env(subkey, state, actions)

            done = dones["__all__"]
            if done:
                rewards_arr = jnp.array([rewards[a] for a in env.agents])
                success = jnp.all(rewards_arr > 0)

            trace.append(state)

        return trace, success

    trace, success = run_episode(env, network, params, key)

    if success:
        print("Success!", len(trace))
        dfa_gym.visualize(config["LAYOUT"], figsize=(9,9), trace=trace, save_path="gifs/2rooms_2agents/helper_agent/trace.gif")


    

