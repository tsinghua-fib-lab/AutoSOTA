import os
import pickle
from typing import TYPE_CHECKING, Literal, Optional, Union

import cv2

import einops

import jax
import jax.numpy as np
import jax.random as jr
from jax import Array, vmap

import numpy as onp

from tqdm import tqdm

from rp_ssm import gym_wrappers

if TYPE_CHECKING:
    from datasets import Dataset

RGB2GRAYSCALE = np.array([0.2989, 0.5870, 0.1140])


class LDSPrior:
    def __init__(self, T):
        self.T = T

    def sample(self, key, stationary_params, num_samples, num_actions: int = -1):
        """
        Actions correspond to the angle of rotation in A.
        num_actions=-1 corresponds to no actions.
        num_actions=0 uniformly samples actions in [-pi/2,pi/2].
        num_actions=n, for n>1, uniformly samples actions in
        np.linspace(-pi/2, pi/2, n).

        """
        m1 = stationary_params["m1"]
        Q1 = stationary_params["Q1"]
        A = stationary_params["A"]
        b = stationary_params["b"]
        Q = stationary_params["Q"]

        As = np.tile(A[None], (self.T, 1, 1))  # TxKxK
        bs = np.concatenate([m1[None], np.tile(b[None], (self.T - 1, 1))])
        Qs = np.concatenate([Q1[None], np.tile(Q[None], (self.T - 1, 1, 1))])

        dim = A.shape[-1]

        # sample T actions for simplicity,
        if num_actions == 0:
            key, subkey = jr.split(key)
            actions = (
                jr.uniform(
                    subkey,
                    shape=(
                        num_samples,
                        self.T,
                    ),
                )
                * np.pi
                - np.pi / 2
            )
            thetas = actions
            key, subkey = jr.split(key)
            Bs = vmap(random_rotations, in_axes=(None, None, 0))(subkey, dim, thetas)
            As = np.einsum("tij,ntjk->ntik", As, Bs)  # multiplicative
        elif num_actions > 0:
            key, subkey = jr.split(key)
            actions = jr.choice(
                subkey,
                num_actions,
                shape=(
                    num_samples,
                    self.T,
                ),
            )
            possible_thetas = np.linspace(-np.pi / 2, np.pi / 2, num_actions)
            key, subkey = jr.split(key)
            possible_Bs = vmap(random_rotation, in_axes=(0, None, 0))(
                jr.split(subkey, num_actions), dim, possible_thetas
            )
            Bs = vmap(lambda act: np.take(possible_Bs, act, axis=0))(actions)
            As = np.einsum("tij,ntjk->ntik", As, Bs)  # multiplicative # MxTxKxK

        def _step(carry, x):
            A, b, noise = x
            zt = carry
            ztt = A @ zt + b + noise
            return ztt, ztt

        def sample_single_sequence(key, As, bs, Qs):
            keys = jr.split(key, self.T)
            noises = vmap(lambda k, b, Q: jr.multivariate_normal(k, b, Q))(keys, bs, Qs)
            zs = jax.lax.scan(
                _step,
                (
                    np.zeros(
                        dim,
                    )
                ),
                (As, bs, noises),
            )[
                1
            ]  # As[0] can be arbitrary
            return zs

        keys = jr.split(key, num_samples)

        if num_actions == -1:
            sample = vmap(lambda k: sample_single_sequence(k, As, bs, Qs))(keys)
            return sample
        else:
            sample = vmap(
                lambda k, A_single_seq: sample_single_sequence(k, A_single_seq, bs, Qs)
            )(keys, As)
            # add an extra dimension to `actions` to make it BxTx1
            # also remove the "0th" action because it doesn't play a role anywhere and we need the # of actions
            # in a sequence to be 1 less than the number of states/observations
            return sample, actions[:, 1:, None]


def random_rotation(key, n, theta=None):
    """Sample a rotation matrix about a random axis in R^n."""
    if n == 1:
        return np.eye(1) * jr.uniform(key)
    if theta is None:
        key, subkey = jr.split(key)
        theta = 0.5 * np.pi * jr.uniform(subkey)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    out = np.eye(n)
    out = out.at[:2, :2].set(rot)
    q = np.linalg.qr(jr.uniform(key, shape=(n, n)))[0]
    return q.dot(out).dot(q.T)


def random_rotations(key, n, thetas):
    """
    Sample multiple rotation matrices about the same axis
    in R^n, one for each theta in thetas.
    """
    if n == 1:
        return jr.uniform(key) * np.ones((len(thetas), 1))

    q = np.linalg.qr(jr.uniform(key, shape=(n, n)))[0]

    def single_rotation(theta, q, n):
        rot = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        out = np.eye(n)
        out = out.at[:2, :2].set(rot)
        return q.dot(out).dot(q.T)

    return vmap(single_rotation, in_axes=(0, None, None))(thetas, q, n)


def generate_linear_data(
    num_factors: int,
    latent_dim: int,
    emission_dim: int,
    num_sequences: int,
    num_timesteps: int,
    emission_cov: float,
    key: Array,
    num_actions: int = -1,
) -> "Dataset":
    from rp_ssm.datasets import Dataset

    J, K, D, N, T = num_factors, latent_dim, emission_dim, num_sequences, num_timesteps

    key, key_A, key_C, key_d = jr.split(key, 4)

    A = 0.95 * random_rotation(key_A, K, theta=np.pi / 5)
    Q = np.eye(K) - A @ A.T

    params = {
        "m1": np.zeros(K),
        "Q1": np.eye(K),
        "A": A,
        "b": np.zeros(K),
        "Q": Q,
        "R": np.eye(D) * np.sqrt(emission_cov)
    }

    lat_key, obs_key = jr.split(key)

    # sample an additional 25% sequences to use as validation data
    M = N + N // 4

    prior = LDSPrior(T)

    if num_actions == -1:
        latent_sample = prior.sample(lat_key, params, M, num_actions) # MxTxK
    else:
        latent_sample, actions = prior.sample(lat_key, params, M, num_actions)

    emission_params = {
        "C": jr.normal(key_C, shape=(J, D, K)),
        "d": jr.normal(key_d, shape=(J, D)),
    }

    def sample_single_factor(emission_params, key):
        C, d, R = emission_params["C"], emission_params["d"], params["R"]
        obs_sample = latent_sample @ C.T + jr.multivariate_normal(
            key, d, R, shape=(M, T)
        )  # MxTxD
        return obs_sample

    obs_samples = vmap(sample_single_factor)(emission_params, jr.split(obs_key, J))

    params.update(emission_params)

    if num_actions == -1:
        train_actions = None
        val_actions = None
    else:
        train_actions = actions[:N]
        val_actions = actions[N:]

    return Dataset(
        train_obs=tuple([o[:N] for o in obs_samples]),
        train_states=latent_sample[:N],
        train_actions=train_actions,
        val_obs=tuple([v[N:] for v in obs_samples]),
        val_states=latent_sample[N:],
        val_actions=val_actions,
        params=params,
    )


def read_double_pend(
    file_ids: list[int], T: int, key: Array, root_dir: str, frameskip: int = 3
) -> "Dataset":
    from rp_ssm.datasets import Dataset

    X, thetas, betas, gammas = [], [], [], []
    for fid in file_ids:
        dataset = np.load(
            f"{root_dir}/double-pendulum-chaotic/processed_frameskip_{frameskip}/{fid}_processed.npz"
        )

        xid = dataset["obs"]

        tot_frames = xid.shape[0]
        rem = tot_frames % T

        xid = xid[0:-rem]
        X.append(einops.rearrange(xid, "(N t) H W -> N t H W", t=T)[..., None])

        states = dataset["states"]
        states = states[0:-rem]
        states = einops.rearrange(states, "(N t) (C X) -> N t C X", t=T, C=3)

        d_1_0 = states[:, :, 1, :] - states[:, :, 0, :]
        d_2_1 = states[:, :, 2, :] - states[:, :, 1, :]
        d_2_0 = states[:, :, 2, :] - states[:, :, 0, :]

        thetas.append(np.arctan2(d_1_0[:, :, 1], d_1_0[:, :, 0]))
        betas.append(np.arctan2(d_2_1[:, :, 1], d_2_1[:, :, 0]))
        gammas.append(np.arctan2(d_2_0[:, :, 1], d_2_0[:, :, 0]))

    X = np.vstack(X)
    thetas = np.vstack(thetas)
    betas = np.vstack(betas)
    gammas = np.vstack(gammas)

    N = X.shape[0]
    shuffle = jr.permutation(key, N)

    X = X[shuffle]
    thetas = thetas[shuffle]
    betas = betas[shuffle]
    gammas = gammas[shuffle]

    # split data into 80% train 20% validation
    M = int(N * 0.8)

    return Dataset(
        train_obs=(X[:M],),
        train_states=(thetas[:M], betas[:M], gammas[:M]),
        val_obs=(X[M:],),
        val_states=(thetas[M:], betas[M:], gammas[M:]),
        params=None,
    )


def generate_cartpole_data(
    num_sequences: int,
    num_timesteps: int,
    policy: Literal["random", "constant", "left", "right"],
    img_shape: Optional[tuple[int, int]] = None,
    save_dir: Optional[str] = None,
    noise_scale: float = 0.0,
    image_obs: bool = False,
) -> dict[str, Optional[Union[tuple[Array], Array]]]:
    """
    Adjust environment to terminate when angle
    exceeds pi/2.

    policy:
        - random: random actions in {L,R}, i.e., {0,1}
        - constant: actions always 0
        - left: actions always L
        - right: actions always R
    """
    if save_dir is not None and os.path.exists(save_dir):
        print("Loading saved data!")
        with open(save_dir, "rb") as f:
            data = pickle.load(f)
        return data

    env = gym_wrappers.ExtendedCartpole(img_shape=img_shape)
    env.theta_threshold_radians = np.pi / 2.0

    zs, xs, all_actions = [], [], []

    # sample an additional 25% sequences to use as validation data
    M = num_sequences + num_sequences // 4

    for i in tqdm(range(M)):
        obs, _ = env.reset(seed=i+10)  # some random seed
        z = [onp.array(env.state)]
        if image_obs:
            frame = env.render()
            frame = cv2.resize(frame, (60, 40))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            x = [frame.flatten()]
        else:
            x = [obs]

        if policy == "random":
            actions = onp.random.choice([-1, 1], size=(num_timesteps - 1,))
        elif policy == "constant":
            actions = onp.zeros((num_timesteps - 1,), dtype=int)
        elif policy == "left":
            actions = onp.ones((num_timesteps - 1,), dtype=int) * -1
        elif policy == "right":
            actions = onp.ones((num_timesteps - 1,), dtype=int)

        for t, action in enumerate(actions):
            obs, _, terminated, _, _ = env.step(action)
            if terminated:
                print(f"Sequence {i} terminated at step {t+1}/{num_timesteps}!")
                return {}
            z.append(onp.array(env.state))
            if image_obs:
                frame = env.render()
                frame = cv2.resize(frame, (60, 40))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                x.append(frame.flatten())
            else:
                x.append(obs)

        print()

        zs.append(np.array(z))
        xs.append(np.array(x))
        all_actions.append(actions)

    zs = np.array(zs)
    xs = np.array(xs)
    xs += jr.normal(jr.PRNGKey(0), xs.shape) * noise_scale
    all_actions = np.array(all_actions)

    data = {
        "train_obs": (xs[:num_sequences],),
        "train_states": zs[:num_sequences],
        "val_obs": (xs[num_sequences:],),
        "val_states": zs[num_sequences:],
        "train_actions": all_actions[:num_sequences],
        "val_actions": all_actions[num_sequences:],
        "params": None,
    }

    if save_dir is not None:
        print("Saving data!")
        with open(save_dir, "wb") as f:
            pickle.dump(data, f)

    return data


def parse_control_distract_data(raw_data, T, task):
    for x in raw_data:
        if task == "walker":
            x["states"] = np.concatenate(
                [x["orientations"][:T],
                 x["velocities"][:T],
                 x["heights"][:T]],
                axis=1
            )
        elif task == "humanoid":
            x["states"] = np.concatenate(
                [x["com_velocity"][:T],
                 x["extremities"][:T],
                 x["head_height"][:T],
                 x["joint_angles"][:T],
                 x["torso_vertical"][:T],
                 x["velocity"][:T]],
                axis=1
            )
        elif task == "cheetah":
            x["states"] = np.concatenate(
                [x["position"][:T],
                 x["velocity"][:T]],
                axis=1
            )

    obs = np.stack([x["obs"][:T] for x in raw_data])
    obs = obs / 255.0
    obs = obs @ RGB2GRAYSCALE
    obs = (obs - obs.mean()) / obs.std()
    
    return (
        obs[..., None],
        np.stack([x["states"] for x in raw_data]),
        np.stack([x["actions"][:T-1] for x in raw_data])
    )