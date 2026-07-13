import os
import pickle
from pathlib import Path
from typing import Optional

import jax.numpy as np
import jax.random as jr
from jax import Array

from flax.struct import dataclass

import rp_ssm
from rp_ssm import datasets_utils

DATA_DIR = os.getenv("RP_GSSM_DATA_DIR") or os.path.join(rp_ssm.__path__[0], "data")
NOISE_SCALE = 0.05


@dataclass
class TrainData:
    obs: tuple[Array, ...]
    actions: Optional[tuple[Array, ...]] = None

    def __getitem__(self, index):
        """Allow indexing over all J data modalities"""
        return TrainData(
            obs=tuple(x[index] for x in self.obs),
            actions=self.actions if self.actions is None else self.actions[index],
        )


@dataclass
class Dataset:
    train_obs: tuple[Array, ...]
    train_states: Array
    val_obs: tuple[Array, ...]
    val_states: Array
    params: dict[str, Array]
    train_actions: Optional[tuple[Array, ...]] = None
    val_actions: Optional[tuple[Array, ...]] = None

    @property
    def train_data(self):
        return TrainData(obs=self.train_obs, actions=self.train_actions)

    @property
    def val_data(self):
        return TrainData(obs=self.val_obs, actions=self.val_actions)

    @property
    def standardized_data(self):
        means = tuple(np.mean(d, keepdims=True) for d in self.train_obs)
        stds = tuple(np.std(d, keepdims=True) for d in self.train_obs)

        scaled_train_obs = tuple(
            (d - m) / s for d, m, s in zip(self.train_obs, means, stds)
        )
        scaled_val_obs = tuple(
            (d - m) / s for d, m, s in zip(self.val_obs, means, stds)
        )

        return Dataset(
            train_obs=scaled_train_obs,
            train_actions=self.train_actions,
            train_states=self.train_states,
            val_obs=scaled_val_obs,
            val_actions=self.val_actions,
            val_states=self.val_states,
            params=self.params,
        )

    @property
    def flatten(self):
        train_shape = self.train_obs[0].shape[:2] + (-1,)
        val_shape = self.val_obs[0].shape[:2] + (-1,)
        train_obs = tuple(np.reshape(x, train_shape) for x in self.train_obs)
        val_obs = tuple(np.reshape(x, val_shape) for x in self.val_obs)

        return Dataset(
            train_obs=train_obs,
            train_actions=self.train_actions,
            train_states=self.train_states,
            val_obs=val_obs,
            val_actions=self.val_actions,
            val_states=self.val_states,
            params=self.params,
        )

    def __getitem__(self, index):
        """Allow indexing over all J data modalities"""
        return Dataset(
            train_obs=tuple(x[index] for x in self.train_obs),
            train_actions=(
                self.train_actions
                if self.train_actions is None
                else self.train_actions[index]
            ),
            train_states=self.train_states[index],
            val_obs=tuple(x[index] for x in self.val_obs),
            val_actions=(
                self.val_actions
                if self.val_actions is None
                else self.val_actions[index]
            ),
            val_states=self.val_states[index],
            params=self.params,
        )


def load_dataset(name: str, seed: int, datadir: str = DATA_DIR) -> Dataset:
    """
    Load dataset from string (for experiments).
    """
    key = jr.PRNGKey(seed + 1)
    if name == "linear_small":
        return datasets_utils.generate_linear_data(1, 3, 5, 200, 100, 0.1, key)
    elif name == "linear_medium":
        return datasets_utils.generate_linear_data(1, 5, 10, 200, 100, 0.1, key)
    elif name == "linear_large":
        return datasets_utils.generate_linear_data(1, 10, 20, 200, 100, 0.1, key)
    elif name == "pendulum":
        num_sequences = 500
        loaded_data = np.load(
            os.path.join(datadir, "datasets", "pendulum_data_no_obs_noise.npz")
        )
        train_key, val_key = jr.split(key)
        train_noise = NOISE_SCALE * jr.normal(
            train_key, loaded_data["train_obs"][:num_sequences].shape
        )
        val_noise = NOISE_SCALE * jr.normal(
            val_key, loaded_data["test_obs"][:num_sequences].shape
        )
        return Dataset(
            train_obs=(loaded_data["train_obs"][:num_sequences] / 255.0 + train_noise,),
            train_states=loaded_data["train_states"][:num_sequences],
            val_obs=(loaded_data["test_obs"][:num_sequences] / 255.0 + val_noise,),
            val_states=loaded_data["test_states"][:num_sequences],
            params=None,
        )
    elif name == "pendulum_distract":
        num_sequences = 500
        loaded_data = np.load(
            os.path.join(datadir, "datasets", "pendulum_data_distract.npz")
        )
        return Dataset(
            train_obs=(loaded_data["train_obs"][:num_sequences] / 255.0,),
            train_states=loaded_data["train_states"][:num_sequences],
            val_obs=(loaded_data["test_obs"][:num_sequences] / 255.0,),
            val_states=loaded_data["test_states"][:num_sequences],
            params=None,
        )
    if name == "tracking":
        loaded_data = np.load(Path(datadir) / "datasets" / "tracking_data.npz")
        return Dataset(
            train_obs=(loaded_data["train_obs"] / 255.0,),
            train_states=loaded_data["train_states"],
            val_obs=(loaded_data["test_obs"] / 255.0,),
            val_states=loaded_data["test_states"],
            params=None,
        ).standardized_data
    if name == "tracking_full_vid":
        loaded_data = np.load(Path(datadir) / "datasets" / "tracking_data_full_vid.npz")
        return Dataset(
            train_obs=(loaded_data["train_obs"] / 255.0,),
            train_states=loaded_data["train_states"],
            val_obs=(loaded_data["test_obs"] / 255.0,),
            val_states=loaded_data["test_states"],
            params=None,
        ).standardized_data
    if name == "tracking_no_distractions":
        loaded_data = np.load(
            Path(datadir) / "datasets" / "tracking_data_no_distractions.npz"
        )
        return Dataset(
            train_obs=(loaded_data["train_obs"] / 255.0,),
            train_states=loaded_data["train_states"],
            val_obs=(loaded_data["test_obs"] / 255.0,),
            val_states=loaded_data["test_states"],
            params=None,
        ).standardized_data
    elif name == "double_pendulum":
        return datasets_utils.read_double_pend(
            list(range(8)), 100, key, datadir, frameskip=3
        )
    elif name == "double_pendulum_distract":
        loaded_data = np.load(
            os.path.join(datadir, "datasets", "double_pendulum_distract_data.npz")
        )
        return Dataset(
            train_obs=(loaded_data["train_obs"],),
            train_states=loaded_data["train_states"],
            val_obs=(loaded_data["test_obs"],),
            val_states=loaded_data["test_states"],
            params=None,
        )
    elif name == "cartpole_sensor":
        raw_data = datasets_utils.generate_cartpole_data(
            num_sequences=200,
            num_timesteps=25,
            policy="random",
            save_dir=os.path.join(datadir, "datasets", "cartpole_sensor_random.pkl"),
            noise_scale=0.01,
        )
        # turn actions from {-1,1} to {0,1}
        raw_data["train_actions"] = (raw_data["train_actions"] + 1) // 2
        raw_data["val_actions"] = (raw_data["val_actions"] + 1) // 2
        return Dataset(**raw_data)
    elif name == "cartpole_image":
        raw_data = datasets_utils.generate_cartpole_data(
            num_sequences=200,
            num_timesteps=25,
            policy="random",
            save_dir=os.path.join(datadir, "datasets", "cartpole_image_random.pkl"),
            noise_scale=0.01,
            img_shape=(48, 48),
        )
        # turn actions from {-1,1} to {0,1}
        raw_data["train_actions"] = (raw_data["train_actions"] + 1) // 2
        raw_data["val_actions"] = (raw_data["val_actions"] + 1) // 2
        return Dataset(**raw_data)
    elif name == "walker_image":
        with open(
            os.path.join(datadir, "datasets", "walker-run", "walker-run_mini.pkl"), "rb"
        ) as f:
            obs, controls, states = pickle.load(f)

        n_train = 290
        n_val = 10
        T = 50
        assert n_train <= 300
        
        return Dataset(
            train_obs=(obs[:n_train, :T].reshape(n_train, T, -1),),
            train_states=states[:n_train, :T],
            train_actions=controls[:n_train, :T-1],
            val_obs=(obs[n_train: n_train+n_val, :T].reshape(n_val, T, -1),),
            val_states=states[n_train: n_train+n_val, :T],
            val_actions=controls[n_train: n_train+n_val, :T-1],
            params={}
        ).standardized_data
    elif name == "walker_distract_dynamic":
        with open(
            os.path.join(datadir, "datasets", "control_distract", "walker_walk", "dynamic_data.pkl"), "rb"
        ) as f:
            raw_data = pickle.load(f)
        with open(
            os.path.join(datadir, "datasets", "control_distract", "walker_walk", "dynamic_data_val.pkl"), "rb"
        ) as f:
            raw_data_val = pickle.load(f)
        train_data = datasets_utils.parse_control_distract_data(raw_data, 100, "walker")
        val_data = datasets_utils.parse_control_distract_data(raw_data_val, 100, "walker")

        return Dataset(
            train_obs=(train_data[0],),
            train_states=train_data[1],
            train_actions=train_data[2],
            val_obs=(val_data[0],),
            val_states=val_data[1],
            val_actions=val_data[2],
            params={},
        )
    else:
        raise NotImplementedError
