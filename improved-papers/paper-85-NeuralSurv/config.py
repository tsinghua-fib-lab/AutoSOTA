import os
import ml_collections
import jax
import jax.numpy as jnp

from simulation.simulation_utils import simulate_data
from data.data_loading import (
    load_support,
    load_metabric,
    load_gbsg,
    load_nwtco,
    load_sac3,
    load_sacadmin,
    load_colon,
    load_whas,
    load_lung,
    load_vlc,
    load_synthetic_data,
)


class DataObsConfig(ml_collections.ConfigDict):

    def __init__(
        self,
        dataset_name,
        path_to_dataset,
        dataset_subsample_n,
        dataset_chosen_fold,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.path_to_dataset = path_to_dataset
        self.subsample_n = dataset_subsample_n
        self.chosen_fold = dataset_chosen_fold
        self._get_data_obs()

    def _get_data_obs(self):
        if self.dataset_name == "support":
            data = load_support(
                chosen_fold=self.chosen_fold, subsample_n=self.subsample_n
            )
        elif self.dataset_name == "metabric":
            data = load_metabric(
                chosen_fold=self.chosen_fold, subsample_n=self.subsample_n
            )
        elif self.dataset_name == "gbsg":
            data = load_gbsg(chosen_fold=self.chosen_fold, subsample_n=self.subsample_n)
        elif self.dataset_name == "nwtco":
            data = load_nwtco(
                chosen_fold=self.chosen_fold, subsample_n=self.subsample_n
            )
        elif self.dataset_name == "sac3":
            data = load_sac3(chosen_fold=self.chosen_fold, subsample_n=self.subsample_n)
        elif self.dataset_name == "sacadmin":
            data = load_sacadmin(
                chosen_fold=self.chosen_fold, subsample_n=self.subsample_n
            )
        elif self.dataset_name == "colon":
            data = load_colon(
                chosen_fold=self.chosen_fold,
                subsample_n=self.subsample_n,
                repodir=self.path_to_dataset,
            )
        elif self.dataset_name == "lung":
            data = load_lung(
                chosen_fold=self.chosen_fold,
                subsample_n=self.subsample_n,
                repodir=self.path_to_dataset,
            )
        elif self.dataset_name == "vlc":
            data = load_vlc(chosen_fold=self.chosen_fold, subsample_n=self.subsample_n)
        elif self.dataset_name == "whas":
            data = load_whas(chosen_fold=self.chosen_fold, subsample_n=self.subsample_n)
        elif self.dataset_name == "synthetic":
            data = load_synthetic_data(
                subsample_n=self.subsample_n,
                repodir=self.path_to_dataset,
            )

        self.data_train = data["train"]["np.array"]
        self.data_val = data["val"]["np.array"]
        self.data_test = data["test"]["np.array"]
        self.dim_x = self.data_train["x"].shape[1]


class SimulationConfig(ml_collections.ConfigDict):

    def __init__(
        self,
        data_obs_time_to_event_name,
        data_obs_time_to_event_params,
        data_obs_time_to_censoring_name,
        data_obs_time_to_censoring_params,
        data_obs_num_samples,
        data_obs_dim_x,
    ):
        super().__init__()
        self.time_to_event_name = data_obs_time_to_event_name
        self.time_to_event_params = data_obs_time_to_event_params
        self.time_to_censoring_name = data_obs_time_to_censoring_name
        self.time_to_censoring_params = data_obs_time_to_censoring_params
        self.num_samples = data_obs_num_samples
        self.dim_x = data_obs_dim_x


class ModelConfig(ml_collections.ConfigDict):
    def __init__(
        self,
        model_name,
        model_n_hidden,
        model_n_layers,
        model_activation_name,
    ):
        super().__init__()
        self.name = model_name
        self.n_hidden = model_n_hidden
        self.n_layers = model_n_layers
        self.activation = self._get_activation_function(model_activation_name)

    def _get_activation_function(self, activation_name):
        if activation_name == "silu":
            return jax.nn.silu
        elif activation_name == "relu":
            return jax.nn.relu
        else:
            raise ValueError(
                f"Activation function {activation_name} not implemented yet"
            )


class ConfigBuilder:
    def __init__(self, **kwargs):

        # create directories
        self.directories = ml_collections.ConfigDict()
        self.directories.workdir = kwargs.get("workdir", "~/.")
        self.directories.dataset = kwargs.get(
            "dataset_dir", os.path.join(os.getcwd(), "data/data_files")
        )
        self.directories.plots = os.path.join(self.directories.workdir, "plots")
        self.directories.outputs = os.path.join(self.directories.workdir, "outputs")
        os.makedirs(self.directories.plots, exist_ok=True)
        os.makedirs(self.directories.outputs, exist_ok=True)

        # Observed data
        simulate_data_obs = kwargs.get("dataset_name", None) is None
        if simulate_data_obs:
            self.data_obs = SimulationConfig(
                data_obs_time_to_event_name=kwargs.get(
                    "data_obs_time_to_event_name", "weibull"
                ),
                data_obs_time_to_event_params=kwargs.get(
                    "data_obs_time_to_event_params", [1.0, 1.0]
                ),
                data_obs_time_to_censoring_name=kwargs.get(
                    "data_obs_time_to_censoring_name", "exponential"
                ),
                data_obs_time_to_censoring_params=kwargs.get(
                    "data_obs_time_to_censoring_params", 0.5
                ),
                data_obs_num_samples=kwargs.get("data_obs_num_samples", 100),
                data_obs_dim_x=kwargs.get("data_obs_dim_x", (10,)),
            )
        else:
            self.data_obs = DataObsConfig(
                path_to_dataset=self.directories.dataset,
                dataset_name=kwargs.get("dataset_name", None),
                dataset_subsample_n=kwargs.get("dataset_subsample_n", None),
                dataset_chosen_fold=kwargs.get("dataset_chosen_fold", None),
            )
        self.data_obs.simulate_data_obs = simulate_data_obs

        # Prior
        self.prior = ml_collections.ConfigDict(
            {
                "p_phi_alpha": kwargs.get("p_phi_alpha", 1.0),
                "p_phi_beta": kwargs.get("p_phi_beta", 1.0),
                "rho": kwargs.get("rho", jnp.float32(1.0)),
            }
        )

        # Model
        self.model = ModelConfig(
            model_name=kwargs.get("model_name", "mlp"),
            model_n_hidden=kwargs.get("model_n_hidden", 28),
            model_n_layers=kwargs.get("model_n_layers", 6),
            model_activation_name=kwargs.get("model_activation_name", "silu"),
        )

        # Sample
        self.sample = ml_collections.ConfigDict({"num_samples": 100})

        # Devices
        self.devices = ml_collections.ConfigDict()
        self.devices.name = kwargs.get("devices", ["cpu"])
        self.devices.index = kwargs.get("devices_index", None)

        # Seeds and misc
        self.algorithm = ml_collections.ConfigDict(
            {
                "seed_train": kwargs.get("seed_train", 104),
                "seed_postprocessing": kwargs.get("seed_postprocessing", 106),
                "seed_data_obs": kwargs.get("seed_data_obs", 107),
                "debug": False,
                "overwrite_em": kwargs.get("overwrite_em", False),
                "overwrite_cavi": kwargs.get("overwrite_cavi", False),
                "batch_size": kwargs.get("batch_size", 100),
                "num_points_integral_cavi": kwargs.get("num_points_integral_cavi", 200),
                "num_points_integral_em": kwargs.get("num_points_integral_em", 200),
                "max_iter_em": kwargs.get("max_iter_em", 100),
                "max_iter_cavi": kwargs.get("max_iter_cavi", 100),
            }
        )

    def build(self):
        config = ml_collections.ConfigDict()
        config.data_obs = self.data_obs
        config.model = self.model
        config.prior = self.prior
        config.algorithm = self.algorithm
        config.devices = self.devices
        config.directories = self.directories
        config.sample = self.sample
        return config


def get_config(**kwargs):
    builder = ConfigBuilder(**kwargs)
    config = builder.build()

    # Simulate data if none were given
    if config.data_obs.simulate_data_obs == True:
        simulate_data(config)

    return config
