import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp

from taming_the_ito_lyon.config.config import load_toml_config
from taming_the_ito_lyon.config.config_options import (
    Datasets,
    ExtrapolationSchemeType,
    ModelType,
)
from taming_the_ito_lyon.data.oxford_multimotion import OxfordMultimotionDataset
from taming_the_ito_lyon.training.runtime import build_runtime
from taming_the_ito_lyon.utils.so3 import rodrigues


def test_oxford_dataset_uses_lie_algebra_driver_for_extrapolation_models() -> None:
    config = load_toml_config("configs/sg_so3_sim/nrde.toml").model_copy(deep=True)
    config.experiment_config.dataset_name = Datasets.OXFORD_MULTIMOTION_STATIC
    config.experiment_config.model_type = ModelType.NRDE
    config.experiment_config.extrapolation_scheme = ExtrapolationSchemeType.HERMITE
    config.experiment_config.n_recon = 12

    dataset = OxfordMultimotionDataset(config, "train")
    sample = dataset[0]

    assert sample["driver"].shape == (21, 3)
    assert sample["solution"].shape == (21, 3, 3)

    reconstructed = rodrigues(sample["driver"][0])
    assert jnp.allclose(reconstructed, sample["solution"][0], atol=1e-5)


def test_oxford_runtime_uses_time_augmented_lie_driver_for_extrapolation_models() -> (
    None
):
    config = load_toml_config("configs/sg_so3_sim/nrde.toml").model_copy(deep=True)
    config.experiment_config.dataset_name = Datasets.OXFORD_MULTIMOTION_STATIC
    config.experiment_config.model_type = ModelType.NRDE
    config.experiment_config.extrapolation_scheme = ExtrapolationSchemeType.HERMITE
    config.experiment_config.n_recon = 12

    runtime = build_runtime(config, jnp.array([0, 1], dtype=jnp.uint32))

    assert runtime.input_path_dim == 4


def test_oxford_dataset_keeps_flat_driver_for_matrix_specific_paths() -> None:
    config = load_toml_config("configs/oxford_mm/m_ode.toml")
    dataset = OxfordMultimotionDataset(config, "train")

    assert dataset[0]["driver"].shape == (21, 9)
