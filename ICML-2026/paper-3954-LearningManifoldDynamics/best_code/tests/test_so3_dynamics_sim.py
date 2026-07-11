import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp

from taming_the_ito_lyon.config.config import load_toml_config
from taming_the_ito_lyon.config.config_options import ExtrapolationSchemeType
from taming_the_ito_lyon.data.so3_dynamics_sim import SO3DynamicsSim
from taming_the_ito_lyon.training.runtime import build_runtime
from taming_the_ito_lyon.utils.so3 import rodrigues


def test_so3_dataset_uses_lie_algebra_driver_for_extrapolation_models() -> None:
    config = load_toml_config("configs/sg_so3_sim/nrde.toml")
    dataset = SO3DynamicsSim(config, "train")

    sample = dataset[0]

    assert sample["driver"].shape == (21, 3)
    assert sample["solution"].shape == (21, 3, 3)

    reconstructed = rodrigues(sample["driver"][0])
    assert jnp.allclose(reconstructed, sample["solution"][0], atol=1e-5)


def test_so3_runtime_uses_time_augmented_lie_driver_for_extrapolation_models() -> None:
    config = load_toml_config("configs/sg_so3_sim/nrde.toml")
    runtime = build_runtime(config, jnp.array([0, 1], dtype=jnp.uint32))

    assert runtime.input_path_dim == 4


def test_so3_dataset_keeps_flat_driver_for_matrix_specific_paths() -> None:
    m_ode_config = load_toml_config("configs/sg_so3_sim/m_ode.toml")
    m_ode_dataset = SO3DynamicsSim(m_ode_config, "train")
    assert m_ode_dataset[0]["driver"].shape == (21, 9)

    so3_sg_config = load_toml_config("configs/sg_so3_sim/nrde.toml").model_copy(
        deep=True
    )
    so3_sg_config.experiment_config.extrapolation_scheme = (
        ExtrapolationSchemeType.SO3_SG
    )
    so3_sg_dataset = SO3DynamicsSim(so3_sg_config, "train")
    assert so3_sg_dataset[0]["driver"].shape == (21, 9)
