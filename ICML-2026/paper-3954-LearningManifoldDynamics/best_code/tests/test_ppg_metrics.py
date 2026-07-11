from pathlib import Path

import diffrax
import jax
import jax.numpy as jnp
import pytest

from taming_the_ito_lyon.config import load_toml_config
from taming_the_ito_lyon.training.factories import (
    create_model,
    create_results_gathering_fn,
)


def test_ppg_uses_loss_space_mse_for_eval_metric() -> None:
    config = load_toml_config(
        str(Path(__file__).parent.parent / "configs" / "ppg_dalia" / "ncde.toml")
    )

    results_fn = create_results_gathering_fn(config)
    results = results_fn(
        preds=jnp.zeros((2, 390, 1)),
        targets=jnp.ones((2, 390, 1)),
        controls=None,
        epoch_idx=0,
        model_name="ncde",
        config=config,
    )

    assert results_fn.__name__ == "get_ppg_dalia_results"
    assert results.eval_metric is None
    assert results.results_times == []
    assert results.results == []


def test_ppg_uses_heun_with_constant_stepsize() -> None:
    config = load_toml_config(
        str(Path(__file__).parent.parent / "configs" / "ppg_dalia" / "ncde.toml")
    )

    model = create_model(
        config=config,
        input_path_dim=6,
        output_path_dim=1,
        key=jax.random.PRNGKey(0),
    )

    assert isinstance(model.solver, diffrax.Heun)
    assert isinstance(model.stepsize_controller, diffrax.ConstantStepSize)


def test_ppg_ncde_scales_dt0_to_observation_grid() -> None:
    config = load_toml_config(
        str(Path(__file__).parent.parent / "configs" / "ppg_dalia" / "ncde.toml")
    )

    model = create_model(
        config=config,
        input_path_dim=6,
        output_path_dim=1,
        key=jax.random.PRNGKey(0),
    )

    scaled_dt0 = model._scaled_dt0_for_observation_grid(49_920)

    assert scaled_dt0 == pytest.approx(0.01 * 49_919.0)
