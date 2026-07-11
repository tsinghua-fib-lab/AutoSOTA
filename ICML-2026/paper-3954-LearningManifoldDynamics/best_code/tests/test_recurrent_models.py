import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import jax.random as jr

from stochastax.manifolds import EuclideanSpace

from taming_the_ito_lyon.models import LSTM, StackedXLSTM, XLSTM


def test_lstm_preserves_sequence_shape() -> None:
    model = LSTM(
        input_path_dim=3,
        lstm_state_dim=8,
        output_path_dim=2,
        mlp_hidden_dim=8,
        initial_cond_mlp_depth=2,
        manifold=EuclideanSpace(),
        hidden_manifold=EuclideanSpace(),
        num_layers=2,
        key=jr.PRNGKey(0),
    )

    control_values = jr.normal(jr.PRNGKey(1), (12, 3))
    outputs = model(control_values)
    assert outputs.shape == (12, 2)


def test_xlstm_parallel_matches_step() -> None:
    model = XLSTM(
        input_path_dim=3,
        output_path_dim=2,
        d_model=16,
        n_heads=4,
        key=jr.PRNGKey(2),
        manifold=EuclideanSpace(),
    )

    control_values = jr.normal(jr.PRNGKey(3), (12, 3))
    outputs_parallel = model(control_values)

    state = None
    outputs_step = []
    for x_t in control_values:
        y_t, state = model.step(x_t, state)
        outputs_step.append(y_t)
    outputs_step = jnp.stack(outputs_step, axis=0)

    assert outputs_parallel.shape == outputs_step.shape == (12, 2)
    assert jnp.allclose(outputs_parallel, outputs_step, rtol=1e-4, atol=1e-5)


def test_stacked_xlstm_parallel_matches_step() -> None:
    model = StackedXLSTM(
        input_path_dim=3,
        output_path_dim=2,
        d_model=16,
        n_heads=4,
        num_layers=2,
        key=jr.PRNGKey(4),
        manifold=EuclideanSpace(),
    )

    control_values = jr.normal(jr.PRNGKey(5), (12, 3))
    outputs_parallel = model(control_values)

    state = None
    outputs_step = []
    for x_t in control_values:
        y_t, state = model.step(x_t, state)
        outputs_step.append(y_t)
    outputs_step = jnp.stack(outputs_step, axis=0)

    assert outputs_parallel.shape == outputs_step.shape == (12, 2)
    assert jnp.allclose(outputs_parallel, outputs_step, rtol=1e-4, atol=1e-5)
