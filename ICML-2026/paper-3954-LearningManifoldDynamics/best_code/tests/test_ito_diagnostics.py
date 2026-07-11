import numpy as np
import jax.numpy as jnp

from taming_the_ito_lyon.training.factories import (
    create_grad_batch_loss_fns,
    _simple_bergomi_joint_driver_output_path,
)
from taming_the_ito_lyon.config import load_toml_config
from taming_the_ito_lyon.training.losses import simple_bergomi_ito_signature_loss
from taming_the_ito_lyon.training.results_gathering_fns import (
    _cumulative_ito_signature_feature_vectors,
    _ito_driver_output_summary,
    _ito_log_martingale_mean_l2,
    _local_ito_projection_feature_vectors,
)


def test_simple_bergomi_joint_path_uses_driver_and_output_channels() -> None:
    t = jnp.array([[0.0, 0.5, 1.0]], dtype=jnp.float32)
    w = jnp.array([[0.0, 0.1, -0.2]], dtype=jnp.float32)
    x = jnp.array([[[0.0], [0.3], [0.4]]], dtype=jnp.float32)
    control = jnp.stack([t, w], axis=-1)

    joint = _simple_bergomi_joint_driver_output_path(
        driver_source=control,
        output_path=x,
        driver_name="control",
        output_name="x",
    )

    np.testing.assert_allclose(
        np.asarray(joint),
        np.array([[[0.0, 0.0], [0.1, 0.3], [-0.2, 0.4]]], dtype=np.float32),
    )


def test_simple_bergomi_sigker_loss_uses_joint_driver_output_path() -> None:
    cfg = load_toml_config("configs/simple_bergomi/bnrde_sigker_joint.toml")
    _, _, loss_on_preds = create_grad_batch_loss_fns(cfg, output_path_dim=1)
    t = jnp.broadcast_to(jnp.linspace(0.0, 1.0, 5, dtype=jnp.float32), (2, 5))
    w = jnp.array(
        [[0.0, 0.1, -0.2, 0.0, 0.3], [0.0, -0.1, 0.2, 0.0, -0.3]],
        dtype=jnp.float32,
    )
    x = jnp.zeros((2, 5, 1), dtype=jnp.float32)
    pred_control = jnp.stack([t, w], axis=-1)
    target_driver = jnp.zeros_like(w)[..., None]

    loss = loss_on_preds(x, x, pred_control, target_driver)

    assert float(loss) > 0.0


def test_cumulative_ito_signature_features_include_log_martingale_coordinate() -> None:
    w = np.array([[0.0, 1.0, 2.0]], dtype=np.float32)
    x = np.array([[0.0, 2.0, 5.0]], dtype=np.float32)

    features = _cumulative_ito_signature_feature_vectors(
        w_paths=w,
        x_paths=x,
        num_eval_times=2,
    )

    np.testing.assert_allclose(
        features,
        np.array(
            [[1.0, 2.0, 2.0, 5.0, 1.0, 2.0, 4.0, 13.0, 2.0, 5.0, 4.0, 11.5]],
            dtype=np.float32,
        ),
    )


def test_ito_log_martingale_mean_l2_uses_realized_quadratic_variation() -> None:
    x = np.array(
        [
            [0.0, 1.0, 1.5],
            [0.0, -1.0, -1.5],
        ],
        dtype=np.float32,
    )

    value = _ito_log_martingale_mean_l2(x_paths=x, num_eval_times=2)

    np.testing.assert_allclose(value, np.sqrt((0.5**2 + 0.625**2) / 2.0))


def test_ito_driver_output_summary_reports_realized_qv_and_covariation() -> None:
    w = np.array([[0.0, 1.0, 2.0]], dtype=np.float32)
    x = np.array([[0.0, 2.0, 5.0]], dtype=np.float32)

    summary = _ito_driver_output_summary(w_paths=w, x_paths=x)

    np.testing.assert_allclose(summary["qv_w_mean"], 2.0)
    np.testing.assert_allclose(summary["qv_x_mean"], 13.0)
    np.testing.assert_allclose(summary["cov_wx_mean"], 5.0)
    np.testing.assert_allclose(summary["beta_mean"], 2.5)
    np.testing.assert_allclose(summary["driver_corr"], 0.0)


def test_local_ito_projection_features_are_blockwise() -> None:
    w = np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float32)
    x = np.array([[0.0, 2.0, 4.0, 6.0]], dtype=np.float32)

    features = _local_ito_projection_feature_vectors(
        w_paths=w,
        x_paths=x,
        block_size=2,
    )

    expected_residual = 4.0
    expected_normalized = expected_residual / np.sqrt(8.0)
    np.testing.assert_allclose(
        features,
        np.array(
            [
                [
                    expected_residual,
                    expected_residual,
                    expected_residual,
                    expected_normalized,
                    expected_normalized,
                ]
            ],
            dtype=np.float32,
        ),
        rtol=1e-6,
    )


def test_simple_bergomi_ito_signature_loss_is_zero_for_identical_pairs() -> None:
    w = jnp.array(
        [
            [0.0, 0.1, -0.2, 0.0, 0.3],
            [0.0, -0.1, 0.2, 0.0, -0.3],
        ],
        dtype=jnp.float32,
    )
    x = 0.2 * w
    t = jnp.linspace(0.0, 1.0, w.shape[1], dtype=w.dtype)
    control = jnp.stack([jnp.broadcast_to(t, w.shape), w], axis=-1)
    target_driver = w[..., None]
    loss_fn = simple_bergomi_ito_signature_loss(
        num_eval_times=2,
        projection_block_size=2,
    )

    loss = loss_fn(x[..., None], x[..., None], control, target_driver)

    np.testing.assert_allclose(np.asarray(loss), 0.0, atol=1e-6)


def test_simple_bergomi_ito_signature_loss_detects_wrong_log_price_pair() -> None:
    w = jnp.array(
        [
            [0.0, 0.1, -0.2, 0.0, 0.3],
            [0.0, -0.1, 0.2, 0.0, -0.3],
        ],
        dtype=jnp.float32,
    )
    x = 0.2 * w
    t = jnp.linspace(0.0, 1.0, w.shape[1], dtype=w.dtype)
    control = jnp.stack([jnp.broadcast_to(t, w.shape), w], axis=-1)
    target_driver = w[..., None]
    loss_fn = simple_bergomi_ito_signature_loss(
        num_eval_times=2,
        projection_block_size=2,
    )

    loss = loss_fn((x + 0.1 * t)[..., None], x[..., None], control, target_driver)

    assert float(loss) > 0.0
