import jax
import jax.numpy as jnp
import pytest

import softjax as sj

from . import common


SOFT_MODES = ["smooth", "c0", "c1", "c2"]
DISTRIBUTIONAL_SAMPLES = 5_000
DISTRIBUTIONAL_TOL = 3e-2


CATEGORICAL_CASES = [
    (jnp.array([0.1, 1.5, -0.2]), -1, None),
    (jnp.array([[0.1, 1.5, -0.2], [1.2, -0.4, 0.3]]), -1, None),
    (jnp.array([[0.1, 1.5], [-0.2, 0.7], [1.2, -0.4]]), 0, None),
    (jnp.array([[0.1, 1.5, -0.2], [1.2, -0.4, 0.3]]), -1, (4, 2)),
]


CATEGORICAL_SOFT_SHAPE_CASES = [
    (jnp.array([0.1, 1.5, -0.2]), -1, 2),
    (jnp.array([[0.1, 1.5, -0.2], [1.2, -0.4, 0.3]]), -1, None),
    (jnp.array([[0.1, 1.5, -0.2], [1.2, -0.4, 0.3]]), -1, (4, 2)),
    (jnp.array([[0.1, 1.5], [-0.2, 0.7], [1.2, -0.4]]), 0, (4, 2)),
]


BERNOULLI_SOFT_SHAPE_CASES = [
    (jnp.array([0.2, 0.7, 0.4]), None),
    (jnp.array([0.2, 0.7, 0.4]), 3),
    (jnp.array([0.2, 0.7, 0.4]), (2, 3)),
    (jnp.array(0.3), (2, 3)),
]


CHOICE_CASES = [
    (5, (), True, None, 0),
    (5, (3,), False, None, 0),
    (jnp.arange(12.0).reshape(3, 4), (2,), True, None, 0),
    (jnp.arange(12.0).reshape(3, 4), (2,), False, None, 1),
    (jnp.array([10.0, 20.0, 30.0]), (4,), True, jnp.array([0.2, 0.3, 0.5]), 0),
]


def _categorical_hard_expected(
    key, logits, axis=-1, shape=None, replace=True, rng_mode=None
):
    indices = jax.random.categorical(
        key, logits, axis=axis, shape=shape, replace=replace, mode=rng_mode
    )
    return jax.nn.one_hot(indices, num_classes=logits.shape[axis], axis=-1)


def _distribution_keys(seed: int) -> jax.Array:
    return jax.random.split(jax.random.key(seed), DISTRIBUTIONAL_SAMPLES)


def _empirical_frequencies(samples, num_classes: int):
    return (
        jnp.bincount(samples.astype(jnp.int32), length=num_classes)
        / DISTRIBUTIONAL_SAMPLES
    )


@pytest.mark.parametrize(
    "logits, axis, shape",
    CATEGORICAL_CASES,
)
@pytest.mark.parametrize("rng_mode", [None, "low", "high"])
def test_categorical_hard_matches_jax(logits, axis, shape, rng_mode):
    key = jax.random.key(0)
    actual = sj.random.categorical(
        key, logits, axis=axis, shape=shape, mode="hard", rng_mode=rng_mode
    )
    expected = _categorical_hard_expected(
        key, logits, axis=axis, shape=shape, rng_mode=rng_mode
    )
    common.assert_jax_parity(actual, expected, msg="categorical hard")


@pytest.mark.parametrize("logits, axis, shape", CATEGORICAL_CASES)
@pytest.mark.parametrize("rng_mode", [None, "low", "high"])
def test_categorical_private_hard_matches_hard(logits, axis, shape, rng_mode):
    key = jax.random.key(1)

    actual = sj.random.categorical(
        key, logits, axis=axis, shape=shape, mode="_hard", rng_mode=rng_mode
    )
    expected = sj.random.categorical(
        key, logits, axis=axis, shape=shape, mode="hard", rng_mode=rng_mode
    )
    common.assert_allclose(actual, expected, tol=0.0)


@pytest.mark.parametrize("logits, axis, shape", CATEGORICAL_SOFT_SHAPE_CASES)
def test_categorical_soft_shape_contract(logits, axis, shape):
    key = jax.random.key(27)

    actual = sj.random.categorical(
        key, logits, axis=axis, shape=shape, softness=1.0, mode="smooth"
    )
    expected_indices = jax.random.categorical(key, logits, axis=axis, shape=shape)

    assert actual.shape == expected_indices.shape + (logits.shape[axis],)
    common.assert_simplex(actual)


@pytest.mark.parametrize("mode", ["hard", "_hard", "smooth"])
@pytest.mark.parametrize("axis", [2, -3])
def test_categorical_invalid_axis_errors_for_all_modes(mode, axis):
    key = jax.random.key(29)
    logits = jnp.zeros((2, 3))

    with pytest.raises((IndexError, ValueError), match="out of bounds"):
        sj.random.categorical(key, logits, axis=axis, mode=mode)


@pytest.mark.parametrize("mode", SOFT_MODES)
def test_categorical_soft_modes_are_simplex_and_differentiable(mode):
    key = jax.random.key(2)
    logits = jnp.array([0.1, 1.5, -0.2])
    weights = jnp.array([0.3, -0.7, 1.1])

    out = sj.random.categorical(key, logits, softness=1.0, mode=mode, method="softsort")
    common.assert_simplex(out)
    common.assert_finite(out, msg=f"categorical {mode}")

    def loss(z):
        sample = sj.random.categorical(
            key, z, softness=1.0, mode=mode, method="softsort"
        )
        return jnp.sum(sample * weights)

    grad = jax.grad(loss)(logits)
    common.assert_finite(grad, msg=f"categorical grad {mode}")


@pytest.mark.parametrize("mode", SOFT_MODES)
def test_categorical_near_hard_matches_hard(mode):
    key = jax.random.key(14)
    logits = jnp.array([0.1, 1.5, -0.2])

    actual = sj.random.categorical(
        key,
        logits,
        softness=common.NEAR_HARD_SOFTNESS,
        mode=mode,
        method="softsort",
    )
    expected = sj.random.categorical(key, logits, mode="hard")
    common.assert_allclose(actual, expected, tol=common.TOLERANCE)


@pytest.mark.parametrize("method", ["softsort", "neuralsort", "sorting_network"])
@pytest.mark.parametrize("mode", SOFT_MODES)
def test_categorical_methods_are_simplex(method, mode):
    key = jax.random.key(3)
    logits = jnp.array([0.1, 1.5, -0.2])

    out = sj.random.categorical(key, logits, softness=1.0, mode=mode, method=method)
    common.assert_simplex(out)


def test_categorical_return_log_probs_matches_probs():
    key = jax.random.key(4)
    logits = jnp.array([0.1, 1.5, -0.2])

    probs = sj.random.categorical(key, logits, softness=1.0, mode="smooth")
    log_probs = sj.random.categorical(
        key, logits, softness=1.0, mode="smooth", return_log_probs=True
    )

    common.assert_allclose(jnp.exp(log_probs), probs)
    common.assert_allclose(jax.nn.logsumexp(log_probs, axis=-1), 0.0)


def test_categorical_return_log_probs_log_prob_eps_matches_manual():
    key = jax.random.key(16)
    logits = jnp.array([0.1, 1.5, -0.2])
    eps = 1e-3

    probs = sj.random.categorical(key, logits, softness=1.0, mode="c0")
    log_probs = sj.random.categorical(
        key,
        logits,
        softness=1.0,
        mode="c0",
        return_log_probs=True,
        log_prob_eps=eps,
    )
    clipped = jnp.maximum(probs, eps)
    expected = jnp.log(clipped / jnp.sum(clipped, axis=-1, keepdims=True))

    common.assert_allclose(log_probs, expected)
    common.assert_allclose(jax.nn.logsumexp(log_probs, axis=-1), 0.0)


def test_categorical_return_log_probs_validation():
    key = jax.random.key(17)
    logits = jnp.array([0.1, 1.5, -0.2])

    with pytest.raises(ValueError, match="return_log_probs=True"):
        sj.random.categorical(key, logits, log_prob_eps=1e-3)

    with pytest.raises(ValueError, match="log_prob_eps must be in"):
        sj.random.categorical(key, logits, return_log_probs=True, log_prob_eps=0.0)


def test_categorical_hard_log_probs_have_negative_infinity():
    key = jax.random.key(18)
    logits = jnp.array([0.1, 1.5, -0.2])

    probs = sj.random.categorical(key, logits, mode="hard")
    log_probs = sj.random.categorical(key, logits, mode="hard", return_log_probs=True)

    common.assert_allclose(jnp.exp(log_probs), probs)
    assert jnp.any(jnp.isneginf(log_probs))


def test_categorical_return_log_probs_smooth_has_finite_gradients():
    key = jax.random.key(24)
    logits = jnp.array([0.1, 1.5, -0.2], dtype=jnp.float64)
    weights = jnp.array([0.3, -0.7, 1.1], dtype=jnp.float64)

    def loss(z):
        log_probs = sj.random.categorical(
            key,
            z,
            softness=1.0,
            mode="smooth",
            return_log_probs=True,
        )
        return jnp.sum(log_probs * weights)

    grad = jax.grad(loss)(logits)
    common.assert_finite(grad, msg="categorical smooth log_probs gradient")


def test_categorical_log_prob_eps_has_finite_sparse_gradients():
    key = jax.random.key(25)
    logits = jnp.array([0.1, 1.5, -0.2], dtype=jnp.float64)
    weights = jnp.array([0.3, -0.7, 1.1], dtype=jnp.float64)

    def loss(z):
        log_probs = sj.random.categorical(
            key,
            z,
            softness=0.01,
            mode="c0",
            return_log_probs=True,
            log_prob_eps=1e-12,
        )
        return jnp.sum(log_probs * weights)

    grad = jax.grad(loss)(logits)
    common.assert_finite(grad, msg="categorical c0 safe log_probs gradient")


@pytest.mark.parametrize(
    "logits, axis, shape",
    [
        (jnp.array([0.1, 1.5, -0.2]), -1, None),
        (jnp.array([0.1, 1.5, -0.2]), -1, (2,)),
        (jnp.array([[0.1, 1.5, -0.2], [1.2, -0.4, 0.3]]), -1, (2, 2)),
        (jnp.array([[0.1, 1.5], [-0.2, 0.7], [1.2, -0.4]]), 0, (2, 2)),
    ],
)
def test_categorical_replace_false_hard_matches_jax(logits, axis, shape):
    key = jax.random.key(5)

    actual = sj.random.categorical(
        key, logits, axis=axis, shape=shape, replace=False, mode="hard"
    )
    expected = _categorical_hard_expected(
        key, logits, axis=axis, shape=shape, replace=False
    )
    common.assert_jax_parity(actual, expected, msg="categorical replace=False hard")


@pytest.mark.parametrize(
    "logits, axis, shape",
    [
        (jnp.array([0.1, 1.5, -0.2]), -1, None),
        (jnp.array([0.1, 1.5, -0.2]), -1, (2,)),
        (jnp.array([[0.1, 1.5, -0.2], [1.2, -0.4, 0.3]]), -1, (2, 2)),
        (jnp.array([[0.1, 1.5], [-0.2, 0.7], [1.2, -0.4]]), 0, (2, 2)),
    ],
)
def test_categorical_replace_false_soft_shape_contract(logits, axis, shape):
    key = jax.random.key(30)

    actual = sj.random.categorical(
        key,
        logits,
        axis=axis,
        shape=shape,
        replace=False,
        softness=1.0,
        mode="smooth",
    )
    expected_indices = jax.random.categorical(
        key, logits, axis=axis, shape=shape, replace=False
    )

    assert actual.shape == expected_indices.shape + (logits.shape[axis],)
    common.assert_simplex(actual)


def test_categorical_replace_false_private_hard_matches_hard():
    key = jax.random.key(31)
    logits = jnp.array([0.1, 1.5, -0.2])

    actual = sj.random.categorical(key, logits, shape=(2,), replace=False, mode="_hard")
    expected = sj.random.categorical(
        key, logits, shape=(2,), replace=False, mode="hard"
    )

    common.assert_allclose(actual, expected, tol=0.0)


@pytest.mark.parametrize("mode", SOFT_MODES)
def test_categorical_replace_false_near_hard_matches_private_hard(mode):
    key = jax.random.key(31)
    logits = jnp.array([0.1, 1.5, -0.2])

    actual = sj.random.categorical(
        key,
        logits,
        shape=(2,),
        replace=False,
        softness=common.NEAR_HARD_SOFTNESS,
        mode=mode,
    )
    expected = sj.random.categorical(
        key, logits, shape=(2,), replace=False, mode="_hard"
    )

    common.assert_allclose(actual, expected, tol=common.TOLERANCE)


def test_categorical_replace_false_too_many_samples_errors():
    key = jax.random.key(32)
    logits = jnp.array([0.1, 1.5, -0.2])

    with pytest.raises(ValueError, match="cannot exceed"):
        sj.random.categorical(key, logits, shape=(4,), replace=False)


@pytest.mark.parametrize("mode", SOFT_MODES)
def test_categorical_replace_false_zero_probability_support_is_finite(mode):
    key = jax.random.key(65)
    logits = jnp.array([-jnp.inf, 0.0, -jnp.inf])

    actual = sj.random.categorical(
        key,
        logits,
        shape=(2,),
        replace=False,
        softness=1.0,
        mode=mode,
    )

    assert actual.shape == (2, 3)
    common.assert_finite(actual, msg=f"categorical zero support {mode}")
    common.assert_simplex(actual)


def test_categorical_st_forward_and_grad():
    key = jax.random.key(6)
    logits = jnp.array([0.1, 1.5, -0.2])
    weights = jnp.array([0.3, -0.7, 1.1])

    actual = sj.random.categorical_st(key, logits, softness=1.0, mode="smooth")
    expected = sj.random.categorical(key, logits, mode="hard")
    common.assert_allclose(actual, expected, tol=0.0)

    def st_loss(z):
        return jnp.sum(
            sj.random.categorical_st(key, z, softness=1.0, mode="smooth") * weights
        )

    def soft_loss(z):
        return jnp.sum(
            sj.random.categorical(key, z, softness=1.0, mode="smooth") * weights
        )

    common.assert_allclose(jax.grad(st_loss)(logits), jax.grad(soft_loss)(logits))


@pytest.mark.parametrize("mode", SOFT_MODES)
def test_categorical_st_return_log_probs_handles_negative_infinity(mode):
    key = jax.random.key(26)
    logits = jnp.array([0.1, 1.5, -0.2])

    log_probs = sj.random.categorical_st(
        key, logits, softness=1.0, mode=mode, return_log_probs=True
    )
    hard_probs = sj.random.categorical(key, logits, mode="hard")

    assert not jnp.any(jnp.isnan(log_probs))
    common.assert_allclose(jnp.exp(log_probs), hard_probs)


def test_categorical_jit_and_vmap():
    key = jax.random.key(12)
    logits = jnp.array([0.1, 1.5, -0.2])

    jitted = jax.jit(
        lambda z: sj.random.categorical(key, z, softness=1.0, mode="smooth")
    )(logits)
    common.assert_simplex(jitted)

    keys = jax.random.split(key, 2)
    batch_logits = jnp.array([[0.1, 1.5, -0.2], [1.2, -0.4, 0.3]])
    vmapped = jax.vmap(
        lambda k, z: sj.random.categorical(k, z, softness=1.0, mode="smooth")
    )(keys, batch_logits)
    assert vmapped.shape == batch_logits.shape
    common.assert_simplex(vmapped)


def test_categorical_grad_matches_finite_diff():
    key = jax.random.key(19)
    logits = jnp.array([0.1, 1.5, -0.2], dtype=jnp.float64)
    weights = jnp.array([0.3, -0.7, 1.1], dtype=jnp.float64)

    def loss(z):
        sample = sj.random.categorical(
            key, z, softness=1.0, mode="smooth", method="softsort"
        )
        return jnp.sum(sample * weights)

    common.assert_grad_matches_finite_diff(loss, logits, msg="categorical")


def test_categorical_take_along_axis_pipeline_grad_matches_finite_diff():
    key = jax.random.key(20)
    x = jnp.array([[1.0, 3.0, 2.0], [5.0, 1.0, 4.0]], dtype=jnp.float64)

    def pipeline(values):
        soft_idx = sj.random.categorical(
            key, values, axis=-1, softness=1.0, mode="smooth"
        )
        soft_idx = jnp.expand_dims(soft_idx, axis=-2)
        selected = sj.take_along_axis(values, soft_idx, axis=-1)
        return jnp.sum(selected)

    grad = jax.grad(pipeline)(x)
    common.assert_finite(grad, msg="categorical take_along_axis pipeline")
    common.assert_grad_matches_finite_diff(pipeline, x, msg="categorical pipeline")


@pytest.mark.parametrize("shape", [None, (3,), (2, 3)])
@pytest.mark.parametrize("rng_mode", ["low", "high"])
def test_bernoulli_hard_matches_jax(shape, rng_mode):
    key = jax.random.key(7)
    p = jnp.array([0.2, 0.7, 0.4])

    actual = sj.random.bernoulli(key, p, shape=shape, mode="hard", rng_mode=rng_mode)
    expected = jax.random.bernoulli(key, p=p, shape=shape, mode=rng_mode).astype(
        p.dtype
    )
    common.assert_jax_parity(actual, expected, msg="bernoulli hard")


@pytest.mark.parametrize("rng_mode", ["low", "high"])
def test_bernoulli_private_hard_matches_hard(rng_mode):
    key = jax.random.key(8)
    p = jnp.array([0.2, 0.7, 0.4])

    actual = sj.random.bernoulli(key, p, mode="_hard", rng_mode=rng_mode)
    expected = sj.random.bernoulli(key, p, mode="hard", rng_mode=rng_mode)
    common.assert_allclose(actual, expected, tol=0.0)


@pytest.mark.parametrize("p, shape", BERNOULLI_SOFT_SHAPE_CASES)
def test_bernoulli_soft_shape_contract(p, shape):
    key = jax.random.key(28)

    actual = sj.random.bernoulli(key, p, shape=shape, softness=1.0, mode="smooth")
    expected = jax.random.bernoulli(key, p=p, shape=shape)

    assert actual.shape == expected.shape
    common.assert_softbool(actual)


@pytest.mark.parametrize("mode", SOFT_MODES)
def test_bernoulli_soft_modes_are_softbool_and_differentiable(mode):
    key = jax.random.key(9)
    p = jnp.array([0.35, 0.5, 0.65])
    weights = jnp.array([0.3, -0.7, 1.1])

    out = sj.random.bernoulli(key, p, softness=1.0, mode=mode)
    common.assert_softbool(out)
    common.assert_finite(out, msg=f"bernoulli {mode}")

    def loss(prob):
        return jnp.sum(
            sj.random.bernoulli(key, prob, softness=1.0, mode=mode) * weights
        )

    grad = jax.grad(loss)(p)
    common.assert_finite(grad, msg=f"bernoulli grad {mode}")


@pytest.mark.parametrize("mode", SOFT_MODES)
def test_bernoulli_near_hard_matches_hard(mode):
    key = jax.random.key(10)
    p = jnp.array([0.2, 0.7, 0.4, 0.9])

    actual = sj.random.bernoulli(key, p, softness=common.NEAR_HARD_SOFTNESS, mode=mode)
    expected = sj.random.bernoulli(key, p, mode="hard")
    common.assert_allclose(actual, expected, tol=common.TOLERANCE)


def test_bernoulli_st_forward_and_grad():
    key = jax.random.key(10)
    p = jnp.array([0.35, 0.5, 0.65])
    weights = jnp.array([0.3, -0.7, 1.1])

    actual = sj.random.bernoulli_st(key, p, softness=1.0, mode="smooth")
    expected = sj.random.bernoulli(key, p, mode="hard")
    common.assert_allclose(actual, expected, tol=0.0)

    def st_loss(prob):
        return jnp.sum(
            sj.random.bernoulli_st(key, prob, softness=1.0, mode="smooth") * weights
        )

    def soft_loss(prob):
        return jnp.sum(
            sj.random.bernoulli(key, prob, softness=1.0, mode="smooth") * weights
        )

    common.assert_allclose(jax.grad(st_loss)(p), jax.grad(soft_loss)(p))


def test_bernoulli_jit_and_vmap():
    key = jax.random.key(13)
    p = jnp.array([0.35, 0.5, 0.65])

    jitted = jax.jit(lambda prob: sj.random.bernoulli(key, prob, softness=1.0))(p)
    common.assert_softbool(jitted)

    keys = jax.random.split(key, 2)
    batch_p = jnp.array([[0.35, 0.5, 0.65], [0.2, 0.7, 0.4]])
    vmapped = jax.vmap(lambda k, prob: sj.random.bernoulli(k, prob, softness=1.0))(
        keys, batch_p
    )
    assert vmapped.shape == batch_p.shape
    common.assert_softbool(vmapped)


def test_bernoulli_grad_matches_finite_diff():
    key = jax.random.key(21)
    p = jnp.array([0.35, 0.5, 0.65], dtype=jnp.float64)
    weights = jnp.array([0.3, -0.7, 1.1], dtype=jnp.float64)

    def loss(prob):
        return jnp.sum(
            sj.random.bernoulli(key, prob, softness=1.0, mode="smooth") * weights
        )

    common.assert_grad_matches_finite_diff(loss, p, msg="bernoulli")


def test_bernoulli_where_pipeline_grad_matches_finite_diff():
    key = jax.random.key(22)
    p = jnp.array([0.35, 0.5, 0.65], dtype=jnp.float64)
    x = jnp.array([1.0, 3.0, 2.0], dtype=jnp.float64)
    y = jnp.array([-1.0, -2.0, 0.5], dtype=jnp.float64)

    def pipeline(prob):
        condition = sj.random.bernoulli(key, prob, softness=1.0, mode="smooth")
        return jnp.sum(sj.where(condition, x, y))

    grad = jax.grad(pipeline)(p)
    common.assert_finite(grad, msg="bernoulli where pipeline")
    common.assert_grad_matches_finite_diff(pipeline, p, msg="bernoulli pipeline")


@pytest.mark.parametrize("a, shape, replace, p, axis", CHOICE_CASES)
def test_choice_hard_matches_jax(a, shape, replace, p, axis):
    key = jax.random.key(33)

    actual = sj.random.choice(
        key, a, shape=shape, replace=replace, p=p, axis=axis, mode="hard"
    )
    expected = jax.random.choice(key, a, shape=shape, replace=replace, p=p, axis=axis)

    common.assert_jax_parity(actual, expected, msg="choice hard")


@pytest.mark.parametrize("a, shape, replace, p, axis", CHOICE_CASES)
def test_choice_soft_shape_contract(a, shape, replace, p, axis):
    key = jax.random.key(34)

    actual = sj.random.choice(
        key,
        a,
        shape=shape,
        replace=replace,
        p=p,
        axis=axis,
        softness=1.0,
        mode="smooth",
    )
    expected = jax.random.choice(key, a, shape=shape, replace=replace, p=p, axis=axis)

    assert actual.shape == expected.shape
    common.assert_finite(actual, msg="choice soft")


def test_choice_soft_is_random_not_plain_expectation():
    a = jnp.array([10.0, 20.0, 30.0])
    p = jnp.array([0.2, 0.3, 0.5])
    expectation = jnp.sum(a * p)

    sample = sj.random.choice(jax.random.key(35), a, p=p, softness=0.25, mode="smooth")

    assert not jnp.isclose(sample, expectation)


@pytest.mark.parametrize("mode", ["_hard", "smooth"])
@pytest.mark.parametrize("a", [0, -1])
def test_choice_zero_draw_empty_scalar_population_matches_jax_shape(mode, a):
    key = jax.random.key(66)

    actual = sj.random.choice(key, a, shape=(0,), mode=mode)
    expected = jax.random.choice(key, a, shape=(0,))

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype


@pytest.mark.parametrize("mode", SOFT_MODES)
def test_choice_replace_false_zero_probabilities_are_finite(mode):
    key = jax.random.key(67)
    values = jnp.array([10.0, 20.0, 30.0])
    p = jnp.array([0.0, 1.0, 0.0])

    actual = sj.random.choice(
        key,
        values,
        shape=(2,),
        replace=False,
        p=p,
        softness=1.0,
        mode=mode,
    )

    assert actual.shape == (2,)
    common.assert_finite(actual, msg=f"choice zero probabilities {mode}")


@pytest.mark.parametrize("replace", [True, False])
def test_choice_private_hard_shape_contract(replace):
    key = jax.random.key(36)
    a = jnp.array([10.0, 20.0, 30.0])
    p = jnp.array([0.2, 0.3, 0.5])

    actual = sj.random.choice(key, a, shape=(2,), replace=replace, p=p, mode="_hard")
    expected_shape = jax.random.choice(key, a, shape=(2,), replace=replace, p=p).shape

    assert actual.shape == expected_shape
    common.assert_finite(actual, msg="choice _hard")
    assert jnp.all(actual >= jnp.min(a))
    assert jnp.all(actual <= jnp.max(a))


@pytest.mark.parametrize("mode", SOFT_MODES)
@pytest.mark.parametrize("replace", [True, False])
def test_choice_near_hard_matches_private_hard(mode, replace):
    key = jax.random.key(36)
    a = jnp.array([10.0, 20.0, 30.0])
    p = jnp.array([0.2, 0.3, 0.5])

    actual = sj.random.choice(
        key,
        a,
        shape=(2,),
        replace=replace,
        p=p,
        softness=common.NEAR_HARD_SOFTNESS,
        mode=mode,
    )
    expected = sj.random.choice(key, a, shape=(2,), replace=replace, p=p, mode="_hard")

    common.assert_allclose(actual, expected, tol=common.TOLERANCE)


def test_choice_private_hard_matches_hard_distributionally():
    keys = _distribution_keys(59)
    a = jnp.arange(4)
    p = jnp.array([0.1, 0.2, 0.3, 0.4])

    sample_hard = jax.jit(
        jax.vmap(lambda key: sj.random.choice(key, a, p=p, mode="hard"))
    )
    sample_private_hard = jax.jit(
        jax.vmap(lambda key: sj.random.choice(key, a, p=p, mode="_hard"))
    )
    hard_freq = _empirical_frequencies(sample_hard(keys), num_classes=4)
    private_hard_freq = _empirical_frequencies(sample_private_hard(keys), num_classes=4)

    common.assert_allclose(hard_freq, p, tol=DISTRIBUTIONAL_TOL)
    common.assert_allclose(private_hard_freq, p, tol=DISTRIBUTIONAL_TOL)
    common.assert_allclose(hard_freq, private_hard_freq, tol=2 * DISTRIBUTIONAL_TOL)


def test_choice_gradient_through_values_and_probabilities():
    key = jax.random.key(37)
    values = jnp.array([10.0, 20.0, 30.0], dtype=jnp.float64)
    p = jnp.array([0.2, 0.3, 0.5], dtype=jnp.float64)

    def value_loss(x):
        return sj.random.choice(key, x, p=p, softness=1.0, mode="smooth")

    def prob_loss(prob):
        return sj.random.choice(key, values, p=prob, softness=1.0, mode="smooth")

    common.assert_finite(jax.grad(value_loss)(values), msg="choice value grad")
    common.assert_finite(jax.grad(prob_loss)(p), msg="choice probability grad")


def test_choice_st_forward_and_grad():
    key = jax.random.key(52)
    values = jnp.array([10.0, 20.0, 30.0], dtype=jnp.float64)
    p = jnp.array([0.2, 0.3, 0.5], dtype=jnp.float64)

    actual = sj.random.choice_st(key, values, p=p, softness=1.0, mode="smooth")
    expected = sj.random.choice(key, values, p=p, mode="_hard")
    common.assert_allclose(actual, expected, tol=0.0)

    def st_loss(prob):
        return sj.random.choice_st(key, values, p=prob, softness=1.0, mode="smooth")

    def soft_loss(prob):
        return sj.random.choice(key, values, p=prob, softness=1.0, mode="smooth")

    common.assert_allclose(jax.grad(st_loss)(p), jax.grad(soft_loss)(p))


@pytest.mark.parametrize(
    "x, axis, independent",
    [
        (5, 0, False),
        (jnp.arange(12).reshape(3, 4), 0, False),
        (jnp.arange(12).reshape(3, 4), 1, False),
        (jnp.arange(12).reshape(3, 4), 1, True),
    ],
)
def test_permutation_hard_matches_jax(x, axis, independent):
    key = jax.random.key(38)

    actual = sj.random.permutation(
        key, x, axis=axis, independent=independent, mode="hard"
    )
    expected = jax.random.permutation(key, x, axis=axis, independent=independent)

    common.assert_jax_parity(actual, expected, msg="permutation hard")


@pytest.mark.parametrize("mode", ["_hard", "smooth"])
def test_permutation_scalar_validates_like_jax(mode):
    key = jax.random.key(63)

    with pytest.raises(ValueError, match="out of bounds"):
        sj.random.permutation(key, 5, axis=1, mode=mode)

    with pytest.raises(TypeError, match="integer or at least 1-dimensional"):
        sj.random.permutation(key, 5.0, mode=mode)


@pytest.mark.parametrize("mode", ["hard", "_hard", "smooth"])
@pytest.mark.parametrize("x", [0, -1])
def test_permutation_empty_scalar_matches_jax_shape(mode, x):
    key = jax.random.key(64)

    scalar_result = sj.random.permutation(key, x, mode=mode)
    array_result = sj.random.permutation(key, jnp.array([]), mode=mode)

    assert scalar_result.shape == jax.random.permutation(key, x).shape
    assert array_result.shape == jax.random.permutation(key, jnp.array([])).shape


@pytest.mark.parametrize("axis, independent", [(0, False), (1, False), (1, True)])
def test_permutation_soft_shape_contract(axis, independent):
    key = jax.random.key(39)
    x = jnp.arange(12.0).reshape(3, 4)

    actual = sj.random.permutation(
        key,
        x,
        axis=axis,
        independent=independent,
        softness=1.0,
        mode="smooth",
    )

    assert actual.shape == x.shape
    common.assert_finite(actual, msg="permutation soft")


@pytest.mark.parametrize("independent", [False, True])
def test_permutation_private_hard_shape_contract(independent):
    key = jax.random.key(40)
    x = jnp.arange(12.0).reshape(3, 4)

    actual = sj.random.permutation(
        key, x, axis=1, independent=independent, mode="_hard"
    )

    assert actual.shape == x.shape
    common.assert_finite(actual, msg="permutation _hard")
    common.assert_allclose(jnp.sort(actual, axis=1), x)


@pytest.mark.parametrize("mode", SOFT_MODES)
@pytest.mark.parametrize("independent", [False, True])
def test_permutation_near_hard_matches_private_hard(mode, independent):
    key = jax.random.key(40)
    x = jnp.arange(12.0).reshape(3, 4)

    actual = sj.random.permutation(
        key,
        x,
        axis=1,
        independent=independent,
        softness=common.NEAR_HARD_SOFTNESS,
        mode=mode,
    )
    expected = sj.random.permutation(
        key, x, axis=1, independent=independent, mode="_hard"
    )

    common.assert_allclose(actual, expected, tol=common.TOLERANCE)


def test_permutation_private_hard_matches_hard_distributionally():
    keys = _distribution_keys(60)
    x = jnp.arange(4)
    expected_position_freq = jnp.full((4, 4), 0.25)

    sample_hard = jax.jit(
        jax.vmap(lambda key: sj.random.permutation(key, x, mode="hard"))
    )
    sample_private_hard = jax.jit(
        jax.vmap(lambda key: sj.random.permutation(key, x, mode="_hard"))
    )
    hard = sample_hard(keys)
    private_hard = sample_private_hard(keys)

    common.assert_allclose(jnp.sort(hard, axis=-1), jnp.broadcast_to(x, hard.shape))
    common.assert_allclose(
        jnp.sort(private_hard, axis=-1), jnp.broadcast_to(x, private_hard.shape)
    )

    hard_position_freq = jnp.mean(
        jax.nn.one_hot(hard.astype(jnp.int32), num_classes=4), axis=0
    )
    private_hard_position_freq = jnp.mean(
        jax.nn.one_hot(private_hard.astype(jnp.int32), num_classes=4), axis=0
    )
    common.assert_allclose(
        hard_position_freq, expected_position_freq, tol=DISTRIBUTIONAL_TOL
    )
    common.assert_allclose(
        private_hard_position_freq,
        expected_position_freq,
        tol=DISTRIBUTIONAL_TOL,
    )
    common.assert_allclose(
        hard_position_freq,
        private_hard_position_freq,
        tol=2 * DISTRIBUTIONAL_TOL,
    )


def test_permutation_gradient_through_values():
    key = jax.random.key(41)
    x = jnp.arange(12.0, dtype=jnp.float64).reshape(3, 4)

    def loss(values):
        return jnp.sum(
            sj.random.permutation(key, values, axis=1, softness=1.0, mode="smooth")
        )

    common.assert_finite(jax.grad(loss)(x), msg="permutation value grad")


def test_permutation_st_forward_and_grad():
    key = jax.random.key(53)
    x = jnp.arange(12.0, dtype=jnp.float64).reshape(3, 4)

    actual = sj.random.permutation_st(key, x, axis=1, softness=1.0, mode="smooth")
    expected = sj.random.permutation(key, x, axis=1, mode="_hard")
    common.assert_allclose(actual, expected, tol=0.0)

    def st_loss(values):
        return jnp.sum(
            sj.random.permutation_st(key, values, axis=1, softness=1.0, mode="smooth")
        )

    def soft_loss(values):
        return jnp.sum(
            sj.random.permutation(key, values, axis=1, softness=1.0, mode="smooth")
        )

    common.assert_allclose(jax.grad(st_loss)(x), jax.grad(soft_loss)(x))


@pytest.mark.parametrize("shape", [(), (2, 3)])
def test_rademacher_hard_matches_jax(shape):
    key = jax.random.key(42)

    actual = sj.random.rademacher(key, shape=shape, mode="hard")
    expected = jax.random.rademacher(key, shape=shape)

    common.assert_jax_parity(actual, expected, msg="rademacher hard")


def test_rademacher_private_hard_matches_hard():
    key = jax.random.key(58)

    actual = sj.random.rademacher(key, shape=(4,), mode="_hard")
    expected = sj.random.rademacher(key, shape=(4,), mode="hard").astype(actual.dtype)

    common.assert_allclose(actual, expected, tol=0.0)


@pytest.mark.parametrize("mode", SOFT_MODES)
def test_rademacher_soft_range_and_near_hard(mode):
    key = jax.random.key(43)

    actual = sj.random.rademacher(key, shape=(4,), softness=1.0, mode=mode)
    assert actual.shape == (4,)
    assert jnp.all(actual >= -1.0)
    assert jnp.all(actual <= 1.0)

    near_hard = sj.random.rademacher(
        key,
        shape=(4,),
        softness=common.NEAR_HARD_SOFTNESS,
        mode=mode,
    )
    expected = sj.random.rademacher(key, shape=(4,), mode="_hard")
    common.assert_allclose(near_hard, expected, tol=common.TOLERANCE)


def test_rademacher_st_forward():
    key = jax.random.key(54)

    actual = sj.random.rademacher_st(key, shape=(4,), softness=1.0, mode="smooth")
    expected = sj.random.rademacher(key, shape=(4,), mode="hard")

    common.assert_allclose(actual, expected, tol=0.0)


@pytest.mark.parametrize("shape", [None, (2,)])
def test_binomial_hard_matches_jax(shape):
    key = jax.random.key(44)
    p = jnp.array([0.2, 0.7])

    actual = sj.random.binomial(key, 4, p, shape=shape, mode="hard")
    expected = jax.random.binomial(key, 4, p, shape=shape)

    common.assert_jax_parity(actual, expected, msg="binomial hard")


@pytest.mark.parametrize("mode", SOFT_MODES)
def test_binomial_soft_counts_and_gradients(mode):
    key = jax.random.key(45)
    p = jnp.array([0.2, 0.7], dtype=jnp.float64)
    weights = jnp.array([0.3, -0.7], dtype=jnp.float64)

    out = sj.random.binomial(key, 4, p, softness=1.0, mode=mode)
    assert out.shape == p.shape
    assert jnp.all(out >= 0.0)
    assert jnp.all(out <= 4.0)
    common.assert_finite(out, msg=f"binomial {mode}")

    def loss(prob):
        return jnp.sum(
            sj.random.binomial(key, 4, prob, softness=1.0, mode=mode) * weights
        )

    common.assert_finite(jax.grad(loss)(p), msg=f"binomial grad {mode}")


def test_binomial_private_hard_count_contract():
    key = jax.random.key(46)
    p = jnp.array([0.2, 0.7])

    actual = sj.random.binomial(key, 4, p, mode="_hard")

    assert actual.shape == p.shape
    assert jnp.all(actual >= 0.0)
    assert jnp.all(actual <= 4.0)
    common.assert_allclose(actual, jnp.round(actual), tol=0.0)


@pytest.mark.parametrize("mode", SOFT_MODES)
def test_binomial_near_hard_matches_private_hard(mode):
    key = jax.random.key(46)
    p = jnp.array([0.2, 0.7])

    actual = sj.random.binomial(
        key, 4, p, softness=common.NEAR_HARD_SOFTNESS, mode=mode
    )
    expected = sj.random.binomial(key, 4, p, mode="_hard")

    common.assert_allclose(actual, expected, tol=common.TOLERANCE)


def test_binomial_private_hard_matches_hard_distributionally():
    keys = _distribution_keys(61)
    n = 5
    p = 0.3
    expected_mean = n * p
    expected_var = n * p * (1.0 - p)

    sample_hard = jax.jit(
        jax.vmap(lambda key: sj.random.binomial(key, n, p, mode="hard"))
    )
    sample_private_hard = jax.jit(
        jax.vmap(lambda key: sj.random.binomial(key, n, p, mode="_hard"))
    )
    hard = sample_hard(keys)
    private_hard = sample_private_hard(keys)

    hard_mean = jnp.mean(hard)
    private_hard_mean = jnp.mean(private_hard)
    hard_var = jnp.var(hard)
    private_hard_var = jnp.var(private_hard)

    common.assert_allclose(hard_mean, expected_mean, tol=3 * DISTRIBUTIONAL_TOL)
    common.assert_allclose(private_hard_mean, expected_mean, tol=3 * DISTRIBUTIONAL_TOL)
    common.assert_allclose(hard_var, expected_var, tol=5 * DISTRIBUTIONAL_TOL)
    common.assert_allclose(private_hard_var, expected_var, tol=5 * DISTRIBUTIONAL_TOL)
    common.assert_allclose(hard_mean, private_hard_mean, tol=3 * DISTRIBUTIONAL_TOL)
    common.assert_allclose(hard_var, private_hard_var, tol=5 * DISTRIBUTIONAL_TOL)


def test_binomial_soft_requires_static_scalar_count():
    key = jax.random.key(47)

    with pytest.raises(NotImplementedError, match="scalar static counts"):
        sj.random.binomial(key, jnp.array([2, 3]), jnp.array([0.2, 0.7]))


def test_binomial_st_forward_and_grad():
    key = jax.random.key(55)
    p = jnp.array([0.2, 0.7], dtype=jnp.float64)
    weights = jnp.array([0.3, -0.7], dtype=jnp.float64)

    actual = sj.random.binomial_st(key, 4, p, softness=1.0, mode="smooth")
    expected = sj.random.binomial(key, 4, p, mode="_hard")
    common.assert_allclose(actual, expected, tol=0.0)

    def st_loss(prob):
        return jnp.sum(
            sj.random.binomial_st(key, 4, prob, softness=1.0, mode="smooth") * weights
        )

    def soft_loss(prob):
        return jnp.sum(
            sj.random.binomial(key, 4, prob, softness=1.0, mode="smooth") * weights
        )

    common.assert_allclose(jax.grad(st_loss)(p), jax.grad(soft_loss)(p))


@pytest.mark.parametrize("shape", [None, (2, 3)])
def test_multinomial_hard_matches_jax(shape):
    key = jax.random.key(48)
    p = jnp.array([0.2, 0.3, 0.5])

    actual = sj.random.multinomial(key, 4, p, shape=shape, mode="hard")
    expected = jax.random.multinomial(key, 4, p, shape=shape)

    common.assert_jax_parity(actual, expected, msg="multinomial hard")


@pytest.mark.parametrize("mode", SOFT_MODES)
def test_multinomial_soft_counts_and_gradients(mode):
    key = jax.random.key(49)
    p = jnp.array([0.2, 0.3, 0.5], dtype=jnp.float64)
    weights = jnp.array([0.3, -0.7, 1.1], dtype=jnp.float64)

    out = sj.random.multinomial(key, 4, p, softness=1.0, mode=mode)
    assert out.shape == p.shape
    common.assert_allclose(jnp.sum(out, axis=-1), 4.0)
    common.assert_finite(out, msg=f"multinomial {mode}")

    def loss(prob):
        return jnp.sum(
            sj.random.multinomial(key, 4, prob, softness=1.0, mode=mode) * weights
        )

    common.assert_finite(jax.grad(loss)(p), msg=f"multinomial grad {mode}")


def test_multinomial_private_hard_count_contract():
    key = jax.random.key(50)
    p = jnp.array([0.2, 0.3, 0.5])

    actual = sj.random.multinomial(key, 4, p, mode="_hard")

    assert actual.shape == p.shape
    common.assert_allclose(jnp.sum(actual, axis=-1), 4.0)
    common.assert_allclose(actual, jnp.round(actual), tol=0.0)


@pytest.mark.parametrize("mode", SOFT_MODES)
def test_multinomial_near_hard_matches_private_hard(mode):
    key = jax.random.key(50)
    p = jnp.array([0.2, 0.3, 0.5])

    actual = sj.random.multinomial(
        key, 4, p, softness=common.NEAR_HARD_SOFTNESS, mode=mode
    )
    expected = sj.random.multinomial(key, 4, p, mode="_hard")

    common.assert_allclose(actual, expected, tol=common.TOLERANCE)


def test_multinomial_private_hard_matches_hard_distributionally():
    keys = _distribution_keys(62)
    n = 5
    p = jnp.array([0.2, 0.3, 0.5])
    expected_mean = n * p
    expected_cov = n * (jnp.diag(p) - p[:, None] * p[None, :])

    sample_hard = jax.jit(
        jax.vmap(lambda key: sj.random.multinomial(key, n, p, mode="hard"))
    )
    sample_private_hard = jax.jit(
        jax.vmap(lambda key: sj.random.multinomial(key, n, p, mode="_hard"))
    )
    hard = sample_hard(keys)
    private_hard = sample_private_hard(keys)

    common.assert_allclose(jnp.sum(hard, axis=-1), n)
    common.assert_allclose(jnp.sum(private_hard, axis=-1), n)

    hard_mean = jnp.mean(hard, axis=0)
    private_hard_mean = jnp.mean(private_hard, axis=0)
    hard_centered = hard - hard_mean
    private_hard_centered = private_hard - private_hard_mean
    hard_cov = hard_centered.T @ hard_centered / DISTRIBUTIONAL_SAMPLES
    private_hard_cov = (
        private_hard_centered.T @ private_hard_centered / DISTRIBUTIONAL_SAMPLES
    )

    common.assert_allclose(hard_mean, expected_mean, tol=3 * DISTRIBUTIONAL_TOL)
    common.assert_allclose(private_hard_mean, expected_mean, tol=3 * DISTRIBUTIONAL_TOL)
    common.assert_allclose(hard_cov, expected_cov, tol=6 * DISTRIBUTIONAL_TOL)
    common.assert_allclose(private_hard_cov, expected_cov, tol=6 * DISTRIBUTIONAL_TOL)
    common.assert_allclose(hard_mean, private_hard_mean, tol=3 * DISTRIBUTIONAL_TOL)
    common.assert_allclose(hard_cov, private_hard_cov, tol=8 * DISTRIBUTIONAL_TOL)


def test_multinomial_soft_requires_static_scalar_count():
    key = jax.random.key(51)

    with pytest.raises(NotImplementedError, match="scalar static counts"):
        sj.random.multinomial(
            key, jnp.array([2, 3]), jnp.array([[0.2, 0.8], [0.7, 0.3]])
        )


def test_multinomial_st_forward_and_grad():
    key = jax.random.key(56)
    p = jnp.array([0.2, 0.3, 0.5], dtype=jnp.float64)
    weights = jnp.array([0.3, -0.7, 1.1], dtype=jnp.float64)

    actual = sj.random.multinomial_st(key, 4, p, softness=1.0, mode="smooth")
    expected = sj.random.multinomial(key, 4, p, mode="_hard")
    common.assert_allclose(actual, expected, tol=0.0)

    def st_loss(prob):
        return jnp.sum(
            sj.random.multinomial_st(key, 4, prob, softness=1.0, mode="smooth")
            * weights
        )

    def soft_loss(prob):
        return jnp.sum(
            sj.random.multinomial(key, 4, prob, softness=1.0, mode="smooth") * weights
        )

    common.assert_allclose(jax.grad(st_loss)(p), jax.grad(soft_loss)(p))


def test_new_random_functions_jit():
    key = jax.random.key(57)
    values = jnp.array([10.0, 20.0, 30.0], dtype=jnp.float64)
    p = jnp.array([0.2, 0.3, 0.5], dtype=jnp.float64)

    choice_out = jax.jit(
        lambda prob: sj.random.choice(key, values, p=prob, softness=1.0)
    )(p)
    permutation_out = jax.jit(lambda x: sj.random.permutation(key, x, softness=1.0))(
        values
    )
    rademacher_out = jax.jit(
        lambda: sj.random.rademacher(key, shape=(3,), softness=1.0)
    )()
    binomial_out = jax.jit(lambda prob: sj.random.binomial(key, 4, prob, softness=1.0))(
        p
    )
    multinomial_out = jax.jit(
        lambda prob: sj.random.multinomial(key, 4, prob, softness=1.0)
    )(p)

    assert choice_out.shape == ()
    assert permutation_out.shape == values.shape
    assert rademacher_out.shape == values.shape
    assert binomial_out.shape == p.shape
    assert multinomial_out.shape == p.shape


def test_random_shape_and_rng_mode_errors():
    key = jax.random.key(23)

    with pytest.raises(ValueError, match="broadcast-compatible"):
        sj.random.categorical(
            key, jnp.array([[0.1, 1.5, -0.2], [1.2, -0.4, 0.3]]), shape=(3,)
        )

    with pytest.raises(ValueError, match="broadcast-compatible"):
        sj.random.bernoulli(key, jnp.array([0.2, 0.7, 0.4]), shape=(2,))

    with pytest.raises(ValueError, match="expected 'high' or 'low'"):
        sj.random.bernoulli(key, jnp.array([0.5]), rng_mode="medium")

    with pytest.raises(TypeError, match="integer type"):
        sj.random.bernoulli(key, jnp.array([0.5]), shape=(1.2,))


def test_random_invalid_noise_errors():
    key = jax.random.key(11)
    with pytest.raises(ValueError, match="noise='gumbel'"):
        sj.random.categorical(key, jnp.array([0.1, 0.2]), noise="uniform")
    with pytest.raises(ValueError, match="noise='gumbel'"):
        sj.random.choice(key, jnp.array([0.1, 0.2]), noise="uniform")
    with pytest.raises(ValueError, match="noise='uniform'"):
        sj.random.bernoulli(key, jnp.array([0.5]), noise="gumbel")
    with pytest.raises(ValueError, match="noise='uniform'"):
        sj.random.rademacher(key, noise="gumbel")
    with pytest.raises(ValueError, match="noise='uniform'"):
        sj.random.rademacher(key, mode="hard", noise="gumbel")
    with pytest.raises(ValueError, match="noise='uniform'"):
        sj.random.binomial(key, 2, jnp.array([0.5]), noise="gumbel")
    with pytest.raises(ValueError, match="noise='uniform'"):
        sj.random.binomial(key, 2, jnp.array([0.5]), mode="hard", noise="gumbel")
    with pytest.raises(ValueError, match="noise='gumbel'"):
        sj.random.multinomial(
            key, 2, jnp.array([0.2, 0.8]), mode="hard", noise="uniform"
        )
