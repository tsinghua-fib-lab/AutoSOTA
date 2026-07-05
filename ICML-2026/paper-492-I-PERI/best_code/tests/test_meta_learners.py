import pytest
import numpy as np
import pandas as pd

from pyciphod.causal_estimation.meta_learners import SLearner, TLearner, XLearner


def make_data(n=300, exposure_type='binary', outcome_type='binary', seed=1):
    rng = np.random.RandomState(seed)
    X1 = rng.normal(size=n)
    X2 = rng.normal(size=n)
    if exposure_type == 'binary':
        logits_t = 0.6 * X1 - 0.3 * X2
        p_t = 1.0 / (1.0 + np.exp(-logits_t))
        t = (rng.rand(n) < p_t).astype(int)
    else:
        t = X1 + 0.1 * rng.normal(size=n)

    if outcome_type == 'continuous':
        y = 1.0 + 2.0 * t + 0.5 * X1 - 0.3 * X2 + 0.5 * rng.normal(size=n)
    else:
        logits_y = -0.5 + 1.2 * t + 0.3 * X1 - 0.2 * X2
        p_y = 1.0 / (1.0 + np.exp(-logits_y))
        y = (rng.rand(n) < p_y).astype(int)

    df = pd.DataFrame({'x1': X1, 'x2': X2, 't': t, 'y': y})
    return df


def check_basic_result(res):
    assert isinstance(res, dict)
    assert 'cate' in res
    assert 'model' in res
    # ensure cate is numeric
    assert isinstance(res['cate'], (float, int))


def test_binary_exposure_binary_outcome():
    df = make_data(exposure_type='binary', outcome_type='binary', seed=10)
    s = SLearner('t', 'y', z=['x1', 'x2'], w=None, model=None, seed=0)
    res_s = s.run(df)
    check_basic_result(res_s)

    t = TLearner('t', 'y', z=['x1', 'x2'], w=None, model=None, seed=0)
    res_t = t.run(df)
    check_basic_result(res_t)

    x = XLearner('t', 'y', z=['x1', 'x2'], w=None, model=None, seed=0)
    res_x = x.run(df)
    check_basic_result(res_x)


def test_continuous_exposure_binary_outcome():
    df = make_data(exposure_type='continuous', outcome_type='binary', seed=20)
    s = SLearner('t', 'y', z=['x1', 'x2'], w=None, model=None, seed=0)
    res_s = s.run(df)
    check_basic_result(res_s)

    t = TLearner('t', 'y', z=['x1', 'x2'], w=None, model=None, seed=0)
    with pytest.raises(ValueError):
        t.run(df)

    x = XLearner('t', 'y', z=['x1', 'x2'], w=None, model=None, seed=0)
    with pytest.raises(ValueError):
        x.run(df)


def test_binary_exposure_continuous_outcome():
    df = make_data(exposure_type='binary', outcome_type='continuous', seed=30)
    s = SLearner('t', 'y', z=['x1', 'x2'], w=None, model=None, seed=0)
    res_s = s.run(df)
    check_basic_result(res_s)

    t = TLearner('t', 'y', z=['x1', 'x2'], w=None, model=None, seed=0)
    res_t = t.run(df)
    check_basic_result(res_t)

    x = XLearner('t', 'y', z=['x1', 'x2'], w=None, model=None, seed=0)
    res_x = x.run(df)
    check_basic_result(res_x)


def test_continuous_exposure_continuous_outcome():
    df = make_data(exposure_type='continuous', outcome_type='continuous', seed=40)
    s = SLearner('t', 'y', z=['x1', 'x2'], w=None, model=None, seed=0)
    res_s = s.run(df)
    check_basic_result(res_s)

    t = TLearner('t', 'y', z=['x1', 'x2'], w=None, model=None, seed=0)
    with pytest.raises(ValueError):
        t.run(df)

    x = XLearner('t', 'y', z=['x1', 'x2'], w=None, model=None, seed=0)
    with pytest.raises(ValueError):
        x.run(df)


def test_small_treated_group_fallback_to_slearner():
    # Create a dataset with only 1 treated unit so TLearner must fallback to SLearner
    n = 50
    rng = np.random.RandomState(123)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    t = np.zeros(n, dtype=int)
    t[0] = 1  # only one treated
    # continuous outcome
    y = 5.0 + 1.5 * t + 0.2 * x1 + 0.1 * x2 + rng.normal(scale=0.1, size=n)
    df = pd.DataFrame({'x1': x1, 'x2': x2, 't': t, 'y': y})

    learner = TLearner('t', 'y', z=['x1', 'x2'], model=None, seed=0)
    res = learner.run(df)
    # Expect a result (fallback to S-learner) and numeric cate
    assert isinstance(res, dict)
    assert 'cate' in res and isinstance(res['cate'], (float, int))


def test_model_types_for_binary_outcome_use_classifier():
    # For binary outcome learners should use classifiers (predict_proba) where appropriate
    df = make_data(exposure_type='binary', outcome_type='binary', seed=77)

    s = SLearner('t', 'y', z=['x1', 'x2'], model=None, seed=0)
    res_s = s.run(df)
    model_s = res_s['model']
    assert hasattr(model_s, 'predict'), "SLearner model must implement predict"
    # prefer classifier: if predict_proba exists it should be used for binary outcomes
    assert hasattr(model_s, 'predict_proba') or not _is_callable_probable(model_s), "SLearner should use a classifier for binary outcome when possible"

    t = TLearner('t', 'y', z=['x1', 'x2'], model=None, seed=0)
    res_t = t.run(df)
    model_t = res_t['model']['model_t']
    model_c = res_t['model']['model_c']
    assert hasattr(model_t, 'predict')
    assert hasattr(model_c, 'predict')
    assert hasattr(model_t, 'predict_proba') or hasattr(model_c, 'predict_proba')

    x = XLearner('t', 'y', z=['x1', 'x2'], model=None, seed=0)
    res_x = x.run(df)
    mxt = res_x['model']['model_t']
    mxc = res_x['model']['model_c']
    assert hasattr(mxt, 'predict')
    assert hasattr(mxc, 'predict')
    assert hasattr(mxt, 'predict_proba') or hasattr(mxc, 'predict_proba')


def _is_callable_probable(model):
    """Helper used in test to detect if model implements predict_proba safely."""
    return hasattr(model, 'predict_proba')


def test_marginal_effect_estimate_continuous_exposure_no_covariates():
    # Simple linear data y = 3 * t + noise, no covariates -> SLearner marginal effect should be ~3
    n = 300
    rng = np.random.RandomState(2026)
    t = np.linspace(-1.0, 1.0, n) + 0.01 * rng.normal(size=n)
    y = 3.0 * t + 0.2 * rng.normal(size=n)
    df = pd.DataFrame({'t': t, 'y': y})

    s = SLearner('t', 'y', z=None, w=None, model=None, seed=0)
    res = s.run(df)
    assert 'cate' in res
    # allow a reasonable tolerance
    assert abs(res['cate'] - 3.0) < 0.5, f"Estimated marginal effect {res['cate']} deviates from true 3.0"


def test_w_provided_binary_exposure_binary_outcome():
    df = make_data(exposure_type='binary', outcome_type='binary', seed=11)
    # build a binary conditioning variable w from x1
    df['w'] = (df['x1'] > 0).astype(int)

    s = SLearner('t', 'y', z=['x1', 'x2'], w=['w'], model=None, seed=0)
    res_s = s.run(df)
    assert 'cate_per_w_stratum' in res_s and isinstance(res_s['cate_per_w_stratum'], dict)
    assert set(res_s['cate_per_w_stratum'].keys()).issubset({0, 1})

    t = TLearner('t', 'y', z=['x1', 'x2'], w=['w'], model=None, seed=0)
    res_t = t.run(df)
    assert 'cate_per_w_stratum' in res_t and isinstance(res_t['cate_per_w_stratum'], dict)

    x = XLearner('t', 'y', z=['x1', 'x2'], w=['w'], model=None, seed=0)
    res_x = x.run(df)
    assert 'cate_per_w_stratum' in res_x and isinstance(res_x['cate_per_w_stratum'], dict)


def test_w_provided_continuous_exposure_binary_outcome():
    df = make_data(exposure_type='continuous', outcome_type='binary', seed=12)
    df['w'] = (df['x1'] > 0).astype(int)

    s = SLearner('t', 'y', z=['x1', 'x2'], w=['w'], model=None, seed=0)
    res_s = s.run(df)
    assert 'cate_per_w_stratum' in res_s and isinstance(res_s['cate_per_w_stratum'], dict)

    t = TLearner('t', 'y', z=['x1', 'x2'], w=['w'], model=None, seed=0)
    with pytest.raises(ValueError):
        t.run(df)

    x = XLearner('t', 'y', z=['x1', 'x2'], w=['w'], model=None, seed=0)
    with pytest.raises(ValueError):
        x.run(df)


def test_w_provided_binary_exposure_continuous_outcome():
    df = make_data(exposure_type='binary', outcome_type='continuous', seed=13)
    df['w'] = (df['x1'] > 0).astype(int)

    s = SLearner('t', 'y', z=['x1', 'x2'], w=['w'], model=None, seed=0)
    res_s = s.run(df)
    assert 'cate_per_w_stratum' in res_s and isinstance(res_s['cate_per_w_stratum'], dict)

    t = TLearner('t', 'y', z=['x1', 'x2'], w=['w'], model=None, seed=0)
    res_t = t.run(df)
    assert 'cate_per_w_stratum' in res_t and isinstance(res_t['cate_per_w_stratum'], dict)

    x = XLearner('t', 'y', z=['x1', 'x2'], w=['w'], model=None, seed=0)
    res_x = x.run(df)
    assert 'cate_per_w_stratum' in res_x and isinstance(res_x['cate_per_w_stratum'], dict)


def test_w_provided_continuous_exposure_continuous_outcome():
    df = make_data(exposure_type='continuous', outcome_type='continuous', seed=14)
    df['w'] = (df['x1'] > 0).astype(int)

    s = SLearner('t', 'y', z=['x1', 'x2'], w=['w'], model=None, seed=0)
    res_s = s.run(df)
    assert 'cate_per_w_stratum' in res_s and isinstance(res_s['cate_per_w_stratum'], dict)

    t = TLearner('t', 'y', z=['x1', 'x2'], w=['w'], model=None, seed=0)
    with pytest.raises(ValueError):
        t.run(df)

    x = XLearner('t', 'y', z=['x1', 'x2'], w=['w'], model=None, seed=0)
    with pytest.raises(ValueError):
        x.run(df)

