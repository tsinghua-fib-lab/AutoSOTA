import numpy as np
from core.computation import compute_subgroup_metrics
from proto.config import ComputationConfig
from proto.subgroup import SubgroupResults
from stats.prevalence import default_prevalence_grid


def _make_subgroup(n=80, seed=123, name="sg"):
    from reproducibility.seed_manager import get_random_generator
    rng = get_random_generator(seed=seed, component_name="test_computation_" + name)
    y_true = rng.integers(0, 2, size=n)
    scores = rng.uniform(0, 1, size=n)
    return SubgroupResults(name=name, y_true=y_true, y_pred_proba=scores)


def test_compute_subgroup_metrics_optional_outputs():
    sg = _make_subgroup()
    grid = default_prevalence_grid(num=50)

    # Baseline: no CI, no calibration, no diamond shift
    cfg = ComputationConfig(
        prevalence_grid=grid,
        cost_ratio=0.4,
        compute_ci=False,
        compute_calibrated=False,
        diamond_shift_amount=None,
        n_bootstrap=10,
    )
    res = compute_subgroup_metrics(sg, cfg)
    assert res.nb_ci_lower is None and res.nb_ci_upper is None
    assert res.calibrated_nb_curve is None
    assert res.shifted_point is None

    # With calibration only
    cfg2 = cfg.__class__(**{**cfg.__dict__, "compute_calibrated": True})
    res2 = compute_subgroup_metrics(sg, cfg2)
    assert res2.calibrated_nb_curve is not None
    assert res2.nb_ci_lower is None

    # With CI only
    cfg3 = cfg.__class__(**{**cfg.__dict__, "compute_ci": True})
    res3 = compute_subgroup_metrics(sg, cfg3)
    assert res3.nb_ci_lower is not None and res3.nb_ci_upper is not None
    assert res3.calibrated_nb_curve is None

    # With diamond shift
    cfg4 = cfg.__class__(**{**cfg.__dict__, "diamond_shift_amount": 0.3})
    res4 = compute_subgroup_metrics(sg, cfg4)
    assert res4.shifted_point is not None


def test_identical_subgroup_names_preserved_order():
    grid = default_prevalence_grid(num=30)
    cfg = ComputationConfig(prevalence_grid=grid, cost_ratio=0.5)

    sg1 = _make_subgroup(seed=1, name="dup")
    sg2 = _make_subgroup(seed=2, name="dup")

    res1 = compute_subgroup_metrics(sg1, cfg)
    res2 = compute_subgroup_metrics(sg2, cfg)

    # Even with identical names, results should be computed independently
    assert not np.array_equal(res1.nb_curve, res2.nb_curve)
    # Ensure order is preserved when placed in list comprehension
    computed = [compute_subgroup_metrics(s, cfg) for s in [sg1, sg2]]
    assert computed[0].name == "dup" and computed[1].name == "dup" 