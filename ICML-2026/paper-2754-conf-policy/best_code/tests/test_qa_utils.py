import numpy as np

from utils import (
    get_taus_grid_from_data,
    hb_p_value,
    loss_factuality,
    split_dataset,
)


class TestHbPValue:
    def test_low_risk_small_pvalue(self):
        # H0: risk >= alpha. When r_hat << alpha, strong evidence to reject → small p
        p = hb_p_value(r_hat=0.01, n=100, alpha=0.1)
        assert p < 0.01

    def test_risk_at_alpha_pvalue_one(self):
        # When r_hat >= alpha, cannot reject H0 → p-value = 1
        p = hb_p_value(r_hat=0.10, n=100, alpha=0.1)
        assert p == 1.0

    def test_returns_finite(self):
        p = hb_p_value(r_hat=0.05, n=50, alpha=0.1)
        assert np.isfinite(p)
        assert p >= 0


class TestSplitDataset:
    def test_split_fractions(self):
        x = list(range(100))
        y = list(range(100, 200))
        rng = np.random.RandomState(42)

        (x_train, y_train), (x_cal, y_cal), train_ind, cal_ind = split_dataset(
            (x, y), rng, train_frac=0.8
        )

        assert len(x_train) == 80
        assert len(x_cal) == 20
        assert len(train_ind) + len(cal_ind) == 100

    def test_no_overlap(self):
        x = list(range(50))
        y = list(range(50))
        rng = np.random.RandomState(0)

        _, _, train_ind, cal_ind = split_dataset((x, y), rng, train_frac=0.6)

        assert len(set(train_ind) & set(cal_ind)) == 0

    def test_explicit_train_num(self):
        x = list(range(20))
        y = list(range(20))
        rng = np.random.RandomState(1)

        (x_train, _), (x_cal, _), _, _ = split_dataset((x, y), rng, train_num=5)

        assert len(x_train) == 5
        assert len(x_cal) == 15


class TestLossFactuality:
    def test_all_true_returns_zero(self):
        scores = np.array([0.8, 0.9, 0.7])
        annotations = np.array([True, True, True])
        assert loss_factuality(scores, annotations, tau=0.5) == 0

    def test_false_claim_above_threshold(self):
        scores = np.array([0.8, 0.9, 0.7])
        annotations = np.array([True, False, True])
        assert loss_factuality(scores, annotations, tau=0.5) == 1

    def test_false_claim_below_threshold_excluded(self):
        scores = np.array([0.8, 0.3, 0.7])
        annotations = np.array([True, False, True])
        # The false claim (score=0.3) is below tau=0.5, so excluded
        assert loss_factuality(scores, annotations, tau=0.5) == 0

    def test_no_claims_above_threshold(self):
        scores = np.array([0.1, 0.2])
        annotations = np.array([True, False])
        assert loss_factuality(scores, annotations, tau=0.5) == 0


class TestGetTausGrid:
    def test_flattens_scores(self):
        scores = [np.array([0.1, 0.2]), np.array([0.3]), np.array([0.4, 0.5])]
        result = get_taus_grid_from_data(scores)
        np.testing.assert_array_equal(result, [0.1, 0.2, 0.3, 0.4, 0.5])

    def test_single_array(self):
        scores = [np.array([1.0, 2.0, 3.0])]
        result = get_taus_grid_from_data(scores)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])
