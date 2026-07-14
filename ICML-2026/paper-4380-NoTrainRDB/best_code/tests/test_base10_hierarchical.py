"""Unit tests for base-10-hierarchical decomposition helpers."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier

_PKG_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "rdblearn.base10_hierarchical",
    _PKG_ROOT / "rdblearn" / "base10_hierarchical.py",
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

Base10Decomposer = _mod.Base10Decomposer
reconstruct_class_proba_from_digit_probs = _mod.reconstruct_class_proba_from_digit_probs


class TestBase10Hierarchical(unittest.TestCase):
    def test_digit_consistency(self):
        for C in (11, 18, 100, 502):
            dec = Base10Decomposer(C)
            for c in range(C):
                d1 = dec.decompose_class(c)
                arr = np.array([c])
                d2 = [dec.digits_for_array(arr)[i][0] for i in range(dec.D)]
                self.assertEqual(d1, d2)

    def test_D(self):
        self.assertEqual(Base10Decomposer(11).D, 2)
        self.assertEqual(Base10Decomposer(10).D, 1)
        self.assertEqual(Base10Decomposer(100).D, 2)
        self.assertEqual(Base10Decomposer(101).D, 3)

    def test_reconstruct_proba_shape_and_normalization(self):
        C = 18
        n_test = 5
        dec = Base10Decomposer(C)
        digit_probs = [np.full((n_test, 10), 0.1) for _ in range(dec.D)]
        p = reconstruct_class_proba_from_digit_probs(dec, digit_probs, C)
        self.assertEqual(p.shape, (n_test, C))
        np.testing.assert_allclose(p.sum(axis=1), 1.0, rtol=1e-5)

    def test_class_17_digits(self):
        dec = Base10Decomposer(18)
        self.assertEqual(dec.decompose_class(17), [7, 1])

    def test_digit_heads_via_dummy_classifier(self):
        C = 12
        n_train, n_test = 80, 10
        rng = np.random.RandomState(0)
        X_train = rng.randn(n_train, 4)
        X_test = rng.randn(n_test, 4)
        y_enc = rng.randint(0, C, n_train)

        dec = Base10Decomposer(C)
        train_digits = dec.digits_for_array(y_enc)
        digit_probs = []
        for i in range(dec.D):
            clf = DummyClassifier(strategy="uniform")
            clf.fit(X_train, train_digits[i])
            digit_probs.append(clf.predict_proba(X_test))

        p = reconstruct_class_proba_from_digit_probs(dec, digit_probs, C)
        self.assertEqual(p.shape, (n_test, C))
        np.testing.assert_allclose(p.sum(axis=1), 1.0, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
