"""Model-rule equivalence tests.

Verifies that rule-based predictions (via explain + predict_rules) exactly match
PyTorch model predictions (forward pass + activation) for all three task types.

This is the contract that makes rule extraction meaningful: the symbolic
representation must reproduce the neural network's output.
"""

import numpy as np
import pytest
import torch
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris

from tt_sparse import TabularEncoder, TTSparseModel, train, prune, explain, predict_rules

SEED = 0


def _build_and_predict(df, task_type, n_nodes=8, n_bits=3, num_bits=3):
    """Build, train, prune, extract rules, return model and rule predictions."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    encoder = TabularEncoder(target="target", task_type=task_type, num_bits=num_bits)
    data = encoder.fit_transform(df)

    n_classes = len(np.unique(data["y"])) if task_type != "regression" else 1

    model = TTSparseModel(
        input_size=encoder.n_ltt_features,
        num_nodes=n_nodes,
        num_classes=n_classes,
        n_bits=n_bits,
        task_type=task_type,
        skip_size=encoder.n_skip_features,
    )

    train(model, data, epochs=20, device="cpu", seed=SEED, verbose=False)
    prune(model, data, max_drop_pct=5.0, finetune_epochs=5, device="cpu")

    # Model predictions
    model.eval()
    with torch.no_grad():
        X_ltt = torch.tensor(data["X_ltt"], dtype=torch.float32)
        X_skip = torch.tensor(data["X_skip"], dtype=torch.float32)
        logits = model(X_ltt, X_skip)
        if task_type == "binary":
            model_preds = torch.sigmoid(logits.view(-1)).numpy()
        elif task_type == "regression":
            model_preds = logits.view(-1).numpy()
        else:
            model_preds = torch.softmax(logits, 1).numpy()

    # Rule predictions
    rules = explain(model, encoder)
    df_features = df.drop(columns=["target"])
    rule_preds = predict_rules(rules, df_features, encoder=encoder)

    return model_preds, rule_preds, rules


class TestBinaryEquivalence:
    """Breast cancer: binary classification equivalence."""

    @pytest.fixture
    def data(self):
        ds = load_breast_cancer(as_frame=True)
        df = ds.frame.rename(columns={"target": "target"})
        return df

    def test_predictions_match(self, data):
        model_p, rule_p, _ = _build_and_predict(data, task_type="binary")
        assert model_p.shape == rule_p.shape
        np.testing.assert_allclose(
            rule_p, model_p, atol=1e-5, rtol=1e-5,
            err_msg="Binary: rule predictions diverge from model predictions",
        )

    def test_probability_bounds(self, data):
        model_p, rule_p, _ = _build_and_predict(data, task_type="binary")
        assert np.all(model_p >= 0) and np.all(model_p <= 1)
        assert np.all(rule_p >= 0) and np.all(rule_p <= 1)


class TestMulticlassEquivalence:
    """Iris: multiclass equivalence."""

    @pytest.fixture
    def data(self):
        ds = load_iris(as_frame=True)
        df = ds.frame.rename(columns={"target": "target"})
        df["target"] = df["target"].map({0: "setosa", 1: "versicolor", 2: "virginica"})
        return df

    def test_predictions_match(self, data):
        model_p, rule_p, _ = _build_and_predict(data, task_type="multiclass")
        assert model_p.shape == rule_p.shape
        np.testing.assert_allclose(
            rule_p, model_p, atol=1e-5, rtol=1e-5,
            err_msg="Multiclass: rule predictions diverge from model predictions",
        )

    def test_probabilities_sum_to_one(self, data):
        model_p, rule_p, _ = _build_and_predict(data, task_type="multiclass")
        np.testing.assert_allclose(model_p.sum(axis=1), 1.0, atol=1e-5)
        np.testing.assert_allclose(rule_p.sum(axis=1), 1.0, atol=1e-5)

    def test_argmax_agreement(self, data):
        model_p, rule_p, _ = _build_and_predict(data, task_type="multiclass")
        assert np.array_equal(model_p.argmax(1), rule_p.argmax(1))


class TestRegressionEquivalence:
    """Diabetes: regression equivalence."""

    @pytest.fixture
    def data(self):
        ds = load_diabetes(as_frame=True)
        df = ds.frame.copy()
        df["target"] = ds.target
        return df

    def test_predictions_match(self, data):
        model_p, rule_p, _ = _build_and_predict(data, task_type="regression")
        assert model_p.shape == rule_p.shape
        np.testing.assert_allclose(
            rule_p, model_p, atol=1e-5, rtol=1e-5,
            err_msg="Regression: rule predictions diverge from model predictions",
        )
