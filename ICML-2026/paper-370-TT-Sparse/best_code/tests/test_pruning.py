"""Pruning tests.

Validates that:
1. fold_classifier produces numerically identical forward passes.
2. _structural_simplify reaches a fixed point (exercises constant propagation + disconnect pruning).
3. prune() end-to-end: sparsity increases, metric respects constraints, Pareto frontier valid.
"""

import numpy as np
import torch
import pytest

from tt_sparse.model import (
    TTSparseModel,
    _structural_simplify,
)


@pytest.fixture
def simple_model():
    """Small trained model with known structure for unit tests."""
    torch.manual_seed(42)
    m = TTSparseModel(
        input_size=10, num_nodes=4, num_classes=1, n_bits=3,
        tau=0.1, task_type="binary", skip_size=0, dropout=0.0,
    )
    m.eval()
    m.freeze_connections()
    m.fold_classifier()
    return m


class TestFoldClassifier:
    """Verify fold_classifier produces numerically identical output."""

    @pytest.mark.parametrize("seed,input_size,num_nodes,num_classes,n_bits,task_type,skip_size", [
        (0, 12, 6, 1, 4, "binary", 0),
        (1, 10, 4, 1, 3, "binary", 3),
        (2, 8, 4, 3, 2, "multiclass", 0),
    ], ids=["binary", "binary-skip", "multiclass"])
    def test_forward_equivalence(self, seed, input_size, num_nodes, num_classes, n_bits, task_type, skip_size):
        torch.manual_seed(seed)
        m = TTSparseModel(
            input_size=input_size, num_nodes=num_nodes, num_classes=num_classes,
            n_bits=n_bits, tau=0.1, task_type=task_type, skip_size=skip_size, dropout=0.0,
        )
        m.eval()
        m.freeze_connections()

        x = torch.randn(16, input_size)
        xs = torch.randn(16, skip_size) if skip_size > 0 else None
        with torch.no_grad():
            out_before = m(x, xs).clone()

        m.fold_classifier()
        with torch.no_grad():
            out_after = m(x, xs)

        np.testing.assert_allclose(
            out_before.numpy(), out_after.numpy(), atol=1e-5, rtol=1e-5
        )

    def test_idempotent(self, simple_model):
        m = simple_model
        w_before = m.clf_linear.weight.data.clone()
        m.fold_classifier()
        torch.testing.assert_close(m.clf_linear.weight.data, w_before)

    def test_is_classifier_folded_flag(self):
        m = TTSparseModel(
            input_size=6, num_nodes=2, num_classes=1, n_bits=2,
            tau=0.1, task_type="binary",
        )
        assert not m.is_classifier_folded
        m.freeze_connections()
        m.fold_classifier()
        assert m.is_classifier_folded

    def test_get_folded_classifier_from_folded(self, simple_model):
        m = simple_model
        W, b = m.get_folded_classifier()
        np.testing.assert_allclose(W, m.clf_linear.weight.detach().numpy(), atol=1e-6)
        np.testing.assert_allclose(b, m.clf_linear.bias.detach().numpy(), atol=1e-6)


class TestStructuralSimplify:
    """Verify iterative simplification reaches fixed point."""

    def test_reaches_fixed_point(self, simple_model):
        m = simple_model
        with torch.no_grad():
            m.logic_weights[:, 0] = 0
            m.ltt_bias[0] = 1.0
            m.clf_linear.weight[:, 0] = 2.0

            m.logic_weights[:, 1] = 0
            m.ltt_bias[1] = -1.0
            m.clf_linear.weight[:, 1] = 3.0

        total = _structural_simplify(m)
        assert total > 0
        assert (m.logic_weights[:, 0] == 0).all()
        assert (m.logic_weights[:, 1] == 0).all()
        assert m.clf_linear.weight[:, 0].item() == 0.0
        assert m.clf_linear.weight[:, 1].item() == 0.0

    def test_cascading_simplification(self, simple_model):
        m = simple_model
        with torch.no_grad():
            m.logic_weights[:] = 0
            m.logic_weights[0, 2] = 1.0
            m.ltt_bias[0] = 1.0
            m.ltt_bias[1] = -1.0
            m.ltt_bias[2] = 0.5
            m.ltt_bias[3] = 0.5
            m.clf_linear.weight[:] = 0
            m.clf_linear.weight[0, 0] = 2.0
            m.clf_linear.weight[0, 1] = 1.0

        _structural_simplify(m)

        assert m.clf_linear.weight[0, 0].item() == 0.0
        assert m.clf_linear.weight[0, 1].item() == 0.0
        assert (m.logic_weights[:, 2] == 0).all()
        assert (m.logic_weights[:, 3] == 0).all()

    def test_already_clean_noop(self, simple_model):
        m = simple_model
        with torch.no_grad():
            m.logic_weights[:] = 0
            m.ltt_bias[:] = 0
            m.clf_linear.weight[:] = 0
            m.clf_linear.bias[:] = 0

        total = _structural_simplify(m)
        assert total == 0


class TestPruneEndToEnd:
    """End-to-end pruning: verify sparsity increases and metric holds."""

    @pytest.fixture
    def trained_model_and_data(self):
        torch.manual_seed(99)
        np.random.seed(99)

        n_samples = 200
        n_features = 6
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        X_bin = (X > 0).astype(np.float32)
        y = ((X_bin[:, 0] + X_bin[:, 1]) % 2).astype(np.float32)

        m = TTSparseModel(
            input_size=n_features, num_nodes=8, num_classes=1, n_bits=3,
            tau=0.1, task_type="binary", skip_size=0, dropout=0.0,
        )

        from tt_sparse.model import train
        data = {"X_ltt": X_bin, "X_skip": None, "y": y}
        train(m, data, epochs=40, batch_size=64, lr=0.01,
              val_split=0.2, patience=20, verbose=False)

        return m, data

    def test_sparsity_increases(self, trained_model_and_data):
        from tt_sparse.model import prune

        m, data = trained_model_and_data
        m.freeze_connections()

        lw_before = (m.logic_weights.detach() != 0).sum().item()

        stats = prune(
            m, data,
            max_drop_pct=5.0,
            collapse_drop_pct=30.0,
            finetune_epochs=5,
            finetune_batch_size=64,
            initial_prune_fraction=0.15,
            max_iterations=10,
            max_fanin=4,
            verbose=False,
        )

        assert stats["ltt_sparsity_pct"] > 0
        assert stats["edges_remaining"] <= lw_before
        assert not m.training
        assert m.is_classifier_folded

        expected_keys = {
            "baseline_metric", "final_metric", "ltt_sparsity_pct",
            "classifier_sparsity_pct", "edges_remaining", "complexity",
            "lut_cost", "boolean_cost", "accepted_steps", "iterations",
            "pareto_frontier",
        }
        assert expected_keys <= set(stats.keys())

    def test_pareto_frontier_no_dominated(self, trained_model_and_data):
        from tt_sparse.model import prune

        m, data = trained_model_and_data
        stats = prune(
            m, data,
            max_drop_pct=5.0,
            collapse_drop_pct=30.0,
            finetune_epochs=5,
            finetune_batch_size=64,
            initial_prune_fraction=0.10,
            max_iterations=10,
            max_fanin=4,
            verbose=False,
        )

        frontier = stats["pareto_frontier"]
        for i, p in enumerate(frontier):
            for j, q in enumerate(frontier):
                if i == j:
                    continue
                dominated = (
                    q["metric"] >= p["metric"]
                    and q["total_complexity"] <= p["total_complexity"]
                    and (q["metric"] > p["metric"] or q["total_complexity"] < p["total_complexity"])
                )
                assert not dominated, f"Point {i} dominated by point {j}"
