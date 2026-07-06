from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

import pytest
import torch

from data.suitesparse.load import (
    SuiteSparseDatasetConfig,
    _compute_submatrix_shape,
    _resolve_max_dense_elems,
    _resolve_submatrix_shape,
)

pytestmark = pytest.mark.small


def test_compute_submatrix_shape_bounds():
    rows, cols = _compute_submatrix_shape(1000, 200, 10_000)
    assert rows > 0
    assert cols > 0
    assert rows <= 1000
    assert cols <= 200
    assert rows * cols <= 10_000


def test_resolve_submatrix_shape_auto():
    cfg = SuiteSparseDatasetConfig(auto_submatrix=True)
    rows, cols, used = _resolve_submatrix_shape(5000, 2000, cfg, 10_000)
    assert used is True
    assert rows > 0
    assert cols > 0
    assert rows <= 5000
    assert cols <= 2000
    assert rows * cols <= 10_000


def test_resolve_submatrix_shape_enforce_max():
    cfg = SuiteSparseDatasetConfig(submatrix_rows=200, submatrix_cols=200, enforce_max_elems=True)
    with pytest.raises(ValueError):
        _resolve_submatrix_shape(1000, 1000, cfg, 10_000)


def test_resolve_submatrix_shape_no_submatrix():
    cfg = SuiteSparseDatasetConfig()
    rows, cols, used = _resolve_submatrix_shape(100, 200, cfg, 50_000)
    assert used is False
    assert rows == 100
    assert cols == 200


def test_resolve_max_dense_elems_conflict():
    cfg = SuiteSparseDatasetConfig(max_dense_elems=100, max_dense_fraction=0.5)
    with pytest.raises(ValueError):
        _resolve_max_dense_elems(cfg, torch.device("cpu"), torch.float32)


def test_resolve_max_dense_elems_fraction_cpu():
    cfg = SuiteSparseDatasetConfig(max_dense_fraction=0.5)
    with pytest.raises(RuntimeError):
        _resolve_max_dense_elems(cfg, torch.device("cpu"), torch.float32)


def test_resolve_max_dense_elems_invalid_fraction():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for fraction validation.")
    cfg = SuiteSparseDatasetConfig(max_dense_fraction=1.5)
    with pytest.raises(ValueError):
        _resolve_max_dense_elems(cfg, torch.device("cuda"), torch.float32)
