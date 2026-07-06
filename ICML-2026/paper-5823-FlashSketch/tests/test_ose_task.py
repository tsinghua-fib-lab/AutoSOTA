from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

import pytest
import torch

import globals as g
from bench.e2e.tasks.ose_error import OseErrorConfig, run_task as run_ose_error
from sketches.gaussian_dense_cublas import GaussianDenseCublasConfig, sketch as gaussian_sketch
from torch_utils import manual_seed

pytestmark = pytest.mark.small


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for OSE task test")
def test_ose_error_decreases_with_k():
    """OSE errors should decrease with larger sketch dimensions on average."""
    device = torch.device("cuda")
    manual_seed(0)
    A = torch.randn((256, 64), device=device, dtype=torch.float32)
    task_cfg = OseErrorConfig(r=32, ose_variant="colspace")

    errors_small = []
    errors_large = []
    for seed in [0, 1, 2]:
        cfg_small = GaussianDenseCublasConfig(k=64, seed=seed, dtype=g.DTYPE_FP32)
        cfg_large = GaussianDenseCublasConfig(k=128, seed=seed, dtype=g.DTYPE_FP32)

        metrics_small = run_ose_error(A, gaussian_sketch, cfg_small, task_cfg)
        metrics_large = run_ose_error(A, gaussian_sketch, cfg_large, task_cfg)

        errors_small.append(metrics_small["ose_spec_err"])
        errors_large.append(metrics_large["ose_spec_err"])

    avg_small = sum(errors_small) / len(errors_small)
    avg_large = sum(errors_large) / len(errors_large)
    assert avg_large <= avg_small * 1.1
