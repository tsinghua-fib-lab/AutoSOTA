"""
Parity test for HVP implementations.

Verifies that FlashSinkhorn, OTT-Hessian (KeOps), and OTT-Hessian (JAX)
all compute the same Hessian-vector product using the Schur complement formula.

All three implementations should give identical results (within numerical tolerance)
since they use the same mathematical formula:
    HVP = R^T @ (H*)^{-1} @ R @ A / eps + E @ A

Where:
    - R = Jacobian of optimality conditions w.r.t. x
    - H* = Schur complement of the Hessian of Lagrangian
    - E = Explicit Hessian term (5 matrix terms)

NOTE: These tests require optional dependencies (KeOps, JAX with GPU).
They will be skipped if those dependencies are not available or fail to initialize.
JAX GPU often fails due to cuDNN version conflicts with PyTorch.
"""

import pytest
import torch
import numpy as np


def test_hvp_parity_flashsinkhorn_vs_ott_hessian_keops():
    """Test parity between FlashSinkhorn and OTT-Hessian (KeOps) HVP."""
    pytest.importorskip("pykeops")

    import sys
    from pathlib import Path

    # Add OTT-Hessian path
    ott_hessian_path = str(Path(__file__).parent.parent.parent.parent / "3rd-party" / "OTT-Hessian")
    if ott_hessian_path not in sys.path:
        sys.path.insert(0, ott_hessian_path)

    try:
        from torch_sinkhorn_hessian import TorchSinkhornHessian, TorchOTResult, _TorchGeometry
    except ImportError:
        pytest.skip("torch_sinkhorn_hessian not available (requires 3rd-party/OTT-Hessian)")
    from flash_sinkhorn.kernels.sinkhorn_flashstyle_sqeuclid import (
        sinkhorn_flashstyle_symmetric,
    )
    from flash_sinkhorn.hvp import hvp_x_sqeuclid_from_potentials, geomloss_to_ott_potentials

    # Setup
    n, d, eps = 256, 16, 1.0
    device = torch.device("cuda")

    torch.manual_seed(42)
    x = torch.randn(n, d, device=device, dtype=torch.float32)
    y = torch.randn(n, d, device=device, dtype=torch.float32)
    a = torch.rand(n, device=device, dtype=torch.float32) + 0.1
    b = torch.rand(n, device=device, dtype=torch.float32) + 0.1
    a = a / a.sum()
    b = b / b.sum()
    A = torch.randn(n, d, device=device, dtype=torch.float32)

    # Get shared potentials
    f_geo, g_geo = sinkhorn_flashstyle_symmetric(
        x, y, a, b,
        use_epsilon_scaling=False,
        eps=eps,
        n_iters=100,
        last_extrapolation=True,
        autotune=False,
    )
    f_ott, g_ott = geomloss_to_ott_potentials(f_geo, g_geo, a, b, eps=eps)

    # FlashSinkhorn HVP
    hvp_flash, _ = hvp_x_sqeuclid_from_potentials(
        x, y, f_ott, g_ott, A,
        eps=eps,
        tau2=1e-5,
        max_cg_iter=50,
        cg_rtol=1e-6,
        cg_atol=1e-6,
    )

    # OTT-Hessian (KeOps) HVP
    C = torch.cdist(x, y, p=2).pow(2)
    P = torch.exp((f_ott[:, None] + g_ott[None, :] - C) / eps)

    geom = _TorchGeometry(x=x, y=y, epsilon=eps)
    ot = TorchOTResult(
        geom=geom, a=a, b=b, matrix=P,
        reg_ot_cost=torch.tensor(0.0, device=device),
        threshold=1e-4, iterations=100,
        f=f_ott, g=g_ott,
    )

    solver = TorchSinkhornHessian(
        svd_thr=1e-10,
        solver="native",
        dtype=torch.float32,
        use_keops=True,
        device=device,
    )

    hvp_keops, _ = solver.hessian_vector_product(
        ot, A,
        tau2=1e-5,
        max_cg_iter=50,
        cg_rtol=1e-6,
        cg_atol=1e-6,
        use_preconditioner=False,
        return_info=True,
    )

    # Compare
    cos_sim = torch.nn.functional.cosine_similarity(
        hvp_flash.flatten(), hvp_keops.flatten(), dim=0
    ).item()

    rel_error = (hvp_flash - hvp_keops).norm() / hvp_flash.norm()

    print(f"\nFlashSinkhorn vs OTT-Hessian (KeOps):")
    print(f"  Cosine similarity: {cos_sim:.6f}")
    print(f"  Relative error: {rel_error:.2e}")

    assert cos_sim > 0.999, f"HVP results not matching: cos_sim={cos_sim}"
    assert rel_error < 0.01, f"HVP relative error too high: {rel_error}"


def test_hvp_parity_flashsinkhorn_vs_ott_hessian_jax():
    """Test parity between FlashSinkhorn and OTT-Hessian (JAX) HVP."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    import sys
    from pathlib import Path

    # Add OTT-Hessian path
    ott_hessian_path = str(Path(__file__).parent.parent.parent.parent / "3rd-party" / "OTT-Hessian")
    if ott_hessian_path not in sys.path:
        sys.path.insert(0, ott_hessian_path)

    try:
        from SinkhornHessian import HessianALineax as HessianA  # Use lineax version
    except ImportError:
        pytest.skip("SinkhornHessian not available (requires 3rd-party/OTT-Hessian)")
    from ott.geometry import pointcloud
    from ott.problems.linear import linear_problem
    from ott.solvers.linear import sinkhorn

    from flash_sinkhorn.kernels.sinkhorn_flashstyle_sqeuclid import (
        sinkhorn_flashstyle_symmetric,
    )
    from flash_sinkhorn.hvp import hvp_x_sqeuclid_from_potentials, geomloss_to_ott_potentials

    # Setup
    n, d, eps = 256, 16, 1.0
    device = torch.device("cuda")

    torch.manual_seed(42)
    np.random.seed(42)

    x_torch = torch.randn(n, d, device=device, dtype=torch.float32)
    y_torch = torch.randn(n, d, device=device, dtype=torch.float32)
    a_torch = torch.rand(n, device=device, dtype=torch.float32) + 0.1
    b_torch = torch.rand(n, device=device, dtype=torch.float32) + 0.1
    a_torch = a_torch / a_torch.sum()
    b_torch = b_torch / b_torch.sum()
    A_torch = torch.randn(n, d, device=device, dtype=torch.float32)

    # Convert to JAX
    x_jax = jnp.array(x_torch.cpu().numpy())
    y_jax = jnp.array(y_torch.cpu().numpy())
    a_jax = jnp.array(a_torch.cpu().numpy())
    b_jax = jnp.array(b_torch.cpu().numpy())
    A_jax = jnp.array(A_torch.cpu().numpy())

    # FlashSinkhorn: Get potentials and compute HVP
    f_geo, g_geo = sinkhorn_flashstyle_symmetric(
        x_torch, y_torch, a_torch, b_torch,
        use_epsilon_scaling=False,
        eps=eps,
        n_iters=100,
        last_extrapolation=True,
        autotune=False,
    )
    f_ott, g_ott = geomloss_to_ott_potentials(f_geo, g_geo, a_torch, b_torch, eps=eps)

    hvp_flash, _ = hvp_x_sqeuclid_from_potentials(
        x_torch, y_torch, f_ott, g_ott, A_torch,
        eps=eps,
        tau2=1e-5,
        max_cg_iter=50,
        cg_rtol=1e-6,
        cg_atol=1e-6,
    )

    # OTT-Hessian JAX: Get potentials and compute HVP
    geom = pointcloud.PointCloud(x_jax, y_jax, epsilon=eps)
    prob = linear_problem.LinearProblem(geom, a=a_jax, b=b_jax)
    solver = sinkhorn.Sinkhorn(
        max_iterations=100,
        threshold=1e-6,
        use_danskin=False,
    )
    ot_result = solver(prob)

    hvp_jax = HessianA(A_jax, ot_result, tau2=1e-5, iter=50)
    hvp_jax_torch = torch.tensor(np.array(hvp_jax), device=device)

    # Compare
    cos_sim = torch.nn.functional.cosine_similarity(
        hvp_flash.flatten(), hvp_jax_torch.flatten(), dim=0
    ).item()

    rel_error = (hvp_flash - hvp_jax_torch).norm() / hvp_flash.norm()

    print(f"\nFlashSinkhorn vs OTT-Hessian (JAX):")
    print(f"  Cosine similarity: {cos_sim:.6f}")
    print(f"  Relative error: {rel_error:.2e}")

    # Note: JAX uses its own Sinkhorn solver, so potentials may differ slightly
    # We expect high but not perfect agreement
    assert cos_sim > 0.99, f"HVP results not matching: cos_sim={cos_sim}"
    assert rel_error < 0.05, f"HVP relative error too high: {rel_error}"


def test_hvp_parity_all_three_methods():
    """Test parity between all three HVP implementations.

    This test uses SHARED potentials computed by FlashSinkhorn for all methods,
    ensuring a fair comparison of the Schur complement + CG implementation only.
    """
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("pykeops")

    import sys
    from pathlib import Path

    # Add OTT-Hessian path
    ott_hessian_path = str(Path(__file__).parent.parent.parent.parent / "3rd-party" / "OTT-Hessian")
    if ott_hessian_path not in sys.path:
        sys.path.insert(0, ott_hessian_path)

    try:
        from torch_sinkhorn_hessian import TorchSinkhornHessian, TorchOTResult, _TorchGeometry
    except ImportError:
        pytest.skip("torch_sinkhorn_hessian not available (requires 3rd-party/OTT-Hessian)")
    try:
        from SinkhornHessian import HessianALineax as HessianA  # Use lineax version
    except ImportError:
        pytest.skip("SinkhornHessian not available (requires 3rd-party/OTT-Hessian)")
    from ott.geometry import pointcloud
    from ott.problems.linear import linear_problem
    from ott.solvers.linear import sinkhorn as ott_sinkhorn

    from flash_sinkhorn.kernels.sinkhorn_flashstyle_sqeuclid import (
        sinkhorn_flashstyle_symmetric,
    )
    from flash_sinkhorn.hvp import hvp_x_sqeuclid_from_potentials, geomloss_to_ott_potentials

    # Setup
    n, d, eps = 256, 16, 1.0
    device = torch.device("cuda")
    tau2 = 1e-5
    cg_iters = 50

    torch.manual_seed(42)
    np.random.seed(42)

    x = torch.randn(n, d, device=device, dtype=torch.float32)
    y = torch.randn(n, d, device=device, dtype=torch.float32)
    a = torch.rand(n, device=device, dtype=torch.float32) + 0.1
    b = torch.rand(n, device=device, dtype=torch.float32) + 0.1
    a = a / a.sum()
    b = b / b.sum()
    A = torch.randn(n, d, device=device, dtype=torch.float32)

    # Get SHARED potentials from FlashSinkhorn
    f_geo, g_geo = sinkhorn_flashstyle_symmetric(
        x, y, a, b,
        use_epsilon_scaling=False,
        eps=eps,
        n_iters=100,
        last_extrapolation=True,
        autotune=False,
    )
    f_ott, g_ott = geomloss_to_ott_potentials(f_geo, g_geo, a, b, eps=eps)

    # ========== FlashSinkhorn HVP ==========
    hvp_flash, _ = hvp_x_sqeuclid_from_potentials(
        x, y, f_ott, g_ott, A,
        eps=eps,
        tau2=tau2,
        max_cg_iter=cg_iters,
        cg_rtol=1e-6,
        cg_atol=1e-6,
    )

    # ========== OTT-Hessian (KeOps) HVP ==========
    C = torch.cdist(x, y, p=2).pow(2)
    P = torch.exp((f_ott[:, None] + g_ott[None, :] - C) / eps)

    geom_torch = _TorchGeometry(x=x, y=y, epsilon=eps)
    ot_torch = TorchOTResult(
        geom=geom_torch, a=a, b=b, matrix=P,
        reg_ot_cost=torch.tensor(0.0, device=device),
        threshold=1e-4, iterations=100,
        f=f_ott, g=g_ott,
    )

    solver_keops = TorchSinkhornHessian(
        svd_thr=1e-10,
        solver="native",
        dtype=torch.float32,
        use_keops=True,
        device=device,
    )

    hvp_keops, _ = solver_keops.hessian_vector_product(
        ot_torch, A,
        tau2=tau2,
        max_cg_iter=cg_iters,
        cg_rtol=1e-6,
        cg_atol=1e-6,
        use_preconditioner=False,
        return_info=True,
    )

    # ========== OTT-Hessian (JAX) HVP ==========
    # Convert to JAX and create OT result with SAME potentials
    x_jax = jnp.array(x.cpu().numpy())
    y_jax = jnp.array(y.cpu().numpy())
    a_jax = jnp.array(a.cpu().numpy())
    b_jax = jnp.array(b.cpu().numpy())
    A_jax = jnp.array(A.cpu().numpy())
    f_jax = jnp.array(f_ott.cpu().numpy())
    g_jax = jnp.array(g_ott.cpu().numpy())

    # Create OTT geometry and manually inject potentials
    geom_jax = pointcloud.PointCloud(x_jax, y_jax, epsilon=eps)
    prob = linear_problem.LinearProblem(geom_jax, a=a_jax, b=b_jax)
    solver_jax = ott_sinkhorn.Sinkhorn(max_iterations=1, threshold=1.0, use_danskin=False)
    ot_jax = solver_jax(prob)
    # Override with shared potentials using _replace on the potentials tuple
    ot_jax = ot_jax._replace(potentials=(f_jax, g_jax))

    hvp_jax = HessianA(A_jax, ot_jax, tau2=tau2, iter=cg_iters)
    hvp_jax_torch = torch.tensor(np.array(hvp_jax), device=device)

    # ========== Compare all pairs ==========
    print("\n" + "="*60)
    print("HVP PARITY TEST: All Three Methods")
    print("="*60)

    methods = {
        "FlashSinkhorn": hvp_flash,
        "OTT-Hessian (KeOps)": hvp_keops,
        "OTT-Hessian (JAX)": hvp_jax_torch,
    }

    results = {}
    for name1, hvp1 in methods.items():
        for name2, hvp2 in methods.items():
            if name1 >= name2:
                continue

            cos_sim = torch.nn.functional.cosine_similarity(
                hvp1.flatten(), hvp2.flatten(), dim=0
            ).item()

            rel_error = (hvp1 - hvp2).norm() / hvp1.norm()

            print(f"\n{name1} vs {name2}:")
            print(f"  Cosine similarity: {cos_sim:.6f}")
            print(f"  Relative error: {rel_error:.2e}")

            results[(name1, name2)] = (cos_sim, rel_error.item())

    print("\n" + "="*60)

    # All pairs should have high agreement
    for (name1, name2), (cos_sim, rel_error) in results.items():
        assert cos_sim > 0.999, f"{name1} vs {name2}: cos_sim={cos_sim} too low"
        assert rel_error < 0.01, f"{name1} vs {name2}: rel_error={rel_error} too high"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
