import torch

from models import FiniteSeparableModel
from tools.utils import moded_max, moded_min


def quadratic_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """1-D kernel used in the tests: -(x - y)^2 per entry."""
    return -(x - y) ** 2


def make_lp_kernel(power: int):
    """Return a separable kernel of the form -||x - y||_p^p in 1-D."""
    def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -(torch.abs(x - y) ** power)
    return kernel


def build_model(
    num_dims: int,
    radius: float = 1.0,
    accuracy: float = 1.0,
    mode: str = "convex",
    kernel_fn=None,
    cache_gradients: bool = False,
    **model_kwargs,
) -> FiniteSeparableModel:
    """Utility that creates a deterministic FiniteSeparableModel."""
    if kernel_fn is None:
        kernel_fn = quadratic_kernel
    model = FiniteSeparableModel(
        kernel=kernel_fn,
        num_dims=num_dims,
        radius=radius,
        y_accuracy=accuracy,
        x_accuracy=accuracy,
        mode=mode,
        temp=1.0,
        cache_gradients=cache_gradients,
        **model_kwargs,
    )
    with torch.no_grad():
        # Zero out underlying intercept parameters (respecting any gauge parametrization).
        intercepts_param = getattr(model, "intercepts_param", None)
        if intercepts_param is not None and hasattr(intercepts_param, "theta"):
            intercepts_param.theta.zero_()
        else:
            model.intercepts.zero_()
    return model


def finite_difference_grad(
    model: FiniteSeparableModel,
    base_points: torch.Tensor,
    selection_mode: str,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Finite-difference approximation of ∂/∂x sum_j f(x_j)."""
    fd_grad = torch.zeros_like(base_points)
    base = base_points.clone()
    flat_grad = fd_grad.view(-1)
    num_samples = base.shape[0]

    def _forward_value(points: torch.Tensor) -> torch.Tensor:
        return separable_forward_value(model, points, selection_mode)

    for idx in range(base.numel()):
        x_plus = base.clone()
        x_minus = base.clone()
        x_plus.view(-1)[idx] += eps
        x_minus.view(-1)[idx] -= eps
        x_plus = model.project(x_plus)
        x_minus = model.project(x_minus)
        f_plus = _forward_value(x_plus)
        f_minus = _forward_value(x_minus)
        flat_grad[idx] = (f_plus.sum() - f_minus.sum()) / (2 * eps)
    return fd_grad


def separable_forward_value(
    model: FiniteSeparableModel,
    X: torch.Tensor,
    selection_mode: str,
) -> torch.Tensor:
    """Compute f(X) using continuous kernel evaluations (no snapping)."""
    num_samples, num_dims = X.shape
    values = torch.zeros(num_samples, dtype=X.dtype, device=X.device)
    Y_candidates = model.Y_grid.view(1, -1, 1)

    for dim in range(num_dims):
        x_vals = X[:, dim]
        kernel_scores = model.kernel_fn(
            x_vals.unsqueeze(-1),
            model.Y_grid.unsqueeze(0),
        )
        b = model.intercepts[:, dim].unsqueeze(0)
        scores = kernel_scores - b

        if model.mode == "convex":
            if selection_mode == "hard":
                val, _ = scores.max(dim=1)
            else:
                _, val, _ = moded_max(
                    scores, Y_candidates, dim=1, temp=model.temp, mode=selection_mode
                )
        else:
            if selection_mode == "hard":
                val, _ = scores.min(dim=1)
            else:
                _, val, _ = moded_min(
                    scores, Y_candidates, dim=1, temp=model.temp, mode=selection_mode
                )

        values = values + val

    return values


def snap_to_grid(values: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Snap values to nearest grid point using the same rounding rule as the model."""
    step = (grid[1] - grid[0]).item()
    radius = grid[-1].item()
    idx = ((values + radius) / step).round()
    idx = idx.clamp(0, grid.numel() - 1).long()
    return grid.to(values)[idx]


def continuous_choice(model: FiniteSeparableModel, X: torch.Tensor) -> torch.Tensor:
    """Returns choices by evaluating the kernel continuously (no x-grid snapping)."""
    num_samples, num_dims = X.shape
    choice = torch.empty(num_samples, num_dims, dtype=X.dtype, device=X.device)

    for dim in range(num_dims):
        x_vals = X[:, dim]
        kernel_scores = model.kernel_fn(
            x_vals.unsqueeze(-1),
            model.Y_grid.unsqueeze(0),
        )
        b = model.intercepts[:, dim].unsqueeze(0)
        scores = kernel_scores - b

        if model.mode == "convex":
            idx = scores.argmax(dim=1)
        else:
            idx = scores.argmin(dim=1)

        choice[:, dim] = model.Y_grid[idx]

    return choice


def select_expected_choice(values: torch.Tensor, grid: torch.Tensor, mode: str) -> torch.Tensor:
    """Select closest (convex) or farthest (concave) grid point, respecting x-grid snapping."""
    grid = grid.to(values)
    x_ref = snap_to_grid(values, grid)
    grid_expanded = grid.view(1, 1, -1).expand(values.shape[0], values.shape[1], -1)
    distances = torch.abs(x_ref.unsqueeze(-1) - grid_expanded)
    if mode == "concave":
        idx = distances.argmax(dim=-1, keepdim=True)
    else:
        idx = distances.argmin(dim=-1, keepdim=True)
    return torch.take_along_dim(grid_expanded, idx, dim=-1).squeeze(-1)


def run_forward_and_gradient_check(
    num_dims: int,
    X: torch.Tensor,
    mode: str = "convex",
    kernel_power: int = 2,
    radius: float = 1.0,
    accuracy: float = 1.0,
    cache_gradients: bool = False,
):
    """Validate forward choices/values and ∂f/∂x against analytic and FD references."""
    print(f"\n[Dimension {num_dims}] Running forward and gradient checks (mode={mode}, p={kernel_power})...")
    kernel_fn = make_lp_kernel(kernel_power)
    model = build_model(
        num_dims,
        radius=radius,
        accuracy=accuracy,
        mode=mode,
        kernel_fn=kernel_fn,
        cache_gradients=cache_gradients,
    )

    # Prepare input tensor and run forward pass
    X = X.clone().requires_grad_(True)
    choice, f_x = model.forward(X, selection_mode="hard")

    # Expected choices and values from analytic solution
    expected_choice = select_expected_choice(X.detach(), model.Y_grid, mode=mode)
    x_snap = snap_to_grid(X.detach(), model.X_grid)
    expected_values = -(torch.abs(x_snap - expected_choice) ** kernel_power).sum(dim=1)

    # Forward assertions
    if not torch.allclose(choice.detach(), expected_choice, atol=1e-6):
        raise AssertionError(f"Choice mismatch:\nExpected: {expected_choice}\nGot: {choice}")

    if not torch.allclose(f_x.detach(), expected_values, atol=1e-6):
        raise AssertionError(f"Value mismatch:\nExpected: {expected_values}\nGot: {f_x}")

    print(f"  Forward check passed. Choices:\n{choice}\n  Values:\n{f_x}")

    # Gradient check: df/dx = -p * (x - y*) * |x - y*|^{p-2}
    loss = f_x.sum()
    loss.backward()
    diff = X.detach() - expected_choice
    if kernel_power == 2:
        pow_term = torch.ones_like(diff)
    else:
        pow_term = torch.abs(diff) ** (kernel_power - 2)
    expected_grad = -kernel_power * diff * pow_term

    if not torch.allclose(X.grad, expected_grad, atol=1e-6):
        raise AssertionError(f"Gradient mismatch:\nExpected: {expected_grad}\nGot: {X.grad}")

    print(f"  Gradient check passed. X.grad:\n{X.grad}")
    if mode == "convex":
        fd_hard = finite_difference_grad(model, X.detach(), selection_mode="hard")
        hard_err = (X.grad - fd_hard).abs().max().item()
        print(f"  Hard-mode FD max error: {hard_err:.2e}")
        if hard_err > 5e-3:
            raise AssertionError(f"Hard-mode finite-difference error too large: {hard_err}")
    else:
        print("  Skipping hard-mode FD check for concave mode (non-smooth argmin).")

    # Soft selection check: should interpolate between neighbors smoothly
    X_soft = X.detach().clone().requires_grad_(True)
    _, f_x_soft = model.forward(X_soft, selection_mode="soft")
    loss_soft = f_x_soft.sum()
    loss_soft.backward()
    if X_soft.grad is None or not torch.all(torch.isfinite(X_soft.grad)):
        raise AssertionError("Soft-mode gradients contain NaNs or infs")
    print(f"  Soft-mode gradients finite. X_soft.grad:\n{X_soft.grad}")
    fd_soft = finite_difference_grad(model, X_soft.detach(), selection_mode="soft")
    soft_err = (X_soft.grad - fd_soft).abs().max().item()
    print(f"  Soft-mode FD max error: {soft_err:.2e}")
    soft_tol = 5e-3 if mode == "convex" else 2e-2
    if soft_err > soft_tol:
        raise AssertionError(f"Soft-mode finite-difference error too large: {soft_err}")

def compute_expected_intercept_grad(model: FiniteSeparableModel, choice: torch.Tensor) -> torch.Tensor:
    expected = torch.zeros_like(model.intercepts)
    for sample in range(choice.shape[0]):
        for dim in range(choice.shape[1]):
            idx = int(model.get_indices_for_y(choice[sample, dim]).item())
            expected[idx, dim] -= 1.0
    return expected


def run_fine_grid_forward_param_gradients(mode: str = "convex", cache_gradients: bool = False):
    """Check parameter gradients on a dense grid against brute-force counts."""
    print(f"\n[Fine Grid] Forward values and parameter gradients (mode={mode})...")
    radius = 0.5
    accuracy = 1e-3
    kernel_fn = make_lp_kernel(2)
    model = build_model(
        num_dims=2,
        radius=radius,
        accuracy=accuracy,
        mode=mode,
        kernel_fn=kernel_fn,
        cache_gradients=cache_gradients,
    )
    X = torch.tensor(
        [
            [0.100, -0.200],
            [-0.300, 0.400],
        ],
        dtype=torch.float32,
    )

    choice, f_x = model.forward(X, selection_mode="hard")
    expected_choice = select_expected_choice(X, model.Y_grid, mode=mode)
    if not torch.allclose(choice.detach(), expected_choice, atol=1e-6):
        raise AssertionError(f"Choice mismatch on fine grid:\nExpected: {expected_choice}\nGot: {choice}")

    model.zero_grad()
    loss = f_x.sum()
    loss.backward()
    expected_grad = compute_expected_intercept_grad(model, choice.detach())

    theta_grad = getattr(model.intercepts_param, "theta", None)
    if theta_grad is None or theta_grad.grad is None:
        raise AssertionError("No gradients collected for intercept parameters.")
    # Parameter gradient corresponds to rows 1: of full intercept gradient (row 0 is fixed gauge).
    if not torch.allclose(theta_grad.grad, expected_grad[1:, :], atol=1e-6):
        raise AssertionError(
            f"Intercept gradient mismatch:\nExpected (rows 1:): {expected_grad[1:, :]}\nGot: {theta_grad.grad}"
        )
    print("  Parameter gradient check passed on fine grid.")


def evaluate_transform_reference(model: FiniteSeparableModel, Z: torch.Tensor, maximize: bool):
    """Compute transform via brute-force grid search for verification."""
    num_samples, num_dims = Z.shape
    X_opts = []
    values = []
    for i in range(num_samples):
        z_i = Z[i]
        x_opt_dims = []
        value_sum = 0.0
        for d in range(num_dims):
            z_val = model.project(z_i[d])
            z_idx = int(model.get_indices_for_y(z_val).item())
            kernel_vals = model.kernel_tensor[:, z_idx]
            scores = model.kernel_tensor - model.intercepts[:, d].unsqueeze(0)
            if model.mode == "convex":
                f_vals = scores.max(dim=1).values
            else:
                f_vals = scores.min(dim=1).values
            objective = kernel_vals - f_vals
            if maximize:
                idx = objective.argmax()
            else:
                idx = objective.argmin()
            x_opt_dims.append(model.X_grid[idx])
            value_sum = value_sum + objective[idx]
        X_opts.append(torch.stack(x_opt_dims))
        values.append(value_sum)
    return torch.stack(X_opts), torch.stack(values)


def run_sup_transform_gradient_relation(mode: str = "convex", kernel_power: int = 2, cache_gradients: bool = False):
    """Verify ∇_b sup_transform = -∇_b f(X*) for sup over the grid."""
    print(f"\n[Sup Transform] Checking value/gradient relation (mode={mode}, p={kernel_power})...")
    radius = 0.5
    accuracy = 1e-3
    kernel_fn = make_lp_kernel(kernel_power)
    model = build_model(
        num_dims=2,
        radius=radius,
        accuracy=accuracy,
        mode=mode,
        kernel_fn=kernel_fn,
        cache_gradients=cache_gradients,
    )
    Z = torch.tensor([[0.123, -0.222], [-0.111, 0.333]], dtype=torch.float32)

    X_opt, values, _ = model.sup_transform(Z)
    ref_x, ref_values = evaluate_transform_reference(model, Z, maximize=True)
    if not torch.allclose(X_opt, ref_x, atol=1e-6):
        raise AssertionError(f"Sup transform optimizer mismatch:\nExpected: {ref_x}\nGot: {X_opt}")
    if not torch.allclose(values.detach(), ref_values, atol=1e-6):
        raise AssertionError(f"Sup transform values mismatch:\nExpected: {ref_values}\nGot: {values}")

    model.zero_grad()
    values.sum().backward()
    theta = getattr(model.intercepts_param, "theta", None)
    if theta is None or theta.grad is None:
        raise AssertionError("No gradients collected for intercept parameters (sup transform).")
    grad_sup = theta.grad.detach().clone()

    model.zero_grad()
    _, f_vals = model.forward(X_opt, selection_mode="hard")
    f_vals.sum().backward()
    if theta.grad is None:
        raise AssertionError("No gradients collected for intercept parameters (forward at X*).")
    grad_forward = theta.grad.detach().clone()

    relation_err = (grad_sup + grad_forward).abs().max().item()
    print(f"  ∥∇sup + ∇f(X*)∥_∞ = {relation_err:.2e}")
    if relation_err > 1e-4:
        raise AssertionError("Sup-transform gradient does not match -∇f(X*) within tolerance.")


def run_inf_transform_value_and_gradients(mode: str = "convex", kernel_power: int = 2, cache_gradients: bool = False):
    """Verify inf_transform optimizer/value and gradient relation mirrors forward at X*."""
    print(f"\n[Inf Transform] Checking value/gradient relation (mode={mode}, p={kernel_power})...")
    radius = 0.5
    accuracy = 1e-3
    kernel_fn = make_lp_kernel(kernel_power)
    model = build_model(
        num_dims=2,
        radius=radius,
        accuracy=accuracy,
        mode=mode,
        kernel_fn=kernel_fn,
        cache_gradients=cache_gradients,
    )
    Z = torch.tensor([[0.200, -0.300], [-0.250, 0.150]], dtype=torch.float32)

    X_opt, values, _ = model.inf_transform(Z)
    ref_x, ref_values = evaluate_transform_reference(model, Z, maximize=False)
    if not torch.allclose(X_opt, ref_x, atol=1e-6):
        raise AssertionError(f"Inf transform optimizer mismatch:\nExpected: {ref_x}\nGot: {X_opt}")
    if not torch.allclose(values.detach(), ref_values, atol=1e-6):
        raise AssertionError(f"Inf transform values mismatch:\nExpected: {ref_values}\nGot: {values}")

    model.zero_grad()
    values.sum().backward()
    theta = getattr(model.intercepts_param, "theta", None)
    if theta is None or theta.grad is None:
        raise AssertionError("No gradients collected for intercept parameters (inf transform).")
    grad_inf = theta.grad.detach().clone()

    model.zero_grad()
    _, f_vals = model.forward(X_opt, selection_mode="hard")
    f_vals.sum().backward()
    if theta.grad is None:
        raise AssertionError("No gradients collected for intercept parameters (forward at X*).")
    grad_forward = theta.grad.detach().clone()

    relation_err = (grad_inf + grad_forward).abs().max().item()
    print(f"  ∥∇inf + ∇f(X*)∥_∞ = {relation_err:.2e}")
    if relation_err > 1e-4:
        raise AssertionError("Inf-transform gradient does not match -∇f(X*) within tolerance.")


def run_high_dim_pnorm_forward_checks():
    """Regression for higher-dimensional p-norm kernels (p=4)."""
    print("\n[High-D] Testing forward/gradients with p=4 kernel...")
    X = torch.tensor(
        [
            [0.40, -0.30, 0.10, -0.20, 0.05],
            [-0.60, 0.80, -0.40, 0.30, -0.10],
            [0.00, -0.20, 0.70, -0.80, 0.20],
        ],
        dtype=torch.float32,
    )
    run_forward_and_gradient_check(
        num_dims=5,
        X=X,
        mode="convex",
        kernel_power=4,
        radius=1.0,
        accuracy=0.1,
    )


def run_concave_forward_checks():
    """Exercise concave mode with p=4 kernel."""
    print("\n[Concave] Testing forward/gradients with p=4 kernel...")
    X = torch.tensor(
        [
            [-0.5, 0.2, -0.1],
            [0.3, -0.7, 0.4],
        ],
        dtype=torch.float32,
    )
    run_forward_and_gradient_check(
        num_dims=3,
        X=X,
        mode="concave",
        kernel_power=4,
        radius=1.0,
        accuracy=0.1,
    )


def run_hard_no_snap_choice_checks(mode: str = "convex"):
    """Ensure hard-mode without snapping uses continuous kernel scores."""
    print(f"\n[No Snap] Verifying hard-mode choices ({mode}) use continuous kernel evaluations...")
    radius = 1.0
    accuracy = 0.3
    model = build_model(
        num_dims=2,
        radius=radius,
        accuracy=accuracy,
        mode=mode,
    )

    # Points deliberately offset from the x-grid so snapping would change the maxima.
    X = torch.tensor(
        [
            [0.12, -0.27],
            [-0.43, 0.37],
            [0.38, -0.12],
        ],
        dtype=torch.float32,
    )

    choice_snap, _ = model.forward(X, selection_mode="hard", snap_to_grid=True)
    choice_no_snap, _ = model.forward(X, selection_mode="hard", snap_to_grid=False)
    continuous = continuous_choice(model, X)

    if not torch.allclose(choice_no_snap, continuous, atol=1e-6):
        raise AssertionError(
            f"No-snap choice mismatch:\nExpected: {continuous}\nGot: {choice_no_snap}"
        )

    if torch.allclose(choice_snap, continuous, atol=1e-6):
        raise AssertionError(
            "Snapped choice should differ from the continuous selection on off-grid inputs."
        )
    print("  Hard-mode no-snap choice matches continuous kernel evaluation.")


def run_cached_gradient_regression():
    """Regression for cached-derivative mode on forward/sup/inf/coarse search."""
    print("\n[Cached Gradients] Verifying cached derivative mode...")
    X = torch.tensor([[-0.45], [0.35]], dtype=torch.float32)
    run_forward_and_gradient_check(
        num_dims=1,
        X=X,
        mode="convex",
        kernel_power=4,
        radius=0.5,
        accuracy=0.01,
        cache_gradients=True,
    )
    run_sup_transform_gradient_relation(
        mode="convex",
        kernel_power=2,
        cache_gradients=True,
    )
    run_inf_transform_value_and_gradients(
        mode="convex",
        kernel_power=2,
        cache_gradients=True,
    )
    validate_coarse_search_random_configs()


def verify_coarse_transform_matches_exact():
    """Coarse-to-fine search should match baseline forward/transform outputs."""
    print("\n[Coarse Search] Verifying coarse-to-fine transform matches baseline...")
    torch.manual_seed(1)
    radius = 1.0
    accuracy = 0.25
    base = build_model(num_dims=1, radius=radius, accuracy=accuracy, mode="convex")
    coarse = build_model(
        num_dims=1,
        radius=radius,
        accuracy=accuracy,
        mode="convex",
        coarse_x_factor=2,
        coarse_top_k=2,
        coarse_window=2,
    )
    with torch.no_grad():
        coarse.intercepts_param.project_from_b(base.intercepts.detach())

    X = torch.tensor([[-0.75], [0.1], [0.6]], dtype=torch.float32)
    base_choice, base_vals = base.forward(X, selection_mode="hard")
    coarse_choice, coarse_vals = coarse.forward(X, selection_mode="hard")
    if not torch.allclose(base_choice, coarse_choice):
        raise AssertionError("Coarse forward choices mismatch baseline.")
    if not torch.allclose(base_vals, coarse_vals):
        raise AssertionError("Coarse forward values mismatch baseline.")

    Z = torch.tensor([[0.2], [-0.4]], dtype=torch.float32)
    _, base_sup, _ = base.sup_transform(Z)
    _, coarse_sup, _ = coarse.sup_transform(Z)
    # Coarse search is an approximation: require close but not exact equality.
    if not torch.allclose(base_sup, coarse_sup, atol=2e-1, rtol=1e-3):
        raise AssertionError("Coarse sup_transform mismatch baseline.")

    _, base_inf, _ = base.inf_transform(Z)
    _, coarse_inf, _ = coarse.inf_transform(Z)
    if not torch.allclose(base_inf, coarse_inf, atol=2e-1, rtol=1e-3):
        raise AssertionError("Coarse inf_transform mismatch baseline.")


def validate_coarse_search_random_configs():
    """Stress coarse search across random configs against full-grid baselines."""
    print("\n[Coarse Search] Stress-testing multiple configs...")
    torch.manual_seed(123)
    configs = [
        {"dim": 1, "mode": "convex", "accuracy": 0.2, "factor": 2, "top_k": 1, "window": 1},
        {"dim": 2, "mode": "convex", "accuracy": 0.15, "factor": 3, "top_k": 2, "window": 2},
        {"dim": 2, "mode": "concave", "accuracy": 0.2, "factor": 2, "top_k": 2, "window": 1},
    ]
    for cfg in configs:
        base = build_model(
            num_dims=cfg["dim"],
            radius=1.0,
            accuracy=cfg["accuracy"],
            mode=cfg["mode"],
        )
        coarse = build_model(
            num_dims=cfg["dim"],
            radius=1.0,
            accuracy=cfg["accuracy"],
            mode=cfg["mode"],
            coarse_x_factor=cfg["factor"],
            coarse_top_k=cfg["top_k"],
            coarse_window=cfg["window"],
        )
        with torch.no_grad():
            coarse.intercepts_param.project_from_b(base.intercepts)

        X = (torch.rand(6, cfg["dim"]) * 2 - 1.0).to(torch.float32)
        base_choice, base_vals = base.forward(X, selection_mode="hard")
        coarse_choice, coarse_vals = coarse.forward(X, selection_mode="hard")
        if not torch.allclose(base_choice, coarse_choice):
            raise AssertionError(f"Coarse forward mismatch for cfg={cfg}")
        if not torch.allclose(base_vals, coarse_vals):
            raise AssertionError(f"Coarse forward values mismatch for cfg={cfg}")

        Z = (torch.rand(4, cfg["dim"]) * 2 - 1.0).to(torch.float32)
        _, base_sup, _ = base.sup_transform(Z)
        _, coarse_sup, _ = coarse.sup_transform(Z)
        if not torch.allclose(base_sup, coarse_sup, atol=2e-1, rtol=1e-3):
            raise AssertionError(f"Coarse sup mismatch for cfg={cfg}")

        _, base_inf, _ = base.inf_transform(Z)
        _, coarse_inf, _ = coarse.inf_transform(Z)
        if not torch.allclose(base_inf, coarse_inf, atol=2e-1, rtol=1e-3):
            raise AssertionError(f"Coarse inf mismatch for cfg={cfg}")

def run_coarse_transform_batch_equivalence():
    """Vectorized coarse transform path should equal scalar full search."""
    print("\n[Coarse Search] Batch inf_transform matches full search...")
    base = build_model(num_dims=1, radius=2.0, accuracy=0.25, mode="concave")
    coarse = build_model(
        num_dims=1,
        radius=2.0,
        accuracy=0.25,
        mode="concave",
        coarse_x_factor=2,
        coarse_top_k=2,
        coarse_window=10,
    )
    with torch.no_grad():
        coarse.intercepts_param.project_from_b(base.intercepts)

    Z = torch.linspace(-1.5, 1.5, steps=10).unsqueeze(1)
    _, base_vals, _ = base.inf_transform(Z)
    _, coarse_vals, _ = coarse.inf_transform(Z)
    if not torch.allclose(base_vals, coarse_vals, atol=1e-5, rtol=1e-5):
        raise AssertionError("Vectorized coarse transform does not match full search.")

def test_main():
    torch.manual_seed(0)

    # Dimension 1 test
    X_1d = torch.tensor([[-0.8], [0.2], [0.9]], dtype=torch.float32)
    run_forward_and_gradient_check(num_dims=1, X=X_1d)

    # Dimension 3 test
    X_3d = torch.tensor(
        [
            [-0.9, 0.1, 0.95],
            [0.6, -0.4, 0.03],
        ],
        dtype=torch.float32,
    )
    run_forward_and_gradient_check(num_dims=3, X=X_3d)

    run_high_dim_pnorm_forward_checks()
    run_concave_forward_checks()

    run_fine_grid_forward_param_gradients(mode="convex")
    run_fine_grid_forward_param_gradients(mode="concave")

    run_sup_transform_gradient_relation(mode="convex", kernel_power=2)
    run_sup_transform_gradient_relation(mode="concave", kernel_power=4)

    run_inf_transform_value_and_gradients(mode="convex", kernel_power=2)
    run_inf_transform_value_and_gradients(mode="concave", kernel_power=4)
    run_cached_gradient_regression()
    verify_coarse_transform_matches_exact()
    run_coarse_transform_batch_equivalence()
    run_hard_no_snap_choice_checks(mode="convex")
    run_hard_no_snap_choice_checks(mode="concave")

    print("\n✓ All separable model checks passed!")
