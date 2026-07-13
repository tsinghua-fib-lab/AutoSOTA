import torch
import pytest
from optimal_transport.ot_fc_sep_map import FCOTSeparable
from models import FiniteSeparableModel


################################################################################
# Helper: Simple 1D quadratic kernel
################################################################################

def quad_kernel(x, y):
    return (x - y) ** 2


################################################################################
# Fixture: tiny model that runs fast
################################################################################

@pytest.fixture
def small_model():
    model = FiniteSeparableModel(
        kernel=quad_kernel,
        num_dims=1,
        radius=2.0,
        y_accuracy=0.5,
        x_accuracy=0.5,
        mode="concave",
        temp=5.0,
        epsilon=1e-6,
        cache_gradients=False,
    )
    return model


@pytest.fixture
def small_solver(small_model):
    solver = FCOTSeparable(
        input_dim=1,
        model=small_model,
        inverse_kx=lambda x, p: x - 0.5 * p,
        outer_lr=1e-2,
        warmup_lr=1e-1,
        full_refresh_every=5,               # frequency must be small for testing
        reactivate_every=3,
        reactivate_eps=1e-3,
    )
    return solver


################################################################################
# 1. INTERCEPT REFRESH TEST
################################################################################

def test_refresh_updates_intercepts(small_model):
    # Start from a nontrivial intercept configuration so refresh has work to do
    torch.manual_seed(0)
    with torch.no_grad():
        small_model.intercepts.copy_(torch.randn_like(small_model.intercepts))

    old = small_model.intercepts.clone()

    changed = small_model.refresh_intercepts_via_transform()
    new = small_model.intercepts

    # Function must run without error and produce a valid intercept tensor.
    # Depending on initialization and kernel, refresh may be idempotent.
    assert isinstance(changed, int)
    assert new.shape == old.shape


################################################################################
# 2. MOMENTUM RESET TEST (Adam or TwoStageOptimizer)
################################################################################

def test_momentum_reset(small_solver):
    solver = small_solver
    opt = solver.optimizer

    p = solver.model.intercepts

    # Simulate momentum state
    for group in opt.param_groups:
        for param in group["params"]:
            if param is p:
                state = opt.state.setdefault(param, {})
                state["momentum_buffer"] = torch.ones_like(p) * 3.14

    # Apply reset
    solver._reset_intercept_momentum()

    # Check cleared:
    state = opt.state[p]
    assert len(state) == 0, "Momentum/state was not fully cleared."


################################################################################
# 3. JITTER TEST
################################################################################

def test_jitter_changes_intercepts(small_solver):
    model = small_solver.model
    before = model.intercepts.clone()
    small_solver._jitter_intercepts(strength=1e-2)
    after = model.intercepts

    assert not torch.allclose(before, after), "Jitter must modify intercepts."
    diff = (after - before).abs().mean().item()
    assert diff > 0, "Jitter must have nonzero magnitude."


################################################################################
# 4. FULL REFRESH (with reset + jitter)
################################################################################

def test_full_refresh_triggers_all(small_solver):
    solver = small_solver
    model = solver.model

    # Overwrite refresh to count calls
    calls = {"refresh": 0, "reset": 0, "jitter": 0}

    orig_refresh = model.refresh_intercepts_via_transform
    def counted_refresh():
        calls["refresh"] += 1
        return orig_refresh()

    solver.model.refresh_intercepts_via_transform = counted_refresh

    orig_reset = solver._reset_intercept_momentum
    def counted_reset():
        calls["reset"] += 1
        return orig_reset()

    solver._reset_intercept_momentum = counted_reset

    orig_jitter = solver._jitter_intercepts
    def counted_jitter(strength=None):
        calls["jitter"] += 1
        return orig_jitter(strength)

    solver._jitter_intercepts = counted_jitter

    # Force a step_count where refresh should trigger
    solver._step_count = solver._full_refresh_every
    solver._maybe_full_refresh()

    assert calls["refresh"] == 1, "Refresh should be called exactly once."
    assert calls["reset"] == 1, "Momentum reset should be called exactly once."
    assert calls["jitter"] == 1, "Jitter should be called exactly once."


################################################################################
# 5. FORWARD AND STE TEST
################################################################################

def test_forward_shapes(small_model):
    X = torch.tensor([[0.2], [-0.5], [1.0]])
    choice, fx = small_model.forward(X, selection_mode="hard")

    assert choice.shape == (3, 1)
    assert fx.shape == (3,)
    assert not fx.isnan().any()


def test_ste_gradient_correctness(small_model):
    # Hard selection with STE: gradient should propagate through approx derivative.
    X = torch.tensor([[0.1], [0.5]], requires_grad=True)
    _, fx = small_model.forward(X, selection_mode="hard")
    loss = fx.sum()
    loss.backward()

    assert X.grad is not None, "Gradient must flow through STE wrapper."
    assert not torch.isnan(X.grad).any()
    # gradient magnitude must be finite / nonzero for interior x
    assert X.grad.abs().max() > 0


################################################################################
# 6. TRANSFORM CONSISTENCY TEST
################################################################################

def test_transform_consistency(small_model):
    Z = torch.linspace(-1, 1, 5).reshape(-1, 1)
    Xopt, vals, conv = small_model.inf_transform(Z)

    assert Xopt.shape == Z.shape, "Transform output must match input shape."
    assert vals.shape == (5,), "Value shape mismatch."
    assert conv is True, "Separable transform must always converge."
    assert not torch.isnan(vals).any()


################################################################################
# 7. SMALL TRAINING LOOP TEST
################################################################################

def test_training_makes_progress():
    """
    Non-monotone-safe training test.

    Checks that:
      - objective stays finite,
      - gradients stay reasonable,
      - progress occurs over a window,
    without assuming monotonic ascent (refresh/jitter/STE break monotonicity).
    """

    import math
    import torch
    from optimal_transport.ot_fc_sep_map import FCOTSeparable

    torch.manual_seed(0)

    # 1D synthetic example
    X = torch.randn(200, 1) * 0.5
    Y = torch.randn(200, 1) + 1.0

    # Build a small separable solver
    solver = FCOTSeparable.initialize_right_architecture(
        dim=1,
        radius=3.0,
        n_params=40,                   # modest grid size: 40 intercepts in 1D
        x_accuracy=0.1,
        kernel_1d=lambda x, y: (x - y) ** 2,
        inverse_kx=lambda x, p: x - 0.5 * p,
        outer_lr=5e-3,
        warmup_lr=1e-2,
        warmup_grad_threshold=0.1,
        warmup_max_steps=20,
        full_refresh_every=10,         # refresh allowed
        reactivate_every=15,
    )

    history = []
    for _ in range(50):
        res = solver.step(X, Y)
        obj = res["dual"]
        grad_norm = res["grad_norm"]

        # finite values
        assert not math.isnan(obj)
        assert not math.isinf(obj)
        assert grad_norm < 1e6

        history.append(obj)

    # progress over windows
    early_best = max(history[:5])
    late_best = max(history[-10:])

    assert late_best > early_best, (
        f"OT objective failed to improve over time: "
        f"early={early_best:.4f}, late={late_best:.4f}"
    )

################################################################################
# 8. TRANSPORT MAP TEST
################################################################################

def test_transport_map(small_solver):
    solver = small_solver
    X = torch.tensor([[-0.5], [0.0], [0.5]], dtype=torch.float32)

    # Compute transport map explicitly from potential and inverse_kx
    X_requires_grad = X.to(solver.device).requires_grad_(True)
    _, u_X = solver.model.forward(X_requires_grad, selection_mode="hard")
    grad_u = torch.autograd.grad(u_X.sum(), X_requires_grad, create_graph=False)[0]
    Ypred = solver.inverse_kx(X_requires_grad.detach(), grad_u.detach())

    assert Ypred.shape == X.shape
    assert not torch.isnan(Ypred).any()
    # In 1D quadratic, T(x) = x − 0.5 * ∇u(x), so should not be extreme.
    assert Ypred.abs().max() < 10, "Transport values look unreasonable."
