#!/usr/bin/env python3
"""Fully analytic JAX shallow-water loss.

Single unified loss for shallow-water system with tied parameters:
- 3 PDE residuals: continuity + x-momentum + y-momentum (interior points)
- 3 IC residuals: rho, rho*ux, rho*uy at t=0 (initial condition points)
- 12 BC residuals: rho, rho*ux, rho*uy at 4 boundaries (boundary points)
Total: 18 RMSE components summed into single scalar loss

All derivatives computed via JAX autodiff (no finite differences).
Optimization uses Adam (Optax) to drive loss → 0 at manufactured parameters.

Initial parameters from best structural optimization (fit_user_tied_20251117_183808.pkl)
with tied centers (cx, cy) and amplitudes across rho, Sx, Sy.
"""
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import optax
import numpy as np
from datetime import datetime
import pickle


# ========== Parametric solution functions ==========

def rho_fn(x, y, t, params):
    """Density field with tied parameters."""
    decay = params['decay']
    amplitude = params['amplitude']
    sigma = params['sigma']
    freq = params['freq']
    phase = params['phase']
    cx = params['cx']
    cy = params['cy']
    
    r_sq = (x - cx)**2 + (y - cy)**2
    temporal_decay = jnp.exp(-decay * t)
    gaussian_envelope = 1.0 + amplitude * jnp.exp(-r_sq / (sigma * (1.0 + t)))
    wave = jnp.cos(freq * jnp.sqrt(r_sq + 1e-10) - phase * t)
    
    return temporal_decay * gaussian_envelope * wave


def Sx_fn(x, y, t, params):
    """X-velocity structure (radial pattern)."""
    cx = params['cx']
    cy = params['cy']
    coeff = params['coeff_x']
    
    r_sq = (x - cx)**2 + (y - cy)**2
    return coeff * x / jnp.sqrt(r_sq + 1e-10)


def Sy_fn(x, y, t, params):
    """Y-velocity structure (radial pattern)."""
    cx = params['cx']
    cy = params['cy']
    coeff = params['coeff_y']
    
    r_sq = (x - cx)**2 + (y - cy)**2
    return coeff * y / jnp.sqrt(r_sq + 1e-10)


def solution_fn(x, y, t, params):
    """Full solution: (rho, rho*ux, rho*uy)."""
    rho = rho_fn(x, y, t, params)
    Sx = Sx_fn(x, y, t, params)
    Sy = Sy_fn(x, y, t, params)
    return rho, rho * Sx, rho * Sy


# ========== PDE residuals via autodiff ==========

def pde_residual_1(x, y, t, params, F1=0.0):
    """Continuity equation: rho_t + (rho*ux)_x + (rho*uy)_y = F1."""
    # rho*ux
    def rho_ux(xx, yy, tt):
        return solution_fn(xx, yy, tt, params)[1]
    
    # rho*uy
    def rho_uy(xx, yy, tt):
        return solution_fn(xx, yy, tt, params)[2]
    
    # rho
    def rho(xx, yy, tt):
        return solution_fn(xx, yy, tt, params)[0]
    
    # Time derivative of rho
    rho_t = grad(lambda tt: rho(x, y, tt), argnums=0)(t)
    
    # Spatial derivatives
    rho_ux_x = grad(lambda xx: rho_ux(xx, y, t), argnums=0)(x)
    rho_uy_y = grad(lambda yy: rho_uy(x, yy, t), argnums=0)(y)
    
    # Residual = LHS - RHS (should be zero)
    return (rho_t + rho_ux_x + rho_uy_y) - F1


def pde_residual_2(x, y, t, params, g=9.81, F2=0.0):
    """X-momentum: (rho*ux)_t + (rho*ux^2 + 0.5*g*rho^2)_x + (rho*ux*uy)_y = F2."""
    def rho_ux(xx, yy, tt):
        return solution_fn(xx, yy, tt, params)[1]
    
    def rho_uy(xx, yy, tt):
        return solution_fn(xx, yy, tt, params)[2]
    
    def rho(xx, yy, tt):
        return solution_fn(xx, yy, tt, params)[0]
    
    # Time derivative
    rho_ux_t = grad(lambda tt: rho_ux(x, y, tt), argnums=0)(t)
    
    # Nonlinear flux term
    def flux_x(xx):
        r = rho(xx, y, t)
        ru = rho_ux(xx, y, t)
        ux = ru / (r + 1e-10)
        return ru * ux + 0.5 * g * r**2
    
    flux_x_x = grad(flux_x, argnums=0)(x)
    
    # Cross term
    def flux_y(yy):
        ru = rho_ux(x, yy, t)
        rv = rho_uy(x, yy, t)
        r = rho(x, yy, t)
        uy = rv / (r + 1e-10)
        return ru * uy
    
    flux_y_y = grad(flux_y, argnums=0)(y)
    
    return (rho_ux_t + flux_x_x + flux_y_y) - F2


def pde_residual_3(x, y, t, params, g=9.81, F3=0.0):
    """Y-momentum: (rho*uy)_t + (rho*ux*uy)_x + (rho*uy^2 + 0.5*g*rho^2)_y = F3."""
    def rho_ux(xx, yy, tt):
        return solution_fn(xx, yy, tt, params)[1]
    
    def rho_uy(xx, yy, tt):
        return solution_fn(xx, yy, tt, params)[2]
    
    def rho(xx, yy, tt):
        return solution_fn(xx, yy, tt, params)[0]
    
    # Time derivative
    rho_uy_t = grad(lambda tt: rho_uy(x, y, tt), argnums=0)(t)
    
    # Cross term
    def flux_x(xx):
        ru = rho_ux(xx, y, t)
        rv = rho_uy(xx, y, t)
        r = rho(xx, y, t)
        ux = ru / (r + 1e-10)
        return rv * ux
    
    flux_x_x = grad(flux_x, argnums=0)(x)
    
    # Nonlinear flux term
    def flux_y(yy):
        r = rho(x, yy, t)
        rv = rho_uy(x, yy, t)
        uy = rv / (r + 1e-10)
        return rv * uy + 0.5 * g * r**2
    
    flux_y_y = grad(flux_y, argnums=0)(y)
    
    return (rho_uy_t + flux_x_x + flux_y_y) - F3


# ========== Combined loss ==========

def compute_loss_analytic(params, sample_points, g=9.81):
    """Single scalar loss combining all residuals.
    
    Parameters
    ----------
    params : dict
        Parameter dictionary with keys: decay, amplitude, sigma, freq, phase, cx, cy, coeff_x, coeff_y
    sample_points : dict
        Dictionary with keys:
        - 'interior': (x, y, t, F1, F2, F3) arrays for PDE residuals and forcing
        - 'ic': (x, y, rho_ic, rho_ux_ic, rho_uy_ic) arrays for initial conditions
        - 'bc_xmin', 'bc_xmax', 'bc_ymin', 'bc_ymax': boundary data with target values
    
    Returns
    -------
    loss : scalar
        Combined RMSE over all residual blocks
    """
    # PDE residuals with forcing
    x_pde, y_pde, t_pde, F1, F2, F3 = sample_points['interior']
    
    res1 = vmap(lambda xx, yy, tt, f1: pde_residual_1(xx, yy, tt, params, f1))(x_pde, y_pde, t_pde, F1)
    res2 = vmap(lambda xx, yy, tt, f2: pde_residual_2(xx, yy, tt, params, g, f2))(x_pde, y_pde, t_pde, F2)
    res3 = vmap(lambda xx, yy, tt, f3: pde_residual_3(xx, yy, tt, params, g, f3))(x_pde, y_pde, t_pde, F3)
    
    rmse_pde1 = jnp.sqrt(jnp.mean(res1**2))
    rmse_pde2 = jnp.sqrt(jnp.mean(res2**2))
    rmse_pde3 = jnp.sqrt(jnp.mean(res3**2))
    
    # IC residuals (at t=0, compare to manufactured IC values)
    x_ic, y_ic, rho_ic_target, rho_ux_ic_target, rho_uy_ic_target = sample_points['ic']
    t_ic = jnp.zeros_like(x_ic)
    
    rho_pred, rho_ux_pred, rho_uy_pred = vmap(lambda xx, yy, tt: solution_fn(xx, yy, tt, params))(x_ic, y_ic, t_ic)
    
    rmse_ic_rho = jnp.sqrt(jnp.mean((rho_pred - rho_ic_target)**2))
    rmse_ic_rho_ux = jnp.sqrt(jnp.mean((rho_ux_pred - rho_ux_ic_target)**2))
    rmse_ic_rho_uy = jnp.sqrt(jnp.mean((rho_uy_pred - rho_uy_ic_target)**2))
    
    # BC residuals (compare to manufactured BC target values)
    rmse_bc_total = 0.0
    
    for bc_key in ['bc_xmin', 'bc_xmax', 'bc_ymin', 'bc_ymax']:
        if bc_key in sample_points:
            bc_data = sample_points[bc_key]
            x_bc, y_bc, t_bc = bc_data['coords']
            rho_bc_t, rho_ux_bc_t, rho_uy_bc_t = bc_data['targets']
            
            rho_bc, rho_ux_bc, rho_uy_bc = vmap(lambda xx, yy, tt: solution_fn(xx, yy, tt, params))(x_bc, y_bc, t_bc)
            
            rmse_bc_total += jnp.sqrt(jnp.mean((rho_bc - rho_bc_t)**2))
            rmse_bc_total += jnp.sqrt(jnp.mean((rho_ux_bc - rho_ux_bc_t)**2))
            rmse_bc_total += jnp.sqrt(jnp.mean((rho_uy_bc - rho_uy_bc_t)**2))
    
    # Total loss: sum of all RMSE components
    total_loss = (rmse_pde1 + rmse_pde2 + rmse_pde3 + 
                  rmse_ic_rho + rmse_ic_rho_ux + rmse_ic_rho_uy + 
                  rmse_bc_total)
    
    return total_loss


# ========== Optimization ==========

def optimize_analytic(initial_params, sample_points, n_iterations=2000, lr=5e-3, print_every=200):
    """Optimize parameters using analytic loss."""
    # Convert to vector
    param_keys = ['decay', 'amplitude', 'sigma', 'freq', 'phase', 'cx', 'cy', 'coeff_x', 'coeff_y']
    vec = jnp.array([initial_params[k] for k in param_keys], dtype=jnp.float32)
    
    @jit
    def loss_fn(v):
        params = {k: v[i] for i, k in enumerate(param_keys)}
        return compute_loss_analytic(params, sample_points)
    
    grad_fn = jit(grad(loss_fn))
    
    # Use Adam with gradient clipping to prevent explosions
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),  # Clip gradients
        optax.adam(lr)
    )
    opt_state = opt.init(vec)
    
    loss_hist = []
    
    for i in range(n_iterations):
        loss_val = loss_fn(vec)
        grads = grad_fn(vec)
        
        # Check for NaN in gradients
        if jnp.any(jnp.isnan(grads)):
            print(f"Warning: NaN in gradients at iteration {i}")
            break
            
        updates, opt_state = opt.update(grads, opt_state)
        vec = optax.apply_updates(vec, updates)
        loss_hist.append(float(loss_val))
        
        if i % print_every == 0:
            print(f"Iteration {i}: loss={float(loss_val):.6f}")
    
    final_params = {k: float(vec[i]) for i, k in enumerate(param_keys)}
    return final_params, loss_hist


# ========== Main ==========

def main():
    """Test analytic loss with manufactured solution."""
    # True manufactured params
    true_params = {
        'decay': 0.686,
        'amplitude': 0.399,
        'sigma': 0.608,
        'freq': 2.249,
        'phase': 0.703,
        'cx': 0.5,
        'cy': -2.35,
        'coeff_x': 0.399,
        'coeff_y': 0.399,
    }
    
    # Sample points
    np.random.seed(42)
    n_interior = 1000
    n_ic = 200
    n_bc = 100
    
    x_pde = np.random.uniform(-10, 10, n_interior).astype(np.float32)
    y_pde = np.random.uniform(-10, 10, n_interior).astype(np.float32)
    t_pde = np.random.uniform(0, 5, n_interior).astype(np.float32)
    
    # Compute forcing terms from manufactured solution
    # For manufactured solution, PDE residuals should equal forcing
    F1 = np.array([pde_residual_1(x, y, t, true_params, 0.0) for x, y, t in zip(x_pde, y_pde, t_pde)], dtype=np.float32)
    F2 = np.array([pde_residual_2(x, y, t, true_params, 9.81, 0.0) for x, y, t in zip(x_pde, y_pde, t_pde)], dtype=np.float32)
    F3 = np.array([pde_residual_3(x, y, t, true_params, 9.81, 0.0) for x, y, t in zip(x_pde, y_pde, t_pde)], dtype=np.float32)
    
    # IC points and targets
    x_ic = np.random.uniform(-10, 10, n_ic).astype(np.float32)
    y_ic = np.random.uniform(-10, 10, n_ic).astype(np.float32)
    t_ic = np.zeros(n_ic, dtype=np.float32)
    
    # Compute IC targets from manufactured solution
    rho_ic_target = np.array([solution_fn(x, y, t, true_params)[0] for x, y, t in zip(x_ic, y_ic, t_ic)], dtype=np.float32)
    rho_ux_ic_target = np.array([solution_fn(x, y, t, true_params)[1] for x, y, t in zip(x_ic, y_ic, t_ic)], dtype=np.float32)
    rho_uy_ic_target = np.array([solution_fn(x, y, t, true_params)[2] for x, y, t in zip(x_ic, y_ic, t_ic)], dtype=np.float32)
    
    # BC points
    y_bc_x = np.random.uniform(-10, 10, n_bc).astype(np.float32)
    t_bc_x = np.random.uniform(0, 5, n_bc).astype(np.float32)
    x_bc_xmin = np.full(n_bc, -10.0, dtype=np.float32)
    x_bc_xmax = np.full(n_bc, 10.0, dtype=np.float32)
    
    x_bc_y = np.random.uniform(-10, 10, n_bc).astype(np.float32)
    t_bc_y = np.random.uniform(0, 5, n_bc).astype(np.float32)
    y_bc_ymin = np.full(n_bc, -10.0, dtype=np.float32)
    y_bc_ymax = np.full(n_bc, 10.0, dtype=np.float32)
    
    # Compute BC targets from manufactured solution
    def get_bc_targets(x, y, t):
        rho = np.array([solution_fn(xi, yi, ti, true_params)[0] for xi, yi, ti in zip(x, y, t)], dtype=np.float32)
        rho_ux = np.array([solution_fn(xi, yi, ti, true_params)[1] for xi, yi, ti in zip(x, y, t)], dtype=np.float32)
        rho_uy = np.array([solution_fn(xi, yi, ti, true_params)[2] for xi, yi, ti in zip(x, y, t)], dtype=np.float32)
        return rho, rho_ux, rho_uy
    
    sample_points = {
        'interior': (x_pde, y_pde, t_pde, F1, F2, F3),
        'ic': (x_ic, y_ic, rho_ic_target, rho_ux_ic_target, rho_uy_ic_target),
        'bc_xmin': {
            'coords': (x_bc_xmin, y_bc_x, t_bc_x),
            'targets': get_bc_targets(x_bc_xmin, y_bc_x, t_bc_x)
        },
        'bc_xmax': {
            'coords': (x_bc_xmax, y_bc_x, t_bc_x),
            'targets': get_bc_targets(x_bc_xmax, y_bc_x, t_bc_x)
        },
        'bc_ymin': {
            'coords': (x_bc_y, y_bc_ymin, t_bc_y),
            'targets': get_bc_targets(x_bc_y, y_bc_ymin, t_bc_y)
        },
        'bc_ymax': {
            'coords': (x_bc_y, y_bc_ymax, t_bc_y),
            'targets': get_bc_targets(x_bc_y, y_bc_ymax, t_bc_y)
        },
    }
    
    # Test 1: Evaluate at true params (should be near zero)
    print("="*60)
    print("Test 1: Loss at manufactured parameters")
    print("="*60)
    loss_true = compute_loss_analytic(true_params, sample_points)
    print(f"Loss at true params: {float(loss_true):.6e}")
    
    # Test 2: Optimize from SIGS-discovered parameters
    print("\n" + "="*60)
    print("Test 2: Optimize from SIGS-discovered triplet (seed=123)")
    print("="*60)
    
    # Parameters extracted from seed=123 run (loss=4.78):
    # rho: (cos(0.249*sqrt((x+2.204)^2+(y-3.344)^2)-0.394*t)) * (1+1.822*exp(-((x-4.082)^2+(y+4.91)^2)/(1.933*(1+t)))) * (exp(-((1^2)*pi^2*0.292*t)/(0.406^2)))
    # Sx: ((x*0.6666666111022222)/(sqrt((x-0.244)^2+(y-2.084)^2)))
    # Sy: ((y*0.5333433433311195)/(sqrt((x-3.499)^2+(y-1.778)^2)))
    init_params = {
        'decay': 17.78,            # (1^2)*pi^2*0.292/(0.406^2)
        'amplitude': 1.822,        # from Gaussian envelope
        'sigma': 1.933,            # from Gaussian width
        'freq': 0.249,             # wave frequency
        'phase': 0.394,            # wave phase velocity
        'cx': -2.204,              # from wave center (x+2.204)
        'cy': 3.344,               # from wave center (y-3.344)
        'coeff_x': 0.667,          # from Sx radial coefficient (independent)
        'coeff_y': 0.533,          # from Sy radial coefficient (independent)
    }
    
    print("\nInitial params:")
    for k, v in init_params.items():
        print(f"  {k}: {v:.4f}")
    
    loss_init = compute_loss_analytic(init_params, sample_points)
    print(f"\nInitial loss: {float(loss_init):.6f}")
    
    print("\nRunning optimization...\n")
    final_params, loss_hist = optimize_analytic(init_params, sample_points, n_iterations=20000, lr=1e-5, print_every=2000)
    
    print("\n" + "="*60)
    print("Final results")
    print("="*60)
    print(f"Final loss: {loss_hist[-1]:.6f}")
    print("\nParameter comparison:")
    print(f"{'Param':<12} {'True':<10} {'Optimized':<10} {'Error':<10}")
    print("-" * 45)
    for k in true_params.keys():
        true_val = true_params[k]
        opt_val = final_params[k]
        error = abs(opt_val - true_val)
        print(f"{k:<12} {true_val:<10.4f} {opt_val:<10.4f} {error:<10.4e}")
    
    # Save results
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = f'analytic_loss_test_{ts}.pkl'
    with open(out, 'wb') as f:
        pickle.dump({
            'true_params': true_params,
            'init_params': init_params,
            'final_params': final_params,
            'loss_hist': loss_hist,
            'loss_true': float(loss_true),
        }, f)
    print(f"\nResults saved to {out}")


if __name__ == '__main__':
    main()
