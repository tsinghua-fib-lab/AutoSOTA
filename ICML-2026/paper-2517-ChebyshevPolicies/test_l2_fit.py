import sys, os
sys.path.insert(0, "/repo")
import numpy as np
import torch

from polyagents.polynomial_basis import MultiVarPoly

C1 = 4.3346
C2 = 4.8358
ALPHA_BOOT = 0.1
X_HAT = -np.pi / 6

def analytic_policy_unnormalized(x, v):
    abs_v = abs(v)
    if abs_v < 1e-10:
        if abs(x - X_HAT) <= 0.01:
            return ALPHA_BOOT
        return 0.0
    is_phase2 = (v > 0 and x > X_HAT - 0.02) or (v < 0 and x < X_HAT + 0.02)
    C = C2 if is_phase2 else C1
    action = np.sign(v) * C * abs_v
    if abs(x - X_HAT) <= 0.01 and abs_v < 0.005:
        action = np.sign(v) * max(C * abs_v, ALPHA_BOOT)
    return float(np.clip(action, -1.0, 1.0))

pos_low, pos_high = -1.2, 0.6
vel_low, vel_high = -0.07, 0.07

def denormalize_state(p_norm, v_norm):
    pos = (p_norm + 1.0) / 2.0 * (pos_high - pos_low) + pos_low
    vel = (v_norm + 1.0) / 2.0 * (vel_high - vel_low) + vel_low
    return pos, vel

N_GRID = 50
p_norm_grid = np.linspace(-1.0, 1.0, N_GRID)
v_norm_grid = np.linspace(-1.0, 1.0, N_GRID)
PP, VV = np.meshgrid(p_norm_grid, v_norm_grid)
norm_points = np.column_stack([PP.ravel(), VV.ravel()])

actions = []
for i in range(len(norm_points)):
    p_n, v_n = norm_points[i]
    pos, vel = denormalize_state(p_n, v_n)
    actions.append(analytic_policy_unnormalized(pos, vel))
actions = np.array(actions, dtype=np.float32)

poly = MultiVarPoly(dim=2, degree=3, basis="chebyshev", initialization="random")
poly.fit_l2(torch.tensor(norm_points, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.float32))

pred_actions = []
for i in range(len(norm_points)):
    with torch.inference_mode():
        pred = poly.evaluate_point(torch.tensor(norm_points[i], dtype=torch.float32)).item()
    pred_actions.append(pred)
pred_actions = np.array(pred_actions)
mse = np.mean((pred_actions - actions)**2)
mae = np.mean(np.abs(pred_actions - actions))
print(f"L2 Fit: MSE={mse:.6f}, MAE={mae:.6f}")

coeffs = poly.coeffs.detach().cpu().numpy()
print(f"Fitted coefficients (first 5): {coeffs[:5]}")
print(f"Coeff range: [{coeffs.min():.4f}, {coeffs.max():.4f}]")

# Quick eval with fitted coeffs
from utils import exp_run
xs = np.linspace(-0.6, -0.4, 5)
model, eval_env = exp_run.get_sb3_polynomial_model_and_eval_env(
    basis="chebyshev", env_name="MountainCarContinuous-v0",
    coeffs=coeffs, algo="ars")
rewards = []
for x in xs:
    r_mean, r_std = exp_run.run_sb3_model(
        model, eval_env, options={"low": x, "high": x})
    rewards.append(r_mean)
mean_r = float(np.mean(rewards))
print(f"L2-fitted policy (no training) mean R on 5 positions: {mean_r:.4f}")
