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
def normalize_state(pos, vel):
    p_norm = 2.0 * (pos - pos_low) / (pos_high - pos_low) - 1.0
    v_norm = 2.0 * (vel - vel_low) / (vel_high - vel_low) - 1.0
    return p_norm, v_norm

N_GRID = 50
p_norm_grid = np.linspace(-1.0, 1.0, N_GRID)
v_norm_grid = np.linspace(-1.0, 1.0, N_GRID)
PP, VV = np.meshgrid(p_norm_grid, v_norm_grid)
norm_points = np.column_stack([PP.ravel(), VV.ravel()])
actions = [analytic_policy_unnormalized(*denormalize_state(p_n, v_n)) for p_n, v_n in norm_points]
actions = np.array(actions, dtype=np.float32)

poly = MultiVarPoly(dim=2, degree=3, basis="chebyshev", initialization="random")
poly.fit_l2(torch.tensor(norm_points, dtype=torch.float32), torch.tensor(actions, dtype=torch.float32))
coeffs = poly.coeffs.detach().cpu().numpy()

# Manual simulation
from utils import exp_run
model, eval_env = exp_run.get_sb3_polynomial_model_and_eval_env(
    basis="chebyshev", env_name="MountainCarContinuous-v0", coeffs=coeffs, algo="ars")
env = eval_env.venv.envs[0]
obs, info = env.reset(options={"low": -0.5, "high": -0.5})
print(f"L2-fit sim from x=-0.5: init_obs={obs}")
total_reward = 0
for step in range(999):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += float(reward)
    if step < 5 or step % 200 == 0:
        print(f"  Step {step}: action={float(action):.4f}, reward={float(reward):.4f}, pos={env.unwrapped.state[0]:.4f}, vel={env.unwrapped.state[1]:.6f}")
    if terminated or truncated:
        print(f"  Done at step {step+1}, total={total_reward:.4f}, terminated={terminated}")
        break

# Compare baseline
print("\nBaseline sim from x=-0.5:")
baseline_coeffs = torch.load("/repo/best_ch3_ars_coeffs.pt", map_location="cpu", weights_only=True)
model_bl, eval_env_bl = exp_run.get_sb3_polynomial_model_and_eval_env(
    basis="chebyshev", env_name="MountainCarContinuous-v0", coeffs=baseline_coeffs, algo="ars")
env_bl = eval_env_bl.venv.envs[0]
obs_bl, info_bl = env_bl.reset(options={"low": -0.5, "high": -0.5})
total_reward_bl = 0
for step in range(999):
    action_bl, _ = model_bl.predict(obs_bl, deterministic=True)
    obs_bl, reward_bl, terminated_bl, truncated_bl, info_bl = env_bl.step(action_bl)
    total_reward_bl += float(reward_bl)
    if step < 5 or step % 200 == 0:
        print(f"  Step {step}: action={float(action_bl):.4f}, reward={float(reward_bl):.4f}, pos={env_bl.unwrapped.state[0]:.4f}, vel={env_bl.unwrapped.state[1]:.6f}")
    if terminated_bl or truncated_bl:
        print(f"  Done at step {step+1}, total={total_reward_bl:.4f}")
        break

print(f"\nL2-fit total: {total_reward:.4f}, Baseline total: {total_reward_bl:.4f}")
