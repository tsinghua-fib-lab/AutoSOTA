#!/usr/bin/env python3
import os, sys, time, json, copy
import numpy as np
import torch
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, '/repo')
sys.path.insert(0, '/repo/algorithms')

from utils import exp_run, plot
from polyagents.polynomial_basis import MultiVarPoly

ENV_NAME = 'MountainCarContinuous-v0'
TENSORBOARD_LOG_DIR = '/repo/tensorboard_logs/'
NUM_POLICIES = 20
BASE_SEED = 42
NAME_PREFIX = 'mountaincar_sota2_'

C1 = 4.3346
C2 = 4.8358
ALPHA_BOOT = 0.1
X_HAT = -np.pi / 6
POS_LOW, POS_HIGH = -1.2, 0.6
VEL_LOW, VEL_HIGH = -0.07, 0.07

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

def fit_warmstart_coeffs():
    N_GRID = 50
    p_norm_grid = np.linspace(-1.0, 1.0, N_GRID)
    v_norm_grid = np.linspace(-1.0, 1.0, N_GRID)
    PP, VV = np.meshgrid(p_norm_grid, v_norm_grid)
    norm_points = np.column_stack([PP.ravel(), VV.ravel()])
    actions = []
    for p_n, v_n in norm_points:
        pos = (p_n + 1.0) / 2.0 * (POS_HIGH - POS_LOW) + POS_LOW
        vel = (v_n + 1.0) / 2.0 * (VEL_HIGH - VEL_LOW) + VEL_LOW
        actions.append(analytic_policy_unnormalized(pos, vel))
    actions = np.array(actions, dtype=np.float32)
    poly = MultiVarPoly(dim=2, degree=3, basis='chebyshev', initialization='random')
    poly.fit_l2(torch.tensor(norm_points, dtype=torch.float32),
                torch.tensor(actions, dtype=torch.float32))
    return poly.coeffs.detach().cpu().numpy().copy()

def main():
    os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)

    print('Fitting warm-start coefficients from analytic policy...')
    warmstart_coeffs = fit_warmstart_coeffs()
    print('  Fitted coefficients, range: [{:.4f}, {:.4f}]'.format(
        warmstart_coeffs.min(), warmstart_coeffs.max()))

    kwargs_ars = dict(
        algo='ars',
        env_name=ENV_NAME,
        degree=3,
        basis='chebyshev',
        tensorboard_log_dir=TENSORBOARD_LOG_DIR + 'ars_mountaincar_sota1',
        delta_std=0.1,
        n_delta=8,
        n_top=2,
        learning_rate=0.018,
        steps=80000,
        evaluate_every_n_steps=0,
        normalize_actions=True,
        min_action=-1.0,
        max_action=1.0,
        initialization='random',
        zero_policy=False,
        seed=None,
        verbose=0,
    )

    NUM_CORES = min(8, mp.cpu_count())
    n_steps_val = kwargs_ars['steps']
    n_delta_val = kwargs_ars['n_delta']
    n_top_val = kwargs_ars['n_top']
    print()
    print('SOTA Iter 2: Training {} policies, {} steps each'.format(NUM_POLICIES, n_steps_val))
    print('  n_delta={}, n_top={}'.format(n_delta_val, n_top_val))
    print('  Using {} cores'.format(NUM_CORES))

    args_list, logdirs = plot.get_kwargs_with_distinct_seeds(
        kwargs=kwargs_ars,
        num_experiments=NUM_POLICIES,
        name_prefix=NAME_PREFIX,
        seed=BASE_SEED,
        degree=3
    )

    for args in args_list:
        args['coeffs'] = warmstart_coeffs.copy()

    try:
        mp.set_start_method('forkserver')
    except RuntimeError:
        pass

    start_time = time.time()
    with mp.Pool(processes=NUM_CORES) as pool:
        results = pool.map(exp_run.run_sb3_polyagent_training, args_list)
    training_time = time.time() - start_time
    print('Training completed in {:.0f} seconds'.format(training_time))

    valid_results = [(r[0], r[1], r[2]) for r in results if not isinstance(r, Exception)]
    print('Valid results: {}/{}'.format(len(valid_results), len(results)))

    if len(valid_results) == 0:
        print('ERROR: No valid training results!')
        for i, r in enumerate(results):
            print('  Result {}: {}: {}'.format(i, type(r).__name__, r))
        sys.exit(1)

    print()
    print('Evaluating on 100 start positions from [-0.6, -0.4]...')
    xs = np.linspace(-0.6, -0.4, 100)
    all_eval_results = []
    for policy_idx, (name, duration, coeffs) in enumerate(valid_results):
        rewards = []
        model, eval_env = exp_run.get_sb3_polynomial_model_and_eval_env(
            basis='chebyshev', env_name=ENV_NAME, coeffs=coeffs, algo='ars')
        for x in xs:
            r_mean, r_std = exp_run.run_sb3_model(model, eval_env, options={'low': x, 'high': x})
            rewards.append(r_mean)
        mean_r = float(np.mean(rewards))
        all_eval_results.append({'name': name, 'mean_R': mean_r, 'duration': duration})
        print('  Policy {}/{}: mean_R={:.4f}'.format(policy_idx+1, len(valid_results), mean_r))

    best = max(all_eval_results, key=lambda x: x['mean_R'])
    best_coeffs = None
    for name, duration, coeffs in valid_results:
        if name == best['name']:
            best_coeffs = coeffs
            break

    print()
    print('Best policy: {}, mean_R={:.4f}'.format(best['name'], best['mean_R']))

    print('Computing t*...')
    times_to_goal = []
    model_t, _ = exp_run.get_sb3_polynomial_model_and_eval_env(
        basis='chebyshev', env_name=ENV_NAME, coeffs=best_coeffs, algo='ars')
    for x in xs:
        _, eval_env = exp_run.get_sb3_polynomial_model_and_eval_env(
            basis='chebyshev', env_name=ENV_NAME, coeffs=best_coeffs, algo='ars')
        env = eval_env.venv.envs[0]
        obs, info = env.reset(options={'low': x, 'high': x})
        steps = 0
        terminated, truncated = False, False
        while not (terminated or truncated) and steps < 999:
            action, _ = model_t.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
        times_to_goal.append(steps)
    mean_t_star = float(np.mean(times_to_goal))

    print('Computing L2 distance...')
    N_GRID = 50
    x_grid = np.linspace(-1.2, 0.45, N_GRID)
    v_grid = np.linspace(-0.07, 0.07, N_GRID)
    XX, VV = np.meshgrid(x_grid, v_grid)
    points = np.column_stack([XX.ravel(), VV.ravel()])
    obs_low = np.array([-1.2, -0.07])
    obs_high = np.array([0.6, 0.07])
    points_norm = 2.0 * (points - obs_low) / (obs_high - obs_low) - 1.0

    model_l2, _ = exp_run.get_sb3_polynomial_model_and_eval_env(
        basis='chebyshev', env_name=ENV_NAME, coeffs=best_coeffs, algo='ars')
    model_actions = []
    for i in range(len(points_norm)):
        action, _ = model_l2.predict(points_norm[i].astype(np.float32), deterministic=True)
        model_actions.append(float(action))
    analytic_actions = [analytic_policy_unnormalized(p[0], p[1]) for p in points]
    diff = np.array(model_actions) - np.array(analytic_actions)
    l2_distance = float(np.sqrt(np.mean(diff ** 2)))

    torch.save(best_coeffs, '/repo/best_ch3_ars_coeffs.pt')
    print('Saved best checkpoint')

    metrics = {'R': best['mean_R'], 't*': mean_t_star, 'L2': l2_distance}
    sep = '=' * 60
    print()
    print(sep)
    print('Iter 2 Results: R={:.2f}, t*={:.1f}, L2={:.4f}'.format(
        metrics['R'], mean_t_star, l2_distance))
    print(sep)

    with open('/repo/sota_iter2_results.json', 'w') as f:
        json.dump({'metrics': metrics, 'best_name': best['name'],
                   'training_time': float(training_time)}, f, indent=2)
    return metrics

if __name__ == '__main__':
    main()
