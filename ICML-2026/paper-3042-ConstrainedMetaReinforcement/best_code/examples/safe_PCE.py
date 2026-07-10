# imports
import sys
import time
import numpy as np
sys.path.append('../..')
from env.gridworld import Gridworld
import copy
import pickle
import os
from algs.cmdp import cmdp_gda, cmdp_gda_occ, regularization
import visualization.gridworld_vis as gv
from einops import rearrange, reduce, repeat, einsum
import random
from pathlib import Path
from scipy.stats import entropy
import math
from typing import List, Callable
import matplotlib.pyplot as plt
from PIL import Image
import visualization.gridworld_vis as gv
from tqdm import tqdm
import numpy.random as rn



def make_env(noise: float):
    float_formatter = "{:.8f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})
    # create environment
    params = {
        'grid_height': 7,
        'grid_width' : 7,
        'noise': float(noise),
        'gamma': 0.9
    }
    n = params['grid_height'] * params['grid_width']
    m = 4 # number of actions
    k = 1 # number of constaints
    Psi = np.zeros((params['grid_height'], params['grid_width'], m, k))
    Psi[2, 1:5, :, 0] = 10
    Psi[4, 1:5, :, 0] = 10
    Psi = rearrange(Psi, 'sx sy a k -> (sy sx) a k')
    b = np.array([1.5])
    r = np.zeros((n, m))
    r[45,:] = 10
    env = Gridworld(**params, constraints = (Psi, b), r = r)
    env.P[45, :, :] = np.zeros_like(env.P[0, :, :])
    env.P[45, :, 45] = 1.0
    return env
# using LP for finding optimal policy
def LP(env):
    beta = 0.1
    eta_p = (1-env.gamma) /beta * 1
    eta_xi = 0.1
    policy, _, _ = cmdp_gda(env, beta , eta_p, eta_xi, max_iters=1e4, tol=1e-4, mode='alt_gda',
                                        n_v_tot_eval_steps=50, n_v_c_eval_steps=50, logging=False, check_steps=1000)
    reward, cost = value_function_r_c(env, policy)

    return (policy, reward, cost) 

# Evaluate the policy to get reward and cost values
def value_function_r_c(env, policy: np.ndarray):   
    occ = env.policy2stateactionocc(policy)
    reward = np.sum(occ * env.r / (1-env.gamma))
    cost = np.einsum('jki,jk->i', env.Psi, occ) / (1-env.gamma)
    return reward, cost 

# ----------------------------------------------------
#  High-level Pretraining Main Loop
# ----------------------------------------------------
def pretrain_stage(delta: float, epsilon: float):
    """
    Implements the top-level algorithm.
    """
    # Set a random seed based on the current time
    random.seed(int(time.time()))
    np.random.seed(int(time.time()))

    # number of sampled CMDPs: N = log(1/delta) / delta^2
    N_f = int(math.log(1 / delta) / (delta ** 2))

    phase = 1
    # Sample N CMDPs by sampling noise ∈ [0, 0.5]
    noises = list(truncated_gaussian(size=N_f))
    N = N_f
    while True:
        print(f"\n=== Phase {phase} ===")

        # --- Subroutine: policy cover set
        U, hat_Pi_size = policy_cover_subroutine(noises, delta, epsilon)

        # Check stopping condition
        lhs = math.sqrt((hat_Pi_size * math.log(2 * N / delta)) / (N - hat_Pi_size))
        if lhs <= delta:
            return U, hat_Pi_size
        print(phase, N, hat_Pi_size, lhs)

        # Generate new set of CMDPs with doubled N
        N = 2 * N
        new_noises = list(truncated_gaussian(size=N_f))
        noises.extend(new_noises)
        phase += 1

def policy_cover_subroutine(noises: List,delta: float,epsilon: float):
    """
    Return the cover set U following Algorithm 2.
    """
    N = len(noises)

    # Initialize
    U = []            # selected cover indices
    T = set(range(N)) # uncovered indices
    A = np.zeros((N, N), dtype=int)

    for i in range(N):
        for j in range(N):
            # Oracle: 1 if M_j covers M_i
            A[i, j] = O_c(noises[i], noises[j],epsilon)

    # Greedy covering
    cover_amounts = []

    for t in range(N):
        if len(T) == 0:
            break

        # find j that covers the most elements in T
        best_j = None
        best_cover = -1

        for j in range(N):
            if j in U:
                continue
            cover_size = sum(A[i, j] for i in T)
            if cover_size > best_cover:
                best_cover = cover_size
                best_j = j

        # Add best_j to cover set
        U.append(best_j)

        # Remove covered elements from T
        newly_covered = {i for i in T if A[i, best_j] == 1}
        T = T - newly_covered

        cover_amounts.append(best_cover)

        # Check total cover
        if sum(cover_amounts) >= (1 - 3 * delta) * N:
            break

    # Build final cover set of CMDP objects
    U_cmdps = [noises[j] for j in U]

    return U_cmdps, len(U)
def O_c(Noise_i, Noise_j, epsilon):
    """
    Example coverage oracle:
    Return 1 if |noise_i - noise_j| < epsilon
    """
    return int(abs(Noise_i - Noise_j) < epsilon)
def truncated_gaussian(size, mean=0.3, std=0.03, low=0.0, high=0.5):
    """
    Draw N samples from a Gaussian and truncate to [low, high].
    """
    samples = np.random.normal(mean, std, size=size)
    return np.clip(samples, low, high)
def policy_set(U: List, safe_policy: np.array, show_progress: bool = True):
    """
    Generate policy set hat_Pi for Test Stage.
    Each element is a tuple (pi, u, v, u_s, v_s).

    If show_progress=True, display a tqdm progress bar.
    """
    hat_Pi = []

    # choose appropriate tqdm for notebook / terminal
    try:
        if show_progress:
            from tqdm.notebook import tqdm as _tqdm
            pbar = _tqdm(U, desc="learning policies")
        else:
            pbar = U
    except Exception:
        if show_progress:
            from tqdm import tqdm as _tqdm
            pbar = _tqdm(U, desc="learning policies")
        else:
            pbar = U

    for idx, noise in enumerate(pbar):
        # Learn optimal policy pi on M
        M = make_env(noise)
        pi, u, v = LP(M)  # safe baseline policy
        u_s, v_s = value_function_r_c(M, safe_policy)

        hat_Pi.append((pi, u, v, u_s, v_s))

        if show_progress:
            # update postfix with recent values
            try:
                pbar.set_postfix({"noise": f"{float(noise):.4f}", "u": f"{u:.4f}", "v": f"{v:.4f}"})
            except Exception:
                pass

    return hat_Pi

def mixture_policy(pi_l, pi_s, alpha):
    """Returns a stochastic mixture policy α π_l + (1−α) π_s."""
    if np.random.rand() <= alpha:
        return pi_l
    else:
        return pi_s

def rollout(env, policy: np.ndarray, horizon: int = 200):
    """
    Generate a trajectory of length T, following the given policy. Return trajectories and corresponding total discounted rewards.
    """
    nu0 = env.nu0
    tot_costs = 0
    tot_rewards = 0
    # sample trajectories
    s = rn.choice(env.n, p=nu0)
    for t in range(horizon):      
        # sample from policy
        a = rn.choice(env.m, p=policy[s, :])
        # sample from dynamics
        s_next = rn.choice(env.n, p=env.P[s, a, :]) 
        c = env.Psi[s, a, 0]
        tot_costs += env.gamma**t * c     
        r = env.r[s, a]
        tot_rewards += env.gamma**t * r
        s = s_next
    return (tot_rewards, tot_costs)
def test_stage(K: int = 50000, noise=None, pi_s=None, hat_Pi=None, L_r = 8, L_c = 1):
    env = make_env(noise)
    epsilon = 0.01
    # Initialization
    l = 1
    k0 = 1
    m = 0
    alpha = {}        # alpha[(l,m)] stores α_{l,m}
    alpha[(l,0)] = 0  # α_{l,0} = 0
    # Buffers for observed rewards & costs
    R_hist = []
    C_hist = []
    real_r_hist = np.zeros(K)
    real_c_hist = np.zeros(K)
    # MAIN LOOP
    reg = 0
    _, u_optimal, _ = LP(env)
    # Cache for value_function_r_c (policies rarely change within elimination rounds)
    _cached_pi_l = None
    _cached_r = _cached_c = _cached_r_s = _cached_c_s = None
    u_pi_s = []
    v_pi_s = []
    for k in range(1, K + 1):
        # --- Step 1: Select policy with maximal u_l
        if not hat_Pi:
            print("No policies left in hat_Pi, exiting loop.")
            break
        else:
            index, (pi_l, u_l, v_l, u_l_s, v_l_s) = max(enumerate(hat_Pi), key=lambda x: x[1][1])
            # --- Step 2: Mixture policy
            alpha_lm = alpha[(l, m)]
            pi_mix = mixture_policy(pi_l, pi_s, alpha_lm)
            # --- Step 3: Compute mixture expected values
            u_lm = alpha_lm * u_l + (1 - alpha_lm) * u_l_s
            v_lm = alpha_lm * v_l + (1 - alpha_lm) * v_l_s
            # calculate the real value (with caching -- policies rarely change)
            if pi_l is not _cached_pi_l:
                _cached_r, _cached_c = value_function_r_c(env, pi_l)
                _cached_r_s, _cached_c_s = value_function_r_c(env, pi_s)
                _cached_pi_l = pi_l
            r, c = _cached_r, _cached_c
            r_s, c_s = _cached_r_s, _cached_c_s
            real_r = alpha_lm * r + (1 - alpha_lm) * r_s
            real_c = alpha_lm * c + (1 - alpha_lm) * c_s
            if real_c > env.b:
                print(f"unsafe, {noise}")
                break
            # Store real_r and real_c for plotting
            reg += u_optimal - real_r
            real_r_hist[k-1] = reg
            real_c_hist[k-1] = real_c
            # --- Step 4: Execute episode
            R_k, C_k = rollout(env, pi_mix)
            R_hist.append(R_k)
            C_hist.append(C_k)
            if m == 0 and l > 1:
                u_pi_s.append(R_k)
                v_pi_s.append(C_k)
                thresh_safe =  np.sqrt(4000/len(u_pi_s)) + epsilon * L_r
                thresh_c_safe = np.sqrt(150/len(v_pi_s))+ epsilon * L_c
                if (k - k0 + 1 >= 1000) and (abs(np.mean(u_pi_s) - u_lm) >= thresh_safe or abs(np.mean(v_pi_s)- v_lm) >= thresh_c_safe):
                    # Remove policy
                    print("safe policy:")
                    print(f"> Real vaue is {u_optimal}, Eliminated policy with value {u_l},{len(u_pi_s)}")
                    print(f"> {abs(np.mean(u_pi_s) - u_lm)}, {thresh_safe}, {u_lm - real_r}")
                    print(f"> {abs(np.mean(v_pi_s) - v_lm)}, {thresh_c_safe}, {v_lm - real_c}")
                    del hat_Pi[index]
                    # Update indices
                    k0 = k + 1
                    l = l + 1
                    m = 0
                    alpha[(l, 0)] = 0
                    R_hist = []
                    C_hist = []
                    continue
            else: 
                # Running averages
                window_R = np.mean(R_hist)
                window_C = np.mean(C_hist)
                # Threshold
                thresh =  np.sqrt(4000/(k - k0 + 1)) + epsilon * L_r
                thresh_c = np.sqrt(300/((k - k0 + 1)))+ epsilon * L_c    
                # --- Step 5: Elimination conditions for not safe policy
                if (k - k0 + 1 >= 1000) and (abs(window_R - u_lm) >= thresh or abs(window_C - v_lm) >= thresh_c):
                    # Remove policy
                    print(f"> Real vaue is {u_optimal}, Eliminated policy with value {u_l},{k - k0 + 1}")
                    print(f"> {abs(window_R - u_lm)}, {thresh}, {u_lm - real_r}")
                    print(f"> {abs(window_C - v_lm)}, {thresh_c}, {v_lm - real_c}")
                    del hat_Pi[index]
                    if m == 0 and l == 1:
                        u_pi_s = R_hist
                        v_pi_s = C_hist
                    # Update indices
                    k0 = k + 1
                    l = l + 1
                    m = 0
                    alpha[(l, 0)] = 0
                    R_hist = []
                    C_hist = []

                    continue

            # --- Step 6: Enough samples collected?
            cond_samples = (k - k0 - 1) >= min(max(100 / (((env.b - v_lm) ** 2)), 1000), 5200)
            m_max = math.log(epsilon, 2/3)
            cond_m = (m <= m_max)

            if cond_samples and cond_m:
                if m == 0 and l == 1:
                    u_pi_s = R_hist
                    v_pi_s = C_hist
                # Move to next m
                k0 = k + 1
                m += 1
                R_hist = []
                C_hist = []

                # Update α_{l,m} with constraint-margin-aware acceleration
                if m == 1:
                    alpha_new = ((env.b - v_l_s) - 2 * epsilon * L_c) / ((env.b - v_l_s) - 2 * epsilon * L_c + 2 )
                else:
                    alpha_prev = alpha[(l, m - 1)]
                    alpha_new = (
                        3 * (env.b - v_l_s) * alpha_prev /
                        ((alpha_prev + 2) * (env.b - v_l_s) + (2 * L_c + 1) * epsilon)
                    )

                # Constraint-margin-aware acceleration:
                # If large safety margin, accelerate alpha to reach optimal mixture faster.
                # If small margin, stay conservative to avoid constraint violations.
                margin = float(env.b - real_c)
                if margin > 0.4:
                    alpha_new = min(1.0, alpha_new * 1.5)
                elif margin > 0.2:
                    alpha_new = min(1.0, alpha_new * 1.2)

                alpha[(l, m)] = max(0, min(1, alpha_new))

                print(f"> Updated α_{l,m} = {alpha[(l,m)]} (margin={margin:.3f}) with {u_optimal - real_r}")
    return real_r_hist, real_c_hist
def test_stage_no_safe(K: int = 50000, noise=None, hat_Pi=None, L_r = 8, L_c = 1):
    env = make_env(noise)
    epsilon = 0.1
    k0 = 1
    # Buffers for observed rewards & costs
    R_hist = []
    C_hist = []
    real_r_hist = np.ones(K)
    real_c_hist = np.ones(K)
    # MAIN LOOP
    reg = 0
    _, u_optimal, v = LP(env)
    print(noise, u_optimal, v)
    for k in range(1, K + 1):
        # --- Step 1: Select policy with maximal u_l
        if not hat_Pi:
            print(f"No policies left in hat_Pi, exiting loop, {noise}")
            break
        else:
            index, (pi_l, u_l, v_l, _, _) = max(enumerate(hat_Pi), key=lambda x: x[1][1])
            # calculate the real value
            r, c = value_function_r_c(env, pi_l)
            # Store real_r and real_c for plotting
            reg += abs(u_optimal - r)
            real_r_hist[k-1] = reg
            real_c_hist[k-1] = c
            # --- Step 4: Execute episode
            R_k, C_k = rollout(env, pi_l)
            R_hist.append(R_k)
            C_hist.append(C_k)
            # Running averages
            window_R = np.mean(R_hist)
            window_C = np.mean(C_hist)
            # Threshold
            thresh =  np.sqrt(3000/(k - k0 + 1)) + epsilon * L_r
            thresh_c = np.sqrt(600/((k - k0 + 1)))+ epsilon * L_c    # --- Step 5: Elimination conditions
            if (k - k0 + 1 >= 100) and (abs(window_R - u_l) >= thresh or abs(window_C - v_l) >= thresh_c):
                print(f"> Real vaue is {u_optimal}, Eliminated policy with value {u_l},{abs(window_R - u_l)}, {thresh}, {abs(window_C - v_l)}, {thresh_c}, {k - k0 + 1}")
                # Remove policy
                k0 = k + 1
                del hat_Pi[index]
                R_hist = []
                C_hist = []

                continue
    return real_r_hist, real_c_hist
