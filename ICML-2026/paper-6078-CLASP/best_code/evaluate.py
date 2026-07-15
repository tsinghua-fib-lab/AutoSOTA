#!/usr/bin/env python3
"""
CLASP Paper Reproduction: Online Linear Regression Experiment
Extracted from OLR-Small-Experiment.ipynb

Runs 50 trials with n=10, k=4, T=100.
Reports Cumulative loss, CCVT,1, and CCVT,2 for CLASP-I and baselines.
"""

import numpy as np
from scipy import stats
from tqdm import tqdm
import cvxpy as cp
import json
import time


# ─── Dataset generation ────────────────────────────────────────────

def new_dataset(n, k, T):
    Hs, Ys, As, Bs = [], [], [], []
    for t in range(1, T + 1):
        h = np.random.uniform(-1.0, 1.0, (k, n))
        Hs.append(h)
        y = h @ np.ones((n, 1))
        epsilon = stats.truncnorm(-1., 1.).rvs(k).reshape(-1, 1)
        y = y + epsilon
        Ys.append(y)
        a = np.random.uniform(0.0, 2.0, (k, n))
        As.append(a)
        b = np.random.uniform(0.0, 1.0, (k, 1))
        Bs.append(b)
    return Hs, Ys, As, Bs


# ─── CLASP-I ────────────────────────────────────────────────────────

class CLASP_I:
    def __init__(self, x_0, T, assumptions_param):
        self.x_0 = x_0
        self.n = x_0.shape[0]

    def projection_step(self, data):
        x, A, b = data
        y = cp.Variable((self.n, 1))
        cost = (1 / 2) * cp.norm(y - x) ** 2
        constraints = [y <= 1, y >= 0]
        constraints.append(A @ y <= b)
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve()
        return y.value.copy()

    def run(self, dataset):
        x_pred = self.x_0.copy()
        Preds = [x_pred.flatten()]
        Hs, Ys, As, Bs = dataset
        for t in range(T):
            H, y, A, b = Hs[t], Ys[t], As[t], Bs[t]
            g_t = A @ x_pred - b
            g_t_idx = np.argmax(g_t)
            g_t_val = max(g_t[g_t_idx], 0.0)
            f_grad = (H @ x_pred - y).T @ H
            f_grad = f_grad.reshape(-1, 1)
            x_pred = x_pred - (1 / np.sqrt(t + 1)) * f_grad
            data_proj = (x_pred, A[g_t_idx].reshape(1, -1), b[g_t_idx].reshape(1, 1))
            x_pred = self.projection_step(data_proj)
            Preds.append(x_pred.flatten())
        return x_pred, Preds


# ─── CLASP-F ────────────────────────────────────────────────────────

class CLASP_F:
    def __init__(self, x_0, T, assumptions_param):
        self.x_0 = x_0
        self.n = x_0.shape[0]

    def projection_step(self, x):
        xc = x.copy()
        for i in range(len(xc)):
            xc[i, 0] = min(1., max(0., xc[i, 0]))
        return xc

    def run(self, dataset):
        x_pred = self.x_0.copy()
        Preds = [x_pred.flatten()]
        Hs, Ys, As, Bs = dataset
        for t in range(T):
            H, y, A, b = Hs[t], Ys[t], As[t], Bs[t]
            g_t = A @ x_pred - b
            g_t_idx = np.argmax(g_t)
            g_t_val = max(g_t[g_t_idx], 0.0)
            g_grad = np.zeros((A.shape[1], 1))
            if g_t_val > 0.0:
                g_grad = A[g_t_idx].reshape(-1, 1)
            bar_x = x_pred
            if g_t_val > 0.0:
                bar_x = bar_x - (g_t_val / np.linalg.norm(g_grad) ** 2) * g_grad
            bar_x = self.projection_step(bar_x)
            f_grad = (H @ bar_x - y).T @ H
            f_grad = f_grad.reshape(-1, 1)
            bar_x = bar_x - (1 / np.sqrt(t + 1)) * f_grad
            x_pred = self.projection_step(bar_x)
            Preds.append(x_pred.flatten())
        return x_pred, Preds


# ─── AdaGrad ────────────────────────────────────────────────────────

class adv_AdaGrad:
    def __init__(self, x_0, T, assumptions_param):
        self.x_0 = x_0
        self.n = x_0.shape[0]
        self.F, self.G, self.D = assumptions_param
        self.V = 1
        self.G_temp = max(self.F, self.G)
        self.alpha = 1 / (2 * self.G_temp * self.D)
        self.lameda = 1 / (2 * np.sqrt(T))
        self.Q = 0.0

    def _Phi_grad(self, x):
        return self.lameda * np.exp(self.lameda * x)

    def run(self, dataset):
        x_pred = self.x_0.copy()
        Preds = [x_pred.flatten()]
        f_t_grads_norms = 0.0
        Hs, Ys, As, Bs = dataset
        for i in range(T):
            H, y, a, b = Hs[i], Ys[i], As[i], Bs[i]
            g_t = a @ x_pred - b
            g_t_idx = np.argmax(g_t)
            g_t_val = max(g_t[g_t_idx], 0.0)
            self.Q = self.Q + self.alpha * g_t_val
            g_grad = np.zeros((a.shape[1], 1))
            if g_t_val > 0.0:
                g_grad = a[g_t_idx].reshape(-1, 1)
            f_grad = (H @ x_pred - y).T @ H
            f_grad = f_grad.reshape(-1, 1)
            g_grad_scaled = self.alpha * g_grad
            f_grad_scaled = self.alpha * f_grad
            f_hat_t_grad = self.V * f_grad_scaled + self._Phi_grad(self.Q) * g_grad_scaled
            f_t_grads_norms += np.linalg.norm(f_hat_t_grad) ** 2
            eta_t = (np.sqrt(2) * self.D) / (2 * np.sqrt(f_t_grads_norms))
            x_pred = x_pred - eta_t * f_hat_t_grad
            for k in range(x_pred.shape[0]):
                x_pred[k][0] = min(1., max(0., x_pred[k][0]))
            Preds.append(x_pred.flatten())
        return x_pred, Preds


# ─── RECOO ──────────────────────────────────────────────────────────

class adv_RECOO:
    def __init__(self, x_0, T, assumptions_param):
        self.x_0 = x_0
        self.n = x_0.shape[0]
        self.F, self.G, self.D = assumptions_param
        self.G_temp = max(self.F, self.G)
        self.Q = 0.0

    def run(self, dataset):
        x_pred = self.x_0.copy()
        Preds = [x_pred.flatten()]
        Hs, Ys, As, Bs = dataset
        for t in range(T):
            alpha_t = np.sqrt(t + 1)
            eta_t = np.sqrt(t + 1)
            epsilon = 1e-3
            gamma_t = np.power(t, 0.5 * epsilon)
            x_var = cp.Variable((self.n, 1))
            H, y, a, b = Hs[t], Ys[t], As[t], Bs[t]
            g_t = a @ x_pred - b
            g_t_idx = np.argmax(g_t)
            g_t_val = max(g_t[g_t_idx][0], 0.0)
            self.Q = max(self.Q + g_t_val, eta_t)
            f_grad = (H @ x_pred - y).T @ H
            f_grad = f_grad.reshape(-1, 1)
            g_t_fun = a[g_t_idx] @ x_var - b[g_t_idx]
            g_t_tilde = gamma_t * cp.maximum(g_t_fun[0], 0.0)
            constraints = [x_var <= 1, x_var >= 0]
            cost = f_grad.T @ x_var + self.Q * g_t_tilde + alpha_t * cp.norm(x_var - x_pred) ** 2
            prob = cp.Problem(cp.Minimize(cost), constraints)
            prob.solve()
            x_pred = x_var.value.copy()
            Preds.append(x_pred.flatten())
        return x_pred, Preds


# ─── Switch ─────────────────────────────────────────────────────────

class adv_Switch:
    def __init__(self, x_0, T, assumptions_param):
        self.x_0 = x_0
        self.n = x_0.shape[0]
        self.F, self.G, self.D = assumptions_param
        self.G_temp = max(self.F, self.G)
        self.V = 1
        self.beta = 1 / (2 * self.G_temp * self.D)
        self.lameda = 1 / (2 * np.sqrt(T))
        self.alpha = 1 / (2 * self.G_temp * self.D)
        self.Q = 0.0

    def _Phi_grad(self, x):
        return self.lameda * np.exp(self.lameda * x)

    def _projection_step(self, data):
        x_var, A_set, b_set, x = data
        cost = (1 / 2) * cp.norm(x_var - x) ** 2
        constraints = [x_var <= 1, x_var >= 0]
        if A_set.shape[0] > 0:
            constraints.append(A_set @ x_var <= b_set)
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(warm_start=True)
        return x_var.value.copy()

    def run(self, dataset):
        x_pred = self.x_0.copy()
        Preds = [x_pred.flatten()]
        f_t_grads_norms = 0.0
        Hs, Ys, As, Bs = dataset
        x_var = cp.Variable(self.x_0.shape)
        A_set = np.array([]).reshape(0, self.x_0.shape[0])
        b_set = np.array([]).reshape(0, 1)
        i = 0
        while i < T and self.Q <= np.sqrt(T) * np.log(T):
            H, y, A, b = Hs[i], Ys[i], As[i], Bs[i]
            g_t = A @ x_pred - b
            g_t_idx = np.argmax(g_t)
            g_t_val = max(g_t[g_t_idx], 0.0)
            self.Q = self.Q + g_t_val
            f_grad = (H @ x_pred - y).T @ H
            eta_t = self.D / (self.G_temp * np.sqrt(i + 1))
            grad_step = x_pred - eta_t * f_grad
            data_proj = (x_var, A_set, b_set, grad_step)
            projected_w = self._projection_step(data_proj)
            A_t, b_t = A[g_t_idx].reshape(1, -1), b[g_t_idx].reshape(1, -1)
            A_set = np.vstack([A_set, A_t])
            b_set = np.vstack([b_set, b_t])
            data_proj2 = (x_var, A_set, b_set, projected_w)
            x_pred = self._projection_step(data_proj2)
            Preds.append(x_pred.flatten())
            i += 1
        self.Q = 0
        while i < T:
            H, y, a, b = Hs[i], Ys[i], As[i], Bs[i]
            g_t = a @ x_pred - b
            g_t_idx = np.argmax(g_t)
            g_t_val = max(g_t[g_t_idx], 0.0)
            self.Q = self.Q + self.beta * g_t_val
            g_grad = np.zeros((a.shape[1], 1))
            if g_t_val > 0.0:
                g_grad = a[g_t_idx].reshape(-1, 1)
            f_grad = (H @ x_pred - y).T @ H
            f_grad = f_grad.reshape(-1, 1)
            g_grad_scaled = self.alpha * g_grad
            f_grad_scaled = self.alpha * f_grad
            f_hat_t_grad = self.V * f_grad_scaled + self._Phi_grad(self.Q) * g_grad_scaled
            f_t_grads_norms += np.linalg.norm(f_hat_t_grad) ** 2
            eta_t = (np.sqrt(2) * self.D) / (2 * np.sqrt(f_t_grads_norms))
            x_pred = x_pred - eta_t * f_hat_t_grad
            Preds.append(x_pred.flatten())
            i += 1
        return x_pred, Preds


# ─── Frank-Wolfe ────────────────────────────────────────────────────

class adv_FrankWolfe:
    def __init__(self, x_0, T, assumptions_param):
        self.x_0 = x_0
        self.n = x_0.shape[0]
        self.F, self.G, self.D = assumptions_param
        self.G_temp = max(self.F, self.G)
        self.beta = 1 / ((2 ** 6) * self.G_temp * self.D)
        self.lameda = 1 / (2 * (T ** (3 / 4)))
        self.gamma = 1
        self.Q = 0.0

    def _Phi_grad(self, x):
        return self.lameda * np.exp(self.lameda * x)

    def _optimization_step(self, data):
        f_t_grads, x_s_k, eta_t, s_k = data
        x_var = cp.Variable(f_t_grads[0].shape)
        F_grad = -2 * x_s_k
        for f_sup_grad in f_t_grads[s_k - 1:]:
            F_grad += eta_t * f_sup_grad
        cost = 2 * cp.square(cp.norm(x_var)) + F_grad.T @ x_var
        constraints = [x_var <= 1, x_var >= 0]
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve()
        return x_var.value.copy()

    def run(self, dataset):
        x_pred = self.x_0.copy()
        Preds = [x_pred.flatten()]
        f_t_grads = []
        Hs, Ys, As, Bs = dataset
        G_k = 1
        s_k = 1
        x_s_k = x_pred.copy()
        for i in range(T):
            H, y, A, b = Hs[i], Ys[i], As[i], Bs[i]
            g_t = A @ x_pred - b
            g_t_idx = np.argmax(g_t)
            g_t_val = max(g_t[g_t_idx], 0.0)
            self.Q = self.Q + g_t_val
            while G_k < self.beta * self.G_temp * (self.gamma + self._Phi_grad(self.beta * self.Q)):
                G_k = G_k * 2
                s_k = i + 1
                x_s_k = x_pred.copy()
            g_grad = np.zeros((A.shape[1], 1))
            if g_t_val > 0.0:
                g_grad = A[g_t_idx].reshape(-1, 1)
            f_grad = (H @ x_pred - y).T @ H
            f_grad = f_grad.reshape(-1, 1)
            eta_t = self.D / (2 * G_k * T ** (3 / 4))
            f_tilde_grad = self.gamma * self.beta * f_grad + self._Phi_grad(self.beta * self.Q) * self.beta * g_grad
            f_t_grads.append(f_tilde_grad)
            data_opt = (f_t_grads, x_s_k, eta_t, s_k)
            v_pred = self._optimization_step(data_opt)
            sigma = 2 / np.sqrt(i + 1 - s_k + 1)
            x_pred = x_pred + sigma * (v_pred - x_pred)
            Preds.append(x_pred.flatten())
        return x_pred, Preds


# ─── Metric computation ─────────────────────────────────────────────

def compute_cost_and_ccv(dataset, Preds):
    Cost_sum = 0.0
    Cost_arr = []
    CCV_sum = 0.0
    CCV_arr = [0.0]
    CCV_2_sum = 0.0
    CCV_2_arr = [0.0]
    Hs, Ys, As, Bs = dataset
    for i in range(T):
        H, y, A, b = Hs[i], Ys[i], As[i], Bs[i]
        x_pred = Preds[i].reshape(-1, 1)
        Cost_sum += 1 / 2 * np.linalg.norm(H @ x_pred - y) ** 2
        Cost_arr.append(Cost_sum)
        g_t = A @ x_pred - b
        g_t_idx = np.argmax(g_t)
        g_t_val = max(g_t[g_t_idx][0], 0.0)
        CCV_sum += max(g_t_val, 0.0)
        CCV_arr.append(CCV_sum)
        CCV_2_sum += max(g_t_val, 0.0) ** 2
        CCV_2_arr.append(CCV_2_sum)
    return Cost_arr, CCV_arr, CCV_2_arr


# ─── Main experiment ────────────────────────────────────────────────

if __name__ == "__main__":
    S = 50  # number of trials
    n = 10  # problem dimension
    k = 4   # constraint dimension
    T = 100  # horizon

    # Assumptions parameters
    D = np.sqrt(2)
    F = n ** 2
    G = np.sqrt(4 * n)
    assumptions_param = (F, G, D)

    Cost_dict = {'CLASP_I': [], 'CLASP_F': [], 'AdaGrad': [], 'RECOO': [], 'Switch': [], 'FW': []}
    CCV_dict = {'CLASP_I': [], 'CLASP_F': [], 'AdaGrad': [], 'RECOO': [], 'Switch': [], 'FW': []}
    CCV_2_dict = {'CLASP_I': [], 'CLASP_F': [], 'AdaGrad': [], 'RECOO': [], 'Switch': [], 'FW': []}

    start_time = time.time()

    for trial in tqdm(range(S), desc="Trials"):
        dataset = new_dataset(n, k, T)
        x_0 = np.random.uniform(0.0, 1.0, size=n).reshape(-1, 1)

        # CLASP-I
        CLASP_I_alg = CLASP_I(x_0, T, assumptions_param)
        _, Preds = CLASP_I_alg.run(dataset)
        Cost_CLASP_I, CCV_CLASP_I, CCV_2_CLASP_I = compute_cost_and_ccv(dataset, Preds)
        Cost_dict['CLASP_I'].append(Cost_CLASP_I)
        CCV_dict['CLASP_I'].append(CCV_CLASP_I)
        CCV_2_dict['CLASP_I'].append(CCV_2_CLASP_I)

        # AdaGrad
        AdaGrad_alg = adv_AdaGrad(x_0, T, assumptions_param)
        _, Preds = AdaGrad_alg.run(dataset)
        Cost_AdaGrad, CCV_AdaGrad, CCV_2_AdaGrad = compute_cost_and_ccv(dataset, Preds)
        Cost_dict['AdaGrad'].append(Cost_AdaGrad)
        CCV_dict['AdaGrad'].append(CCV_AdaGrad)
        CCV_2_dict['AdaGrad'].append(CCV_2_AdaGrad)

        # CLASP-F
        CLASP_F_alg = CLASP_F(x_0, T, assumptions_param)
        _, Preds = CLASP_F_alg.run(dataset)
        Cost_CLASP_F, CCV_CLASP_F, CCV_2_CLASP_F = compute_cost_and_ccv(dataset, Preds)
        Cost_dict['CLASP_F'].append(Cost_CLASP_F)
        CCV_dict['CLASP_F'].append(CCV_CLASP_F)
        CCV_2_dict['CLASP_F'].append(CCV_2_CLASP_F)

        # RECOO
        RECOO_alg = adv_RECOO(x_0, T, assumptions_param)
        _, Preds = RECOO_alg.run(dataset)
        Cost_RECOO, CCV_RECOO, CCV_2_RECOO = compute_cost_and_ccv(dataset, Preds)
        Cost_dict['RECOO'].append(Cost_RECOO)
        CCV_dict['RECOO'].append(CCV_RECOO)
        CCV_2_dict['RECOO'].append(CCV_2_RECOO)

        # Switch
        Switch_alg = adv_Switch(x_0, T, assumptions_param)
        _, Preds = Switch_alg.run(dataset)
        Cost_Switch, CCV_Switch, CCV_2_Switch = compute_cost_and_ccv(dataset, Preds)
        Cost_dict['Switch'].append(Cost_Switch)
        CCV_dict['Switch'].append(CCV_Switch)
        CCV_2_dict['Switch'].append(CCV_2_Switch)

        # Frank-Wolfe
        FW_alg = adv_FrankWolfe(x_0, T, assumptions_param)
        _, Preds = FW_alg.run(dataset)
        Cost_FW, CCV_FW, CCV_2_FW = compute_cost_and_ccv(dataset, Preds)
        Cost_dict['FW'].append(Cost_FW)
        CCV_dict['FW'].append(CCV_FW)
        CCV_2_dict['FW'].append(CCV_2_FW)

    elapsed = time.time() - start_time
    print(f"\nExperiment completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Convert to arrays
    for alg in Cost_dict:
        Cost_dict[alg] = np.array(Cost_dict[alg])
        CCV_dict[alg] = np.array(CCV_dict[alg])
        CCV_2_dict[alg] = np.array(CCV_2_dict[alg])

    # Report results at T=100
    print("\n" + "=" * 80)
    print("Results at T=100 (50 trials, mean ± 2*std for ~95% CI)")
    print("=" * 80)

    results = {}
    for alg_name in ['CLASP_I', 'CLASP_F', 'AdaGrad', 'RECOO', 'Switch', 'FW']:
        mean_cost = np.mean(Cost_dict[alg_name][:, -1])
        std_cost = 2 * np.std(Cost_dict[alg_name][:, -1])
        mean_ccv1 = np.mean(CCV_dict[alg_name][:, -1])
        std_ccv1 = 2 * np.std(CCV_dict[alg_name][:, -1])
        mean_ccv2 = np.mean(CCV_2_dict[alg_name][:, -1])
        std_ccv2 = 2 * np.std(CCV_2_dict[alg_name][:, -1])

        results[alg_name] = {
            'cumulative_loss': float(mean_cost),
            'cumulative_loss_2std': float(std_cost),
            'CCVT_1': float(mean_ccv1),
            'CCVT_1_2std': float(std_ccv1),
            'CCVT_2': float(mean_ccv2),
            'CCVT_2_2std': float(std_ccv2),
        }

        print(f"\n{alg_name}:")
        print(f"  Cumulative loss = {mean_cost:.2f} ± {std_cost:.2f}")
        print(f"  CCVT,1          = {mean_ccv1:.2f} ± {std_ccv1:.2f}")
        print(f"  CCVT,2          = {mean_ccv2:.2f} ± {std_ccv2:.2f}")

    # Print comparison against rubric targets
    print("\n" + "=" * 80)
    print("Rubric Comparison (CLASP-I):")
    print(f"  Cumulative loss: ours={results['CLASP_I']['cumulative_loss']:.1f}, paper≈608, CI bounds [204.0, 648.4]")
    print(f"  CCVT,1:          ours={results['CLASP_I']['CCVT_1']:.1f}, paper≈63, CI bounds [32.0, 66.1]")
    print(f"  CCVT,2:          ours={results['CLASP_I']['CCVT_2']:.1f}, paper≈126, CI bounds [40.0, 134.6]")

    # Save results to JSON
    output_path = "/repo/results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
