# acia/core/algorithms.py

from causalspace import MeasurableSet
from typing import Dict, List, Tuple, Callable, Set
import torch
import torch.nn as nn
from causalkernel import CausalKernel
from cvxopt import matrix, solvers
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from causalspace import MeasurableSpace, CausalSpace, ProductCausalSpace
from causalkernel import CausalKernel
from anticausal import *
from typing import Dict, List, Tuple, Callable

class CausalDynamics:
    def __init__(self):
        self.causal_spaces = {}
        self.product_space = None
        self.empirical_measure = None
        self.interventional_kernel = None

    def create_causal_space(self, data: torch.Tensor, labels: torch.Tensor, env: str) -> Tuple[torch.Tensor, List[MeasurableSet],
    Callable, CausalKernel]:
        sample_space = data
        kernel = CausalKernel(sample_space, labels, torch.full_like(labels, float(env == 'e2')))
        sigma_algebra = kernel._generate_sigma_algebra()
        probability_measure = kernel._compute_probability_measure()
        return sample_space, sigma_algebra, probability_measure, kernel

    def compute_empirical_measure(self, V_L: torch.Tensor) -> Callable:
        def Q(A: MeasurableSet) -> float:
            indicator = A.data
            return float(torch.sum(indicator)) / len(V_L)
        return Q

    def compute_interventional_kernel(self, omega: torch.Tensor, A: MeasurableSet, s: Set[str]) -> float:
        kernels = [self.causal_spaces[env][3] for env in s]
        measures = [self.causal_spaces[env][2] for env in s]
        kernel_values = []
        for k, p in zip(kernels, measures):
            k_value = k.compute_kernel(omega, A)
            p_value = p(A)
            kernel_values.append(k_value * p_value)
        return sum(kernel_values) / len(kernel_values)

    def construct_low_level_representation(self, V_L: torch.Tensor, envs: Dict[str, Tuple[torch.Tensor, torch.Tensor]]):
        for env, (data, labels) in envs.items():
            self.causal_spaces[env] = self.create_causal_space(data, labels, env)
        self.empirical_measure = self.compute_empirical_measure(V_L)
        return (V_L,
                self.empirical_measure,
                lambda omega, A, s: self.compute_interventional_kernel(omega, A, s))

class LowLevelEncoder(nn.Module):
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim))

    def forward(self, x):
        return self.encoder(x)

@dataclass
class HighLevelRepresentation:
    V_H: torch.Tensor
    k_H: 'HighLevelKernel'

class DynamicsFunction(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, latent_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class HighLevelKernel:
    def __init__(self, dynamics_func: DynamicsFunction):
        self.tau = dynamics_func
        self.weights = {}

    def compute_kernel(self, Z_L: torch.Tensor, A: torch.Tensor, s_k: int) -> torch.Tensor:
        V_L = self.tau(Z_L)
        return self._compute_kernel_integral(V_L, A, s_k)

    def _compute_kernel_integral(self, V_L: torch.Tensor, A: torch.Tensor, s_k: int) -> torch.Tensor:
        n_samples = 1000
        samples = torch.randn_like(V_L.unsqueeze(0).repeat(n_samples, 1))
        kernel_values = torch.zeros(n_samples)
        for i in range(n_samples):
            kernel_values[i] = self._evaluate_interventional_kernel(samples[i], A, s_k)
        return kernel_values.mean()

class CausalAbstraction:
    def __init__(self, phi_L: List[Tuple[torch.Tensor, Callable, Callable]]):
        self.phi_L = phi_L
        self.dynamics = self._construct_dynamics_function()
        self.high_level_kernel = None

    def _construct_dynamics_function(self) -> Callable:
        def tau(v_l: torch.Tensor) -> torch.Tensor:
            measurable_sets = [MeasurableSet(v_l == v, f"set_{i}")
                               for i, v in enumerate(torch.unique(v_l))]
            pushforward = torch.zeros_like(v_l)
            for set_idx, mset in enumerate(measurable_sets):
                for phi in self.phi_L:
                    V_L, Q, k_s = phi
                    pushforward[mset.data] = Q(mset)
            return pushforward
        return tau

    def compute_high_level_kernel(self, Z_L: torch.Tensor, A: MeasurableSet, s_k: int) -> float:
        V_L, Q, k_s = self.phi_L[s_k]
        tau_inverse = self.dynamics(Z_L)
        integral = 0.0
        measurable_sets = self._generate_partition(A)
        for mset in measurable_sets:
            kernel_value = k_s(tau_inverse, mset, {s_k})
            integral += kernel_value * Q(mset)
        return integral

    def _generate_partition(self, A: MeasurableSet) -> List[MeasurableSet]:
        indicator = A.data
        unique_values = torch.unique(indicator)
        return [MeasurableSet(indicator == v, f"partition_{v}")
                for v in unique_values]

    def optimize_weights(self) -> Dict[int, float]:
        n_datasets = len(self.phi_L)
        P = torch.zeros((n_datasets, n_datasets))
        for i in range(n_datasets):
            for j in range(n_datasets):
                for A in self._generate_base_sets():
                    k_i = self.compute_high_level_kernel(self.phi_L[i][0], A, i)
                    k_j = self.compute_high_level_kernel(self.phi_L[j][0], A, j)
                    P[i, j] += (k_i - k_j) ** 2
        P = matrix(P.numpy())
        q = matrix(torch.zeros(n_datasets).numpy())
        G = matrix(-torch.eye(n_datasets).numpy())
        h = matrix(torch.zeros(n_datasets).numpy())
        A = matrix(torch.ones(1, n_datasets).numpy())
        b = matrix([1.0])
        solution = solvers.qp(P, q, G, h, A, b)
        optimal_weights = torch.tensor(solution['x']).squeeze()
        return {k: w.item() for k, w in enumerate(optimal_weights)}

    def _generate_base_sets(self) -> List[MeasurableSet]:
        base_sets = []
        for V_L, _, _ in self.phi_L:
            values = torch.unique(V_L)
            for v in values:
                base_sets.append(MeasurableSet(V_L == v, f"base_{v}"))
        return base_sets

    def construct_high_level_representation(self) -> Tuple[torch.Tensor, Callable]:
        optimal_weights = self.optimize_weights()
        V_H = sum(w * phi[0] for w, phi in zip(optimal_weights.values(), self.phi_L))
        def k_H(omega: torch.Tensor, A: MeasurableSet) -> float:
            return sum(w * self.compute_high_level_kernel(omega, A, k)
                       for k, w in optimal_weights.items())
        return V_H, k_H


class InterventionalKernel:
    def __init__(self, dataset_e1: ColoredMNIST, dataset_e2: ColoredMNIST):
        self.e1_data = dataset_e1
        self.e2_data = dataset_e2
        self.base_kernel = ColoredMNISTKernel(dataset_e1, dataset_e2)

    def compute_do_X_kernel(self, omega: torch.Tensor, A: torch.Tensor, env: str) -> float:
        kernel = self.base_kernel.kernel_e1 if env == 'e1' else self.base_kernel.kernel_e2
        omega_flat = omega.reshape(-1)
        red_sum = omega_flat[:784].sum()
        label_idx = (kernel.sample_space @ omega_flat.float()) / (kernel.sample_space @ kernel.sample_space[0])
        label = kernel.Y[label_idx.argmax()]
        return float(torch.sum(kernel.Y == label)) / len(kernel.Y)

    def compute_do_Y_kernel(self, omega: torch.Tensor, A: torch.Tensor, env: str) -> float:
        kernel = self.base_kernel.kernel_e1 if env == 'e1' else self.base_kernel.kernel_e2
        if env == 'e1':
            p_red = 0.5
        else:
            p_red = 0.5
        omega_flat = omega.reshape(-1)
        is_red = omega_flat[:784].sum() > omega_flat[784:1568].sum()
        A_flat = A.reshape(A.size(0), -1)
        A_red_sum = A_flat[:, :784].sum(1)
        A_green_sum = A_flat[:, 784:1568].sum(1)
        A_is_red = A_red_sum > A_green_sum
        return p_red if is_red == A_is_red.any() else (1 - p_red)

    def verify_intervention_effects(self, omega: torch.Tensor, A: torch.Tensor):
        results = {}
        for env in ['e1', 'e2']:
            obs_kernel = self.base_kernel.compute_environment_kernel(omega, A, env)
            do_x_kernel = self.compute_do_X_kernel(omega, A, env)
            do_y_kernel = self.compute_do_Y_kernel(omega, A, env)
            results[f'{env}_obs'] = obs_kernel
            results[f'{env}_do_x'] = do_x_kernel
            results[f'{env}_do_y'] = do_y_kernel
            results[f'{env}_x_invariant'] = abs(do_x_kernel - obs_kernel) < 1e-6
            results[f'{env}_y_different'] = abs(do_y_kernel - obs_kernel) >= 1e-6
        return results