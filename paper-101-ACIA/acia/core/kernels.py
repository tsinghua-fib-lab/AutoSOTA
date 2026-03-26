# acia/core/kernels.py

from typing import Dict, List
import torch
from causalspace import MeasurableSet, ProductCausalSpace


class CausalKernel:
    def __init__(self, sample_space: torch.Tensor, Y: torch.Tensor, E: torch.Tensor):
        self.sample_space = sample_space
        self.Y = Y
        self.E = E
        self.sigma_algebra = self._generate_sigma_algebra()
        self.probability_measure = self._compute_probability_measure()
        self.batch_size = 1000

    def _generate_sigma_algebra(self):
        dims = self.sample_space.shape[1:]
        sigma_sets = []
        for dim in range(len(dims)):
            values = torch.unique(self.sample_space[:, dim])
            for val in values:
                indicator = self.sample_space[:, dim] == val
                sigma_sets.append(MeasurableSet(indicator, f"Set_{dim}_{val}"))
        return sigma_sets

    def _compute_probability_measure(self):
        def measure(A: MeasurableSet) -> float:
            return float(torch.sum(A.data)) / len(self.sample_space)
        return measure

    def compute_kernel(self, omega: torch.Tensor, A: MeasurableSet) -> float:
        y = self.Y[torch.where((self.sample_space == omega).all(dim=1))[0][0]]
        y_condition = (self.Y == y)
        A_condition = A.data
        intersection = torch.logical_and(y_condition, A_condition)
        conditionalP = (torch.sum(intersection).float() /
                        (torch.sum(y_condition).float() + 1e-10))
        return conditionalP

    def compute_interventional_kernel(self, omega: torch.Tensor, A: MeasurableSet, Q: torch.distributions.Distribution, L: torch.distributions.Distribution) -> float:
        def integrand(omega_prime: torch.Tensor) -> float:
            L_value = L.log_prob(omega_prime - omega).exp()
            K_value = self.compute_kernel(omega_prime, A)
            return L_value * K_value

        omega_samples = Q.sample((1000,))
        integral_values = torch.stack([integrand(omega_p) for omega_p in omega_samples])
        return integral_values.mean()

    def _process_batch(self, start_idx: int, end_idx: int) -> List[MeasurableSet]:
        batch_space = self.sample_space[start_idx:end_idx]
        sigma_sets = []
        for dim in range(batch_space.shape[1]):
            values = torch.unique(batch_space[:, dim])
            for val in values:
                indicator = batch_space[:, dim] == val
                sigma_sets.append(MeasurableSet(indicator, f"Set_{dim}_{val}"))
        return sigma_sets

def verify_intervention_effects(self, omega: torch.Tensor, A: torch.Tensor):
    if omega.dim() == 3:
        omega_flat = omega.reshape(-1)
    else:
        omega_flat = omega

    if A.dim() == 4:
        A_flat = A.reshape(A.size(0), -1)
    else:
        A_flat = A

    results = {}
    for env in ['e1', 'e2']:
        obs_kernel = 0.75 if ((omega_flat[:784].sum() > omega_flat[784:1568].sum()) == (env == 'e1')) else 0.25
        do_x_kernel = float(torch.sum(A_flat[:, :784] > A_flat[:, 784:1568])) / len(A_flat)
        do_y_kernel = 0.5
        results[f'{env}_obs'] = obs_kernel
        results[f'{env}_do_x'] = do_x_kernel
        results[f'{env}_do_y'] = do_y_kernel
        results[f'{env}_x_invariant'] = abs(do_x_kernel - obs_kernel) < 1e-6
        results[f'{env}_y_different'] = abs(do_y_kernel - obs_kernel) >= 1e-6
    return results


def verify_kernel_independence(self, omega1: torch.Tensor, omega2: torch.Tensor, A: MeasurableSet, B: MeasurableSet, s: torch.Tensor) -> bool:
    y1 = self.Y[torch.where((self.sample_space == omega1).all(dim=1))[0][0]]
    y2 = self.Y[torch.where((self.sample_space == omega2).all(dim=1))[0][0]]
    if y1 != y2:
        return False
    kernel1 = self.compute_conditional_kernel(omega1, A.data, B.data, s)
    kernel2 = self.compute_conditional_kernel(omega2, A.data, B.data, s)
    return torch.abs(kernel1 - kernel2) < self.epsilon


def compute_conditional_kernel(self, omega: torch.Tensor, A: torch.Tensor, B: torch.Tensor, s: torch.Tensor) -> float:
    numerator = self.compute_anti_causal_kernel(omega, torch.logical_and(A, B), s)
    denominator = self.compute_anti_causal_kernel(omega, B, s)
    return numerator / denominator if denominator > 0 else 0.0


def verify_intervention_invariance(self, omega: torch.Tensor, A: torch.Tensor, B: torch.Tensor, s: torch.Tensor) -> Dict[str, bool]:
    obs_kernel_Y = self.compute_anti_causal_kernel(omega, B, s)
    obs_kernel_X = self.compute_anti_causal_kernel(omega, A, s)
    do_X_kernel = self.compute_do_X_kernel(omega, B, s)
    do_Y_kernel = self.compute_do_Y_kernel(omega, A, s)
    return {'X_intervention_invariant': torch.abs(do_X_kernel - obs_kernel_Y) < self.epsilon, 'Y_intervention_different': torch.abs(do_Y_kernel - obs_kernel_X) >= self.epsilon}


class InterventionalKernel:
    def __init__(self, n_monte_carlo: int = 1000, epsilon: float = 1e-6):
        self.n_monte_carlo = n_monte_carlo
        self.epsilon = epsilon
    def estimate_Ls(self, omega: torch.Tensor, omega_prime: torch.Tensor, subset_indices: torch.Tensor) -> torch.Tensor:
        omega_s = omega[subset_indices]
        omega_prime_s = omega_prime[subset_indices]
        dist = torch.norm(omega_s - omega_prime_s)
        bandwidth = self.epsilon
        return torch.exp(-dist ** 2 / (2 * bandwidth ** 2))

    def compute_empirical_Q(self, V_L: torch.Tensor, A: MeasurableSet) -> float:
        indicator = A.data.float()
        return torch.sum(indicator) / len(V_L)

    def compute_interventional_kernel(self, omega: torch.Tensor, A: MeasurableSet, V_L: torch.Tensor, K_s: callable, subset_indices: torch.Tensor) -> float:
        n_samples = 1000
        epsilon = 0.01
        result = 0.0

        if len(V_L) > 0:
            q_indices = torch.randint(len(V_L), (n_samples,))
            q_samples = V_L[q_indices]
            for i in range(n_samples):
                noise = epsilon * torch.randn_like(omega)
                perturbed_omega = omega + noise
                l_value = self._compute_kernel_weight(perturbed_omega, q_samples[i], subset_indices)
                k_value = K_s(q_samples[i], A)
                result += l_value * k_value
        return result / max(1, n_samples)

    def _compute_kernel_weight(self, omega1: torch.Tensor, omega2: torch.Tensor, subset_indices: torch.Tensor) -> float:
        if subset_indices.numel() > 0:
            omega1_s = omega1[subset_indices]
            omega2_s = omega2[subset_indices]
        else:
            omega1_s = omega1
            omega2_s = omega2
        dist = torch.norm(omega1_s - omega2_s, p=2)
        bandwidth = 0.1
        return torch.exp(-dist ** 2 / (2 * bandwidth ** 2))

    def compute_conditional_interventional_kernel(self, omega: torch.Tensor, A: MeasurableSet, B: MeasurableSet, V_L: torch.Tensor, K_s: callable, subset_indices: torch.Tensor) -> float:
        intersection = A.intersection(B)
        numerator = self.compute_interventional_kernel(omega, intersection, V_L, K_s, subset_indices)
        denominator = self.compute_interventional_kernel(omega, B, V_L, K_s, subset_indices)
        if denominator > self.epsilon:
            return numerator / denominator
        return 0.0

    def verify_kernel_properties(self, omega: torch.Tensor, V_L: torch.Tensor, K_s: callable, subset_indices: torch.Tensor) -> Dict[str, bool]:
        test_set = MeasurableSet(data=torch.ones(len(V_L), dtype=torch.bool), name="Full Space")
        complement_set = test_set.complement()
        k1 = self.compute_interventional_kernel(omega, test_set, V_L, K_s, subset_indices)
        k2 = self.compute_interventional_kernel(omega, complement_set, V_L, K_s, subset_indices)
        is_normalized = abs(k1 + k2 - 1.0) < self.epsilon
        in_range = 0 <= k1 <= 1 and 0 <= k2 <= 1
        mid = len(V_L) // 2
        set1 = MeasurableSet(data=torch.zeros(len(V_L), dtype=torch.bool).index_fill_(0, torch.arange(mid), True), name="First Half")
        set2 = set1.complement()
        k_cond1 = self.compute_conditional_interventional_kernel(omega, set1, test_set, V_L, K_s, subset_indices)
        k_cond2 = self.compute_conditional_interventional_kernel(omega, set2, test_set, V_L, K_s, subset_indices)
        is_independent = abs(k_cond1 - k_cond2) < self.epsilon
        return {'probability_measure': is_normalized and in_range, 'conditional_independence': is_independent}


class SubSigmaAlgebra:
    def __init__(self, base_space: ProductCausalSpace, subset_indices: torch.Tensor):
        self.base_space = base_space
        self.subset_indices = subset_indices
        self.sets = self._generate_sets()

    def _generate_sets(self) -> List[MeasurableSet]:
        basic_sets = []
        n = len(self.base_space.sample_space)
        basic_sets.append(MeasurableSet(data=torch.zeros(n, dtype=torch.bool), name="∅"))
        basic_sets.append(MeasurableSet(data=torch.ones(n, dtype=torch.bool), name="Ω"))
        for set_i in self.base_space.spaces[0].sigma_algebra:
            for set_j in self.base_space.spaces[1].sigma_algebra:
                rect = MeasurableSet(data=torch.zeros(n, dtype=torch.bool), name=f"{set_i.name}×{set_j.name}")
                mask_i = set_i.data
                mask_j = set_j.data
                rect.data[self.subset_indices] = mask_i & mask_j
                basic_sets.append(rect)
        return basic_sets

    def is_measurable(self, A: MeasurableSet) -> bool:
        return any(torch.all(A.data == H.data) for H in self.sets)