# acia/core/spaces.py
from typing import List, Set

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd
import torch
import numpy as np

from measuretheory import MeasurableSet


class CausalSpace:
    def __init__(self, sample_space: torch.Tensor, index_set: torch.Tensor):
        self.sample_space = sample_space
        self.index_set = index_set
        self.H_ei = self._generate_sigma_algebra()
        self.P_ei = self._compute_probability_measure()
        self.K_ei = self._init_causal_mechanism()

    def _generate_sigma_algebra(self) -> List[MeasurableSet]:
        sigma_sets = []
        for t in self.index_set:
            E_t = self.sample_space[:, t]
            for value in torch.unique(E_t):
                indicator = (E_t == value)
                sigma_sets.append(MeasurableSet(indicator, f"E_{t}_{value}"))
        n_base = len(sigma_sets)
        for i in range(n_base):
            for j in range(i + 1, n_base):
                sigma_sets.append(sigma_sets[i].union(sigma_sets[j]))
                sigma_sets.append(sigma_sets[i].intersection(sigma_sets[j]))
        return sigma_sets

    def _compute_probability_measure(self) -> callable:
        def P_ei(A: MeasurableSet) -> float:
            if not self.is_measurable(A):
                raise ValueError("Set is not measurable")
            return float(A.data.sum()) / len(self.sample_space)
        return P_ei

    def _init_causal_mechanism(self) -> callable:
        def K_ei(omega: torch.Tensor, A: MeasurableSet) -> float:
            y = self.get_Y_component(omega)
            e = self.get_E_component(omega)
            y_mask = (self.get_Y_component(self.sample_space) == y)
            e_mask = (self.get_E_component(self.sample_space) == e)
            conditional_mask = y_mask & e_mask
            if conditional_mask.sum() == 0:
                return 0.0
            return float((conditional_mask & A.data).sum()) / float(conditional_mask.sum())
        return K_ei

    def is_measurable(self, A: MeasurableSet) -> bool:
        return any((A.data == H.data).all() for H in self.H_ei)

    def get_Y_component(self, omega: torch.Tensor) -> torch.Tensor:
        return omega[..., -1]

    def get_E_component(self, omega: torch.Tensor) -> torch.Tensor:
        return omega[..., -2]


class ProductCausalSpace:
    def __init__(self, spaces: List[CausalSpace]):
        self.spaces = spaces
        self.sample_space = self._product_sample_space()
        self.sigma_algebra = self._product_sigma_algebra()
        self.probability = self._product_probability()
        self.kernel = self._product_kernel()

    def _product_sample_space(self) -> torch.Tensor:
        return torch.cartesian_prod(*[space.sample_space for space in self.spaces])

    def _product_sigma_algebra(self) -> List[MeasurableSet]:
        product_sets = []
        for sets in zip(*[space.H_ei for space in self.spaces]):
            indicator = torch.ones(len(self.sample_space), dtype=torch.bool)
            for set_i in sets:
                indicator &= set_i.data
            product_sets.append(MeasurableSet(indicator, "× ".join(s.name for s in sets)))

        return product_sets

    def _product_probability(self) -> callable:
        def P(A: MeasurableSet) -> float:
            if not self.is_measurable(A):
                raise ValueError("Set is not measurable in product space")
            return float(A.data.sum()) / len(self.sample_space)
        return P

    def _product_kernel(self) -> callable:
        def K(omega: torch.Tensor, A: MeasurableSet, s: Set[int]) -> float:
            if not s:
                return self.probability(A)
            result = 1.0
            for idx, space in enumerate(self.spaces):
                if idx in s:
                    omega_idx = self._get_component(omega, idx)
                    A_idx = self._project_set(A, idx)
                    k_value = space.K_ei(omega_idx, A_idx)
                    result *= k_value
            return result
        return K

    def _get_component(self, omega: torch.Tensor, idx: int) -> torch.Tensor:
        dim_sizes = [space.sample_space.shape[1] for space in self.spaces]
        start_idx = sum(dim_sizes[:idx])
        end_idx = start_idx + dim_sizes[idx]
        return omega[start_idx:end_idx]

    def _project_set(self, A: MeasurableSet, idx: int) -> MeasurableSet:
        space = self.spaces[idx]
        indicator = torch.zeros(len(space.sample_space), dtype=torch.bool)
        for i, point in enumerate(space.sample_space):
            full_point = self._embed_component(point, idx)
            if torch.any(torch.all(full_point == self.sample_space[A.data], dim=1)):
                indicator[i] = True
        return MeasurableSet(indicator, f"proj_{idx}({A.name})")

    def _embed_component(self, component: torch.Tensor, idx: int) -> torch.Tensor:
        full_point = torch.zeros_like(self.sample_space[0])
        dim_sizes = [space.sample_space.shape[1] for space in self.spaces]
        start_idx = sum(dim_sizes[:idx])
        end_idx = start_idx + dim_sizes[idx]
        full_point[start_idx:end_idx] = component
        return full_point

    def is_measurable(self, A: MeasurableSet) -> bool:
        return any((A.data == H.data).all() for H in self.sigma_algebra)

    def verify_properties(self) -> bool:
        for A in self.sigma_algebra:
            assert A.complement() in self.sigma_algebra
        empty_set = MeasurableSet(torch.zeros(len(self.sample_space), dtype=torch.bool), "∅")
        full_set = MeasurableSet(torch.ones(len(self.sample_space), dtype=torch.bool), "Ω")
        assert self.probability(empty_set) == 0
        assert abs(self.probability(full_set) - 1.0) < 1e-6
        return True


class SubSigmaAlgebra:
    def __init__(self, product_space, subset_indices):
        self.product_space = product_space
        self.subset_indices = subset_indices
        self.sets = self._generate_sets()
        self._verify_closure_properties()

    def _generate_sets(self):
        base_sets = []
        n_samples = len(self.product_space.sample_space)
        empty_set = MeasurableSet(torch.zeros(n_samples, dtype=torch.bool), "∅")
        full_set = MeasurableSet(torch.ones(n_samples, dtype=torch.bool), "Ω")
        base_sets = [empty_set, full_set]
        component_spaces = [self.product_space.spaces[i] for i in self.subset_indices]
        component_sigma_algebras = [space.H_ei for space in component_spaces]
        from itertools import product
        for sets in product(*component_sigma_algebras):
            indicator = torch.ones(n_samples, dtype=torch.bool)
            for idx, set_i in zip(self.subset_indices, sets):
                mapped_indicator = self._map_indicator_to_product_space(set_i.data, idx)
                indicator &= mapped_indicator
            rectangle = MeasurableSet(indicator, "×".join(s.name for s in sets))
            base_sets.append(rectangle)
        sets = self._ensure_closure(base_sets)
        return sets

    def _map_indicator_to_product_space(self, indicator, component_idx):
        mapped = torch.zeros(len(self.product_space.sample_space), dtype=torch.bool)
        for i, sample in enumerate(self.product_space.sample_space):
            component_value = sample[component_idx]
            for j, value in enumerate(self.product_space.spaces[component_idx].sample_space):
                if value.item() == component_value.item() and indicator[j]:
                    mapped[i] = True
                    break
        return mapped

    def _ensure_closure(self, base_sets):
        result_sets = base_sets.copy()
        current_size = 0
        while current_size < len(result_sets):
            current_size = len(result_sets)
            for i in range(current_size):
                complement = result_sets[i].complement()
                if not any((complement.data == s.data).all() for s in result_sets):
                    result_sets.append(complement)

                for j in range(i + 1, current_size):
                    union = result_sets[i].union(result_sets[j])
                    if not any((union.data == s.data).all() for s in result_sets):
                        result_sets.append(union)
                    intersection = result_sets[i].intersection(result_sets[j])
                    if not any((intersection.data == s.data).all() for s in result_sets):
                        result_sets.append(intersection)
        return result_sets

    def _verify_closure_properties(self):
        assert any(not s.data.any() for s in self.sets), "Missing empty set"
        for s in self.sets:
            complement = s.complement()
            assert any((complement.data == t.data).all() for t in self.sets), f"Missing complement of {s.name}"

        for i in range(len(self.sets)):
            for j in range(i + 1, len(self.sets)):
                union = self.sets[i].union(self.sets[j])
                assert any((union.data == s.data).all() for s in
                           self.sets), f"Missing union of {self.sets[i].name} and {self.sets[j].name}"

    def is_measurable(self, A):
        return any((A.data == s.data).all() for s in self.sets)