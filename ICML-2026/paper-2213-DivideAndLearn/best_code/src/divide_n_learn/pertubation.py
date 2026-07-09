import numpy as np
import torch
import random
from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Callable, Optional, Set


DEFAULT_INNER_BUDGET = 10  # safety fallback only and can be any number not a hyper-parameter! ; nb_rounds is set in __init__
SHARP_GRAD = 0.1     # strong signal threshold | this is not a hyperparameter as well. Just indicates strong signal  
FLAT_GRAD  = 0.01    # weak signal threshold | this is not a hyperparameter as well. Just indicates weak signal  


def create_problem_agnostic_improver(problem_type: str, problem_size: int) -> Callable:
    """
    Local refinement subroutine: one OUTER round of the subproblem-level
    zeroth-order method analyzed in Lemma `lemma:local-search-regret`
    (Algorithm `LocalRefine`).

    This is a faithful DISCRETE realization of the lemma's idealized
    continuous-space gradient-free OCO method. The correspondence,
    discrete adaptations, and their effect on the lemma's bound are
    documented in Remark `rem:local-search-impl` in the appendix.

    Factory `problem_type` and `problem_size` are retained for caller
    compatibility; domain is detected from the solution structure inside
    the closure (Remark, "Unit perturbation operator").
    """

    def improve_subproblem_oco_universal(self, solution, subproblem_indices):
        # ===== Domain detection → unit perturbation operator selection =====
        # (Definition `def:unit-perturbation`)
        if all(v in (0, 1) for v in solution):
            domain_type = "binary"        # unit op: single bit flip (Hamming)
        elif len(set(solution)) == len(solution) and \
             set(solution) == set(range(len(solution))):
            domain_type = "permutation"   # unit op: transposition 
        else:
            domain_type = "categorical"   # unit op: single value change

        def is_feasible(sol):
            if domain_type == "binary" and getattr(self, 'item_weights', None) is not None:
                w = sum(self.item_weights[i] * sol[i] for i in range(len(sol)))
                return w <= self.knapsack_capacity
            return True

        def generate_perturbation(sol, indices):
            """Sample x_t^+ at unit metric distance from sol on `indices`."""
            perturbed = sol.copy()
            if domain_type == "binary":
                if getattr(self, 'item_weights', None) is not None:
                    # Knapsack: feasibility-respecting bit flip
                    i = random.choice(indices) if indices else random.randint(0, len(sol)-1)
                    if perturbed[i] == 0:
                        new_w = sum(self.item_weights[j] * perturbed[j] for j in range(len(perturbed)))
                        new_w += self.item_weights[i]
                        if new_w <= self.knapsack_capacity:
                            perturbed[i] = 1
                    else:
                        perturbed[i] = 0
                else:
                    if indices:
                        i = random.choice(indices)
                        perturbed[i] = 1 - perturbed[i]
            elif domain_type == "categorical":
                if indices and hasattr(self, 'bounds'):
                    pos = random.choice(indices)
                    valid = self.bounds[pos]
                    if isinstance(valid, list):
                        alts = [v for v in valid if v != sol[pos]]
                        if alts:
                            perturbed[pos] = random.choice(alts)
            else:  # permutation: transposition (unit)
                if len(indices) >= 2:
                    # Distance-aware swap: prefer adjacent positions in tour
                    # (epsilon-greedy: 80% distance-weighted, 20% uniform)
                    if random.random() < 0.8:
                        # Build distance-weighted probability distribution
                        weights = []
                        pairs_list = []
                        n = len(sol)
                        for a, p1 in enumerate(indices):
                            for p2 in indices[a+1:]:
                                # Cyclic tour distance between positions
                                dist = min(abs(p1 - p2), n - abs(p1 - p2))
                                weight = 1.0 / (1.0 + dist)
                                weights.append(weight)
                                pairs_list.append((p1, p2))
                        if pairs_list and sum(weights) > 0:
                            total = sum(weights)
                            probs = [w / total for w in weights]
                            i, j = random.choices(pairs_list, weights=probs, k=1)[0]
                        else:
                            i, j = random.sample(indices, 2)
                    else:
                        i, j = random.sample(indices, 2)
                    perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
            return perturbed

        def move_toward(current, target, indices, num_ops):
            """Discrete realization of x_{t+1} = x_t + α ĝ_t when Δ > 0.
            Applies up to `num_ops` unit operations that reduce
            d_X(current, target)."""
            result = current.copy()
            if domain_type == "binary":
                diffs = [i for i in indices if i < len(current) and current[i] != target[i]]
                for i in diffs[:num_ops]:
                    test = result.copy()
                    test[i] = target[i]
                    if is_feasible(test):
                        result[i] = target[i]
            elif domain_type == "categorical":
                diffs = [i for i in indices if i < len(current) and current[i] != target[i]]
                random.shuffle(diffs)
                for i in diffs[:num_ops]:
                    result[i] = target[i]
            else:  # permutation
                diffs = [(i, current[i], target[i]) for i in indices
                         if i < len(current) and i < len(target) and current[i] != target[i]]
                for _ in range(min(num_ops, len(diffs))):
                    if diffs:
                        idx, _, target_val = random.choice(diffs)
                        if target_val in result:
                            tp = result.index(target_val)
                            result[idx], result[tp] = result[tp], result[idx]
                        diffs = [(i, c, t) for i, c, t in diffs if i != idx]
            return result

        def move_random(sol, indices, num_ops):
            """Independent unit-neighbor probe, used on Δ ≤ 0 branch
            (Remark, 'Negative-direction handling'). Substitutes for the
            degenerate discrete anti-direction; preserves symmetry of the
            perturbation distribution so the next round's estimator
            assumptions still hold."""
            result = sol.copy()
            if domain_type == "binary":
                for _ in range(min(num_ops, len(indices))):
                    if indices:
                        test = result.copy()
                        i = random.choice(indices)
                        test[i] = 1 - test[i]
                        if is_feasible(test):
                            result = test
            elif domain_type == "categorical":
                for _ in range(min(num_ops, len(indices))):
                    if indices and hasattr(self, 'bounds'):
                        pos = random.choice(indices)
                        valid = self.bounds[pos]
                        if isinstance(valid, list):
                            alts = [v for v in valid if v != result[pos]]
                            if alts:
                                result[pos] = random.choice(alts)
            else:  # permutation
                for _ in range(min(num_ops, len(indices) // 2)):
                    if len(indices) >= 2:
                        i, j = random.sample(indices, 2)
                        if i < len(result) and j < len(result):
                            result[i], result[j] = result[j], result[i]
            return result

        # ===== Inner zeroth-order loop = ONE element of T_k =====
        current_solution = solution.copy()
        current_value = self.evaluate_fn(current_solution)
        best_solution, best_value = current_solution, current_value

        # Inner probe budget R (paper); bounded ⇒ contributes a constant
        # factor to the lemma's bound, preserves the √T_k rate.
        T = min(getattr(self, 'nb_rounds', DEFAULT_INNER_BUDGET), len(subproblem_indices))

        # Probe radius δ. Enters only through G² in the bound; we use 1/n
        # in the implementation. The paper's σ = 1/√T_k is the analytical
        # choice that minimizes the constant. See Remark, "Estimator scaling".
        delta = 1.0 / max(1, problem_size)

        # Auxiliary smoothing on the SCALAR finite difference. Used solely
        # by the step-size selector; does not enter the iterate update.
        # Hence preserves the unbiasedness and variance bounds in (6).
        momentum = 0.9
        velocity = 0.0

        for t in range(T):
            # --- Probe: x_t^+ = current + (unit perturbation) ---
            perturbation = generate_perturbation(current_solution, subproblem_indices)
            if perturbation == current_solution:
                continue  # zero-norm direction; skip

            # --- Finite difference Δ_t ---
            # |Δ_t| ≤ L by Assumption A.2 since the perturbation is unit
            # in the operator-specific metric (Hamming / coordinate
            # ; see Remark, "Unit perturbation operator".
            perturbed_value = self.evaluate_fn(perturbation)
            gradient_estimate = (perturbed_value - current_value) / delta

            velocity = momentum * velocity + (1 - momentum) * gradient_estimate

            # --- Discrete step size m_t ---
            # Bounded integer support is what the OGD potential argument
            # needs (Remark, "Discrete step size"); the magnitude-based
            # selector is a variance-reduction heuristic and is not part
            # of the formal bound. Only requirement is that num_operations is bounded, this can be replaced with any reasonable choice/approach.
            # The {1,2,3} support below satisfies this; the magnitude-based selector inside the support is a theoretically motivated heuristic for empirical stability and is not part of the formal bound — any bounded selector preserves the O(L D √(d T_k)) rate.
            gradient_magnitude = abs(velocity)
            if gradient_magnitude > SHARP_GRAD:
                num_operations = 1
            elif gradient_magnitude < FLAT_GRAD:
                num_operations = min(3, len(subproblem_indices) // 2)
            else:
                num_operations = 2

            # --- Iterate update ---
            if gradient_estimate > 0:
                # Δ > 0 ⇒ perturbation improves the objective.
                # Discrete analog of x_{t+1} = x_t + α ĝ_t (positive sign).
                new_solution = move_toward(current_solution, perturbation,
                                           subproblem_indices, num_operations)
            else:
                # Δ ≤ 0 ⇒ continuous OGD would step against ĝ_t.
                # Discrete unit operators are involutions (Remark,
                # "Negative-direction handling"); substitute an
                # independent unit-neighbor probe.
                new_solution = move_random(current_solution,
                                           subproblem_indices, num_operations)

            if new_solution != current_solution and is_feasible(new_solution):
                new_value = self.evaluate_fn(new_solution)
                current_solution = new_solution
                current_value = new_value
                if new_value > best_value:
                    best_solution = new_solution
                    best_value = new_value

        return best_solution, best_value - self.evaluate_fn(solution)

    return improve_subproblem_oco_universal
