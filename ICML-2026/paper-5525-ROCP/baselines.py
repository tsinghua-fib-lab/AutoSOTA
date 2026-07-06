"""
Baseline methods for conformal decision-making.

Currently includes a Risk-Averse Calibration (RAC) baseline that mirrors the
utilities-based implementation in Risk-Averse-Calibration/Covid_EXP.ipynb.
"""

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from numba import jit
import torch
import torch.optim as optim


class RACBaseline:
    """
    RAC baseline for discrete labels using a utility matrix U[y, a].
    """

    def __init__(self, actions: Sequence[object], utility_matrix: Sequence[Sequence[float]]) -> None:
        self.actions: List[object] = list(actions)
        self.action_to_index: Dict[object, int] = {a: i for i, a in enumerate(self.actions)}
        self.utility_matrix = np.asarray(utility_matrix, dtype=float)
        if self.utility_matrix.ndim != 2:
            raise ValueError("utility_matrix must be a 2D array [num_labels, num_actions].")
        if self.utility_matrix.shape[1] != len(self.actions):
            raise ValueError("utility_matrix second dimension must match number of actions.")

    def _utility(self, action: object, true_label: int) -> float:
        a_idx = self.action_to_index[action]
        return float(self.utility_matrix[true_label, a_idx])

    def hbtheta_and_arg(self, t: float, x_probs: Sequence[float]) -> Tuple[float, object]:
        """
        Compute h_btheta(x,t) and argmax action (utilities-based).
        """
        x_probs = np.asarray(x_probs, dtype=float)
        num_labels = len(x_probs)
        labels = np.arange(num_labels)

        best_u = -np.inf
        best_action = None

        for action in self.actions:
            utilities_for_a = np.array([self._utility(action, y) for y in labels], dtype=float)
            unique_util_vals = np.unique(utilities_for_a)
            unique_util_vals = np.sort(unique_util_vals)[::-1]

            candidate_u = -np.inf
            for u_candidate in unique_util_vals:
                mask = utilities_for_a >= u_candidate
                sum_prob = np.sum(x_probs[mask])
                if sum_prob >= t:
                    candidate_u = u_candidate
                    break

            if candidate_u > best_u:
                best_u = candidate_u
                best_action = action

        if best_action is None:
            raise RuntimeError("Failed to compute hbtheta_and_arg.")
        return float(best_u), best_action

    def compute_g_hat(self, x_probs: Sequence[float], beta: float) -> float:
        """
        Compute g_hat(x, beta) = argmax_s [ beta*s + h_btheta(x, s) ].
        """
        x_probs = np.asarray(x_probs, dtype=float)
        num_labels = len(x_probs)

        action_thresholds: Dict[object, List[Tuple[float, float]]] = {}
        all_s_candidates = {0.0, 1.0}

        for action in self.actions:
            utilities_for_a = np.array(
                [self._utility(action, y) for y in range(num_labels)], dtype=float
            )
            unique_utils = np.unique(utilities_for_a)
            unique_utils = np.sort(unique_utils)[::-1]

            pairs: List[Tuple[float, float]] = []
            for u_candidate in unique_utils:
                mask = utilities_for_a >= u_candidate
                t_val = float(x_probs[mask].sum())
                pairs.append((float(u_candidate), t_val))
                all_s_candidates.add(t_val)

            action_thresholds[action] = pairs

        s_candidates = sorted(all_s_candidates)
        s_star = 0.0
        best_score = -np.inf

        for s in s_candidates:
            best_u_for_s = -np.inf
            for action, pairs in action_thresholds.items():
                feasible_u = -np.inf
                for u_candidate, t_val in pairs:
                    if t_val >= s:
                        feasible_u = u_candidate
                        break
                if feasible_u > best_u_for_s:
                    best_u_for_s = feasible_u

            phi_s = beta * s + best_u_for_s
            if phi_s > best_score:
                best_score = phi_s
                s_star = s

        return float(s_star)

    def get_conformal_set(
        self, x_probs: Sequence[float], s_value: float
    ) -> Set[int]:
        """
        Construct the RAC conformal set for a given x and s.
        """
        x_probs = np.asarray(x_probs, dtype=float)
        num_labels = len(x_probs)
        # _, best_action = self.hbtheta_and_arg(s_value, x_probs)
        best_u, best_action = self.hbtheta_and_arg(s_value, x_probs)


        u_vec = np.array([self._utility(best_action, y) for y in range(num_labels)], dtype=float)
        # c_star: Set[int] = set()

        # for y in range(num_labels):
        #     val_y = u_vec[y]
        #     mask = u_vec > val_y
        #     sum_prob = x_probs[mask].sum()
        #     if sum_prob <= s_value:
        #         c_star.add(y)

        # return c_star

        return set(np.where(u_vec >= best_u)[0])

    def predict_set(self, x_probs: Sequence[float], beta: float) -> Set[int]:
        s_star = self.compute_g_hat(x_probs, beta)
        return self.get_conformal_set(x_probs, s_star)

    def coverage_for_beta(
        self,
        calib_probs: Sequence[Sequence[float]],
        calib_labels: Sequence[int],
        beta: float,
    ) -> float:
        hits = 0
        n = len(calib_probs)
        for x_probs, y in zip(calib_probs, calib_labels):
            c_star = self.predict_set(x_probs, beta)
            if int(y) in c_star:
                hits += 1
        return hits / max(n, 1)

    def calibrate_beta(
        self,
        calib_probs: Sequence[Sequence[float]],
        calib_labels: Sequence[int],
        alpha: float,
        max_iter: int = 50,
        beta_init: float = 1.0,
        beta_max: float = 1e6,
    ) -> float:
        """
        Find beta (q) using doubling + bisection, matching RAC code.

        Notes:
        - Empirical coverage may plateau below the target (1 - alpha) for some
          datasets/utilities. To avoid an infinite loop, we cap the doubling
          search at beta_max; if the target is still unreachable, we return the
          capped beta.
        """
        target = 1.0 - alpha
        cov0 = self.coverage_for_beta(calib_probs, calib_labels, 0.0)
        if cov0 >= target:
            return 0.0

        beta_low = 0.0
        beta_high = float(beta_init)
        cov = self.coverage_for_beta(calib_probs, calib_labels, beta_high)
        while cov < target and beta_high < beta_max:
            beta_low = beta_high
            beta_high = min(beta_high * 2.0, beta_max)
            cov = self.coverage_for_beta(calib_probs, calib_labels, beta_high)

        if cov < target:
            return float(beta_high)

        for _ in range(max_iter):
            beta_mid = 0.5 * (beta_low + beta_high)
            if beta_high - beta_low < 1e-8 * max(abs(beta_mid), 1e-6):
                break
            cov = self.coverage_for_beta(calib_probs, calib_labels, beta_mid)
            if cov >= target:
                beta_high = beta_mid
            else:
                beta_low = beta_mid

        return float(0.5 * (beta_low + beta_high))

    def a_RAC(self, label_set: Iterable[int]) -> Tuple[object, float]:
        """
        Max-min action rule: argmax_a min_{y in S} U[y, a].
        Returns (best_action, best_min_utility).
        """
        label_set = list(label_set)
        if not label_set:
            raise ValueError("label_set must be non-empty.")

        best_action = None
        best_min_u = -np.inf

        for action in self.actions:
            a_idx = self.action_to_index[action]
            utilities = self.utility_matrix[label_set, a_idx]
            min_u = float(np.min(utilities))
            if min_u > best_min_u:
                best_min_u = min_u
                best_action = action

        if best_action is None:
            raise RuntimeError("Failed to compute a_RAC.")
        return best_action, best_min_u


class _BaseConformalScorer:
    """
    Base helper for split conformal prediction with a scalar score.

    Subclasses must implement _score(x_probs, label) returning a numeric
    nonconformity score. Lower scores are “better”; the conformal set is
    {y : score(x, y) <= q_hat}.
    """

    def __init__(self, actions: Sequence[object]) -> None:
        self.actions = list(actions)
        self.q_hat: Optional[float] = None

    def calibrate(self, cal_probs: Sequence[Sequence[float]], cal_labels: Sequence[int], alpha: float) -> float:
        """
        Compute and store conformal quantile: k=ceil((n+1)(1-alpha)).
        """
        scores = []
        for probs, y in zip(cal_probs, cal_labels):
            scores.append(self._score(np.asarray(probs, dtype=float), int(y)))
        scores_arr = np.asarray(scores, dtype=float)
        scores_sorted = np.sort(scores_arr)
        n = len(scores_sorted)
        k = int(np.ceil((n + 1) * (1.0 - alpha)))
        k = min(max(k, 1), n)
        self.q_hat = float(scores_sorted[k - 1])
        return self.q_hat

    def predict_set(self, x_probs: Sequence[float]) -> Set[int]:
        """
        Return prediction set {y: score(x, y) <= q_hat}.
        """
        if self.q_hat is None:
            raise RuntimeError("Call calibrate() before predict_set().")
        probs = np.asarray(x_probs, dtype=float)
        return {y for y in range(len(probs)) if self._score(probs, y) <= self.q_hat}

    # to be implemented by subclasses
    def _score(self, x_probs, label):  # type: ignore[no-untyped-def]
        raise NotImplementedError


class LASConformal(_BaseConformalScorer):
    """
    LAS (Least Ambiguous Set, Sadinle et al. 2019): 1 - p_true.
    """

    def _score(self, x_probs, label):
        return 1.0 - float(x_probs[label])


class APSConformal(_BaseConformalScorer):
    """
    APS (Adaptive Prediction Sets, Romano et al. 2020):
    sum_{y' : p(y'|x) > p(label|x)} p(y'|x).
    """

    def _score(self, x_probs, label):
        p_y = float(x_probs[label])
        mask = x_probs > p_y
        mask[label] = False
        return float(x_probs[mask].sum())


class Score3CortesConformal(_BaseConformalScorer):
    """
    Cortés-Gómez et al. (ICLR'25) greedy score (score-3).

    Steps per x:
      1) Build permutation sigma_x with greedy rule (their Eq. 2):
         start S=∅; while labels remain, pick y maximizing
            (M - L(S∪{y})) / (1 - p_y)
         subject to p_y <= alpha - p(S). If no feasible y, drop the
         constraint and pick the max ratio.
      2) Nonconformity score s(x,y) = sum_{i <= k(y)} p(sigma_x(i))
         (APS-style cumulative mass along the greedy order).

    We set the (bounded) set loss to the risk-averse certificate:
      L(S) = min_a [ ell_in(a,S) + alpha * (ell_out(a,S) - ell_in(a,S))_+ ]
    where ell_in(a,S) = max_{y in S} loss[y,a] (0 if S=∅),
          ell_out(a,S) = max_{y not in S} loss[y,a] (0 if S=Y).
    M is max loss over all (y,a) and is only used to form the greedy ratio.
    """

    def __init__(
        self,
        actions: Sequence[object],
        loss_matrix: Sequence[Sequence[float]],
        alpha: float,
    ) -> None:
        super().__init__(actions)
        self.loss_matrix = np.asarray(loss_matrix, dtype=float)
        if self.loss_matrix.ndim != 2:
            raise ValueError("loss_matrix must be 2D [num_labels, num_actions].")
        if self.loss_matrix.shape[1] != len(self.actions):
            raise ValueError("loss_matrix second dimension must match actions.")
        if not (0 < alpha < 1):
            raise ValueError("alpha must be in (0,1).")
        self.alpha = float(alpha)
        self.M = float(np.max(self.loss_matrix))

    def _set_loss(self, subset: np.ndarray) -> float:
        """
        Risk-averse set loss L(S).
        """
        if subset.size == 0:
            return 0.0
        num_labels = self.loss_matrix.shape[0]
        all_labels = np.arange(num_labels)
        comp = np.setdiff1d(all_labels, subset, assume_unique=True)

        best_val = np.inf
        for a_idx in range(len(self.actions)):
            ell_in = float(np.max(self.loss_matrix[subset, a_idx])) if subset.size else 0.0
            ell_out = float(np.max(self.loss_matrix[comp, a_idx])) if comp.size else 0.0
            val = ell_in + self.alpha * max(ell_out - ell_in, 0.0)
            if val < best_val:
                best_val = val
        return best_val

    def _greedy_order(self, x_probs: np.ndarray) -> np.ndarray:
        """
        Build sigma_x using greedy rule with probability budget (phase 1),
        then append remaining labels in fixed order (phase 2).
        """
        K = len(x_probs)
        remaining = np.ones(K, dtype=bool)
        order = []
        p_sum = 0.0
        # Phase 1: constrained greedy
        while len(order) < K:
            feasible_mask = (x_probs <= self.alpha - p_sum) & remaining
            if not feasible_mask.any():
                break
            candidates = np.where(feasible_mask)[0]

            best_ratio = -np.inf
            best_y = candidates[0]
            for y in candidates:
                s_temp = np.array(order + [y], dtype=int)
                l_val = self._set_loss(s_temp)
                ratio = (self.M - l_val) / max(1e-12, 1.0 - float(x_probs[y]))
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_y = y

            order.append(int(best_y))
            remaining[best_y] = False
            p_sum += float(x_probs[best_y])

        # Phase 2: append any leftover labels in descending probability (fixed tie-break).
        if remaining.any():
            tail = np.where(remaining)[0]
            tail_sorted = tail[np.argsort(-x_probs[tail], kind="mergesort")]
            order.extend(tail_sorted.tolist())

        return np.array(order, dtype=int)

    def _score(self, x_probs, label):
        x_probs = np.asarray(x_probs, dtype=float)
        sigma = self._greedy_order(x_probs)
        pos = int(np.where(sigma == label)[0][0])
        return float(np.sum(x_probs[sigma[: pos + 1]]))

    def calibrate(self, cal_probs: Sequence[Sequence[float]], cal_labels: Sequence[int], alpha: float) -> float:
        # enforce alpha consistency with the greedy construction
        if abs(alpha - self.alpha) > 1e-12:
            raise ValueError("Alpha for calibrate must match the alpha used in the greedy score.")
        return super().calibrate(cal_probs, cal_labels, alpha)


# ============================================================================
# Numba JIT-accelerated Convex Hull for SOCOP
# ============================================================================

@jit(nopython=True, cache=True)
def _compute_lower_convex_hull_numba(p_hat_sorted: np.ndarray, penalty_lambda: float):
    """
    Computes the lower convex hull of the point set P_k = (S_k, g_k).
    This is a numba-accelerated implementation.
    """
    K = len(p_hat_sorted)
    S = np.zeros(K + 1, dtype=np.float64)
    g = np.zeros(K + 1, dtype=np.float64)
    
    current_S = 0.0
    for k in range(K):
        current_S += p_hat_sorted[k]
        S[k + 1] = current_S
    for k in range(K + 1):
        g[k] = (1.0 if k > 1 else 0.0) + penalty_lambda * k

    lower_hull_indices_arr = np.zeros(K + 1, dtype=np.int64)
    hull_size = 0
    for k in range(K + 1):
        while hull_size >= 2:
            i = lower_hull_indices_arr[hull_size - 1]
            j = lower_hull_indices_arr[hull_size - 2]
            cross_product = (S[i] - S[j]) * (g[k] - g[i]) - (S[k] - S[i]) * (g[i] - g[j])
            if cross_product <= 1e-9:
                hull_size -= 1
            else:
                break
        lower_hull_indices_arr[hull_size] = k
        hull_size += 1
    return lower_hull_indices_arr[:hull_size], S, g


class SOCOPConformal(_BaseConformalScorer):
    """
    SOCOP: Singleton-Optimized Conformal Prediction score.

    Based on the geometric Convex Hull approach from:
    Singleton-Optimized-Conformal-Prediction.

    The nonconformity score is computed as the slope on the lower convex hull
    of points (S_k, g_k) where:
      - S_k = cumulative probability of top-k labels
      - g_k = 1{k>1} + lambda * k

    This method uses numba for acceleration.

    Parameters:
        actions: Sequence of action labels (for compatibility with base class).
        lambda_param: Penalty parameter for SOCOP (default 0.125).
    """

    def __init__(
        self,
        actions: Sequence[object],
        lambda_param: float = 0.0625,
    ) -> None:
        super().__init__(actions)
        if lambda_param <= 0:
            raise ValueError("lambda_param must be positive for SOCOP.")
        self.lambda_param = lambda_param
        # Warm-up call to compile the numba function
        _ = _compute_lower_convex_hull_numba(np.array([0.5, 0.3, 0.2]), self.lambda_param)

    def _score(self, x_probs, label):
        """
        Compute nonconformity score for SOCOP.
        Returns the slope at the segment covering label's rank.
        """
        x_probs = np.asarray(x_probs, dtype=np.float64)
        sorted_indices = np.argsort(x_probs)[::-1]
        p_sorted = x_probs[sorted_indices]

        # Find 1-based rank of the label
        rank_positions = np.where(sorted_indices == label)[0]
        if len(rank_positions) == 0:
            return np.inf
        i = int(rank_positions[0]) + 1  # 1-based rank

        hull_indices, S, g = _compute_lower_convex_hull_numba(p_sorted, self.lambda_param)

        # Find the hull segment that covers index i
        found_j = -1
        for j, v_j in enumerate(hull_indices):
            if v_j >= i:
                found_j = j
                break

        if found_j <= 0:
            # Edge case: check if label falls in the first segment
            if len(hull_indices) > 1 and hull_indices[0] < i <= hull_indices[1]:
                found_j = 1
            else:
                return 0.0

        v_j = hull_indices[found_j]
        v_prev = hull_indices[found_j - 1]

        delta_S = S[v_j] - S[v_prev]

        if delta_S < 1e-12:
            return np.inf
        return float((g[v_j] - g[v_prev]) / delta_S)


class BestResponseBaseline:
    """
    Decision calibration (soft partition, Algorithm-2 style) on calibration data,
    then best-response under expected loss.
    """

    def __init__(
        self,
        actions: Sequence[object],
        loss_matrix: Sequence[Sequence[float]],  # [C, K]
        device: str = "cpu",
        ridge: float = 1e-6,
    ) -> None:
        self.actions: List[object] = list(actions)
        self.K = len(self.actions)
        self.device = device
        self.ridge = float(ridge)

        self.loss_matrix = torch.tensor(loss_matrix, dtype=torch.float32, device=device)  # [C, K]
        if self.loss_matrix.ndim != 2:
            raise ValueError("loss_matrix must be 2D [num_labels, num_actions].")
        self.C = int(self.loss_matrix.shape[0])
        if self.loss_matrix.shape[1] != self.K:
            raise ValueError("loss_matrix second dimension must match actions.")

        self.layers: List[Dict[str, torch.Tensor]] = []

    @staticmethod
    def _project_simplex(P: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
        """
        Euclidean projection onto the simplex {p >= 0, sum p = 1} row-wise (Duchi et al., 2008).
        P: [N,C]
        """
        if P.ndim == 1:
            P = P.unsqueeze(0)
        U, _ = torch.sort(P, dim=1, descending=True)
        cssv = torch.cumsum(U, dim=1) - 1.0
        ind = torch.arange(1, P.size(1) + 1, device=P.device, dtype=P.dtype).unsqueeze(0)
        cond = U - cssv / ind > 0
        rho = cond.sum(dim=1, keepdim=True).clamp(min=1)
        theta = cssv.gather(1, rho - 1) / rho.to(P.dtype)
        W = (P - theta).clamp_min(0.0)
        if eps > 0:
            W = W.clamp_min(eps)
            W = W / W.sum(dim=1, keepdim=True)
        return W

    def _gate(self, P: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """B = softmax(P @ W^T), with W: [K,C], P: [N,C] => B: [N,K]."""
        return torch.softmax(P @ W.t(), dim=1)

    def fit(
        self,
        val_probs: np.ndarray,  # [N,C]
        val_labels: np.ndarray,  # [N] class index or [N,C] one-hot
        outer_iters: int = 10,
        inner_steps: int = 50,
        lr: float = 0.1,
        epsilon: float = 1e-3,
    ) -> "BestResponseBaseline":
        self.layers = []

        val_probs = np.asarray(val_probs)
        val_labels = np.asarray(val_labels)
        P = torch.tensor(val_probs, dtype=torch.float32, device=self.device)  # [N,C]
        P = self._project_simplex(P)  # ensure valid probs

        if val_labels.ndim == 1:
            y = torch.tensor(val_labels, dtype=torch.long, device=self.device)
            Y = torch.nn.functional.one_hot(y, self.C).float()
        else:
            Y = torch.tensor(val_labels, dtype=torch.float32, device=self.device)
            if Y.shape[1] != self.C:
                raise ValueError(f"val_labels one-hot must have {self.C} columns.")

        N = P.shape[0]
        I_K = torch.eye(self.K, device=self.device)
        threshold = (float(epsilon) ** 2) / max(self.K, 1)

        for _ in range(int(outer_iters)):
            # Step 1: adversary - maximize v(W)=||R(W)||_F^2
            W = torch.randn(self.K, self.C, device=self.device) * 0.01
            W.requires_grad_()
            opt = optim.Adam([W], lr=lr)

            P_det = P.detach()
            E_det = (Y - P_det)  # [N,C]

            for _ in range(int(inner_steps)):
                opt.zero_grad(set_to_none=True)
                B = self._gate(P_det, W)                  # [N,K]
                Rk = (E_det.t() @ B) / N                  # [C,K]
                v = (Rk * Rk).sum()                       # scalar
                (-v).backward()                           # maximize v
                opt.step()

            W_fixed = W.detach()

            # Step 1b: compute v and stop if below paper threshold
            with torch.no_grad():
                B_det = self._gate(P_det, W_fixed)
                Rk_det = (E_det.t() @ B_det) / N
                v_value = float((Rk_det * Rk_det).sum().item())
            if v_value < threshold:
                break

            # Step 2: compute U = R D^{-1} (Moore-Penrose when singular)
            B = self._gate(P, W_fixed)                    # [N,K]
            E = (Y - P)                                   # [N,C]
            D = (B.t() @ B) / N                           # [K,K]
            R = (E.t() @ B) / N                           # [C,K]
            D_inv = torch.linalg.pinv(D)
            U = (R @ D_inv).contiguous()                  # [C,K]

            self.layers.append({"W": W_fixed, "U": U})

            # Step 3: update P for next round: P <- proj(P + B U^T)
            P = self._project_simplex(P + B @ U.t())

        return self

    def _apply_layers(self, P: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            B = self._gate(P, layer["W"])                 # [N,K]
            P = self._project_simplex(P + B @ layer["U"].t())
        return P

    def predict(self, test_probs: np.ndarray) -> np.ndarray:
        if not self.layers:
            raise RuntimeError("Call fit() first.")

        P = torch.tensor(test_probs, dtype=torch.float32, device=self.device)
        if P.ndim == 1:
            P = P.unsqueeze(0)
        P = self._project_simplex(P)

        with torch.no_grad():
            P = self._apply_layers(P)
            exp_losses = P @ self.loss_matrix             # [N,K]
            best_actions = torch.argmin(exp_losses, dim=1)

        action_arr = np.asarray(self.actions, dtype=object)
        return action_arr[best_actions.cpu().numpy()]

    def predict_action(self, x_probs: Sequence[float]) -> object:
        actions = self.predict(np.asarray(x_probs))
        return actions[0]


class RAPSConformal(_BaseConformalScorer):
    """RAPS (Regularized Adaptive Prediction Sets, Romano et al. 2020).
    Adds regularization penalty to APS: score = APS_score + lambda * max(0, rank - k_reg)."""

    def __init__(self, actions, lambda_reg=0.01, k_reg=1):
        super().__init__(actions)
        if lambda_reg < 0: raise ValueError("lambda_reg must be >= 0")
        if k_reg < 0: raise ValueError("k_reg must be >= 0")
        self.lambda_reg = float(lambda_reg)
        self.k_reg = int(k_reg)

    def _score(self, x_probs, label):
        x_probs = np.asarray(x_probs, dtype=float)
        p_y = float(x_probs[label])
        mask = x_probs > p_y
        mask[label] = False
        aps = float(x_probs[mask].sum())
        rank_val = int(np.sum(x_probs > p_y))
        penalty = self.lambda_reg * max(0, rank_val - self.k_reg)
        return aps + penalty


class DecisionFocusedConformal(_BaseConformalScorer):
    """Decision-Focused Conformal Score. Weights APS by loss importance of excluded labels."""

    def __init__(self, actions, loss_matrix, gamma=1.0):
        super().__init__(actions)
        self.loss_matrix = np.asarray(loss_matrix, dtype=float)
        if gamma < 0: raise ValueError("gamma must be >= 0")
        self.gamma = float(gamma)
        self.max_loss = float(np.max(self.loss_matrix))
        if self.max_loss <= 0:
            self.loss_weights = np.zeros(self.loss_matrix.shape[0])
        else:
            self.loss_weights = np.max(self.loss_matrix, axis=1) / self.max_loss

    def _score(self, x_probs, label):
        x_probs = np.asarray(x_probs, dtype=float)
        p_y = float(x_probs[label])
        mask = x_probs > p_y
        mask[label] = False
        weights = 1.0 + self.gamma * self.loss_weights
        return float(np.sum(x_probs[mask] * weights[mask]))


__all__ = ["RACBaseline", "LASConformal", "APSConformal", "Score3CortesConformal", "SOCOPConformal", "RAPSConformal", "DecisionFocusedConformal", "BestResponseBaseline"]
