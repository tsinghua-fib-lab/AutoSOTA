from __future__ import annotations

from typing import  Sequence, Union, Callable, List, Tuple, Optional
import math
import numpy as np
import torch

from likelihood.preference_likelihood_threeway_gpu import (bt_threeway_hier,
                                                           bt_threeway_hier_pairs,
                                                           log_expert_likelihood_bt_threeway)
from dag.dag_ops_gpu import is_acyclic

class ParticlePosterior:
    """Torch-backed particle posterior.

    - Particles stored as a tensor (S,D,D) for GPU-friendly ops.
    - Weights stored as a tensor (S,).
    - Provides batched EIG over many candidate pairs (K) with optional chunking.

    Conventions follow the baseline:
      y=0: i->j
      y=1: j->i
      y=2: none
    """

    def __init__(
        self,
        particles: Sequence[np.ndarray],
        weights: Optional[np.ndarray] = None,
        *,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float64,
    ):
        if len(particles) == 0:
            raise ValueError("particles must be non-empty")

        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = dtype

        # Stack to (S,D,D)
        P = torch.stack([torch.as_tensor(W, dtype=dtype) for W in particles], dim=0)
        self.particles = P.to(self.device)

        S = self.particles.shape[0]
        if weights is None:
            self.weights = torch.full((S,), 1.0 / S, dtype=dtype, device=self.device)
        else:
            w = torch.as_tensor(weights, dtype=dtype, device=self.device)
            s = torch.sum(w)
            if float(s.item()) <= 0:
                self.weights = torch.full((S,), 1.0 / S, dtype=dtype, device=self.device)
            else:
                self.weights = w / s

    # ---------- Core GPU-accelerated utilities ----------

    @torch.no_grad()
    def edge_marginals(self) -> np.ndarray:
        """Posterior probability an edge exists i->j (ignoring orientation)."""
        A = (torch.abs(self.particles) > 1e-12).to(self.dtype)
        marg = torch.sum(self.weights.view(-1, 1, 1) * A, dim=0)
        return marg.detach().cpu().numpy()

    # ---------- Predictive distributions ----------

    def predictive_answer_dist(self, i: int, j: int, beta_edge: float, beta_dir: float, lam: float = 0.0) -> np.ndarray:
        """Posterior predictive distribution over the expert's 3-way response for one pair."""

        with torch.no_grad():
            P = bt_threeway_hier(self.particles, i, j, beta_edge=beta_edge, beta_dir=beta_dir, lam=lam)
            pred = torch.sum(self.weights[:, None] * P, dim=0)
            return pred.detach().cpu().numpy()

    def predictive_answer_dist_pairs(
        self,
        pairs: Union[np.ndarray, List[Tuple[int, int]]],
        beta_edge: float,
        beta_dir: float,
        lam: float = 0.0,
        *,
        chunk_size: int = 4096,
    ) -> np.ndarray:
        """Posterior predictive distributions for K queried pairs.

        Returns an array of shape (K,3).
        """
        pairs_arr = np.asarray(pairs, dtype=np.int64)
        if pairs_arr.ndim != 2 or pairs_arr.shape[1] != 2:
            raise ValueError("pairs must have shape (K,2)")

        K = pairs_arr.shape[0]

        with torch.no_grad():
            i_idx_full = torch.as_tensor(pairs_arr[:, 0], dtype=torch.long, device=self.device)
            j_idx_full = torch.as_tensor(pairs_arr[:, 1], dtype=torch.long, device=self.device)

            preds = []
            for start in range(0, K, chunk_size):
                end = min(K, start + chunk_size)
                i_idx = i_idx_full[start:end]
                j_idx = j_idx_full[start:end]
                P = bt_threeway_hier_pairs(self.particles, i_idx, j_idx, beta_edge=beta_edge, beta_dir=beta_dir, lam=lam)
                pred = torch.sum(self.weights[:, None, None] * P, dim=0)  # (k,3)
                preds.append(pred)
            return torch.cat(preds, dim=0).detach().cpu().numpy()

    # ---------- Expected information gain ----------

    def eig_for_pair(self, i: int, j: int, beta_edge: float, beta_dir: float, lam: float = 0.0) -> float:
        """Expected information gain for querying (i,j)."""

        with torch.no_grad():
            P = bt_threeway_hier(self.particles, i, j, beta_edge=beta_edge, beta_dir=beta_dir, lam=lam)
            pred = torch.sum(self.weights[:, None] * P, dim=0)
            pred_c = torch.clamp(pred, min=1e-12, max=1.0)
            H_pred = -torch.sum(pred_c * torch.log(pred_c))

            P_c = torch.clamp(P, min=1e-12, max=1.0)
            H_each = -torch.sum(P_c * torch.log(P_c), dim=-1)  # (S,)
            H_cond = torch.sum(self.weights * H_each)
            return float((H_pred - H_cond).detach().cpu().item())

    def eig_for_pairs(
        self,
        pairs: Union[np.ndarray, List[Tuple[int, int]]],
        beta_edge: float,
        beta_dir: float,
        lam: float = 0.0,
        *,
        chunk_size: int = 2048,
    ) -> np.ndarray:
        """Compute EIG for K candidate pairs.

        Returns:
            eig: numpy array of shape (K,)

        Notes:
            - Uses chunking over K to limit memory: each chunk materializes (S,chunk,3).
            - Fully vectorized over particles within each chunk.
        """
        pairs_arr = np.asarray(pairs, dtype=np.int64)
        if pairs_arr.ndim != 2 or pairs_arr.shape[1] != 2:
            raise ValueError("pairs must have shape (K,2)")

        K = pairs_arr.shape[0]

        with torch.no_grad():
            i_idx_full = torch.as_tensor(pairs_arr[:, 0], dtype=torch.long, device=self.device)
            j_idx_full = torch.as_tensor(pairs_arr[:, 1], dtype=torch.long, device=self.device)

            out_chunks = []
            for start in range(0, K, chunk_size):
                end = min(K, start + chunk_size)
                i_idx = i_idx_full[start:end]
                j_idx = j_idx_full[start:end]

                P = bt_threeway_hier_pairs(
                    self.particles,
                    i_idx,
                    j_idx,
                    beta_edge=beta_edge,
                    beta_dir=beta_dir,
                    lam=lam,
                )  # (S,chunk,3)

                # predictive per pair
                pred = torch.sum(self.weights[:, None, None] * P, dim=0)  # (chunk,3)
                pred_c = torch.clamp(pred, min=1e-12, max=1.0)
                H_pred = -torch.sum(pred_c * torch.log(pred_c), dim=-1)  # (chunk,)

                # expected conditional entropy
                P_c = torch.clamp(P, min=1e-12, max=1.0)
                H_each = -torch.sum(P_c * torch.log(P_c), dim=-1)  # (S,chunk)
                H_cond = torch.sum(self.weights[:, None] * H_each, dim=0)  # (chunk,)

                out_chunks.append(H_pred - H_cond)

            return torch.cat(out_chunks, dim=0).detach().cpu().numpy()

    # ---------- Update ----------

    def update_with_observation(self, i: int, j: int, y_idx: int, beta_edge: float, beta_dir: float, lam: float = 0.0):
        """Bayes update of particle weights given an expert response y_idx."""
        if y_idx not in (0, 1, 2):
            raise ValueError("y_idx must be 0 (i->j), 1 (j->i), or 2 (none)")


        with torch.no_grad():
            P = bt_threeway_hier(self.particles, i, j, beta_edge=beta_edge, beta_dir=beta_dir, lam=lam)
            new_w = self.weights * P[:, y_idx]
            s = torch.sum(new_w)
            if float(s.detach().cpu().item()) <= 0.0:
                self.weights = torch.full_like(self.weights, 1.0 / self.weights.numel())
            else:
                self.weights = new_w / s

        return


    # ---------- Resampling / rejuvenation (kept faithful, numpy-based) ----------
    def resample(self, torch_gen: torch.Generator):
        """
        Systematic resampling using torch.Generator for determinism.

        Assumes:
          - self.particles: torch.Tensor (S,D,D) on device
          - self.weights: torch.Tensor (S,) on device (or will be moved)
        """
        if not isinstance(self.particles, torch.Tensor):
            raise TypeError("resample expects torch.Tensor particles")

        device = self.particles.device
        S = int(self.particles.shape[0])

        # Enforce generator/device consistency (fail early with a useful message)
        # if torch_gen.device != device:
        #     raise ValueError(
        #         f"torch_gen is on {torch_gen.device} but particles are on {device}. "
        #         "Create torch_gen with torch.Generator(device=device)."
        #     )

        # Normalize weights ON DEVICE
        w = self.weights.detach().to(device=device, dtype=torch.float64)
        w = w / w.sum().clamp_min(1e-300)

        # Systematic positions ON DEVICE
        u0 = torch.rand((), device=device, generator=torch_gen, dtype=torch.float64)
        positions = (u0 + torch.arange(S, device=device, dtype=torch.float64)) / float(S)

        csum = torch.cumsum(w, dim=0)

        idx = torch.searchsorted(csum, positions, right=False).clamp_max(S - 1).to(torch.long)

        self.particles = self.particles.index_select(0, idx).clone()
        self.weights = torch.full((S,), 1.0 / float(S), device=device, dtype=torch.float64)


    def rejuvenate_particles(
        self,
        q0_logprob: Callable[[torch.Tensor], torch.Tensor],
        expert_history: List[Tuple[int, int, int]],
        beta_edge: float,
        beta_dir: float,
        lam: float,
        *,
        n_steps: int = 1,
        mutate_frac: float = 0.5,
        round: Optional[int] = None,
        torch_gen: Optional[torch.Generator] = None,
    ):
        if torch_gen is None:
            raise ValueError("torch_gen must be provided for deterministic behavior")

        W_all = self.particles
        device = W_all.device
        dtype = W_all.dtype
        S, D, _ = W_all.shape

        # Enforce generator/device match early
        # if torch_gen.device != device:
        #     raise ValueError(
        #         f"torch_gen is on {torch_gen.device} but particles are on {device}. "
        #         "Create torch_gen with torch.Generator(device=device)."
        #     )

        # Decide which particles mutate (ON DEVICE)
        mutate_mask = (torch.rand((S,), device=device, generator=torch_gen) < mutate_frac)

        new_particles = W_all.clone()

        for s in range(S):
            if not bool(mutate_mask[s].item()):
                continue

            W_new = new_particles[s].clone()

            for _ in range(n_steps):
                # sample i != j cheaply without randperm
                i = int(torch.randint(0, D, (), device=device, generator=torch_gen).item())
                j = int(torch.randint(0, D - 1, (), device=device, generator=torch_gen).item())
                if j >= i:
                    j += 1

                move_type = int(torch.randint(0, 3, (), device=device, generator=torch_gen).item())
                W_prop = W_new.clone()

                if move_type == 0:
                    # flip
                    if W_prop[i, j] != 0 and W_prop[j, i] == 0:
                        tmp = W_prop[i, j].clone()
                        W_prop[i, j] = 0
                        W_prop[j, i] = tmp
                    elif W_prop[j, i] != 0 and W_prop[i, j] == 0:
                        tmp = W_prop[j, i].clone()
                        W_prop[j, i] = 0
                        W_prop[i, j] = tmp
                    else:
                        continue

                elif move_type == 1:
                    # add
                    if W_prop[i, j] == 0 and W_prop[j, i] == 0:
                        val = 0.5 + torch.rand((), device=device, dtype=dtype, generator=torch_gen)
                        W_prop[i, j] = val
                    else:
                        continue
                else:
                    # remove
                    W_prop[i, j] = 0
                    W_prop[j, i] = 0

                if not is_acyclic(W_prop):
                    continue

                with torch.no_grad():
                    logp_old = q0_logprob(W_new)
                    logp_new = q0_logprob(W_prop)

                    for (a, b, y) in expert_history:
                        logp_old = logp_old + log_expert_likelihood_bt_threeway(
                            y=y, W=W_new, i=a, j=b, beta_edge=beta_edge, beta_dir=beta_dir, lam=lam
                        )
                        logp_new = logp_new + log_expert_likelihood_bt_threeway(
                            y=y, W=W_prop, i=a, j=b, beta_edge=beta_edge, beta_dir=beta_dir, lam=lam
                        )

                    log_alpha = (logp_new - logp_old).item()

                u = torch.rand((), device=device, generator=torch_gen).item()
                if math.log(u) < log_alpha:
                    W_new = W_prop

            new_particles[s] = W_new

        self.particles = new_particles
        print(f"Rejuvenated {S} particles at round {round}")
