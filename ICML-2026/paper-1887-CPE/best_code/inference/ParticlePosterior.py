import numpy as np
from typing import List, Optional
from utils.utils import normalize, entropy_categorical
from likelihood.preference_likelihood_threeway import bt_threeway_hier
from dag.dag_ops import is_acyclic
from likelihood.preference_likelihood_threeway import log_expert_likelihood_bt_threeway

class ParticlePosterior:
    def __init__(self, particles: List[np.ndarray], weights: Optional[np.ndarray] = None):
        self.particles = particles  # list of W matrices (DxD)
        S = len(particles)
        if weights is None:
            self.weights = np.full(S, 1.0 / S, dtype=float)
        else:
            self.weights = normalize(weights.astype(float))

    def predictive_answer_dist(self, i: int, j: int, beta_edge: float, beta_dir: float, lam: float = 0.0) -> np.ndarray:
        p = np.zeros(3, dtype=float)
        for w, W in zip(self.weights, self.particles):
            p += w * bt_threeway_hier(W, i, j, beta_edge=beta_edge, beta_dir=beta_dir, lam=lam)
        return p

    def eig_for_pair(self, i: int, j: int, beta_edge: float, beta_dir: float, lam: float = 0.0) -> float:
        pred = self.predictive_answer_dist(i, j, beta_edge, beta_dir, lam)
        H_pred = entropy_categorical(pred)
        H_cond = 0.0
        for w, W in zip(self.weights, self.particles):
            p = bt_threeway_hier(W, i, j, beta_edge=beta_edge, beta_dir=beta_dir, lam=lam)
            H_cond += w * entropy_categorical(p)
        return float(H_pred - H_cond)

    def update_with_observation(self, i: int, j: int, y_idx: int, beta_edge: float, beta_dir: float, lam: float = 0.0):
        S = len(self.particles)
        new_w = np.empty(S, dtype=float)
        for s, W in enumerate(self.particles):
            p = bt_threeway_hier(W, i, j, beta_edge=beta_edge, beta_dir=beta_dir, lam=lam)
            new_w[s] = self.weights[s] * p[y_idx]
        self.weights = normalize(new_w)

    def resample(self, rng: np.random.Generator):
        S = len(self.weights)

        # Systematic resampling
        positions = (rng.random() + np.arange(S)) / S
        csum = np.cumsum(self.weights)
        idx = np.zeros(S, dtype=int)
        i = j = 0
        while i < S:
            if positions[i] < csum[j]:
                idx[i] = j
                i += 1
            else:
                j += 1
        self.particles = [self.particles[k].copy() for k in idx]
        self.weights = np.full(S, 1.0 / S, dtype=float)

        print(f"resampled {len(self.particles)} particles")

    def edge_marginals(self) -> np.ndarray:
        """Posterior probability an edge exists i->j (ignoring orientation conflicts)."""
        Wsum = np.zeros_like(self.particles[0], dtype=float)
        for w, W in zip(self.weights, self.particles):
            A = (np.abs(W) > 1e-12).astype(float)
            Wsum += w * A
        return Wsum

    def rejuvenate_particles(self, q0_logprob,
                             expert_history,
                             beta_edge, beta_dir, lam,
                             n_steps=1, mutate_frac=0.5,
                             round=None, rng=None):
        """
        Lightweight MCMC rejuvenation after resampling.
        Mutates a subset of particles with random DAG edits,
        accepted under MH ratio using current posterior.
        """
        particles = self.particles
        if rng is None:
            rng = np.random.default_rng()
        D = particles[0].shape[0]
        new_particles = []

        for W in particles:
            W_new = W.copy()
            if rng.random() > mutate_frac:
                new_particles.append(W_new)
                continue

            for _ in range(n_steps):
                i, j = rng.choice(D, 2, replace=False)
                move_type = rng.integers(3)
                W_prop = W_new.copy()

                if move_type == 0:
                    # flip direction if one direction exists
                    if W_prop[i,j] != 0 and W_prop[j,i] == 0:
                        W_prop[i,j], W_prop[j,i] = 0, W_prop[i,j]
                    elif W_prop[j,i] != 0 and W_prop[i,j] == 0:
                        W_prop[j,i], W_prop[i,j] = 0, W_prop[j,i]
                    else:
                        continue
                elif move_type == 1:
                    # add edge
                    if W_prop[i,j] == 0 and W_prop[j,i] == 0:
                        W_prop[i,j] = rng.uniform(0.5, 1.5)
                else:
                    # remove edge
                    W_prop[i,j] = 0
                    W_prop[j,i] = 0

                if not is_acyclic(W_prop):
                    continue

                # Approximate MH ratio using prior + expert history
                logp_old = q0_logprob(W_new)
                logp_new = q0_logprob(W_prop)
                for (a,b,y) in expert_history:
                    logp_old += log_expert_likelihood_bt_threeway(y=y, W=W_new, i=a, j=b, beta_edge=beta_edge,
                                                                  beta_dir=beta_dir, lam=lam)
                    logp_new += log_expert_likelihood_bt_threeway(y=y, W=W_prop, i=a, j=b, beta_edge=beta_edge,
                                                                  beta_dir=beta_dir, lam=lam)

                log_alpha = logp_new - logp_old
                if np.log(rng.random()) < log_alpha:
                    W_new = W_prop  # accept

            new_particles.append(W_new)

        self.particles = new_particles

        print(f"Rejuvenated {len(self.particles)} particles at round {round}")



