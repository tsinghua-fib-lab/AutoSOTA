"""
Fit a Gaussian to the cosine/sine components of the lagged angles.

"""
__date__ = "August - September 2025"

from dataclasses import dataclass
import jax
from jax import tree_util
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from tqdm import tqdm



def get_lag_statistics(
        loader,
        L: int,
        accumulator_dtype=jnp.float32,
        max_num_batches=None,
    ):
    """
    Get the lag statistics that are used for the imputation model.
    
    For each frequency f and each batch (B,C,F) of phases:
      - Convert angles to [cos, sin], shape per window = (C, L, 2).
      - Flatten to size D = C*L*2 in order [C, L, 2].
    Accumulate first and uncentered second moments across all batches,
    then center at the end.

    Returns:
      means: (F, D)   where D = C*L*2   (flattened [C, L, 2])
      covs:  (F, D, D)
    """
    C, F = loader.C, loader.F
    D = C * L * 2
    B = loader.batch_size

    S1 = jnp.zeros((F, D), dtype=accumulator_dtype)          # sum x
    S2 = jnp.zeros((F, D, D), dtype=accumulator_dtype)       # sum x x^T
    n_blocks = 0

    @jax.jit
    def process_batch(batch_phases: jnp.ndarray):
        """
        batch_phases: (B, L, C, F) angles
        Returns:
          S1_b: (F, D), S2_b: (F, D, D), n_b: int (number of windows)
        """
        B, L, C, F = batch_phases.shape

        cosv = jnp.cos(batch_phases)
        sinv = jnp.sin(batch_phases)
        z = jnp.stack([cosv, sinv], axis=-1)           # (B, L, C, F, 2)
        z = jnp.transpose(z, (0, 3, 2, 1, 4))          # (B, F, C, L, 2)
        Z = z.reshape(B, F, D).astype(accumulator_dtype)  # (B, F, D)

        # Accumulate per frequency
        X = jnp.swapaxes(Z, 0, 1)            # (F, B, D)
        S1_b = X.sum(axis=1)                 # (F, D)
        S2_b = jnp.einsum('fwd,fwe->fde', X, X)  # (F, D, D)
        return S1_b, S2_b

    # Stream batches; only slide within each batch
    for batch_num, batch in tqdm(enumerate(loader)):
        # batch: (B, L+1, C, F)
        B = len(batch)
        S1_b, S2_b = process_batch(batch[:,:-1])
        S1 += S1_b
        S2 += S2_b
        n_blocks += B

        if batch_num + 1 == max_num_batches:
            break

    if n_blocks == 0:
        raise ValueError("No L-length sliding windows found in any batch.")

    means = S1 / n_blocks
    ExxT  = S2 / n_blocks
    covs  = ExxT - jnp.einsum('fd,fe->fde', means, means)

    return means, covs, n_blocks


@tree_util.register_pytree_node_class
@dataclass
class GaussianChannelConditioner:
    """Precompute precision + block Cholesky for fast per-channel conditionals."""
    Lambda: jnp.ndarray      # [RD, RD] = Sigma^{-1}
    L_blocks: jnp.ndarray    # [R, D, D] Cholesky of each block Lambda_rr
    R: int
    D: int

    def tree_flatten(self):
        children = (self.Lambda, self.L_blocks)
        aux = (self.R, self.D)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        R, D = aux
        Lambda, L_blocks = children
        return cls(Lambda, L_blocks, R, D)

    @classmethod
    def from_cov(cls, Sigma: jnp.ndarray, R: int, D: int):
        RD = R * D
        assert Sigma.shape == (RD, RD)
        # Invert Sigma stably via Cholesky (Σ^{-1} = L^{-T} L^{-1})
        L = jnp.linalg.cholesky(Sigma)
        I = jnp.eye(RD, dtype=Sigma.dtype)
        inv_L = solve_triangular(L, I, lower=True)
        Lambda = inv_L.T @ inv_L

        # Extract Λ_rr and factor each once
        def chol_block(r):
            # starting indices along each axis
            start_idx = jnp.array([r * D, r * D])
            # size of the slice along each axis
            slice_size = (D, D)
            block = jax.lax.dynamic_slice(Lambda, start_idx, slice_size)
            return jnp.linalg.cholesky(block)
        L_blocks = jax.vmap(chol_block)(jnp.arange(R))
        return cls(Lambda=Lambda, L_blocks=L_blocks, R=R, D=D)

    @jax.jit
    def conditional_means(self, x: jnp.ndarray, mu: jnp.ndarray):
        """
        x: [..., R, D]
        mu: [R, D] or broadcastable to x
        returns: [..., R, D] of E[x_r | x_{-r}] for each r
        """
        R, D = self.R, self.D
        RD = R * D

        # Flatten batch, keep (R,D) trailing
        x_shape = x.shape
        x_ = x.reshape(-1, R, D)
        mu_ = jnp.broadcast_to(mu, (R, D)).reshape(1, R, D)
        d = (x_ - mu_).reshape(-1, RD)                # [B, RD]

        # v = Λ (x - μ) via matmul once per sample
        v = d @ self.Lambda.T                          # [B, RD]
        v = v.reshape(-1, R, D)                        # [B, R, D]

        # Solve y_r = Λ_rr^{-1} v_r using precomputed Cholesky of Λ_rr
        def solve_block(Lr, br):                       # both [D] (or [D,K])
            y = solve_triangular(Lr, br, lower=True)
            y = solve_triangular(Lr.T, y, lower=False)
            return y

        solve_r = jax.vmap(solve_block, in_axes=(0, 0))           # over r
        def solve_obs(v_one):                                     # v_one: [R,D]
            return solve_r(self.L_blocks, v_one)                  # [R,D]
        y = jax.vmap(solve_obs)(v)                                # [B,R,D]

        cond_means = x_ - y                                       # [B,R,D]
        return cond_means.reshape(x_shape)

    @jax.jit
    def conditional_sample(self, key: jax.random.PRNGKey, x: jnp.ndarray, mu: jnp.ndarray):
        """
        Sample one draw per channel from x_r | x_{-r} for each r.
        Shapes:
          x: [..., R, D]
          mu: [R, D] or broadcastable to x
        Returns:
          sample: [..., R, D]   (same as conditional_means)
        """
        R, D = self.R, self.D
        x_shape = x.shape
        x_ = x.reshape(-1, R, D)

        # 1) conditional mean
        m = self.conditional_means(x, mu).reshape(-1, R, D)

        # 2) standard normal noise shaped like x_
        eps = jax.random.normal(key, shape=x_.shape, dtype=x.dtype)  # [B,R,D]

        # 3) transform per r with Λ_rr^{-1/2} via triangular solves
        eps_RBD = jnp.transpose(eps, (1, 0, 2))  # [R,B,D]

        def noise_r(Lr, eps_r):                  # eps_r: [B,D]
            ZT = solve_triangular(Lr.T, eps_r.T, lower=False)
            return ZT.T                           # [B,D]

        Z_RBD = jax.vmap(noise_r, in_axes=(0, 0))(self.L_blocks, eps_RBD)
        Z = jnp.transpose(Z_RBD, (1, 0, 2))      # [B,R,D]

        return (Z + m).reshape(x_shape)

    def conditional_cov_blocks(self) -> jnp.ndarray:
        """Optional: return Σ_{r|−r} blocks (size [R,D,D]), where Σ_{r|−r} = Λ_rr^{-1}."""
        I = jnp.eye(self.D, dtype=self.L_blocks.dtype)
        def inv_from_chol(Lr):
            Y = solve_triangular(Lr, I, lower=True)
            return solve_triangular(Lr.T, Y, lower=False)
        return jax.vmap(inv_from_chol)(self.L_blocks)

