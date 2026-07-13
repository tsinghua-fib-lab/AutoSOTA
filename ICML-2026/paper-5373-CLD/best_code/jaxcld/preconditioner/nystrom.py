import jax
import jax.numpy as jnp
import jax.random as jrn
from jax import jit, lax, vmap
from jax.scipy.linalg import solve_triangular
from ..utils.linops_utils import tensor_to_vec, vec_to_tensor
from typing import NamedTuple
from jax import tree_util

class Nys_Precond(NamedTuple):
            U: jnp.ndarray
            S: jnp.ndarray
            d: float
            rho: float
            P_S: int
            
            @jit
            def apply(self,u):
                u = tensor_to_vec(u)
                Utu = self.U.T @ u
                u = (self.S[-1] + self.rho) * (self.U @ (Utu / (self.S + self.rho))) + u - self.U @ Utu
                return vec_to_tensor(u, self.d, self.P_S)
            
            def batch_apply(self,u_s):
              return vmap(self.apply)(u_s)
            
            def _tree_flatten(self):
                children = (self.U, self.S)  # arrays / dynamic values
                aux_data = {'d': self.d, 'rho': self.rho,'P_S': self.P_S}  # static values
                return (children, aux_data)
    
            @classmethod
            def _tree_unflatten(cls, aux_data, children):
                return cls(*children, **aux_data)

tree_util.register_pytree_node(Nys_Precond,
                                Nys_Precond._tree_flatten,
                                Nys_Precond._tree_unflatten)


def rand_nys_appx(model, rank: int, key): #-> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes the Nystrom approximation via sketch A.T@(A@Omega) following Tropp et al. 2017
    Uses linops to compute Y from A; memory efficient
    
    Parameters
    ----------
    OPS: FG_Operators
        FG operators containing A operation
    rank : int
        number of top eigenvalues to flatten in preconditioning
    key: jax.random.PRNGKey
        PRNG key for random number generation

    Returns
    ----------
    U : jnp.ndarray
        first preconditioning matrix
    S : jnp.ndarray
        second preconditioning matrix
    """
    d = model.X.shape[1]
    N = 2 * model.P_S * d
    key,subkey = jrn.split(key)
    Omega = jrn.normal(subkey, (N, rank))  # Generate test matrix
    Omega = jnp.linalg.qr(Omega)[0]

    # Define a function to compute the sketch for a single column
    def compute_sketch(col):
        col_tensor = vec_to_tensor(col, d, model.P_S)
        col_A = model.matvec_A(col_tensor)
        return tensor_to_vec(col_A)

    # Vectorize the function over all columns
    compute_sketch_vmap = jax.vmap(compute_sketch)

    # Compute the sketch for all columns at once
    Y = compute_sketch_vmap(Omega.T).T

    #v = jnp.sqrt(rank) * mnp.jax_spacing(jnp.linalg.norm(Y))
    v = jnp.sqrt(rank) *10**-16*(jnp.linalg.norm(Y))
    Y += v * Omega  # Add shift

    Core = Omega.T @ Y

    
    C = jnp.linalg.cholesky(Core) #Do Cholesky on Core
    #print("Finished computing cholesky on core!")
    
    #print(" Finished computing cholesky on core finally! Now computing B with SVD...")
    # C and Y are already JAX arrays
    B = solve_triangular(C, Y.T, lower=True)

    # Compute SVD
    U, S, _ = lax.linalg.svd(B.T, full_matrices=False)

    S = jax.nn.relu(S**2 - v) # Subtract off shift
    
    return U, S, key

            
            