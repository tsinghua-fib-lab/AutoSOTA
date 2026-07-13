import jax.numpy as jnp
from jax import jit
from functools import partial
from typing import NamedTuple

class PCG_State(NamedTuple):
    u: jnp.ndarray
    r: jnp.ndarray
    z: jnp.ndarray
    p: jnp.ndarray
    r_dot_z: float
    k: int

def pcg(b: jnp.ndarray,
        model: object,
        M: object,
        max_iter: int):
    
    return _pcg(b, model, M ,max_iter)


@partial(jit,static_argnames=['max_iter'])
def _pcg(b: jnp.ndarray,
        model,
        M,
        max_iter: int):
    
    
    #nits = 0

    # def _cond_fun(state):
    #   return (jnp.linalg.norm(state.r)>tol) & (state.k<max_iter)
    
    def _init_pcg():
      r = b
      z = M.batch_apply(r)
      p = jnp.copy(z)
      r_dot_z = jnp.sum(r*z)
      k = 0
      return PCG_State(jnp.zeros_like(b),r, z, p, r_dot_z, k)
    
    def _pcg_step(state):
      w = model.batch_matvec_A(state.p)
      # Update solution and residual
      alpha = state.r_dot_z / jnp.sum(w * state.p)
      u = state.u+alpha * state.p
      r = state.r-alpha * w
      # Apply preconditioner
      z = M.batch_apply(r)

      # Update search direction
      rnp1_dot_znp1 = jnp.sum(r * z)
      p = z + (rnp1_dot_znp1 / state.r_dot_z) * state.p
      return PCG_State(u = u,r= r,z = z, p = p,r_dot_z = rnp1_dot_znp1,k =state.k+1)
    
    state = _init_pcg()

    # while _cond_fun(state) == True:
    #   state = _pcg_step(state)
    # state =lax.while_loop(_cond_fun,_pcg_step,_init_pcg())
  
    for i in range(max_iter):
      state = _pcg_step(state)

    return state.u, state.r, state.k




