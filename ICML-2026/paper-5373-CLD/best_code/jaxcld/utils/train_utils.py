import jax.random as jrn
from jax import jit
from functools import partial

@partial(jit, static_argnames = ['ntr', 'batch_size'])
def get_batch(key, ntr, batch_size): 
    return jrn.choice(key, ntr, (batch_size,))