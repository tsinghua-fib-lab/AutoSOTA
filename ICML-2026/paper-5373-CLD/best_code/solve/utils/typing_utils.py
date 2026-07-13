"""
Typing functions to convert and infer backend type or datatype of array_like objects
This mostly deals with converting between numpy, torch, and jax datatypes, as of jaxv3 not needed
"""

from typing import Union, Callable
import numpy as np
import jax.numpy as jnp
import torch

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap,  pmap
from jax import random
from jax import lax
from jax import make_jaxpr 
from jax import config
from jax import device_put


CPUS = jax.devices("cpu")
GPUS = jax.devices("gpu")

DATATYPE_BACKENDS = ["numpy", "torch", "jax"]

# default precision as 32; can change to 64 but may mess up memory
TORCH_DTYPE = torch.float32
JAX_DTYPE = jnp.float32
NP_DTYPE = np.float32

# type assertion for any kind of array
ArrayType = Union[np.ndarray, np.array, torch.Tensor, jnp.ndarray, jnp.array]

# type for any scalars
ScalarTypes = [int, float,
               np.int16, np.int32, np.int64, np.float16, np.float32, np.float64,
               jnp.int16, jnp.int32, jnp.int64, jnp.float16, jnp.float32, jnp.float64,
               torch.int16, torch.int32, torch.int64, torch.float16, torch.float32, torch.float64]

# type assertion for loss or accuracy function
EvalFunction = Callable[[ArrayType, ArrayType], float]

def get_backend_type(x: ArrayType) -> str:
    """
    Get the backend type for an array_like object
    """
        
    if isinstance(x, np.ndarray):
        return "numpy"
    elif isinstance(x, jnp.ndarray):
        return "jax"
    elif torch.is_tensor(x):
        return "torch"
    elif type(x) in ScalarTypes:
        return "scalar"
    else:
        raise NotImplementedError(f"type of input not valid backend datatype. got type: {type(x)}")
    
def convert_backend_type(x: ArrayType, 
                        target_backend: str,
                        device: str = "cpu",
                        verbose: bool = False) -> ArrayType:
    """
    Convert input data to the right type
    Input x must be one of [np.ndarray, np.array, torch.tensor, jax.numpy.ndarray, jax.numpy.array]
    If jax or torch, load to the device too
    """
        
    assert target_backend in DATATYPE_BACKENDS, f"Parameter \"target_backend\" must be one of {DATATYPE_BACKENDS}." 

    current_backend = get_backend_type(x)
    
    if target_backend == "torch":
        # convert numpy to torch
        if current_backend == "numpy":
            if verbose: print("\tConverting data from numpy to torch.")
            return torch.from_numpy(x).to(TORCH_DTYPE).to(device)
        # convert jax to torch
        elif current_backend == "jax":
            if verbose: print("\tConverting data from jax to torch.")
            return torch.from_numpy(np.asarray(x)).to(TORCH_DTYPE).to(device)
        
    if target_backend== "jax":
        # convert numpy to jax
        if current_backend == "numpy":
            if verbose: print("\tConverting data from numpy to jax.")
            return jnp.array(x, dtype=JAX_DTYPE)
        # convert torch to jax
        elif current_backend == "torch":
            if verbose: print("\tConverting data from torch to jax.")
            return jnp.array(x.cpu().numpy(), dtype=JAX_DTYPE)
        
    if target_backend == "numpy":
        # convert torch to numpy
        if current_backend == "torch":
            if verbose: print("\tConverting data from torch to numpy.")
            return x.cpu().numpy().astype(NP_DTYPE)
        # convert jax to numpy
        elif current_backend == "jax":
            if verbose: print("\tConverting data from jax to numpy.")
            return np.asarray(x, dtype=NP_DTYPE)
        
    # if no conversion needed, just return identity
    if verbose: print("\tNo conversion needed; data already on correct backend.")
    return x
    
def as_default_datatype(x: ArrayType) -> ArrayType:
    """
    Convert array x of any datatype to its default datatype listed in constants section above
    """

    backend_type = get_backend_type(x)

    if backend_type == "torch":
        return x.to(TORCH_DTYPE)
    elif backend_type == "jax":
        return x.astype(JAX_DTYPE)
    else:
        return x.astype(NP_DTYPE)

    