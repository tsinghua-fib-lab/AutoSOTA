import jax.numpy as jnp

def tensor_to_vec(tensor: jnp.ndarray) -> jnp.ndarray:
    """
    Flatten a (2,d,P_S) tensor into a (2*d*P_S,) vector

    Parameters
    ----------
    tensor : ArrayType
        data in shape (2, d, P_S)

    Returns
    ----------
    vec: ArrayType
        data in shape (2 * d * P_S), with columns stacked sequentially by axis 0, then axis 2
    """

    if len(tensor.shape) == 4:
        B = tensor.shape[0]
        vec = jnp.reshape(tensor, (B, -1))
    else:
        vec = jnp.reshape(tensor, (-1,))
    return vec

def vec_to_tensor(vec: jnp.ndarray, 
                  d: int, 
                  P_S: int) -> jnp.ndarray:
    """
    Tensorize a (2*d*P_S) or (B,2*d*P_S) vector

    Parameters
    ----------
    vec: ArrayType
        data in shape (2 * d * P_S)
    d : int
        feature dimension
    P_S : int
        hyperplane samples 

    Returns
    ----------
    tensor : ArrayType
        data in shape (2, d, P_S)
    """
    
    if len(vec.shape) > 1:
        tensor = jnp.reshape(vec, (-1, 2, d, P_S))
    else:
        tensor = jnp.reshape(vec, (2, d, P_S))
    return tensor
