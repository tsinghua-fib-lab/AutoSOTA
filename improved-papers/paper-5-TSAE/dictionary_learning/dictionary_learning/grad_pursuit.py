"""
Implements batched gradient pursuit algorithm here:
https://www.lesswrong.com/posts/C5KAZQib3bzzpeyrg/full-post-progress-update-1-from-the-gdm-mech-interp-team#Inference_Time_Optimisation:~:text=two%20seem%20promising.-,Details%20of%20Sparse%20Approximation%20Algorithms%20(for%20accelerators),-This%20section%20gets
"""

import torch as t


def _grad_pursuit_update_step(signal, weights, dictionary, batch_arange, selected_features):
    """
    signal: b x d, weights: b x n, dictionary: d x n, batch_arange: b, selected_features: b x n
    """
    residual = signal - t.einsum('bn,dn -> bd', weights, dictionary)
    # choose the element with largest inner product with residual, as in matched pursuit.
    inner_products = t.einsum('dn,bd -> bn', dictionary, residual)
    idxs = t.argmax(inner_products, dim=1)
    # add the new feature to the active set. 
    selected_features[batch_arange, idxs] = 1

    # the gradient for the weights is the inner product, restricted to the chosen features
    grad = selected_features * inner_products
    # the next two steps compute the optimal step size
    c = t.einsum('bn,dn -> bd', grad, dictionary)
    step_size = t.einsum('bd,bd -> b', c, residual) / t.einsum('bd,bd -> b ', c, c)
    weights = weights + t.einsum('b,bn -> bn', step_size, grad)
    weights = t.clip(weights, min=0) # clip the weights to be positive
    return weights, selected_features

def grad_pursuit(signal, dictionary, target_l0 : int = 20, device : str = 'cpu'):
    """
    Inputs: signal: b x d, dictionary: d x n, target_l0: int, device: str
    Outputs: weights: b x n
    """
    assert len(signal.shape) == 2  # makes sure this a batch of signals
    with t.no_grad():
        batch_arange = t.arange(signal.shape[0]).to(device)
        weights = t.zeros((signal.shape[0], dictionary.shape[1])).to(device)
        selected_features = t.zeros((signal.shape[0], dictionary.shape[1])).to(device)
        for _ in range(target_l0):
            weights, selected_features = _grad_pursuit_update_step(
                signal, weights, dictionary, batch_arange, selected_features)
    return weights