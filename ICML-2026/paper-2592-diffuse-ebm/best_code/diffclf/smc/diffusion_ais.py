# Diffusion Annealed Importance sampler implementation

# Libraries
import torch
from tqdm import trange

def diffusion_ais_sampler(x_init, times, base_log_prob_and_grad, target_log_prob_and_grad, grads, sde,
                          integrator_fn, forward_kernel=None, kernel_aux_fn=None,
                          return_intermediates=False, verbose=True):
    """Diffusion Annealed Importance sampler

    Args:
        * x_init (torch.Tensor of shape (n_particles, *data_shape)): Initial samples
        * times (torch.Tensor of shape (n_levels, n_particles, *data_shape_ones)): Value of the intermediate noise levels
        * base_log_prob_and_grad (function): Log-likelihood and score of the base distribution
        * target_log_prob_and_grad (function): Log-likelihood and score of the target distribution
        * grads (function): Intermediate scores
        * sde (LinearSDE): SDE
        * integrator_fn (callable): Integration function
        * forward_kernel (callable): Custom forward kernel (default is None)
        * kernel_aux_fn (callable): Auxiliary kernel quantities (default is None)
        * return_intermediates (bool): Whether to return all the intermediates samples (default is False)
        * verbose (bool): Whether to display a progress bar (default is True)

    Returns:
        * samples (torch.Tensor of shape (n_particles, *data_shape)): Samples at the last noise level
        * weights (torch.Tensor of shape (n_particles,)): Importance weights at level 0
    """
    # Parse the initial point shape
    n_particles = x_init.shape[0]
    data_shape = x_init.shape[1:]
    sum_indexes = tuple(range(1, len(data_shape)+1))
    # Initialize the storage for the weights, gradients and points
    x_prev = x_init.clone()
    log_weights, grad_x_prev = base_log_prob_and_grad(x_prev)
    log_weights *= -1.
    if kernel_aux_fn is not None:
        aux_x_prev = kernel_aux_fn(times[-1], x_prev)
    else:
        aux_x_prev = None
    # Initialize the sample buffer
    if return_intermediates:
        samples = torch.empty((times.shape[0], n_particles, *data_shape), device=x_init.device)
    else:
        samples = torch.empty((n_particles, *data_shape), device=x_init.device)
    x = x_init.clone()
    if verbose:
        r = trange(times.shape[0]-2, -1, -1)
    else:
        r = range(times.shape[0]-2, -1, -1)
    for time_id in r:
        # Set the first point
        x = x.clone()
        # Perform the transition
        x, log_prob_transition_backward = integrator_fn(x_prev, times[time_id], times[time_id+1],
            grad_x_prev, aux_t=aux_x_prev)
        # Update the grad_x
        if time_id == 0:
            def cur_grad(y): return target_log_prob_and_grad(y)[1]
        else:
            def cur_grad(y): return grads(times[time_id], y)
        grad_x = cur_grad(x)
        if kernel_aux_fn is not None:
            aux_x = kernel_aux_fn(times[time_id], x)
        else:
            aux_x = None
        # Compute the forward transition probability
        if forward_kernel:
            log_prob_transition_forward = forward_kernel(x_prev, x, grad_x, times[time_id], times[time_id+1],
                aux_s=aux_x)
        else:
            mean_factor_forward, var_factor_forward = sde.transition_params(times[time_id], times[time_id+1])
            log_prob_transition_forward = -0.5 * \
                torch.sum(torch.square(mean_factor_forward * x - x_prev) / var_factor_forward, dim=sum_indexes)
        # Update the weights
        log_weights += log_prob_transition_forward - log_prob_transition_backward
        # Save the samples
        if return_intermediates:
            samples[time_id] = x.clone()
        elif time_id == 0:
            samples = x.clone()
        # Rename the log-prob
        x_prev = x.clone()
        grad_x_prev = grad_x.clone()
        if kernel_aux_fn is not None:
            aux_x_prev = { k : v.clone() for k,v in aux_x.items() }
    # Evaluate the target log-likelihood
    log_weights += target_log_prob_and_grad(x)[0]
    # Return everything
    return samples, torch.nn.functional.softmax(log_weights, dim=0)
