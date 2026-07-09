# Classic Diffusion Replica Exchange implementation

# Libraries
import torch
from tqdm import trange
from .utils import make_re_pairings
from ..mcmc.utils import heuristics_step_size
from ..mcmc.samplers import mala_step
from ..utils.se3_utils import remove_mean

def diffusion_re_sde_step(x, log_prob_x, grad_x, log_prob_and_grads, times, idx_i, idx_j,
        forward_kernel, forward_kernel_log_prob, backward_kernel, backward_kernel_log_prob,
        kernel_aux_x=None, kernel_aux_fn=None, kernel_aux_shapes_ones=None, **kwargs):
    """Make a diffusion swap step with stochastic transition kernels

    Args:
        * x (torch.Tensor of shape (n_levels, batch_size, *data_shape)): Input samples (at each noise level)
        * log_prob_x (torch.Tensor of shape (n_levels, batch_size,)): Log-prob of the input samples at each noise level
        * grad_x (torch.Tensor of shape (n_levels, batch_size, *data_shape)): Score of the input samples at each noise level
        * log_prob_and_grads (function taking t of shape (K, batch_size, *data_shape_ones) and x of shape (K, batch_size, *data_shape)): Intermediate distributions
        * times (torch.Tensor of shape (n_levels, batch_size, *data_shape_ones)): Intermediate times
        * idx_i, idx_j (torch.Tensor of shape (n_pairs,)): Indexes of the different pairs
        * forward_kernel (callable): Forward kernel
        * forward_kernel_log_prob (callable): Log-likelihood of the forward kernel
        * backward_kernel (callable): Backward kernel
        * backward_kernel_log_prob (callable): Log-likelihood of the backward kernel
        * kernel_aux_x (dict of torch.Tensor): Auxiliary objects (default is None)
        * kernel_aux_fn (callable): Function to compute the auxiliary objects (default is None)
        * kernel_aux_shapes_ones (dict of tuple): Shapes of the kernel aux objects (default is None)

    Returns:
        * x (torch.Tensor of shape (n_levels * batch_size, *data_shape)): Input samples (at each noise level)
        * log_prob_x (torch.Tensor of shape (n_levels * batch_size,)): Log-prob of the input samples at each noise level
        * grad_x (torch.Tensor of shape (n_levels * batch_size, *data_shape)): Score of the input samples at each noise level
        * aux_x (dict of torch.Tensor): Auxiliary kernel stuff
        * re_acc (float): Mean swap acceptance
    """
    # Get the times
    t_i, t_j = times[idx_i], times[idx_j]
    # Extract and initialize x_i and x_j tensors
    x_i, x_j = x[idx_i], x[idx_j]
    log_prob_i_x_i, log_prob_j_x_j = log_prob_x[idx_i], log_prob_x[idx_j]
    grad_i_x_i, grad_j_x_j = grad_x[idx_i], grad_x[idx_j]
    if kernel_aux_x is not None:
        aux_i_x_i = { k : v[idx_i] for k,v in kernel_aux_x.items() }
        aux_j_x_j = { k : v[idx_j] for k,v in kernel_aux_x.items() }
    else:
        aux_i_x_i, aux_j_x_j = None, None
    # Get y_i from x_j by denoising
    y_i, log_prob_kernel_y_i_x_j = backward_kernel(x_j, t_i, t_j, grad_j_x_j, aux_t_x_t=aux_j_x_j)
    # Get y_j from x_i by noising
    y_j, log_prob_kernel_y_j_x_i = forward_kernel(x_i, t_i, t_j, grad_i_x_i, aux_s_x_s=aux_i_x_i)
    # Compute the log-likelihood and scores of y_i and y_j
    log_prob_i_y_i, grad_i_y_i = log_prob_and_grads(t_i, y_i)
    log_prob_j_y_j, grad_j_y_j = log_prob_and_grads(t_j, y_j)
    if kernel_aux_x is not None:
        aux_i_y_i = kernel_aux_fn(t_i, y_i)
        aux_j_y_j = kernel_aux_fn(t_j, y_j)
    else:
        aux_i_y_i, aux_j_y_j = None, None
    # Log-likelihood of getting x_i from y_j from denoising
    log_prob_kernel_x_i_y_j = backward_kernel_log_prob(x_i, y_j, t_i, t_j, grad_j_y_j, aux_t_x_t=aux_j_y_j)
    # Log-likelihood of getting x_j by noising y_i
    log_prob_kernel_x_j_y_i = forward_kernel_log_prob(x_j, y_i, t_i, t_j, grad_i_y_i, aux_s_x_s=aux_i_y_i)
    # Compute the acceptance ratio
    with torch.no_grad():
        log_acc = (log_prob_i_y_i + log_prob_j_y_j) - (log_prob_i_x_i + log_prob_j_x_j)
        log_acc += log_prob_kernel_x_i_y_j - log_prob_kernel_y_i_x_j
        log_acc += log_prob_kernel_x_j_y_i - log_prob_kernel_y_j_x_i
    mask = (torch.log(torch.rand_like(log_acc)) < log_acc).float()
    # Update acceptance counter
    swap_acc = mask.mean()
    # Use the acceptance mask to swap values in-place
    mask_shape = mask.shape
    data_shape_ones = (1,) * (len(x.shape)-2)
    mask = mask.view((*mask_shape, *data_shape_ones))  # Expand dimensions to match data shape
    x[idx_i] = mask * y_i + (1 - mask) * x_i
    x[idx_j] = mask * y_j + (1 - mask) * x_j
    grad_x[idx_i] = mask * grad_i_y_i + (1 - mask) * grad_i_x_i
    grad_x[idx_j] = mask * grad_j_y_j + (1 - mask) * grad_j_x_j
    mask = mask.view(mask_shape)
    log_prob_x[idx_i] = mask * log_prob_i_y_i + (1 - mask) * log_prob_i_x_i
    log_prob_x[idx_j] = mask * log_prob_j_y_j + (1 - mask) * log_prob_j_x_j
    if kernel_aux_x is not None:
        for k in kernel_aux_x.keys():
            mask = mask.view((*mask_shape, *kernel_aux_shapes_ones[k]))
            kernel_aux_x[k][idx_i] = mask * aux_i_y_i[k] + (1 - mask) * aux_i_x_i[k]
            kernel_aux_x[k][idx_j] = mask * aux_j_y_j[k] + (1 - mask) * aux_j_x_j[k]
    # Return the updated variables and the average acceptance rate
    return x, log_prob_x, grad_x, kernel_aux_x, swap_acc

def diffusion_re_sampler(x_init, forward_kernel, forward_kernel_log_prob, backward_kernel,
                        backward_kernel_log_prob, times, log_prob_and_grads, swap_frequency,
                        n_warmup_mcmc_steps, n_mcmc_steps, step_sizes_per_noise, kernel_aux_fn=None,
                        per_noise_init=False, ignore_mcmc=False, target_acceptance=0.75,
                        is_particles=False, return_intermediates=False, verbose=True, **kwargs):
    """Diffusion Replica-Exchange sampler

    Args:
        * x_init (torch.Tensor of shape (n_levels, batch_size, *data_shape) if per_noise_init and (batch_size, *data_shape) otherwise): Initial samples
        * forward_kernel (callable): Forward kernel
        * forward_kernel_log_prob (callable): Log-likelihood of the forward kernel
        * backward_kernel (callable): Backward kernel
        * backward_kernel_log_prob (callable): Log-likelihood of the backward kernel
        * times (torch.Tensor of shape (n_levels, batch_size, *data_shape_ones)): Value of the intermediate noise levels
        * log_prob_and_grads (function taking t of shape (batch_size, *data_shape_ones) and x of shape (batch_size, *data_shape)): Intermediate distributions
        * swap_frequency (int): Number of MCMC steps between each local step
        * n_warmup_mcmc_steps (int): Number of initial warmup steps for each noise level
        * n_mcmc_steps (int): Total number of MCMC steps (Swap + Local)
        * kernel_aux_fn (callable): Auxiliary kernel quantities (default is None)
        * step_sizes_per_noise (torch.Tensor of size (n_levels, batch_size, *data_shape_ones)): Step size for each noise level
        * per_noise_init (bool): Whether x_init contains per noise initialization (default is False)
        * ignore_mcmc (bool): Whether to ignore MCMC steps (default is False)
        * target_acceptance (float): Target acceptance rate for local steps (default is 0.75)
        * is_particles (bool): Whether it is a particle system (default is False)
        * return_intermediates (bool): Whether to return all the intermediates samples (default is False)
        * verbose (bool): Whether to display a progress bar (default is True)

    Returns:
        if return_intermediates:
            * samples (torch.Tensor of shape (n_levels, n_mcmc_steps, batch_size, *data_shape)): Samples at each noise level
        else
            * samples (torch.Tensor of shape (n_mcmc_steps, batch_size, *data_shape)): Samples at the last noise level
        * step_sizes_per_noise (torch.Tensor of size (n_levels, batch_sizes, *data_shape_ones)): Updated step size for each noise level
        * diags (dict): Dictionnary of diagnostics
    """
    # Get the shapes
    if per_noise_init:
        batch_size = x_init.shape[1]
        data_shape = x_init.shape[2:]
    else:
        batch_size = x_init.shape[0]
        data_shape = x_init.shape[1:]
    data_shape_ones = (1,) * len(data_shape)
    # Initalize the storage
    if return_intermediates:
        samples = torch.empty((times.shape[0], n_mcmc_steps, batch_size, *data_shape), device=x_init.device)
    else:
        samples = torch.empty((n_mcmc_steps, batch_size, *data_shape), device=x_init.device)
    mean_local_accs = torch.zeros((times.shape[0],))
    mean_swap_acc = 0.0
    # Define the scores and log_probs
    time_ones = times.reshape((-1, *data_shape_ones))
    def local_log_prob_and_grads(y):
        return log_prob_and_grads(time_ones, y)
    def log_prob_and_grads_batched(t, y):
        log_prob, grad = log_prob_and_grads(t.view((-1, *data_shape_ones)), y.view((-1, *data_shape)))
        return log_prob.view(y.shape[:-len(data_shape)]), grad.view(y.shape)
    # Get the initial point
    if per_noise_init:
        x = x_init.clone()
    else:
        x = x_init.unsqueeze(0).repeat((times.shape[0], 1, *data_shape_ones))
    x = x.view((-1, *data_shape))
    if not ignore_mcmc:
        step_sizes_per_noise = step_sizes_per_noise.view((-1, *data_shape_ones))
    # Initialize log_prob_x and grad_x
    log_prob_x, grad_x = local_log_prob_and_grads(x)
    # Compute the auxiliaries
    if kernel_aux_fn is not None:
        aux_x = kernel_aux_fn(time_ones, x)
        aux_shapes = { k : v.shape[1:] for k,v in aux_x.items() }
        aux_shapes_ones = { k : (1,) * len(v) for k,v in aux_shapes.items() }
        def kernel_aux_fn_batched(t, y):
            ret = kernel_aux_fn(t.view((-1, *data_shape_ones)), y.view((-1, *data_shape)))
            return { k : v.view((*y.shape[:-len(data_shape)], *aux_shapes[k])) for k,v in ret.items() }
    else:
        aux_x = None
        aux_shapes = None
        aux_shapes_ones = None
        kernel_aux_fn_batched = None
    # Make the pairings
    pairs = make_re_pairings(times.shape[0], x_init.device)
    # Run the algorithm
    if verbose:
        r = trange(n_warmup_mcmc_steps+n_mcmc_steps)
    else:
        r = range(n_warmup_mcmc_steps+n_mcmc_steps)
    for step_id in r:
        if (step_id % swap_frequency == 0) or ignore_mcmc:
            # Select the pairs
            if ignore_mcmc:
                swap_id = step_id % 2
            else:
                swap_id = int(step_id // swap_frequency) % 2
            # Get the auxiliary object
            if aux_x is not None:
                if (step_id > 0) and not ignore_mcmc:
                    aux_x = kernel_aux_fn(time_ones, x)
                aux_x = { k : v.view((-1, batch_size, *aux_shapes[k])) for k,v in aux_x.items() }
            # Reshape the data into (n_levels, batch_size, ...)
            x = x.view((-1, batch_size, *data_shape))
            grad_x = grad_x.view((-1, batch_size, *data_shape))
            log_prob_x = log_prob_x.view((-1, batch_size))
            # Perform the step
            x, log_prob_x, grad_x, aux_x, re_acc = diffusion_re_sde_step(
                x=x, log_prob_x=log_prob_x, grad_x=grad_x, log_prob_and_grads=log_prob_and_grads_batched,
                times=times, idx_i=pairs[swap_id][:, 0], idx_j=pairs[swap_id][:, 1],
                forward_kernel=forward_kernel, forward_kernel_log_prob=forward_kernel_log_prob,
                backward_kernel=backward_kernel, backward_kernel_log_prob=backward_kernel_log_prob,
                kernel_aux_x=aux_x, kernel_aux_fn=kernel_aux_fn_batched, kernel_aux_shapes_ones=aux_shapes_ones
            )
            # Reshape the data into (n_levels * batch_size, ....)
            x = x.view((-1, *data_shape))
            grad_x = grad_x.view((-1, *data_shape))
            log_prob_x = log_prob_x.flatten()
            if aux_x is not None:
                aux_x = { k : v.view((-1, *aux_shapes[k])) for k,v in aux_x.items() }
            mean_swap_acc = re_acc
        else:
            # Perform the local step
            x, log_prob_x, grad_x, log_acc = mala_step(
                x, log_prob_x, grad_x, local_log_prob_and_grads, step_sizes_per_noise,
                is_particles=is_particles
            )
            # Adapt the step-size
            if target_acceptance > 0.0:
                step_sizes_per_noise = heuristics_step_size(
                    step_sizes_per_noise, log_acc, target_acceptance=target_acceptance)
            # Log the acceptance
            acc = torch.exp(torch.minimum(torch.zeros_like(log_acc), log_acc))
            mean_local_accs = acc.view((-1, batch_size)).mean(dim=-1)        
        # Store the samples
        if step_id >= n_warmup_mcmc_steps:
            if return_intermediates:
                samples[:, step_id-n_warmup_mcmc_steps] = x.reshape((-1, batch_size, *data_shape)).clone()
            else:
                samples[step_id-n_warmup_mcmc_steps] = x.view((-1, batch_size, *data_shape))[0].clone()
        if verbose:
            diags = { 'swap_acc': mean_swap_acc.item() }
            if not ignore_mcmc:
                diags['local_acc'] = mean_local_accs.mean().item()
            r.set_postfix(**diags)
    # Return the samples
    final_diags = { 'swap_acc': mean_swap_acc }
    if not ignore_mcmc:
        final_diags['local_acc'] = mean_local_accs
    return samples, step_sizes_per_noise, final_diags
