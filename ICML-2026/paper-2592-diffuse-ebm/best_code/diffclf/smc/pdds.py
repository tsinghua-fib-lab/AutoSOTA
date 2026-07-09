# Particle Denoising Diffusion Sampler implementation

# Libraries
import torch
from tqdm import trange
from ..mcmc.utils import heuristics_step_size
from ..mcmc.samplers import mala_step

def pdds_sampler(x_init, times, log_prob_and_grads, sde, integrator_fn, n_warmup_mcmc_steps, n_mcmc_steps,
    step_sizes_per_noise, reweight_threshold=1.0, target_acceptance=0.75, ignore_mcmc=False,
    forward_kernel=None, kernel_aux_fn=None, is_particles=False, stop_ressampling_fn=None, return_intermediates=False, verbose=True):
    """Particle Denoising Diffusion Sampler 

    Args:
        * x_init (torch.Tensor of shape (n_particles, *data_shape)): Initial samples
        * times (torch.Tensor of shape (n_levels, n_particles, *data_shape_ones)): Value of the intermediate noise levels
        * log_prob_and_grads (function taking t of shape (n_particles, *data_shape_ones) and x of shape (n_particles, *data_shape)): Intermediate distributions
        * sde (LinearSDE): SDE
        * integrator_fn (callable): Integration function
        * n_warmup_mcmc_steps (int): Number of initial warmup steps for each noise level
        * n_mcmc_steps (int): Number of steps for each noise level
        * step_sizes_per_noise (torch.Tensor of size (n_levels, n_particles, *data_shape_ones)): Step size for each noise level
        * reweight_threshold (float): ESS threshold for reweighting (default is 1.0)
        * ignore_mcmc (bool): Whether to ignore MCMC steps (default is False)
        * target_acceptance (float): Target acceptance rate for local steps (default is 0.75)
        * forward_kernel (callable): Custom forward kernel (default is None)
        * kernel_aux_fn (callable): Auxiliary kernel quantities (default is None)
        * is_particles (bool): Whether it is a particle system (default is False)
        * stop_ressampling_fn (bool): Function to forbid samples (default is None)
        * return_intermediates (bool): Whether to return all the intermediates samples (default is False)
        * verbose (bool): Whether to display a progress bar (default is True)

    Returns:
        if return_intermediates:
            * samples (torch.Tensor of shape (n_levels, n_mcmc_steps, n_particles, *data_shape)): Samples at each noise level
        else
            * samples (torch.Tensor of shape (n_mcmc_steps, n_particles, *data_shape)): Samples at the last noise level
        * weights (torch.Tensor of shape (n_particles,)): Importance weights at the 0 level
        * step_sizes_per_noise (torch.Tensor of size (n_levels, n_particles, *data_shape_ones)): Updated step size for each noise level
        * diags (dict): Dictionnary of diagnostics
    """
    # Parse the initial point shape
    n_particles = x_init.shape[0]
    data_shape = x_init.shape[1:]
    sum_indexes = tuple(range(1, len(data_shape)+1))
    # Initialize the storage for the weights, gradiens and points
    log_weights = torch.zeros((n_particles,), device=x_init.device)
    log_prob_x_prev = torch.empty((n_particles,), device=x_init.device)
    x_prev = torch.empty_like(x_init)
    grad_x_prev = torch.empty_like(x_init)
    aux_x_prev = None
    ess_logs = torch.ones((times.shape[0],))
    if stop_ressampling_fn is None:
        stop_ressampling_fn = lambda k : False
    # Initialize the sample buffer
    if ignore_mcmc:
        n_mcmc_steps_ = 1
    else:
        n_mcmc_steps_ = n_mcmc_steps
    if return_intermediates:
        samples = torch.empty((times.shape[0], n_mcmc_steps_, n_particles, *data_shape), device=x_init.device)
    else:
        samples = torch.empty((n_mcmc_steps_, n_particles, *data_shape), device=x_init.device)
    mean_accs = torch.empty((times.shape[0],))
    x = x_init.clone()
    if verbose:
        r = trange(times.shape[0]-1, -1, -1)
    else:
        r = range(times.shape[0]-1, -1, -1)
    for time_id in r:
        # Set the first point
        x = x.clone()
        # Define the current log-prob and score

        def cur_log_prob_and_grad(y):
            return log_prob_and_grads(times[time_id], y)
        # Select the current step size
        if not ignore_mcmc:
            if step_sizes_per_noise.shape == times.shape:
                cur_step_size = step_sizes_per_noise[time_id]
            else:
                cur_step_size = step_sizes_per_noise
        # Compute the inital values
        log_prob_x, grad_x = cur_log_prob_and_grad(x)
        if kernel_aux_fn is not None:
            aux_x = kernel_aux_fn(times[time_id], x)
        else:
            aux_x = None
        # Move the particles with the SDE reverse kernel
        if (time_id != times.shape[0]-1):
            # Perform the transition
            x, log_prob_transition_backward = integrator_fn(x_prev, times[time_id], times[time_id+1],
                grad_x_prev, aux_t=aux_x_prev)
            # Update the log_prob_x and grad_x
            log_prob_x, grad_x = cur_log_prob_and_grad(x)
            if kernel_aux_fn is not None:
                aux_x = kernel_aux_fn(times[time_id], x)
            # Compute the forward transition probability
            if forward_kernel:
                log_prob_transition_forward = forward_kernel(x_prev, x, grad_x, times[time_id], times[time_id+1],
                    aux_s=aux_x)
            else:
                mean_factor_forward, var_factor_forward = sde.transition_params(times[time_id], times[time_id+1])
                log_prob_transition_forward = -0.5 * \
                    torch.sum(torch.square(mean_factor_forward * x - x_prev) / var_factor_forward, dim=sum_indexes)
            # Compute the weights
            with torch.no_grad():
                log_weights += log_prob_x - log_prob_x_prev
                log_weights += log_prob_transition_forward - log_prob_transition_backward
            # Perform ressampling
            if reweight_threshold > 0.0:
                # Compute the ESS
                weights = torch.nn.functional.softmax(log_weights, dim=0)
                ess = (1.0 / torch.sum(torch.square(weights))) / n_particles
                ess_logs[time_id] = ess.cpu().clone()
                # Ressample the particles
                if (ess < reweight_threshold) and (not stop_ressampling_fn(time_id)):
                    idx = torch.multinomial(weights, n_particles, replacement=True)
                    x = x[idx]
                    log_prob_x = log_prob_x[idx]
                    grad_x = grad_x[idx]
                    if kernel_aux_fn is not None:
                        aux_x = { k : v[idx] for k,v in aux_x.items() }
                    log_weights.zero_()
        # Run the local sampler
        if ignore_mcmc:
            if return_intermediates:
                samples[time_id, 0] = x.clone()
            else:
                samples[0] = x.clone()
        else:
            sum_acc = 0.0
            for step_id in range(n_warmup_mcmc_steps+n_mcmc_steps):
                # Perform the MCMC step
                x, log_prob_x, grad_x, log_acc = mala_step(x, log_prob_x, grad_x, cur_log_prob_and_grad,
                    cur_step_size, is_particles=is_particles)
                # Store the acceptance and update the step size
                acc = torch.exp(torch.minimum(torch.zeros_like(log_acc), log_acc))
                if step_id >= n_warmup_mcmc_steps:
                    sum_acc += acc
                if target_acceptance > 0.0:
                    cur_step_size = heuristics_step_size(cur_step_size, log_acc, target_acceptance=target_acceptance)
                # Store the sample
                if step_id >= n_warmup_mcmc_steps:
                    if return_intermediates:
                        samples[time_id, step_id - n_warmup_mcmc_steps] = x.clone()
                    else:
                        if time_id == 0:
                            samples[step_id - n_warmup_mcmc_steps] = x.clone()
            if kernel_aux_fn is not None:
                aux_x = kernel_aux_fn(times[time_id], x)
            # Store the mean acceptance and the step size
            mean_accs[time_id] = (sum_acc / n_mcmc_steps).mean()
            if step_sizes_per_noise.shape == times.shape:
                step_sizes_per_noise[time_id] = cur_step_size.clone()
            else:
                step_sizes_per_noise = cur_step_size.clone()
        # Display the logs
        if verbose:
            if ignore_mcmc:
                logs = {}
            else:
                logs = {'local_acc': mean_accs[time_id].item()}
            if reweight_threshold > 0.0:
                logs['ess'] = ess_logs[time_id].item()
            r.set_postfix(**logs)
        # Rename the log-prob
        x_prev = x.clone()
        grad_x_prev = grad_x.clone()
        log_prob_x_prev = log_prob_x.clone()
        if kernel_aux_fn is not None:
            aux_x_prev = { k : v.clone() for k,v in aux_x.items() }
    # Return everything
    if ignore_mcmc:
        diags = {}
    else:
        diags = {'local_acc': mean_accs}
    if reweight_threshold > 0.0:
        diags['ess'] = ess_logs
    return samples, torch.nn.functional.softmax(log_weights, dim=0), step_sizes_per_noise, diags


def pdds_annealed_sampler(x_init, times, log_prob_and_grads, sde, integrator_fn, n_warmup_mcmc_steps, n_mcmc_steps,
    step_sizes_per_noise, target_acceptance=0.75, is_particles=False, return_intermediates=False, verbose=True):
    """Particle Denoising Diffusion Sampler with no reweightinh

    Args:
        * x_init (torch.Tensor of shape (n_particles, *data_shape)): Initial samples
        * times (torch.Tensor of shape (n_levels, n_particles, *data_shape_ones)): Value of the intermediate noise levels
        * log_prob_and_grads (function taking t of shape (n_particles, *data_shape_ones) and x of shape (n_particles, *data_shape)): Intermediate distributions
        * sde (LinearSDE): SDE
        * integrator_fn (callable): Integration function
        * n_warmup_mcmc_steps (int): Number of initial warmup steps for each noise level
        * n_mcmc_steps (int): Number of steps for each noise level
        * step_sizes_per_noise (torch.Tensor of size (n_levels, n_particles, *data_shape_ones)): Step size for each noise level
        * target_acceptance (float): Target acceptance rate for local steps (default is 0.75)
        * is_particles (bool): Whether it is a particle system (default is False)
        * return_intermediates (bool): Whether to return all the intermediates samples (default is False)
        * verbose (bool): Whether to display a progress bar (default is True)

    Returns:
        if return_intermediates:
            * samples (torch.Tensor of shape (n_levels, n_mcmc_steps, n_particles, *data_shape)): Samples at each noise level
        else
            * samples (torch.Tensor of shape (n_mcmc_steps, n_particles, *data_shape)): Samples at the last noise level
        * step_sizes_per_noise (torch.Tensor of size (n_levels, n_particles, *data_shape_ones)): Updated step size for each noise level
        * diags (dict): Dictionnary of diagnostics
    """
    # Parse the initial point shape
    n_particles = x_init.shape[0]
    data_shape = x_init.shape[1:]
    # Initialize the storage for the weights, gradiens and points
    # if reweight_threshold > 0.0:
    x_prev = torch.empty_like(x_init)
    grad_x_prev = torch.empty_like(x_init)
    # Initialize the sample buffer
    if return_intermediates:
        samples = torch.empty((times.shape[0], n_mcmc_steps, n_particles, *data_shape), device=x_init.device)
    else:
        samples = torch.empty((n_mcmc_steps, n_particles, *data_shape), device=x_init.device)
    mean_accs = torch.empty((times.shape[0],))
    x = x_init.clone()
    if verbose:
        r = trange(times.shape[0]-2, -1, -1)
    else:
        r = range(times.shape[0]-2, -1, -1)
    for time_id in r:
        # Set the first point
        x = x.clone()
        # Define the current log-prob and score
        def cur_log_prob_and_grad(y):
            return log_prob_and_grads(times[time_id], y)
        # Select the current step size
        if step_sizes_per_noise.shape == times.shape:
            cur_step_size = step_sizes_per_noise[time_id]
        else:
            cur_step_size = step_sizes_per_noise
        # Move the particles with the SDE reverse kernel
        x = integrator_fn(x_prev, times[time_id], times[time_id+1], grad_x_prev)
        # Compute update the log_prob and gradient
        log_prob_x, grad_x = cur_log_prob_and_grad(x)
        if return_intermediates:
            samples[time_id, 0] = x.clone()
        else:
            samples[0] = x.clone()
        # Run the MCMC steps
        sum_acc = 0.0
        for step_id in range(n_warmup_mcmc_steps+n_mcmc_steps):
            # Perform the MCMC step
            x, log_prob_x, grad_x, log_acc = mala_step(x, log_prob_x, grad_x,
                cur_log_prob_and_grad, cur_step_size, is_particles=is_particles)
            # Store the acceptance and update the step size
            acc = torch.exp(torch.minimum(torch.zeros_like(log_acc), log_acc))
            if step_id >= n_warmup_mcmc_steps:
                sum_acc += acc
            if target_acceptance > 0.0:
                cur_step_size = heuristics_step_size(cur_step_size, log_acc, target_acceptance=target_acceptance)
            # Store the sample
            if step_id >= n_warmup_mcmc_steps:
                if return_intermediates:
                    samples[time_id, step_id - n_warmup_mcmc_steps] = x.clone()
                else:
                    if time_id == 0:
                        samples[step_id - n_warmup_mcmc_steps] = x.clone()
        # Store the mean acceptance and the step size
        mean_accs[time_id] = (sum_acc / n_mcmc_steps).mean()
        if step_sizes_per_noise.shape == times.shape:
            step_sizes_per_noise[time_id] = cur_step_size.clone()
        else:
            step_sizes_per_noise = cur_step_size.clone()
        # Display the logs
        if verbose:
            r.set_postfix(local_acc=mean_accs[time_id].item())
        # Rename the log-prob
        x_prev = x.clone()
        grad_x_prev = grad_x.clone()
    # Return everything
    diags = { 'local_acc': mean_accs }
    return samples, step_sizes_per_noise, diags
