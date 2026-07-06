import os, yaml
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib as mpl
import matplotlib.pyplot as plt
import imageio
from einops import reduce, rearrange
from src_pr.unet_new import generalized_image_to_b_xy_c, generalized_b_xy_c_to_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# fancy plots
def hdr_plot_style():
    plt.style.use('dark_background')
    mpl.rcParams.update({'font.size': 18, 'lines.linewidth': 3, 'lines.markersize': 15})
    mpl.rcParams['ps.useafm'] = True
    mpl.rcParams['pdf.use14corefonts'] = True
    mpl.rcParams['text.usetex'] = False
    plt.rc('legend', facecolor='#666666EE', edgecolor='white', fontsize=16)
    plt.rc('grid', color='white', linestyle='solid')
    plt.rc('text', color='white')
    plt.rc('xtick', direction='out', color='white')
    plt.rc('ytick', direction='out', color='white')
    plt.rc('patch', edgecolor='#E6E6E6')
hdr_plot_style()

def fix_seeds(seed=42):
    # Fold seeds into the range accepted by NumPy legacy RNG.
    s = int(seed) % (2**32 - 1)
    if s <= 0:
        s = 1
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    np.random.seed(s)

def image_to_b_xy_c(tensor):
    """
    Transpose the tensor from [batch, channels, x, y] to [batch, x*y, channels].
    """
    assert len(tensor.shape) == 4, 'Input tensor must have shape [batch, channels, x, y].'
    batch_size, channels, pixels_x, pixels_y = tensor.shape
    return torch.permute(tensor, (0,2,3,1)).view(batch_size, pixels_x*pixels_y, channels)

def b_xy_c_to_image(tensor, pixels_x = None, pixels_y = None):
    """
    Transpose the tensor from [batch, x*y, channels] to [batch, channels, x, y].
    """
    assert len(tensor.shape) == 3, 'Input tensor must have shape [batch, x*y, channels].'
    batch_size, pixels_x_times_y, channels = tensor.shape
    if pixels_x is None and pixels_y is None:
        assert np.sqrt(pixels_x_times_y) % 1 == 0, 'Number of pixels must be a perfect square.'
        pixels_x = pixels_y = int(np.sqrt(pixels_x_times_y))
    else:
        assert pixels_x*pixels_y == pixels_x_times_y, 'Number of given pixels must match dim 1 of input tensor.'
    return torch.permute(tensor.view(batch_size, pixels_x, pixels_y, channels), (0,3,1,2))

def resize_image(tensor, target_size):
    """
    Transpose the tensor from [batch, channels, ..., pixel_x, pixel_y] to [batch, pixel_x*pixel_y, channels, ...]. We assume two pixel dimensions.
    """
    assert len(tensor.shape) > 3, f"Expected image, got {tensor.shape}"
    original_shape = tensor.shape
    batch_size = original_shape[0]
    num_dims = len(tensor.shape) - 3  # Subtracting batch and pixel dimensions
    pattern = 'b ' + ' '.join([f'c{i}' for i in range(num_dims)]) + ' x y -> b' + ' (' + ' '.join([f'c{i}' for i in range(num_dims)]) + ') ' + 'x y'
    tensor = rearrange(tensor, pattern)
    tensor = transforms.Resize((target_size, target_size), antialias=False)(tensor).view(batch_size, *original_shape[1:-2], target_size, target_size)
    return tensor

def noop(*args, **kwargs):
    pass

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

def plot_data(data):
    plt.figure(figsize=(16, 12))
    plt.scatter(*data, s=10)
    plt.title(r'Ground truth $q(\mathbf{x}_{0})$')
    plt.show()
    exit()

def sample_zeros(size):
    return np.zeros((size, 2))

def sample_gaussian(size, dim=2):
    return np.random.randn(size, dim)

def sample_circle(size):
    # sample points from a circle
    theta = np.random.uniform(0, 2*np.pi, size)
    x = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    return x

def sample_hypersphere(size, dim):
    # sample points from a hypersphere surface in `dim` dimensions
    x = np.random.normal(0, 1, (size, dim))
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    x_normalized = x / norm    
    return x_normalized

def sample_two_points(size):
    # two points in 2D
    x = np.array([[-0.5, -0.5], [0.5, 0.5]])
    # random selection of these two points
    return x[np.random.randint(2, size=size)]

def sample_four_points(size):
    # four points in 2D
    x = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    # random selection of these four points
    return x[np.random.randint(4, size=size)]

def sample_images_with_squares(no_points, pixels_per_dim, dim, frame_dim = False, use_double = True):

    if use_double:
        dtype = np.float64
    else:
        dtype = np.float32

    # Define the size of the square (e.g., a quarter of the image dimension)
    square_size = pixels_per_dim // 4

    # Initialize an array to store the images
    # Shape: (no_points, pixels_per_dim, pixels_per_dim, dim)
    if frame_dim:
        images = np.zeros((no_points, dim, 1, pixels_per_dim, pixels_per_dim), dtype=dtype)
    else:
        images = np.zeros((no_points, dim, pixels_per_dim, pixels_per_dim), dtype=dtype)

    for i in range(no_points):
        # Randomly choose the top-left corner of the square
        x_start = np.random.randint(0, pixels_per_dim - square_size)
        y_start = np.random.randint(0, pixels_per_dim - square_size)

        for j in range(dim):
            # Draw the square in each channel of the image
            # You can modify the pattern per channel as needed
            if frame_dim:
                images[i, j, :, x_start:x_start + square_size, y_start:y_start + square_size] = 1.
            else:
                images[i, j, x_start:x_start + square_size, y_start:y_start + square_size] = 1.
                
    return images

def sample_ones(no_points, pixels_per_dim, dim, frame_dim = False):
    if frame_dim:
        return np.ones((no_points, dim, 1, pixels_per_dim, pixels_per_dim), dtype=float)
    else:
        return np.ones((no_points, dim, pixels_per_dim, pixels_per_dim), dtype=float)

class EMA(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}
        self.backup = {}

    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module, backup=True):
        for name, param in module.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                if backup:
                    self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name].data)

    def restore(self, module):
        assert hasattr(self, 'backup')
        for name, param in module.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data.copy_(self.backup[name])
        self.backup = {}

    def ema_copy(self, module):
        module_copy = type(module)(module.config).to(module.config.device)
        module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

def remove_outliers(data, percentile = 0.01, also_lower_bound = False):    
    percentile *= 100    
    if data.size == 0:
        return data  # Return the empty array as is
    
    norms = np.linalg.norm(data, axis=1)
    
    # compute the lower and upper bounds for filtering based on norms
    lower_bound = np.percentile(norms, percentile) if also_lower_bound else 0.
    upper_bound = np.percentile(norms, 100 - percentile)
    mask = (norms > lower_bound) & (norms < upper_bound)
    return data[mask]

# tensor of shape (channels, frames, height, width) -> gif
def array_to_gif(data, output_save_dir, x_lim, y_lim, label = None, duration = 0.05):    
    # Create a GIF writer object
    with imageio.get_writer(output_save_dir, mode='I', duration = duration, loop=1) as writer:
        for step in range(data.shape[0]):
            fig, ax = plt.subplots()
            ax.scatter(data[step, :, 0], data[step, :, 1], s=10)
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            if label is not None:
                if label == 'sampled':
                    ax.set_title(r'$p(\mathbf{x}_{' + str(len(data)-step-1)+'})$')
                elif label == 'pred':
                    ax.set_title('Model pred. step ' + str(len(data)-step-1))
                else:
                    ax.set_title(label)
            # Save the current figure directly to the GIF, without an intermediate file
            gif_frame_path = output_save_dir[:-4] + f'_gif_frame_{step}.png'
            plt.savefig(gif_frame_path)
            plt.close(fig)
            writer.append_data(imageio.imread(gif_frame_path))
            # remove the temporary image file after adding to the GIF
            os.remove(gif_frame_path)

def image_array_to_gif(image_array, output_file, frame_duration=0.05, normalization_mode='final_pred', given_min_max=None):
    """
    Create a GIF from a numpy array of images with different normalization modes.

    :param image_array: Numpy array of shape (frames, pixel, pixel)
    :param output_file: Output file path for the GIF
    :param frame_duration: Duration of each frame in the GIF
    :param normalization_mode: Mode of normalization ('given', 'global', 'individual', 'none')
    :param given_min_max: Tuple of (min, max) values for 'given' normalization mode
    """
    if normalization_mode == 'final_pred':
        min_val, max_val = image_array[-1].min(), image_array[-1].max()
    elif normalization_mode == 'global':
        min_val, max_val = image_array.min(), image_array.max()
    elif normalization_mode == 'given':
        if given_min_max is None:
            raise ValueError("Please provide min and max values for 'given' normalization mode.")
        min_val, max_val = given_min_max

    with imageio.get_writer(output_file, mode='I', duration=frame_duration) as writer:
        for frame in image_array:
            if normalization_mode == 'individual':
                min_val, max_val = frame.min(), frame.max()
            if normalization_mode != 'none':
                denom = max_val - min_val
                if denom < 1e-12:
                    frame = np.zeros_like(frame)
                else:
                    frame = (frame - min_val) / denom
                frame = (frame * 255).astype(np.uint8)
            writer.append_data(frame)

def save_model(config, model, train_iterations, output_save_dir, optimizer=None, ema=None):

    os.makedirs(Path(output_save_dir, 'model/'), exist_ok=True)

    # save yaml file for later runs
    with open(output_save_dir + '/model/model.yaml', 'w') as yaml_file:
        yaml.dump(dict(config), yaml_file, default_flow_style=False)    
    save_dir_model = output_save_dir + '/model/checkpoint_' + str(train_iterations) + '.pt'
    save_obj = dict(
        model = model.state_dict()
    )
    if optimizer is not None:
        save_obj['optimizer'] = optimizer.state_dict()
    if ema is not None:
        save_obj['ema'] = ema.state_dict()
    # save model to path
    with open(save_dir_model, 'wb') as f:
        torch.save(save_obj, f)
    print(f'\ncheckpoint saved to {output_save_dir}/.')

def load_model(path, model, optimizer=None, ema=None, strict = True, use_ema = False):

    with open(path, 'rb') as f:
        loaded_obj = torch.load(f, map_location='cpu')
    state_key = 'ema' if use_ema and 'ema' in loaded_obj else 'model'
    if use_ema and state_key != 'ema':
        print(f'\nEMA weights requested but not found in {path}; loading model weights.')
    try:
        missing, unexpected = model.load_state_dict(loaded_obj[state_key], strict=strict)
    except (KeyError, RuntimeError) as e:
        raise RuntimeError(f'Failed to load checkpoint from {path}: {e}') from e

    if not strict:
        print(f'Missing keys: {missing}')
        print(f'Unexpected keys: {unexpected}')
    if optimizer is not None and 'optimizer' in loaded_obj:
        optimizer.load_state_dict(loaded_obj['optimizer'])
    if ema is not None:
        if 'ema' in loaded_obj:
            ema.load_state_dict(loaded_obj['ema'])
        else:
            ema.register(model)
    print('\nCheckpoint loaded from {}'.format(path))

    return model

def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)

class DenoisingDiffusion(nn.Module):
    def __init__(self, n_steps, device, residual_grad_guidance = False):
        self.n_steps = n_steps
        self.device = device
        self.diff_dict = self.create_diff_dict()
        self.residual_grad_guidance = residual_grad_guidance

    def create_diff_dict(self): 
        diff_dict = {
            'betas': self.make_beta_schedule(schedule='cosine', n_timesteps=self.n_steps, start=1e-5, end=1e-2).to(self.device),
        }
        diff_dict['alphas'] = 1. - diff_dict['betas']
        diff_dict['sqrt_recip_alphas'] = torch.sqrt(1. / diff_dict['alphas'])
        diff_dict['alphas_prod'] = torch.cumprod(diff_dict['alphas'], 0)
        diff_dict['alphas_prod_p'] = torch.cat([torch.tensor([1], device=self.device).float(), diff_dict['alphas_prod'][:-1]], 0)
        diff_dict['alphas_bar_sqrt'] = torch.sqrt(diff_dict['alphas_prod'])
        diff_dict['sqrt_recip_alphas_cumprod'] = torch.sqrt(1. / diff_dict['alphas_prod'])
        diff_dict['sqrt_recipm1_alphas_cumprod'] = torch.sqrt(1. / diff_dict['alphas_prod'] - 1)
        diff_dict['one_minus_alphas_bar_log'] = torch.log(1 - diff_dict['alphas_prod'])
        diff_dict['one_minus_alphas_bar_sqrt'] = torch.sqrt(1 - diff_dict['alphas_prod'])
        diff_dict['alphas_prod_prev'] = F.pad(diff_dict['alphas_prod'][:-1], (1, 0), value=1.)
        diff_dict['posterior_mean_coef1'] = diff_dict['betas'] * torch.sqrt(diff_dict['alphas_prod_prev']) / (1. - diff_dict['alphas_prod'])
        diff_dict['posterior_mean_coef2'] = (1. - diff_dict['alphas_prod_prev']) * torch.sqrt(diff_dict['alphas']) / (1. - diff_dict['alphas_prod'])

        diff_dict['noise_mean_coeff'] = torch.sqrt(1. / diff_dict['alphas']) * (1. - diff_dict['alphas']) / torch.sqrt(1. - diff_dict['alphas_prod'])

        # posterior variance
        diff_dict['posterior_variance'] = diff_dict['betas'] * (1. - diff_dict['alphas_prod_prev']) / (1. - diff_dict['alphas_prod'])
        # clip this since it is 0 at the beginning
        diff_dict['posterior_variance_clipped'] = diff_dict['posterior_variance'].clone()
        diff_dict['posterior_variance_clipped'][0] = diff_dict['posterior_variance'][1]

        # NOTE Ho et al. also have a version that clips the log to 1.e-20.
        diff_dict['posterior_log_variance_clipped'] = torch.log(diff_dict['posterior_variance_clipped'])

        use_constant_p2_weight = False
        if use_constant_p2_weight:
            p2_loss_weight_gamma = 1.0  # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
            p2_loss_weight_k = 1.0
            diff_dict['p2_loss_weight'] = (p2_loss_weight_k + diff_dict['alphas_prod'] / (1. - diff_dict['alphas_prod'])) ** -p2_loss_weight_gamma
        else:
            snr = diff_dict['alphas_prod'] / (1. - diff_dict['alphas_prod'])
            diff_dict['p2_loss_weight'] = torch.minimum(snr, torch.ones_like(snr) * 5.0) # from https://arxiv.org/pdf/2303.09556.pdf

        return diff_dict

    def make_beta_schedule(self, schedule='linear', n_timesteps=1000, start=1e-5, end=1e-2):
        if schedule == 'linear':
            betas = torch.linspace(start, end, n_timesteps)
        elif schedule == "quad":
            betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
        elif schedule == 'sigmoid':
            betas = torch.linspace(-6, 6, n_timesteps)
            betas = torch.sigmoid(betas) * (end - start) + start
        elif schedule == 'cosine':
            s = 0.008
            steps = n_timesteps + 1
            x = torch.linspace(0, n_timesteps, steps)
            alphas_cumprod = torch.cos(((x / n_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0, 0.999)
        return betas

    # Sampling function
    def q_sample(self, x_0, t, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        alphas_t = extract(alphas_bar_sqrt, t, x_0)
        alphas_1_m_t = extract(one_minus_alphas_bar_sqrt, t, x_0)
        return (alphas_t * x_0 + alphas_1_m_t * noise)

    def plot_diffusion(self, dataset, alphas_bar_sqrt, one_minus_alphas_bar_sqrt):
        fig, axs = plt.subplots(1, 10, figsize=(18, 2))
        for i in range(10):
            q_i = self.q_sample(dataset, torch.tensor([i * 10]), alphas_bar_sqrt, one_minus_alphas_bar_sqrt)
            axs[i].scatter(q_i[:, 0], q_i[:, 1], s=10)
            axs[i].set_axis_off(); axs[i].set_title(r'$q(\mathbf{x}_{'+str(i*10)+'})$', fontsize=10)
        plt.show()

    def p_sample(self, x, conditioning_input, t, 
                save_output = False, surpress_noise = False, 
                use_dynamic_threshold = False, residual_func = None, eval_residuals = False,
                return_optimizer = False, return_inequality = False, residual_correction = False,
                correction_mode = 'none', collect_physics_per_step = False,
                trace_c_residual: Optional[float] = None,
                model_timestep: Optional[int] = None):
        """
        model_timestep: if set, UNet / residual sees this discrete time index instead of `t`.
        Used when sampling uses T_sample > T_train: diffusion coefs use `t` into this object's
        diff_dict (length T_sample), while the network is conditioned on train-time steps.
        """
        x_init = x.clone().detach()        
        if conditioning_input is not None:
            if residual_func.gov_eqs == 'poisson':
                rho_cond_sample = conditioning_input
                x = torch.cat((x, rho_cond_sample), dim=1)
            elif residual_func.gov_eqs == 'mechanics':
                conditioning, bcs, solution = conditioning_input
                x = torch.cat((x, conditioning), dim=1)
        batch_size = len(x)
        assert correction_mode in ['x0', 'xt'] or not residual_correction, 'Correction mode unknown or not given.'
        
        t_coef = torch.tensor([t], device=x.device, dtype=torch.long)
        t_model = int(model_timestep) if model_timestep is not None else int(t)
        t_model_batch = torch.full((batch_size,), t_model, device=x.device, dtype=torch.long)
        model_input = image_to_b_xy_c(x) # we reshape this later to an image in U-net model class but let's be consistent here with the operator model
        model_input = (model_input, t_model_batch)

        model_intermediate = None
        
        # model output
        # evaluate residuals at last timestep if required
        if residual_func.gov_eqs in ('darcy', 'turbulent'):
            residual_input = (model_input, )
            sample = True
        if residual_func.gov_eqs == 'poisson':
            residual_input = (model_input, rho_cond_sample)
            sample = True
        if residual_func.gov_eqs == 'mechanics':       
            vf = conditioning[:,0,0,0]
            # vf = x_0[:,2].mean((1,2))
            residual_input = (model_input, bcs, vf, solution)
            if t_coef[0] == 0:
                sample = True
            else:
                sample = False
        out_dict = residual_func.compute_residual(  residual_input,
                                                    reduce='per-batch',
                                                    return_model_out = True,
                                                    return_optimizer = return_optimizer,
                                                    return_inequality = return_inequality,
                                                    sample = sample,
                                                    ddim_func = self.ddim_sample_x0)
        
        output, residual = out_dict['model_out'], out_dict['residual']
        model_out = output
        if len(model_out.shape) == 3:
            # convert to image [batch_size, channels, pixels, pixels]
            model_out = generalized_b_xy_c_to_image(model_out)

        if residual_correction and correction_mode == 'x0':
            model_out, residual = residual_func.residual_correction(generalized_image_to_b_xy_c(model_out))
            model_out = generalized_b_xy_c_to_image(model_out)
            
        if save_output:
            model_intermediate = model_out.clone().detach()
        x0_pred = model_out
        mean = (
            extract(self.diff_dict['posterior_mean_coef1'], t_coef, x_init) * x0_pred +
            extract(self.diff_dict['posterior_mean_coef2'], t_coef, x_init) * x_init
        )

        # Generate z
        z = torch.randn_like(x_init, device=x.device)
        # Fixed sigma
        sigma_t = extract(self.diff_dict['betas'], t_coef, x_init).sqrt()
        # no noise when t == 0
        if surpress_noise:
            nonzero_mask = (1. - (t_coef == 0).float())
        else:
            nonzero_mask = 1.
        sample = mean + nonzero_mask * sigma_t * z

        if residual_correction and correction_mode == 'xt':
            sample, residual = residual_func.residual_correction(generalized_image_to_b_xy_c(sample))
            sample = generalized_b_xy_c_to_image(sample)

        dynamic_thres_percentile = 0.9
        if use_dynamic_threshold:
            def maybe_clip(x):
                s = torch.quantile(
                    rearrange(x.float(), "b ... -> b (...)").abs(),
                    dynamic_thres_percentile,
                    dim=-1,
                )
                s.clamp_(min=1.0)
                s = right_pad_dims_to(x, s)
                x = x.clamp(-s, s) / s
                return x
            sample = maybe_clip(sample)

        trace_dict = None
        if collect_physics_per_step:
            with torch.no_grad():
                trace_dict = {
                    't': int(t_coef[0].item()),
                    'physics_residual_mean_abs': float(residual.detach().float().abs().mean().cpu()),
                }
                # Match the residual term used in model_estimation_loss.
                if trace_c_residual is not None:
                    t_batch = torch.full(
                        (batch_size,), int(t_coef[0].item()), device=residual.device, dtype=torch.long
                    )
                    var = extract(self.diff_dict['posterior_variance_clipped'], t_batch, residual)
                    residual_log_likelihood = self.gaussian_log_likelihood(
                        torch.zeros_like(residual), means=residual, variance=var
                    )
                    physics_loss = (float(trace_c_residual) * (-1.0) * residual_log_likelihood).mean()
                    trace_dict['physics_loss_mean'] = float(physics_loss.detach().cpu())
        
        if (t_coef[0] == 0 and eval_residuals):
            aux_out = {}
            aux_out['residual'] = residual
            if return_optimizer and ('optimizer' in out_dict):
                aux_out['optimized_quant'] = out_dict['optimizer']
            if return_inequality:
                aux_out['inequality_quant'] = out_dict['inequality']
            if residual_func.gov_eqs == 'mechanics':
                if residual_func.topopt_eval:
                    aux_out['rel_CE_error_full_batch'] = out_dict['rel_CE_error_full_batch']
                    aux_out['vf_error_full_batch'] = out_dict['vf_error_full_batch']
                    aux_out['fm_error_full_batch'] = out_dict['fm_error_full_batch']
            if trace_dict is not None:
                aux_out['physics_per_step_t'] = trace_dict['t']
                aux_out['physics_residual_mean_abs'] = trace_dict['physics_residual_mean_abs']
                if 'physics_loss_mean' in trace_dict:
                    aux_out['physics_loss_mean'] = trace_dict['physics_loss_mean']
                # Keep the timestep field available for per-step physics traces.
                aux_out['t'] = trace_dict['t']

            return (sample, model_intermediate), aux_out
        elif trace_dict is not None:
            return (sample, model_intermediate), trace_dict
        else:
            return (sample, model_intermediate), None

    # NOTE we do not use @torch.inference_mode() since we need gradients to obtain residual
    # to free up memory, we manually call .detach() where appropriate
    def p_sample_loop(self,
                    conditioning_input,
                    shape,
                    save_output = False, 
                    surpress_noise = True,
                    use_dynamic_threshold = False,
                    residual_func = None,
                    eval_residuals = False,
                    return_optimizer = False,
                    return_inequality = False,
                    M_correction = 0,
                    N_correction = 0,
                    correction_mode = 'none',
                    collect_physics_per_step = False,
                    trace_c_residual: Optional[float] = None,
                    model_timestep_override: Optional[Sequence[int]] = None):

        if model_timestep_override is not None:
            assert len(model_timestep_override) == self.n_steps, (
                f"model_timestep_override length {len(model_timestep_override)} != n_steps {self.n_steps}"
            )

        cur_x = torch.randn(shape, device=self.diff_dict['alphas'].device)
        x_seq = [cur_x.detach().cpu()]

        if save_output:
            interm_imgs = [torch.zeros(shape)]
        else:
            interm_imgs = []

        interm_img = None
        physics_trajectory = []
        for i in reversed(range(self.n_steps)):
            
            # CoCoGen correction
            residual_correction = False
            if i < N_correction:
                residual_correction = True
                eval_residuals = True

            mt = model_timestep_override[i] if model_timestep_override is not None else None
                
            output = self.p_sample(cur_x.detach(), conditioning_input, i, save_output, surpress_noise, use_dynamic_threshold,
                                        residual_func = residual_func, eval_residuals = eval_residuals,
                                        return_optimizer = return_optimizer, return_inequality = return_inequality,
                                        residual_correction = residual_correction, correction_mode = correction_mode,
                                        collect_physics_per_step = collect_physics_per_step,
                                        trace_c_residual = trace_c_residual,
                                        model_timestep = mt)
            cur_x, interm_img = output[0]
            if collect_physics_per_step and output[1] is not None and 'physics_residual_mean_abs' in output[1]:
                row = {'t': output[1]['t'], 'physics_residual_mean_abs': output[1]['physics_residual_mean_abs']}
                if 'physics_loss_mean' in output[1]:
                    row['physics_loss_mean'] = output[1]['physics_loss_mean']
                physics_trajectory.append(row)

            x_seq.append(cur_x.detach().cpu())
            if interm_img is not None:
                interm_imgs.append(interm_img.detach().cpu())

        # CoCoGen correction
        for i in range(M_correction):
            cur_x, residual = residual_func.residual_correction(generalized_image_to_b_xy_c(cur_x))
            cur_x = generalized_b_xy_c_to_image(cur_x)
            x_seq.append(cur_x.detach().cpu())  
            if eval_residuals and i == M_correction - 1:
                output[1]['residual'] = residual
            
        if eval_residuals:
            final_aux = output[1]
            if collect_physics_per_step and final_aux is not None:
                final_aux = dict(final_aux)
                final_aux['physics_trajectory'] = physics_trajectory
            return (x_seq, interm_imgs), final_aux
        if collect_physics_per_step:
            return (x_seq, interm_imgs), {'physics_trajectory': physics_trajectory}
        else:
            return x_seq, interm_imgs

    def normal_kl(self, mean1, logvar1, mean2, logvar2):
        """
        KL divergence between normal distributions parameterized by mean and log-variance.
        """
        kl = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))
        return kl

    def gaussian_log_likelihood(self, x, means, variance):
        centered_x = x - means    
        squared_diffs = (centered_x ** 2) / variance
        log_probs = -0.5 * squared_diffs
        return log_probs

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.diff_dict['sqrt_recip_alphas_cumprod'], t, x_t) * x_t -
            extract(self.diff_dict['sqrt_recipm1_alphas_cumprod'], t, x_t) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract(self.diff_dict['sqrt_recip_alphas_cumprod'], t, x_t) * x_t - x0
        ) / extract(self.diff_dict['sqrt_recipm1_alphas_cumprod'], t, x_t)

    def predict_noise_from_mean(self, x_t, t, mean_t):
        return (
            extract(self.diff_dict['sqrt_recip_alphas'], t, mean_t) * x_t - mean_t
        ) / extract(self.diff_dict['noise_mean_coeff'], t, mean_t)

    def loss_variational(self, output, x_0, x_t, t, base_2 = False):    
        batch_size = x_0.shape[0]

        # Compute the true mean and variance
        true_mean = (
        extract(self.diff_dict['posterior_mean_coef1'], t, x_t) * x_0 +
        extract(self.diff_dict['posterior_mean_coef2'], t, x_t) * x_t
        )
        
        true_var = extract(self.diff_dict['posterior_variance_clipped'], t, x_t)
        model_var = true_var

        # Infer the mean and variance with our model
        model_mean = output

        # Compute the KL loss
        true_var_log = torch.log(true_var)
        model_var_log = torch.log(model_var)
        kl = self.normal_kl(true_mean, true_var_log, model_mean, model_var_log)
        kl = torch.mean(kl.view(batch_size, -1), dim=1)
        if base_2:
            kl = kl / np.log(2.)

        # define p(x_0|x_1) simply as a gaussian
        log_likelihood = self.gaussian_log_likelihood(x_0, means=model_mean, variance=model_var)
        log_likelihood = torch.mean(log_likelihood.view(batch_size, -1), dim=1) 
        if base_2:
            log_likelihood = log_likelihood / np.log(2.)

        # At the first timestep return the log likelihood, otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        # Check invalid values before torch.where because inactive branches can still affect gradients.
        assert log_likelihood.isnan().any() == False, 'Log likelihood is nan.'
        assert log_likelihood.isinf().any() == False, 'Log likelihood is inf.'

        loss_log_likelihood = -1. * log_likelihood # since we minimize loss (instead of maximizing likelihood)

        loss = torch.where(t == 0, loss_log_likelihood, kl)

        return loss.mean(-1)

    def model_estimation_loss(self,
                              input,
                              residual_func = None, 
                              c_data = 1.,
                              c_residual = 0.,
                              c_ineq = 0., 
                              lambda_opt = 0.,
                              use_projection_heads = False,
                              c_projection = 0.1,
                              t_fixed: Optional[int] = None):
        """
        t_fixed: if provided, use the same diffusion index for the full batch.
        Default None: sample a random timestep per example, as in standard training.
        """
        batch_size = len(input)
        if t_fixed is not None:
            ti = int(t_fixed)
            if ti < 0 or ti >= self.n_steps:
                raise ValueError(f"t_fixed={ti} must be in [0, {self.n_steps - 1}]")
            t = torch.full((batch_size,), ti, device=input.device, dtype=torch.long)
        else:
            t = torch.randint(0, self.n_steps, size=(batch_size,), device=input.device)
        
        if residual_func.gov_eqs in ('darcy', 'turbulent'):
            x_0 = input
        if residual_func.gov_eqs == 'poisson':
            # input: [B, 2, H, W] = (rho, U)
            rho_cond = input[:, 0:1]  # charge density condition
            x_0 = input[:, 1:2]       # potential U (target)
        if residual_func.gov_eqs == 'mechanics':            
            conditioning, x_0, bcs = torch.tensor_split(input, (3, 6), dim=1) # vf_arr, strain_energy_density_fem, von_mises_stress, disp_x, disp_y, E_field, BC_node_x, BC_node_y, load_x_img, load_y_img

        # x0 multiplier
        a = extract(self.diff_dict['alphas_bar_sqrt'], t, x_0)
        # eps multiplier
        am1 = extract(self.diff_dict['one_minus_alphas_bar_sqrt'], t, x_0)
        e = torch.randn_like(x_0, device=x_0.device)
        # model input
        x = x_0 * a + e * am1 # previous x_t

        if residual_func.gov_eqs == 'mechanics':
            x = torch.cat((x, conditioning), dim=1)
        if residual_func.gov_eqs == 'poisson':
            x = torch.cat((x, rho_cond), dim=1)  # [U_t, rho]

        x = image_to_b_xy_c(x) # we reshape this later to an image in U-net model class but let's be consistent here with the operator model
        model_input = (x, t)

        return_inequality = False
        return_optimizer = False
        if c_ineq > 0. and residual_func.gov_eqs == 'mechanics':
            return_inequality = True
        # `optimizer` quantity is only implemented for mechanics residuals.
        # Avoid requesting it for darcy/turbulent (their compute_residual does not return it).
        if lambda_opt > 0. and residual_func.gov_eqs == 'mechanics':
            return_optimizer = True

        if residual_func.gov_eqs in ('darcy', 'turbulent'):
            residual_input = (model_input, )
        if residual_func.gov_eqs == 'poisson':
            residual_input = (model_input, rho_cond)
        if residual_func.gov_eqs == 'mechanics':
            vf = conditioning[:,0,0,0]
            residual_input = (model_input, bcs, vf, x_0)

        projection_loss_required = bool(use_projection_heads and c_projection > 0.)
        if use_projection_heads and not getattr(residual_func.model, 'use_projection_heads', False):
            raise RuntimeError('use_projection_heads=True, but the residual model was built without projection heads.')

        # Request projection-head outputs when enabled.
        if use_projection_heads:
            # Model call with projection outputs.
            out_dict = residual_func.compute_residual(residual_input,
                                                      reduce='per-batch', 
                                                      return_model_out = True, 
                                                      return_optimizer = return_optimizer, 
                                                      return_inequality = return_inequality,
                                                      ddim_func = self.ddim_sample_x0,
                                                      return_projections = True)
            residual, output = out_dict['residual'], out_dict['model_out']
            projections = out_dict.get('projections', {})
            if projection_loss_required and not projections:
                raise RuntimeError(
                    'Projection heads are enabled with c_projection > 0, but no projection outputs were returned.'
                )
        else:
            # Standard model call.
            out_dict = residual_func.compute_residual(residual_input,
                                                      reduce='per-batch', 
                                                      return_model_out = True, 
                                                      return_optimizer = return_optimizer, 
                                                      return_inequality = return_inequality,
                                                      ddim_func = self.ddim_sample_x0)
            residual, output = out_dict['residual'], out_dict['model_out']
            projections = {}

        # reshape output to image (batch_size, channels, pixels, pixels)
        if len(output.shape) == 3:
            output = b_xy_c_to_image(output)
            
        target = x_0
        loss_fn = nn.MSELoss(reduction='none')
        loss = loss_fn(target, output)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss * extract(self.diff_dict['p2_loss_weight'], t, loss)
        loss = loss.mean()

        # adjust data-driven loss term
        data_loss = c_data * loss
        data_loss_track = data_loss.item()
        loss = data_loss

        # add negative residual log-likelihood, i.e., - log p(r|x_0_pred(x_0))
        var = extract(self.diff_dict['posterior_variance_clipped'], t, residual)
        
        # residual_loss_track = residual.mean().item()
        residual_loss_track = residual.abs().mean().item()

        residual_log_likelihood = self.gaussian_log_likelihood(torch.zeros_like(residual), means=residual, variance=var)
        residual_loss = c_residual * -1. * residual_log_likelihood
        # Residual likelihood term.
        loss += residual_loss.mean()

        ineq_loss_track = 0.
        if return_inequality and 'inequality' in out_dict:
            # add negative inequality residual log-likelihood, i.e., - log p(r_ineq|x_0_pred(x_0)) (similar to above)
            ineq_log_likelihood = self.gaussian_log_likelihood(torch.zeros_like(out_dict['inequality']), means=out_dict['inequality'], variance=var)
            ineq_loss = c_ineq * -1. * ineq_log_likelihood
            ineq_loss_track = out_dict['inequality'].mean().item()
            # Inequality likelihood term.
            loss += ineq_loss.mean()

        opt_loss_track = 0.
        if return_optimizer and ('optimizer' in out_dict):
            # add optimization log-likelihood, i.e., log p(c=c_min|x_0_pred(x_0)) (where p is Expon. distribution)
            opt_log_likelihood = -1. * out_dict['optimizer']
            opt_loss = -1. * lambda_opt * opt_log_likelihood
            opt_loss_track = out_dict['optimizer'].mean().item()
            # Optimization likelihood term.
            loss += opt_loss.mean()
        elif return_optimizer and ('optimizer' not in out_dict):
            # Graceful fallback: residual implementation didn't provide optimizer term.
            opt_loss_track = 0.

        # Projection-head physics loss.
        projection_loss_track = 0.
        if projection_loss_required:
            projection_losses_nll = []  # training objective
            projection_losses_abs = []  # logging metric
            
            for pos_name, proj_output in projections.items():
                # Convert projection output to the residual input format.
                if len(proj_output.shape) == 4:  # image format [B, C, H, W]
                    proj_input_reshaped = image_to_b_xy_c(proj_output)
                elif len(proj_output.shape) == 5:  # video format [B, C, F, H, W]  
                    proj_input_reshaped = generalized_image_to_b_xy_c(proj_output)
                else:
                    raise RuntimeError(
                        f'Projection output "{pos_name}" has unsupported shape {tuple(proj_output.shape)}.'
                    )
                
                # Build residual inputs for projection-head outputs.
                proj_model_input = (proj_input_reshaped, t)
                
                if residual_func.gov_eqs in ('darcy', 'turbulent'):
                    proj_residual_input = (proj_model_input, )
                elif residual_func.gov_eqs == 'poisson':
                    proj_residual_input = (proj_model_input, rho_cond)
                elif residual_func.gov_eqs == 'mechanics':
                    proj_residual_input = (proj_model_input, bcs, vf, x_0)
                else:
                    raise RuntimeError(f'Projection loss is not implemented for gov_eqs={residual_func.gov_eqs}.')
                
                # Compute residuals directly on projection-head outputs. This must succeed
                # when REPA-P is enabled; otherwise training would silently become PIDM.
                proj_residual_dict = residual_func.compute_residual(
                    proj_residual_input,
                    reduce='per-batch',
                    return_model_out = False,
                    return_optimizer = False,
                    return_inequality = False,
                    ddim_func = self.ddim_sample_x0,
                    skip_model_call = True,
                    given_model_output = proj_input_reshaped
                )
                proj_residual = proj_residual_dict['residual']
                
                # Training term: NLL.
                var = extract(self.diff_dict['posterior_variance_clipped'], t, proj_residual)
                proj_residual_log_likelihood = self.gaussian_log_likelihood(
                    torch.zeros_like(proj_residual), means=proj_residual, variance=var
                )
                proj_loss_nll = -1. * proj_residual_log_likelihood.mean()
                projection_losses_nll.append(proj_loss_nll)

                # Logging term: mean absolute residual.
                proj_loss_abs = proj_residual.abs().mean()
                projection_losses_abs.append(proj_loss_abs)
            
            if not projection_losses_nll:
                raise RuntimeError('Projection heads are enabled with c_projection > 0, but no projection loss was computed.')

            total_projection_loss_nll = torch.stack(projection_losses_nll).mean()
            loss += c_projection * total_projection_loss_nll

            if projection_losses_abs:
                total_projection_loss_abs = torch.stack(projection_losses_abs).mean()
                projection_loss_track = total_projection_loss_abs.item()

        return loss, data_loss_track, residual_loss_track, ineq_loss_track, opt_loss_track, projection_loss_track
    
    def ddim_sample_x0(self, xt, t, model, shape, reduced_n_steps, ddim_sampling_eta, gov_eqs = None, self_cond = None):

        batch, device, sample_timesteps, eta = shape[0], self.diff_dict['alphas'].device, reduced_n_steps, ddim_sampling_eta

        if len(t) == 1:
            batch_t = torch.ones(batch, device=device, dtype=torch.long)*t
        else:
            batch_t = t

        batch_t = batch_t.cpu().numpy()
        seqs = []
        seqs_next = []
        for t_idx, t in enumerate(batch_t):
            seq = list(map(int, np.linspace(0, batch_t[t_idx], sample_timesteps+2, endpoint=True, dtype=float))) # evenly spread from 0 to current t
            seqs.append(list(reversed(seq)))
            seq_next = [-1] + list(seq[:-1])
            seqs_next.append(list(reversed(seq_next)))
            seq = None

        # tranpose to have time as first dimension
        cur_times = torch.tensor(seqs, device=device).T
        next_times = torch.tensor(seqs_next, device=device).T

        time_pairs = list(zip(cur_times, next_times)) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        if len(xt.shape) == 3:            
            xt = generalized_b_xy_c_to_image(xt)

        if gov_eqs == 'mechanics':
            model_input = xt
            cur_x = xt[:, :3] # consider only solution fields
        else:
            model_input = xt
            cur_x = xt
        x0_pred = None

        model_output = None
        for fwd_idx, (t, t_next) in enumerate(time_pairs):

            # create mask for those timesteps that are equal
            mask = (t == t_next).float().view(-1, 1, 1, 1)
            model_output = model(model_input, t, self_cond)
            x0_pred = model_output            
            mean = (
                extract(self.diff_dict['posterior_mean_coef1'], t, cur_x) * x0_pred +
                extract(self.diff_dict['posterior_mean_coef2'], t, cur_x) * cur_x
            )
            # noise estimate
            eps_theta = self.predict_noise_from_mean(cur_x, t, mean)

            if fwd_idx == 0:
                model_out = model_output

            if t_next[0] < 0: # this happens when we predict x0, should never happen during training
                # assert that all next timesteps are equal
                assert torch.all(t_next == -1), 'Next timesteps should be -1, otherwise this is inconsistent.'
                cur_x = x0_pred
                continue

            alpha = extract(self.diff_dict['alphas_prod'], t, cur_x)
            alpha_next = extract(self.diff_dict['alphas_prod'], t_next, cur_x)

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(cur_x)

            # only update cur_x where t != t_next
            # NOTE: this is the standard update (not using larger variance)
            cur_x_ = x0_pred * alpha_next.sqrt() + \
                     c * eps_theta + \
                     sigma * noise
            
            cur_x = mask * cur_x + (1 - mask) * cur_x_

        assert model_out is not None, 'Model output not given.'
        return cur_x, model_out