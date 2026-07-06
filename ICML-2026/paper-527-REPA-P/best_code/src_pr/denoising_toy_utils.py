import os, dill
import imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce, rearrange

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
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def fix_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def noop(*args, **kwargs):
    pass

def plot_data(data):
    plt.figure(figsize=(16, 12))
    plt.scatter(*data, s=10)
    plt.title(r'Ground truth $q(\mathbf{x}_{0})$')
    plt.show()
    exit()

def create_diff_dict(n_steps, device): 

    diff_dict = {
        'betas': make_beta_schedule(schedule='cosine', n_timesteps=n_steps, start=1e-5, end=1e-2).to(device),
    }
    diff_dict['alphas'] = 1. - diff_dict['betas']
    diff_dict['sqrt_recip_alphas'] = torch.sqrt(1. / diff_dict['alphas'])
    diff_dict['alphas_prod'] = torch.cumprod(diff_dict['alphas'], 0)
    diff_dict['alphas_prod_p'] = torch.cat([torch.tensor([1], device=device).float(), diff_dict['alphas_prod'][:-1]], 0)
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

    # calculate p2 reweighting
    use_constant_p2_weight = False
    # use_constant_p2_weight = True
    if use_constant_p2_weight:
        p2_loss_weight_gamma = 1.0  # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1.0
        diff_dict['p2_loss_weight'] = (p2_loss_weight_k + diff_dict['alphas_prod'] / (1. - diff_dict['alphas_prod'])) ** -p2_loss_weight_gamma
    else:
        snr = diff_dict['alphas_prod'] / (1. - diff_dict['alphas_prod'])
        diff_dict['p2_loss_weight'] = torch.minimum(snr, torch.ones_like(snr) * 5.0)  # https://arxiv.org/pdf/2303.09556.pdf

    return diff_dict

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

def sample_zeros(size):
    return np.zeros((size, 2))

def sample_gaussian(size, dim=2):
    return np.random.randn(size, dim)

def sample_hypersphere(size, dim):
    # Sample points from a hypersphere surface in `dim` dimensions
    # Generate normally distributed points
    x = np.random.normal(0, 1, (size, dim))
    
    # Normalize each point to lie on the surface of the hypersphere
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

def make_beta_schedule(schedule='linear', n_timesteps=1000, start=1e-5, end=1e-2):
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

def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)

# Sampling function
def q_sample(x_0, t, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0)
    alphas_t = extract(alphas_bar_sqrt, t, x_0)
    alphas_1_m_t = extract(one_minus_alphas_bar_sqrt, t, x_0)
    return (alphas_t * x_0 + alphas_1_m_t * noise)

def plot_diffusion(dataset, alphas_bar_sqrt, one_minus_alphas_bar_sqrt):
    # fig, axs = plt.subplots(1, 10, figsize=(28, 3))
    fig, axs = plt.subplots(1, 10, figsize=(18, 2))
    for i in range(10):
        q_i = q_sample(dataset, torch.tensor([i * 10]), alphas_bar_sqrt, one_minus_alphas_bar_sqrt)
        axs[i].scatter(q_i[:, 0], q_i[:, 1], s=10)
        axs[i].set_axis_off(); axs[i].set_title(r'$q(\mathbf{x}_{'+str(i*10)+'})$', fontsize=10)
    plt.show()

class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, y):
        out = self.lin(x)
        gamma = self.embed(y)
        out = gamma.view(-1, self.num_out) * out
        return out
    
class ConditionalModel(nn.Module):
    def __init__(
            self, 
            dim,
            n_steps,
        ):
        super(ConditionalModel, self).__init__()
        self.lin1 = ConditionalLinear(dim, 128, n_steps)
        self.lin2 = ConditionalLinear(128, 128, n_steps)
        self.lin3 = nn.Linear(128, dim)
    
    def forward(self, x, y):
        x = F.softplus(self.lin1(x, y))
        x = F.softplus(self.lin2(x, y))
        return self.lin3(x)

def p_sample(model, x, t, diff_dict, model_pred_mode = 'eps', save_output = False, surpress_noise = False, use_dynamic_threshold = False, reduced_ddim_steps = 0):
    t = torch.tensor([t], device=x.device)

    model_output = None
    if model_pred_mode == 'eps':
        # Factor to the model output
        eps_factor = ((1 - extract(diff_dict['alphas'], t, x)) / extract(diff_dict['one_minus_alphas_bar_sqrt'], t, x))
        # Model output
        eps_theta = model(x, t)
        if save_output:
            model_output = eps_theta.clone().detach()
        # Final values
        mean = (1 / extract(diff_dict['alphas'], t, x).sqrt()) * (x - (eps_factor * eps_theta))
        x0_pred = predict_start_from_noise(x, t, eps_theta, diff_dict)
    elif model_pred_mode == 'x0':
        # model output
        model_pred = model(x, t)
        if save_output:
            model_output = model_pred.clone().detach()
        x0_pred = model_pred
        mean = (
            extract(diff_dict['posterior_mean_coef1'], t, x) * x0_pred +
            extract(diff_dict['posterior_mean_coef2'], t, x) * x
        )
    elif model_pred_mode == 'mu':
        # model output
        model_pred = model(x, t)
        if save_output:
            model_output = model_pred.clone().detach()
        mean = model_pred
        eps_theta = predict_noise_from_mean(x, t, model_pred, diff_dict)
        x0_pred = predict_start_from_noise(x, t, eps_theta, diff_dict)
    else:
        raise ValueError('model_pred_mode not recognized.')
    # Generate z
    z = torch.randn_like(x, device=x.device)
    # Fixed sigma
    sigma_t = extract(diff_dict['betas'], t, x).sqrt()
    # no noise when t == 0
    if surpress_noise:
        nonzero_mask = (1. - (t == 0).float())
    else:
        nonzero_mask = 1.
    sample = mean + nonzero_mask * sigma_t * z

    dynamic_thres_percentile = 0.9
    if use_dynamic_threshold:
        # s is the dynamic threshold, determined by percentile of absolute values of reconstructed sample per batch element
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
    # evaluate DDIM x0
    x0_estimation = None
    if save_output:
        if t > 0:
            x0_estimation = ddim_sample_x0(x, t, model, x.shape, reduced_ddim_steps, 0, diff_dict, model_pred_mode = model_pred_mode)
        else:
            x0_estimation = x0_pred
    return (sample, model_output, x0_estimation)

def p_sample_loop(model, shape, n_steps, diff_dict, model_pred_mode = 'x0', save_output = False, surpress_noise = True, use_dynamic_threshold = False, reduced_ddim_steps = 0):

    cur_x = torch.randn(shape, device=diff_dict['alphas'].device)

    x_seq = [cur_x.detach().cpu()]

    if save_output:
        model_outputs = [torch.zeros(shape, device='cpu')]
        x0_estimations = [torch.zeros(shape, device='cpu')]
    else:
        model_outputs = []
        x0_estimations = []

    model_output = None
    for i in reversed(range(n_steps)):
        cur_x, model_output, ddim_x0 = p_sample(model, cur_x.detach(), i, diff_dict, model_pred_mode, save_output, surpress_noise, use_dynamic_threshold, reduced_ddim_steps=reduced_ddim_steps)
        x_seq.append(cur_x.detach().cpu())
        if save_output:
            model_outputs.append(model_output.detach().cpu())
            x0_estimations.append(ddim_x0.detach().cpu())

    return x_seq, model_outputs, x0_estimations

def ddim_sample_x0(xt, t, model, shape, reduced_n_steps, ddim_sampling_eta, diff_dict, model_pred_mode = 'eps'):

    batch, device, sampling_timesteps, eta = shape[0], diff_dict['alphas'].device, reduced_n_steps, ddim_sampling_eta

    if len(t) == 1:
        batch_t = torch.ones(batch, device=device, dtype=torch.long)*t
    else:
        batch_t = t
        
    batch_t = batch_t.cpu().numpy()
    seqs = []
    seqs_next = []
    for t_idx, t in enumerate(batch_t):
        seq = list(np.linspace(0, batch_t[t_idx], sampling_timesteps+2, endpoint=True, dtype=float)) # evenly spread from 0 to current t
        seq = list(map(int, seq))
        seqs.append(list(reversed(seq)))
        seq_next = [-1] + list(seq[:-1])
        seqs_next.append(list(reversed(seq_next)))
        seq = None

    # tranpose to have time as first dimension
    cur_times = torch.tensor(seqs, device=device).T
    next_times = torch.tensor(seqs_next, device=device).T

    time_pairs = list(zip(cur_times, next_times)) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

    cur_x = xt
    x0_pred = None

    for t, t_next in time_pairs:

        # create mask for those timesteps that are equal
        mask = (t == t_next).float().unsqueeze(-1)

        if model_pred_mode == 'eps':
            eps_theta = model(cur_x, t)
            x0_pred = predict_start_from_noise(cur_x, t, eps_theta, diff_dict)
        elif model_pred_mode == 'x0':
            model_pred = model(cur_x, t)
            x0_pred = model_pred
            mean = (
                extract(diff_dict['posterior_mean_coef1'], t, cur_x) * x0_pred +
                extract(diff_dict['posterior_mean_coef2'], t, cur_x) * cur_x
            )
            eps_theta = predict_noise_from_mean(cur_x, t, mean, diff_dict)
        elif model_pred_mode == 'mu':
            model_pred = model(cur_x, t)
            eps_theta = predict_noise_from_mean(cur_x, t, model_pred, diff_dict)
            x0_pred = predict_start_from_noise(cur_x, t, eps_theta, diff_dict)
        else:
            raise ValueError('model_pred_mode not recognized.')

        if t_next[0] < 0: # this happens when we predict x0, should never happen during training
            # assert that all next timesteps are equal
            assert torch.all(t_next == -1), 'Next timesteps should be -1, otherwise this is inconsistent.'
            cur_x = x0_pred
            continue

        alpha = extract(diff_dict['alphas_prod'], t, cur_x)
        alpha_next = extract(diff_dict['alphas_prod'], t_next, cur_x)

        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()

        noise = torch.randn_like(cur_x)

        # only change cur_x where t != t_next
        cur_x_ = x0_pred * alpha_next.sqrt() + \
                c * eps_theta + \
                sigma * noise

        cur_x = mask * cur_x + (1 - mask) * cur_x_
        
    return cur_x

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    kl = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))
    return kl

def gaussian_log_likelihood(x, means, variance, return_full = False):
    centered_x = x - means    
    squared_diffs = (centered_x ** 2) / variance
    if return_full:
        log_likelihood = -0.5 * (squared_diffs + torch.log(variance) + torch.log(2 * torch.pi)) # full log likelihood with constant terms
    else:
        log_likelihood = -0.5 * squared_diffs

    # avoid log(0)
    log_likelihood = torch.clamp(log_likelihood, min=-27.6310211159)

    return log_likelihood

def predict_start_from_noise(x_t, t, noise, diff_dict):
    return (
        extract(diff_dict['sqrt_recip_alphas_cumprod'], t, x_t) * x_t -
        extract(diff_dict['sqrt_recipm1_alphas_cumprod'], t, x_t) * noise
    )

def predict_noise_from_mean(x_t, t, mean_t, diff_dict):
    return (
        extract(diff_dict['sqrt_recip_alphas'], t, mean_t) * x_t - mean_t
    ) / extract(diff_dict['noise_mean_coeff'], t, mean_t)

def loss_variational(output, x_0, x_t, t, diff_dict, base_2 = False):    
    batch_size = x_0.shape[0]

    # Compute the true mean and variance
    true_mean = (
    extract(diff_dict['posterior_mean_coef1'], t, x_t) * x_0 +
    extract(diff_dict['posterior_mean_coef2'], t, x_t) * x_t
    )
    
    true_var = extract(diff_dict['posterior_variance_clipped'], t, x_t)
    model_var = true_var

    # Infer the mean and variance with our model
    model_mean = output

    # Compute the KL loss
    true_var_log = torch.log(true_var)
    model_var_log = torch.log(model_var)
    kl = normal_kl(true_mean, true_var_log, model_mean, model_var_log)
    kl = torch.mean(kl.view(batch_size, -1), dim=1)
    if base_2:
        kl = kl / np.log(2.)

    # define p(x_0|x_1) simply as a gaussian
    log_likelihood = gaussian_log_likelihood(x_0, means=model_mean, variance=model_var)
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

def model_estimation_loss(model, x_0, n_steps, diff_dict, model_pred_mode = 'eps', residual_func = None, 
                          ineq_func = None, opt_func = None, c_data = 1., c_residual = 0., c_ineq = 0., lambda_opt = 0., use_ddim_x0 = False, reduced_ddim_steps = 0):
    batch_size = x_0.shape[0]
    # Select a random step for each example
    t = torch.randint(0, n_steps, size=(batch_size // 2 + 1,), device=x_0.device)
    t = torch.cat([t, n_steps - t - 1], dim=0)[:batch_size].long()

    # x0 multiplier
    a = extract(diff_dict['alphas_bar_sqrt'], t, x_0)
    # eps multiplier
    am1 = extract(diff_dict['one_minus_alphas_bar_sqrt'], t, x_0)
    e = torch.randn_like(x_0, device=x_0.device)
    # model input
    x = x_0 * a + e * am1

    output = model(x, t)
    if model_pred_mode == 'eps':
        target = e
        loss_fn = nn.MSELoss()
        loss = loss_fn(target, output)

        # predict x_0 (of which we evaluate log p(r|x_0)) based on eps 
        x_0_pred = predict_start_from_noise(x, t, output, diff_dict)

    elif model_pred_mode == 'x0':
        target = x_0
        loss_fn = nn.MSELoss(reduction='none')
        loss = loss_fn(target, output)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss * extract(diff_dict['p2_loss_weight'], t, loss)
        loss = loss.mean()

        # predict x_0 (of which we evaluate log p(r|x_0)) based on eps
        x_0_pred = output

    elif model_pred_mode == 'mu':
        # use consistent true variation loss when predicting mu
        loss = loss_variational(output, x_0, x, t, diff_dict)

        # predict x_0 (of which we evaluate log p(r|x_0)) based on eps 
        noise_pred = predict_noise_from_mean(x, t, output, diff_dict)
        x_0_pred = predict_start_from_noise(x, t, noise_pred, diff_dict)
    else:
        raise ValueError('model_pred_mode not recognized.')

    # adjust data-driven loss term
    loss = c_data * loss
    data_loss = loss    

    # add residual loss
    # add - log p(r|x_0_pred(x_0)) to loss
    if use_ddim_x0:
        eval_residual_x0 = ddim_sample_x0(x, t, model, x.shape, reduced_ddim_steps, 0, diff_dict, model_pred_mode = model_pred_mode)
    else:
        eval_residual_x0 = x_0_pred

    residual = residual_func(eval_residual_x0)
    var = extract(diff_dict['posterior_variance_clipped'], t, residual)
    residual_log_likelihood = gaussian_log_likelihood(torch.zeros_like(residual), means=residual, variance=var)
    residual_loss = c_residual * -1. * residual_log_likelihood.mean() # add negative sign since we minimize loss (instead of maximizing likelihood)
    loss += residual_loss

    # add inequality loss
    # add - log p(r_ineq|x_0_pred(x_0)) to loss (same as above)
    ineq, _ = ineq_func(eval_residual_x0)
    ineq_log_likelihood = gaussian_log_likelihood(torch.zeros_like(ineq), means=ineq, variance=var)
    ineq_loss = c_ineq * -1. * ineq_log_likelihood.mean() # add negative sign since we minimize loss (instead of maximizing likelihood)
    loss += ineq_loss

    # add optimization loss
    # add log p(c=c_min|x_0_pred(x_0)) to loss (where p is Expon. distribution)
    opt_log_likelihood = -1. * lambda_opt * opt_func(eval_residual_x0)
    opt_loss = -1. * opt_log_likelihood.mean() # add negative sign since we minimize loss (instead of maximizing likelihood)
    loss += opt_loss

    return loss, data_loss.item(), torch.abs(residual).mean().item(), ineq.mean().item(), opt_func(eval_residual_x0).mean().item()

def remove_outliers(data, percentile = 0.01, also_lower_bound = False):
    
    percentile *= 100
    
    if data.size == 0:
        return data  # return the empty array as is
    
    norms = np.linalg.norm(data, axis=1)
    lower_bound = np.percentile(norms, percentile) if also_lower_bound else 0.
    upper_bound = np.percentile(norms, 100 - percentile)
    mask = (norms > lower_bound) & (norms < upper_bound)
    
    return data[mask]

def save_model(model, name, diff_dict, step, n_steps, dim, model_pred_mode, residual_func, ineq_func, opt_func):

    save_dir = './trained_models/toy/' + name + '/model'
    os.makedirs(save_dir, exist_ok=True)
    
    save_dir_model = save_dir + '/checkpoint_' + str(step) + '.pt'
    save_obj = dict(
        model = model.state_dict(),
        n_steps = n_steps,
        dim = dim,
        model_pred_mode = model_pred_mode,
        diff_dict = diff_dict,
    )
    # save model to path
    with open(save_dir_model, 'wb') as f:
        torch.save(save_obj, f)

    # save residual function with dill
    save_dir_residual_func = save_dir + '/checkpoint_' +str(step) + '_residual_func.pkl'
    with open(save_dir_residual_func, 'wb') as f:
        dill.dump(residual_func, f)

    # save inequality function with dill
    save_dir_ineq_func = save_dir + '/checkpoint_' +str(step) + '_ineq_func.pkl'
    with open(save_dir_ineq_func, 'wb') as f:
        dill.dump(ineq_func, f)

    # save optimization function with dill
    save_dir_opt_func = save_dir + '/checkpoint_' +str(step) + '_opt_func.pkl'
    with open(save_dir_opt_func, 'wb') as f:
        dill.dump(opt_func, f)

    print(f'checkpoint saved to {save_dir}')

def load_model(path, strict = True):

    with open(path, 'rb') as f:
        loaded_obj = torch.load(f, map_location='cpu')

    model = ConditionalModel(loaded_obj['dim'], loaded_obj['n_steps'])

    try:
        missing, unexpected = model.load_state_dict(loaded_obj['model'], strict=strict)
    except (KeyError, RuntimeError) as e:
        raise RuntimeError(f'Failed to load checkpoint from {path}: {e}') from e
    if not strict:
        print(f'Missing keys: {missing}')
        print(f'Unexpected keys: {unexpected}')
        
    try:
        with open(path.replace('.pt', '_residual_func.pkl'), 'rb') as f:
            residual_func = dill.load(f)
    except RuntimeError:
        print("Failed loading residual function.")

    try:
        with open(path.replace('.pt', '_ineq_func.pkl'), 'rb') as f:
            ineq_func = dill.load(f)
    except RuntimeError:
        print("Failed loading inequality function.")

    try:
        with open(path.replace('.pt', '_opt_func.pkl'), 'rb') as f:
            opt_func = dill.load(f)
    except RuntimeError:
        print("Failed loading optimization function.")

    return model, loaded_obj['diff_dict'], loaded_obj['n_steps'], loaded_obj['dim'], loaded_obj['model_pred_mode'], residual_func, ineq_func, opt_func

# tensor of shape (channels, frames, height, width) -> gif
def array_to_gif(data, output_save_dir, x_lim, y_lim, label = None, duration = 0.05, s=10):    
    # Create a GIF writer object
    with imageio.get_writer(output_save_dir, mode='I', duration = duration, loop=1) as writer:
        for step in range(data.shape[0]):
            fig, ax = plt.subplots()
            ax.scatter(data[step, :, 0], data[step, :, 1], s=s)
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