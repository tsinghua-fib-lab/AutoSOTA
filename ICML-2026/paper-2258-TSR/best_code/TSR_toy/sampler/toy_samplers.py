import math
import torch


# noise scheduling
def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.

    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """
    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


# Define the noise schedule parameters
T = 1000

# cosine schedule
beta = betas_for_alpha_bar(T)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)


def alpha_bar_t(t):
    """
    Returns alpha_bar for a given timestep t.
    Handles both scalar and tensor inputs.
    """
    t = t.long()
    return alpha_bar[t]

def null_step(x_t, score, t, prev_t, k=1.0, sigma=1.0):
    return x_t

def ddpm_step(x_t, score, t, prev_t, k=1.0, sigma=1.0):
    alpha_prod_t = alpha_bar_t(torch.tensor(t).to(x_t.device))
    alpha_prod_prev_t = alpha_bar_t(torch.tensor(prev_t).to(x_t.device))
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_prev_t = 1 - alpha_prod_prev_t
    current_alpha_t = alpha_prod_t / alpha_prod_prev_t
    current_beta_t = 1 - current_alpha_t

    epsilon = - score * ((1 - alpha_prod_t) ** 0.5)
    x_0_pred = (x_t - beta_prod_t ** (0.5) * epsilon) / alpha_prod_t ** (0.5)
    x_0_coeff = (alpha_prod_prev_t ** (0.5) * current_beta_t) / beta_prod_t
    x_t_coeff = current_alpha_t ** (0.5) * beta_prod_prev_t / beta_prod_t
    x_prev_pred = x_0_coeff * x_0_pred + x_t_coeff * x_t

    noise = torch.randn_like(x_t)
    noise_coeff = current_beta_t ** 0.5
    x_prev = x_prev_pred + noise_coeff * noise
    return x_prev

def CNS_step(x_t, score, t, prev_t, k=1.0, sigma=1.0):
    alpha_prod_t = alpha_bar_t(torch.tensor(t).to(x_t.device))
    alpha_prod_prev_t = alpha_bar_t(torch.tensor(prev_t).to(x_t.device))
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_prev_t = 1 - alpha_prod_prev_t
    current_alpha_t = alpha_prod_t / alpha_prod_prev_t
    current_beta_t = 1 - current_alpha_t

    epsilon = - score * ((1 - alpha_prod_t) ** 0.5)
    x_0_pred = (x_t - beta_prod_t ** (0.5) * epsilon) / alpha_prod_t ** (0.5)
    x_0_coeff = (alpha_prod_prev_t ** (0.5) * current_beta_t) / beta_prod_t
    x_t_coeff = current_alpha_t ** (0.5) * beta_prod_prev_t / beta_prod_t
    x_prev_pred = x_0_coeff * x_0_pred + x_t_coeff * x_t

    noise = torch.randn_like(x_t)
    noise_coeff = current_beta_t ** 0.5
    x_prev = x_prev_pred + noise_coeff * noise / k**0.5
    return x_prev

def NSS_ddpm_step(x_t, score, t, prev_t, k=1.0, sigma=1.0):
    alpha_prod_t = alpha_bar_t(torch.tensor(t).to(x_t.device))
    alpha_prod_prev_t = alpha_bar_t(torch.tensor(prev_t).to(x_t.device))
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_prev_t = 1 - alpha_prod_prev_t
    current_alpha_t = alpha_prod_t / alpha_prod_prev_t
    current_beta_t = 1 - current_alpha_t

    epsilon = - score * ((1 - alpha_prod_t) ** 0.5)
    epsilon = epsilon * k

    x_0_pred = (x_t - beta_prod_t ** (0.5) * epsilon) / alpha_prod_t ** (0.5)
    x_0_coeff = (alpha_prod_prev_t ** (0.5) * current_beta_t) / beta_prod_t
    x_t_coeff = current_alpha_t ** (0.5) * beta_prod_prev_t / beta_prod_t
    x_prev_pred = x_0_coeff * x_0_pred + x_t_coeff * x_t

    noise = torch.randn_like(x_t)
    noise_coeff = current_beta_t ** 0.5
    x_prev = x_prev_pred + noise_coeff * noise
    return x_prev

def TSR_ddpm_step(x_t, score, t, prev_t, k=1.0, sigma=1.0):
    alpha_prod_t = alpha_bar_t(torch.tensor(t).to(x_t.device))
    alpha_prod_prev_t = alpha_bar_t(torch.tensor(prev_t).to(x_t.device))
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_prev_t = 1 - alpha_prod_prev_t
    current_alpha_t = alpha_prod_t / alpha_prod_prev_t
    current_beta_t = 1 - current_alpha_t

    epsilon = - score * ((1 - alpha_prod_t) ** 0.5)
    ratio = (1 - alpha_prod_t + alpha_prod_t * sigma ** 2) / (
            1 - alpha_prod_t + alpha_prod_t * sigma ** 2 / k
        )
    epsilon = epsilon * ratio

    x_0_pred = (x_t - beta_prod_t ** (0.5) * epsilon) / alpha_prod_t ** (0.5)
    x_0_coeff = (alpha_prod_prev_t ** (0.5) * current_beta_t) / beta_prod_t
    x_t_coeff = current_alpha_t ** (0.5) * beta_prod_prev_t / beta_prod_t
    x_prev_pred = x_0_coeff * x_0_pred + x_t_coeff * x_t

    noise = torch.randn_like(x_t)
    noise_coeff = current_beta_t ** 0.5
    x_prev = x_prev_pred + noise_coeff * noise
    return x_prev

def score_to_velocity(score, x, t):
    v = - (x + t * score) / (1 - t)
    return v

def velocity_to_score(velocity, x, t):
    score = (- x - (1 - t) * velocity) / t
    return score


def flow_ode_step_score(x_t, score, t, prev_t, k=1.0, sigma=1.0):
    velocity = score_to_velocity(score, x_t, t)
    dt = prev_t - t
    x_prev = x_t + velocity * dt
    return x_prev

def flow_ode_step(x_t, velocity, t, prev_t, k=1.0, sigma=1.0):
    dt = prev_t - t
    x_prev = x_t + velocity * dt
    return x_prev

def TSR_flow_ode_step(x_t, velocity, t, prev_t, k=1.0, sigma=1.0):
    score = velocity_to_score(velocity, x_t, t)
    snr_t = (1 - t)**2 / t**2
    ratio = (snr_t * sigma**2 + 1) / (snr_t * sigma**2 / k + 1)
    score = score * ratio
    velocity = score_to_velocity(score, x_t, t)
    
    dt = prev_t - t
    x_prev = x_t + velocity * dt
    return x_prev

def flow_sde_step(x_t, velocity, t, prev_t, k=1.0, sigma=1.0, eta=1.0):
    delta_t = prev_t - t

    score = velocity_to_score(velocity, x_t, t)
    f_t = - 1 / (1 - t)
    g_t_sq = 2 * t / (1 - t)
    g_t = torch.sqrt(g_t_sq)

    drift = f_t * x_t - ((1 + eta**2) / 2) * g_t_sq * score
    x_prev_mean = x_t + drift * delta_t
    
    noise = torch.randn_like(x_t)
    delta_w = noise * torch.sqrt(delta_t.abs())
    x_prev = x_prev_mean + eta * g_t * delta_w
    return x_prev

def CNS_flow_sde_step(x_t, velocity, t, prev_t, k=1.0, sigma=1.0, eta=1.0):
    delta_t = prev_t - t

    score = velocity_to_score(velocity, x_t, t)
    f_t = - 1 / (1 - t)
    g_t_sq = 2 * t / (1 - t)
    g_t = torch.sqrt(g_t_sq)

    drift = f_t * x_t - ((1 + eta**2) / 2) * g_t_sq * score
    x_prev_mean = x_t + drift * delta_t
    
    noise = torch.randn_like(x_t)
    delta_w = noise * torch.sqrt(delta_t.abs())
    x_prev = x_prev_mean + eta * g_t * delta_w / (k**0.5)
    return x_prev

def TSR_flow_sde_step(x_t, velocity, t, prev_t, k=1.0, sigma=1.0, eta=1.0):
    delta_t = prev_t - t

    score = velocity_to_score(velocity, x_t, t)
    snr_t = (1 - t)**2 / t**2
    ratio = (snr_t * sigma**2 + 1) / (snr_t * sigma**2 / k + 1)
    score = score * ratio
    
    f_t = - 1 / (1 - t)
    g_t_sq = 2 * t / (1 - t)
    g_t = torch.sqrt(g_t_sq)

    drift = f_t * x_t - ((1 + eta**2) / 2) * g_t_sq * score
    x_prev_mean = x_t + drift * delta_t
    
    noise = torch.randn_like(x_t)
    delta_w = noise * torch.sqrt(delta_t.abs())
    x_prev = x_prev_mean + eta * g_t * delta_w
    return x_prev


@torch.no_grad()
def pc_learned_sampling(x_T, score_fn, T, num_inf_steps, step_fn, k=1.0, sigma=1.0, log_times=[], corrector_steps=0, snr=0.0):
    x_t = x_T
    all_x_t = []

    timesteps = torch.linspace(0, T, num_inf_steps+1).type(torch.int64).flip(dims=[0])
    timesteps[0] = T - 1

    for tid, t in enumerate(timesteps[:-1]):
        prev_t = timesteps[tid + 1] if tid < len(timesteps) - 1 else 0

        alpha_bar_t_val = alpha_bar_t(torch.tensor(t))
        score = score_fn(x_t, t)
        x_t = step_fn(x_t, score, t, prev_t, k=k, sigma=sigma)

        for j in range(corrector_steps):
            langevin_noise = torch.randn_like(x_t)
            score = score_fn(x_t, t) * k
            noise_norm = torch.norm(langevin_noise.reshape(langevin_noise.shape[0], -1), dim=-1).mean()
            score_norm = torch.norm(score.reshape(score.shape[0], -1), dim=-1).mean()

            step_size = 2 * alpha_bar_t_val * (snr * noise_norm / score_norm) ** 2
            # Anneal step size
            # step_size = step_size * ((corrector_steps - j) / corrector_steps) ** 2
            x_t = x_t + step_size * score + torch.sqrt(2 * step_size) * langevin_noise
        
        if prev_t in log_times:
            all_x_t.append(x_t.clone().detach().cpu())

    return x_t, all_x_t


def pc_flow_learned_sampling(x_T, velocity_fn, num_inf_steps, step_fn, k=1.0, sigma=1.0, log_times=[], corrector_steps=1, snr=0.01):
    with torch.no_grad():
        x_t = x_T
        all_x_t = []

        timesteps = torch.linspace(0, 1, num_inf_steps+1).flip(dims=[0])
        timesteps[0] = 0.999


        for tid, t in enumerate(timesteps[:-1]):
            prev_t = timesteps[tid + 1] if tid < len(timesteps) - 1 else 0

            velocity = velocity_fn(x_t, t)
            x_t = step_fn(x_t, velocity, t, prev_t, k=k, sigma=sigma)

            for j in range(corrector_steps):
                langevin_noise = torch.randn_like(x_t)
                score = velocity_to_score(velocity, x_t, t) * k
                noise_norm = torch.norm(langevin_noise.reshape(langevin_noise.shape[0], -1), dim=-1).mean()
                score_norm = torch.norm(score.reshape(score.shape[0], -1), dim=-1).mean()

                step_size = 2 * (1-t)**2 * (snr * noise_norm / score_norm) ** 2
                # Anneal step size
                # step_size = step_size * ((corrector_steps - j) / corrector_steps) ** 2
                x_t = x_t + step_size * score + torch.sqrt(2 * step_size) * langevin_noise

            if prev_t in log_times:
                all_x_t.append(x_t.clone().detach().cpu())

    return x_t, all_x_t

