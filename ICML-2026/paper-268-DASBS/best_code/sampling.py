import torch
import torch.nn as nn
from noise import Noise
from tqdm import tqdm


def ref_trans_log_p(s: torch.Tensor, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
                    noise: Noise, vocab_size: int) -> torch.Tensor:
    r'''
    Transition log probability log p^r_{t|s}(y|x), assuming 0 <= s < t <= 1
    
    p^r_{t|s}(y|x) = B(s, t) ^ D * C(s, t) ^ d_H(x, y), see paper for formula.

    Args:
        s: start time, shape (..., 1)
        t: end time, shape (..., 1)
        x: start state, shape (..., D), values in range(N)
        y: end state, shape (..., D), values in range(N)
        noise: Noise object
        vocab_size: size of the discrete state space N
    Returns:
        Transition log probability log p^r_{t|s}(y|x), shape (..., 1)
    '''
    hamming_dist = (x != y).sum(dim=-1, keepdim=True) # (..., 1)
    exp_minus_gamma = (-noise.integral_noise(s, t)).exp() # (..., 1)
    b = (1 + (vocab_size - 1) * exp_minus_gamma) / vocab_size # (..., 1)
    c = (1 - exp_minus_gamma) / (1 + (vocab_size - 1) * exp_minus_gamma) # (..., 1)
    return x.shape[-1] * b.log() + hamming_dist * c.log()


@torch.no_grad()
def sample_tau_leaping(batch_size: int, steps: int, args, controller: nn.Module, noise: Noise,
                       cond: torch.Tensor = None, return_all_steps: bool = False,
                       log_prob: callable = None, 
                       init_sample_dataset: torch.Tensor = None,
                       use_tqdm: bool = False):
    r'''
    Tau-leaping sampler for CTMC with transition rate
    u_t(x^{d<-n}, x) = gamma_t / N * varPhi_t(x)_{d,n} for n != x^d.

    Args:
        batch_size: number of samples to generate
        steps: number of time steps. Use uniform discretization from 0 to 1 by default.
        args: training arguments
        controller: model for log varPhi_t(x), takes inputs (x, t, cond) and outputs (B, D, N)
        noise: Noise object
        cond: optional conditioning information for the model, shape (B, ...)
        return_all_steps: whether to return samples at all time steps or only the initial and final samples
        log_prob: log probability function of the target distribution, input is (B, D), output is (B,)
            used for computing log dp*/dp^r for the memoryless case, since
            log p^*/p^r (x) = log nu/p^r_1 (x_1) + const = log nu(x_1) + const
        init_sample_dataset: optional initial sample dataset to draw initial samples from, shape (-1, D)
    
    Returns:
        if return_all_steps:
            [x0, ..., x1]: list of length (steps + 1), each of shape (B, D), values in range(N)
            info: dict of additional info
        else:
            [x0, x1]: Samples at time t = 1, shape (B, D), values in range(N)
            info: dict of additional info
        
        The info dict contains:
            'log_p_u': log p_u(x) on the time grid
            'log_p_r': log p_r(x) on the time grid
            'log_dp_star_dp_r': log dp*/dp_r (x)
            'log_dp_star_dp_u': log dp*/dp_u (x)
            'num_jumps_per_dim': average number of state changes per dimension in each sample, shape (B,)
    '''
    device = args.device

    if args.get("beta_init", 0) == 0: # uniform initial distribution
        x = torch.randint(0, args.vocab_size, (batch_size, args.seq_len)).to(device) # (B, D)
    elif args.beta_init in [float('inf'), 'inf']: # zero-temperature initial distribution (assume h=0 for simplicity)
        x = torch.randint(0, args.vocab_size, (batch_size, 1)).to(device).repeat(1, args.seq_len) # (B, D)
    else:
        assert init_sample_dataset is not None
        idx = torch.randperm(init_sample_dataset.shape[0])[:batch_size]
        x = init_sample_dataset[idx].to(device)  # (B, D)
    samples = [x]

    if not args.memoryless:
        ts = torch.linspace(0, 1, steps + 1).to(device)
    else:
        ts = torch.linspace(0.01, 1, steps + 1).to(device) # gammabar_{0,t} >> 1 when alpha ~ 0
    info = {k: torch.zeros(batch_size).to(device) for k in [
        'log_dp_star_dp_r', 'log_p_u', 'log_p_r', 'log_dp_star_dp_u', 'num_jumps_per_dim']}

    pbar = range(steps)
    if use_tqdm: pbar = tqdm(pbar, desc='Sampling', leave=False)
    for i in pbar:
        t, dt = ts[i], ts[i+1] - ts[i] # scalars
        t = t.repeat(batch_size, 1) # (B, 1)

        varPhi_t = controller(x, t, cond=cond).exp() # (B, D, N)
        varPhi_t.scatter_(dim=2, index=x[..., None], value=0)
        gammabar = noise.integral_noise(t, t + dt) # (B, 1)
        trans_prob = gammabar[..., None] / args.vocab_size * varPhi_t # (B, D, N)
        row_sum = trans_prob.sum(dim=-1, keepdim=True) # (B, D, 1)
        invalid_mask = ((row_sum > 1) | row_sum.isnan() | row_sum.isinf())[... , 0] # (B, D)
        trans_prob.scatter_(dim=2, index=x[..., None], src=1-row_sum)
        
        if invalid_mask.any(): 
            if torch.any(row_sum > 1):
                print(f"!!Warning!! Step {i+1} of tau-leaping: Transition probability row sum exceeds 1.")
            if torch.any(row_sum.isnan()):
                print(f"!!Warning!! Step {i+1} of tau-leaping: NaN detected in transition probabilities.")
            if torch.any(row_sum.isinf()):
                print(f"!!Warning!! Step {i+1} of tau-leaping: Inf detected in transition probabilities.")
            # set trans_prob at invalid_mask to be one-hot at current state x, i.e., no state change
            trans_prob[invalid_mask] = 0
            batch_indices, seq_indices = torch.where(invalid_mask)
            trans_prob[batch_indices, seq_indices, x[batch_indices, seq_indices]] = 1

        x_new = torch.multinomial(trans_prob.view(batch_size * args.seq_len, args.vocab_size),
                                  num_samples=1).view(batch_size, args.seq_len)
        info['num_jumps_per_dim'] += (x_new != x).sum(dim=-1) / args.seq_len  # (B,)

        prob_jump_to = trans_prob.gather(dim=2, index=x_new[..., None])[..., 0]  # (B, D)
        info['log_p_u'] += prob_jump_to.clamp(min=1e-10).log().sum(dim=-1)  # (B,)

        info['log_p_r'] += ref_trans_log_p(t, t+dt, x, x_new, noise, vocab_size=args.vocab_size)[..., 0]  # (B,)

        term1 = gammabar[..., 0] / args.vocab_size * (
            args.seq_len * (args.vocab_size - 1) - varPhi_t.sum(dim=(1, 2))
        )  # (B,)
        term2 = (varPhi_t.gather(dim=2, index=x_new[..., None])[..., 0]
                 .clamp(min=1e-10).log() * (x_new != x)).sum(dim=-1)  # (B,)
        info['log_dp_star_dp_r'] += term1 + term2

        samples.append(x_new); x = x_new

    if args.memoryless and log_prob is not None:
        info['log_dp_star_dp_r'] = log_prob(samples[-1])

    info['log_dp_star_dp_u'] = info['log_dp_star_dp_r'] + info['log_p_r'] - info['log_p_u']

    if return_all_steps:
        return samples, info
    else:
        return [samples[0], samples[-1]], info


def batched_sample_tau_leaping(rounds: int, batch_size: int, steps: int, args, controller: nn.Module,
                               noise: Noise, cond: torch.Tensor = None, return_all_steps: bool = False,
                               log_prob: callable = None,
                               init_sample_dataset: torch.Tensor = None,
                               use_tqdm: bool = False,
                               ):
    '''
    Batched version of sample_tau_leaping to reduce memory usage.
    '''
    all_samples = []; all_info = {}
    pbar = range(rounds)
    if use_tqdm: pbar = tqdm(pbar, desc='Sampling Rounds', leave=False)
    for _ in pbar:
        samples, info = sample_tau_leaping(
            batch_size=batch_size, steps=steps, args=args,
            controller=controller, noise=noise, cond=cond,
            return_all_steps=return_all_steps,
            log_prob=log_prob,
            init_sample_dataset=init_sample_dataset,
            )
        all_samples.append(samples)
        for k, v in info.items():
            if k not in all_info:
                all_info[k] = []
            all_info[k].append(v)

    for k in all_info:
        all_info[k] = torch.cat(all_info[k], dim=0)

    if return_all_steps:
        cat_samples = []
        for step in range(steps + 1):
            cat_samples.append(torch.cat([all_samples[r][step] for r in range(rounds)], dim=0))
        return cat_samples, all_info
    else:
        x0 = torch.cat([all_samples[r][0] for r in range(rounds)], dim=0)
        x1 = torch.cat([all_samples[r][1] for r in range(rounds)], dim=0)
        return [x0, x1], all_info
