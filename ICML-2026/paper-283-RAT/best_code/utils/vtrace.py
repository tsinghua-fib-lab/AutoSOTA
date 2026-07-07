import torch

def compute_td(rew, val, n_val, done, gamma):
    target = rew + gamma * n_val * (1-done)
    adv = target - val
    return target, adv

def compute_v_trace(rew, val, n_val, done, rhos, gamma, lam, nsteps, clip_range=0.2):
    # clipped_rhos = torch.clamp(rhos, 1 - clip_range, 1 + clip_range)
    # clipped_rhos = torch.clamp(rhos, min=0.01, max = 1 + clip_range) 
    clipped_rhos = torch.clamp(rhos, max = 1 + clip_range) 
    deltas = clipped_rhos * (rew + gamma * n_val * (1-done) - val) 
    # cs = torch.clamp(rhos, min=0.01, max=lam)
    cs = torch.clamp(rhos, max = lam)

    # v-trace
    vs_minus_v_xs = torch.zeros_like(deltas)
    for t in reversed(range(nsteps)):
        if t == nsteps - 1:
            vs_minus_v_xs[:, t] = deltas[:, t]
        else:
            vs_minus_v_xs[:, t] = gamma * cs[:, t] * vs_minus_v_xs[:, t+1] * (1-done[:, t]) + deltas[:, t]

    v_trace = vs_minus_v_xs + val

    # new value at t+1
    vs_t_plus_1 = torch.cat((v_trace[:, 1:], n_val[:, -1:]), dim=1)
    adv = rew + gamma * vs_t_plus_1 * (1-done) - val

    return v_trace, adv