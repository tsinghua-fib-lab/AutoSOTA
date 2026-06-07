import torch


def logit_normal_timestep_sample(P_mean: float, P_std: float, num_samples: int, device: torch.device) -> torch.Tensor:
    rnd_normal = torch.randn((num_samples,), device=device)
    time = torch.sigmoid(rnd_normal * P_std + P_mean)
    time = torch.clip(time, min=0.0, max=1.0)
    return time


def sample_two_timesteps(args, num_samples: int, device: torch.device):
    if args.tr_sampler == "v0":
        t, r = sample_two_timesteps_t_r_v0(args, num_samples, device=device)
        return t, r
    elif args.tr_sampler == "v1":
        t, r = sample_two_timesteps_t_r_v1(args, num_samples, device=device)
        return t, r
    else:
        raise ValueError(f"Unknown joint time sampler: {args.tr_sampler}")
    

def sample_two_timesteps_t_r_v0(args, num_samples: int, device: torch.device):
    """
    Sampler (t, r): independently sample t and r, with post-processing.
    Version 0: used in paper.
    """
    # step 1: sample two independent timesteps
    t = logit_normal_timestep_sample(args.P_mean_t, args.P_std_t, num_samples, device=device)
    r = logit_normal_timestep_sample(args.P_mean_r, args.P_std_r, num_samples, device=device)

    # step 2: ensure t >= r
    t, r = torch.maximum(t, r), torch.minimum(t, r)

    # step 3: make t and r different with a probability of args.ratio
    prob = torch.rand(num_samples, device=device)
    mask = prob < 1 - args.ratio
    r = torch.where(mask, t, r)

    return t, r


def sample_two_timesteps_t_r_v1(args, num_samples: int, device: torch.device):
    """
    Sampler (t, r): independently sample t and r, with post-processing.
    Version 1: different post-processing to ensure t >= r.
    """
    # step 1: sample two independent timesteps
    t = logit_normal_timestep_sample(args.P_mean_t, args.P_std_t, num_samples, device=device)
    r = logit_normal_timestep_sample(args.P_mean_r, args.P_std_r, num_samples, device=device)

    # step 2: make t and r different with a probability of args.ratio
    prob = torch.rand(num_samples, device=device)
    mask = prob < 1 - args.ratio
    r = torch.where(mask, t, r)

    # step 3: ensure t >= r
    r = torch.minimum(t, r)

    return t, r    


