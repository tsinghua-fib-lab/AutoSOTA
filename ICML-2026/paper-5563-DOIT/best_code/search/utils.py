import torch
from search.configs import Arguments
from typing import List
from copy import deepcopy

def get_hyper_params(args: Arguments) -> List[Arguments]:
    valid_datasets = {
        "halfcheetah-medium-expert-v2",
        "hopper-medium-expert-v2",
        "walker2d-medium-expert-v2",
        "halfcheetah-medium-v2",
        "hopper-medium-v2",
        "walker2d-medium-v2",
        "halfcheetah-medium-replay-v2",
        "hopper-medium-replay-v2",
        "walker2d-medium-replay-v2",
    }
    if args.dataset not in valid_datasets:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    args_grid = []
    for particles in [4, 8]:
        cur_args = deepcopy(args)
        cur_args.per_sample_batch_size = particles
        args_grid.append(cur_args)

    return args_grid

def ban_requires_grad(module):
    for param in module.parameters():
        param.requires_grad = False

def check_grad_fn(x_need_grad):
    assert x_need_grad.requires_grad, "x_need_grad should require grad"


def rescale_grad(
    grad: torch.Tensor, clip_scale, **kwargs
):  # [B, N, 3+5]
    node_mask = kwargs.get('node_mask', None)

    scale = (grad ** 2).mean(dim=-1)
    if node_mask is not None:  # [B, N, 1]
        scale: torch.Tensor = scale.sum(dim=-1) / node_mask.float().squeeze(-1).sum(dim=-1)  # [B]
        clipped_scale = torch.clamp(scale, max=clip_scale)
        co_ef = clipped_scale / scale  # [B]
        grad = grad * co_ef.view(-1, 1, 1)

    return grad

def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """    
    section_counts = [section_counts]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return sorted(list(set(all_steps)), reverse=True)