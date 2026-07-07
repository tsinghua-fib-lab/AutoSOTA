import torch
from .ising import ising2d_ham, ising2d_log_discrete_score, ising2d_visualize, ising2d_plot_2pt_corr, ising2d_mag
from .potts import potts2d_ham, potts2d_log_discrete_score, potts2d_visualize, potts2d_plot_2pt_corr, potts2d_mag, potts2d_balance
from utils import compute_energy_w2_distance


def get_log_prob(args):
    '''To accommodate possible conditional variables, use kwargs.'''
    if args.dist == 'ising':
        return lambda x, **kwargs: -args.beta * ising2d_ham(2*x-1, args.J, args.h)
    elif args.dist == 'potts':
        return lambda x, **kwargs: -args.beta * potts2d_ham(x, args.J)
    else:
        raise NotImplementedError(f"Unknown target distribution: {args.dist}")


def get_log_discrete_score(args):
    '''To accommodate possible conditional variables, use kwargs.'''
    if args.dist == 'ising':
        return lambda x, **kwargs: ising2d_log_discrete_score(2*x-1, args.J, args.h, args.beta)
    elif args.dist == 'potts':
        return lambda x, **kwargs: potts2d_log_discrete_score(x, args.J, args.beta, args.vocab_size)
    else:
        raise NotImplementedError(f"Unknown target distribution: {args.dist}")


def eval_samples(x: torch.Tensor, args, gt_sample_dataset: torch.Tensor = None) -> dict:
    r'''
    Evaluate generated samples during training.
    
    Args:
        x: generated samples, shape (B, D), values in range(N)
        args: training arguments
        gt_sample_dataset: ground truth samples, shape (., D), optional
    
    Returns:
        info: A dictionary containing evaluation metrics and visualization
    '''

    info = {}
    if args.dist == 'ising':
        info.update({
            'fig_samples': ising2d_visualize(x[:64]*2-1, num_per_row=16),
            'fig_2pt_corr': ising2d_plot_2pt_corr(x*2-1),
            'controller/mag': ising2d_mag(x*2-1).mean().item(),
            })
        if gt_sample_dataset is not None:
            info['controller/energy_w2_distance'] = compute_energy_w2_distance(
                x*2-1, gt_sample_dataset,
                energy_fn=lambda samp: ising2d_ham(samp, J=args.J, h=args.h),
                )
    elif args.dist == 'potts':
        info.update({
            'fig_samples': potts2d_visualize(x[:64], num_per_row=16, q=args.vocab_size),
            'fig_2pt_corr': potts2d_plot_2pt_corr(x, q=args.vocab_size),
            'controller/mag': potts2d_mag(x, q=args.vocab_size).mean().item(),
            'controller/balance': potts2d_balance(x, q=args.vocab_size),
            })
        if gt_sample_dataset is not None:
            info['controller/energy_w2_distance'] = compute_energy_w2_distance(
                x, gt_sample_dataset,
                energy_fn=lambda samp: potts2d_ham(samp, J=args.J),
                )
    return info
