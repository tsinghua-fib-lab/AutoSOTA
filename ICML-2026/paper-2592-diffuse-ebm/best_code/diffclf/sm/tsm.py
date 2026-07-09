# Target Score Matching (TSM) loss

# Libraries
import torch
from ..utils.se3_utils import remove_mean

def compute_loss(ts, data, net, sde, target_score, antithetic=True, is_particles=False):
    """Compute the Target Score Matching (TSM) loss

    This corresponds to the orignal loss from ArXiv:2011.13456 with s^2_t weighting.

    NOTE: Here the squared norm is divided by the dimension.

    Args:
            * ts (torch.Tensor of shape (batch_size, *data_shape_ones)): Evaluation times
            * data (torch.Tensor of shape (batch_size, *data_shape)): Evaluation points
            * net (function): Network modeling the score
            * sde (LinearSDE): SDE
            * target_score (function): Score of the target distribution
            * antithetic (bool): Whether to apply the antithetic trick (default is True)
            * is_particles (bool): Whether it is a particle system (default is False)

    Returns:
            * loss (torch.Tensor of shape (batch_size,)): Value of the loss
    """

    # Get the shapes
    data_shape = data.shape[1:]
    sum_indexes = tuple([-(i + 1) for i in range(len(data_shape))])
    # Noise the data
    loc, var = sde.noise_sample_params(ts, data)
    zs = torch.randn_like(data)
    if is_particles:
        zs = remove_mean(zs)
    ys = loc + torch.sqrt(var) * zs
    # Compute the loss
    score_ = target_score(data)
    if is_particles:
        score_ = remove_mean(score_)
    loss = torch.mean(torch.square(sde.s(ts) * net(ts, ys) - score_), dim=sum_indexes)
    # Apply the antithetic trick
    if antithetic:
        ys_antithetic = loc - torch.sqrt(var) * zs
        loss += torch.mean(torch.square(sde.s(ts) * net(ts, ys_antithetic) - score_), dim=sum_indexes)
        loss /= 2.0
    return loss
