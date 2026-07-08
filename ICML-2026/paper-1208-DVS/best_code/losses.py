import torch
from utils.graph_utils import node_flags, mask_x, mask_adjs, gen_noise

def get_pred_loss_fn(mix_x, mix_adj, train=True, reduce_mean=False, eps=1e-3, lambda_train=1, loss_type=None):
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: torch.sum(*args, **kwargs)

  def compute_loss(pred, target, loss_coeff):
    losses = torch.square( (pred - target) * loss_coeff[:, None, None] ) 
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * 0.5
    return losses

  def get_loss_coeff(sde, t, loss_type):
    if loss_type=='default':
        loss_coeff = sde.loss_coeff(t)
    elif loss_type=='const':
        loss_coeff = torch.ones_like(t)
    else:
        raise NotImplementedError(f'Loss type: {loss_type} not implemented.')
    return loss_coeff

  def loss_fn(model, x, adj, prior_samples=None):
    sde_x = mix_x.bridge(x)
    sde_adj = mix_adj.bridge(adj)
    t = torch.rand(adj.shape[0], device=adj.device) * (sde_adj.T - eps) 
    flags = node_flags(adj)

    if prior_samples is None:
        x0 = sde_x.prior_sampling(x.shape, x.device)
        adj0 = sde_adj.prior_sampling_sym(adj.shape, adj.device)
    else:
        x0, adj0 = prior_samples
    x0 = mask_x(x0, flags)
    adj0 = mask_adjs(adj0, flags)

    mean_x, std_x = sde_x.marginal_prob(x0, t)
    xt = mean_x + std_x[:, None, None] * gen_noise(x, flags, sym=False)
    xt = mask_x(xt, flags)

    mean_adj, std_adj = sde_adj.marginal_prob(adj0, t)
    adjt = mean_adj + std_adj[:, None, None] * gen_noise(adj, flags, sym=True) 
    adjt = mask_adjs(adjt, flags)

    pred_x, pred_adj = model(xt, adjt, t.unsqueeze(-1), flags)

    loss_coeff_x = get_loss_coeff(sde_x, t, loss_type.x)
    loss_coeff_adj = get_loss_coeff(sde_adj, t, loss_type.adj)

    losses_x = compute_loss(pred_x, x, loss_coeff_x)
    losses_adj = compute_loss(pred_adj, adj, loss_coeff_adj)
    losses = torch.mean(losses_x) + torch.mean(losses_adj) * lambda_train
    return losses, torch.mean(losses_x).detach().cpu(), torch.mean(losses_adj).detach().cpu()
  return loss_fn