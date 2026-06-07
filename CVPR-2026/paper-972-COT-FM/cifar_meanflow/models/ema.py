import torch
import logging


def init_ema(net, net_ema, ema_decay):
    for p, p_ema in zip(net.parameters(), net_ema.parameters()):
        p_ema.data.copy_(p.data)
        p_ema.requires_grad = False
    net_ema.ema_decay = ema_decay
    return net_ema


def update_ema_net(net, net_ema, num_updates, period=16):
    decay_effective = net_ema.ema_decay ** period  # period update to speedup
    if num_updates % period == 0:
        with torch.no_grad():
            for p, p_ema in zip(net.parameters(), net_ema.parameters()):
                # double precision to avoid numerical issues
                delta = p.data.double() - p_ema.data.double()
                p_ema_new = p_ema.data.double() + (1 - decay_effective) * delta
                p_ema.data.copy_(p_ema_new.float())