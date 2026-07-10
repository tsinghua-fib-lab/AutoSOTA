import torch.nn.functional as F


def p_xt_g_x1(X1, E1, t, limit_dist):
    # x1 (B, D)
    # t float
    # returns (B, D, S) for varying x_t value
    """
    X1.shape: [B, N, 1]
    E1.shape: [B, N, N, 1]
    t.shape: [B, 1]
    """
    device = X1.device
    limit_dist.X = limit_dist.X.to(device)
    limit_dist.E = limit_dist.E.to(device)

    t_time = t.squeeze(-1)[:, None, None]
    """
    t_time.shape: [B, 1, 1]
    """
    X1_onehot = F.one_hot(X1, num_classes=len(limit_dist.X)).float()
    E1_onehot = F.one_hot(E1, num_classes=len(limit_dist.E)).float()

    Xt = t_time * X1_onehot + (1 - t_time) * limit_dist.X[None, None, :]
    Et = (
        t_time[:, None] * E1_onehot
        + (1 - t_time[:, None]) * limit_dist.E[None, None, None, :]
    )

    assert ((Xt.sum(-1) - 1).abs() < 1e-4).all() and (
        (Et.sum(-1) - 1).abs() < 1e-4
    ).all()

    return Xt.clamp(min=0.0, max=1.0), Et.clamp(min=0.0, max=1.0)


def dt_p_xt_g_x1(X1, E1, limit_dist):
    # x1 (B, D)
    # returns (B, D, S) for varying x_t value
    device = X1.device
    limit_dist.X = limit_dist.X.to(device)
    limit_dist.E = limit_dist.E.to(device)

    X1_onehot = F.one_hot(X1, num_classes=len(limit_dist.X)).float()
    E1_onehot = F.one_hot(E1, num_classes=len(limit_dist.E)).float()

    dX = X1_onehot - limit_dist.X[None, None, :]
    dE = E1_onehot - limit_dist.E[None, None, None, :]

    assert (dX.sum(-1).abs() < 1e-4).all() and (dE.sum(-1).abs() < 1e-4).all()

    return dX, dE
