import torch


def calc_residuals(Y_hat: torch.Tensor, Y: torch.Tensor):
    """
    :param Y_hat:   estimated values
    :param Y:       real values
    """
    assert Y_hat.shape == Y.shape
    return Y - Y_hat


def unfold_window(M, window_len: int, M_past=None, stride=1):
    """
    :param M:           [time_steps, #features]
    :param window_len:  int
    :param M_past:      None or [past_time_steps, #features] with past_time_steps >= window_len
    :return:
        With M_past:    [time_steps, window_len, #features]
        Without M_past: [time_steps - window_len + 1, window_len, #features]
    """
    if M_past is not None:
        return unfold_window(M=torch.concat((M_past[-window_len:], M), dim=0), window_len=window_len)
    else:
        return M.unfold(size=window_len, step=stride, dimension=0).transpose(1, 2)


def calc_stats(M, stats=('moment1', 'moment2')):
    """
    :param M: [time_steps, window_len, #features]
    :param stats: Tuple of Moments to calcualte ('moment1', 'moment2', 'moment3' 'moment4')
    :return: [time_steps, no_of_moments, #features]
    """
    stats_result = torch.empty((M.shape[0], len(stats), M.shape[2]), dtype=torch.float, device=M.device)
    for i, stat in enumerate(stats):
        stats_result[:, i, :] = _calc_stat(M, stat)
    return stats_result


def _calc_stat(M, stat: str):
    if stat == 'moment1':
        return M.mean(dim=1, keepdim=True)
    elif stat == 'moment2':
        return torch.std(dim=1, keepdim=True)
    else:
        raise NotImplemented(f"Stat Measure {stat} not implemented!")
