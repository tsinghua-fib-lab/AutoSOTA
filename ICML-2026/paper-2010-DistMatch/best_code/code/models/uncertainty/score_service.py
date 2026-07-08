import torch

from utils.calc_torch import calc_residuals

MODE_NORM_BY_TS_SD = "norm_by_ts_sd"


class ScoreService:

    def __init__(self) -> None:
        self._score_mode = None

    def set_mode(self, mode):
        assert mode in [None, MODE_NORM_BY_TS_SD]
        self._score_mode = mode

    @property
    def mode(self):
        return self._score_mode

    def get(self, Y_hat, Y, **kwargs):
        if self._score_mode is None:
            return calc_residuals(Y_hat=Y_hat, Y=Y)
        elif self._score_mode == MODE_NORM_BY_TS_SD:
            if len(kwargs['ts_sd'].shape) == 0:
                return torch.div(calc_residuals(Y_hat=Y_hat, Y=Y), kwargs['ts_sd'])
            else:
                return torch.div(calc_residuals(Y_hat=Y_hat, Y=Y), kwargs['ts_sd'][:, None])
        else:
            raise NotImplemented("Not implemented")

    def resolve(self, score, **kwargs):
        if self._score_mode is None:
            return score
        elif self._score_mode == MODE_NORM_BY_TS_SD:
            if 'repeat' in kwargs:
                return torch.mul(score, torch.repeat_interleave(kwargs['ts_sd'], kwargs['interleave'], dim=0).repeat(kwargs['repeat'], 1))
            else:
                return torch.mul(score, kwargs['ts_sd'])
        else:
            raise NotImplemented("Not implemented")


score = ScoreService()


def get_score_param(dataset):
    if score.mode is None:
        return {}
    elif score.mode == MODE_NORM_BY_TS_SD:
        return dict(ts_sd=dataset.Y_std)
