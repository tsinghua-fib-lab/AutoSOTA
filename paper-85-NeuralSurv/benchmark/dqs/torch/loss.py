import numpy as np
import torch


class NegativeLogLikelihood:
    '''
    Negative log-likelihood
    self.distribution : distribution
    self.loss_boundaries : Tensor (dtype=torch.float)
        self.loss_boundaries.shape = [ # bin ]
    self.EPSILON : float
        Small positive float number to avoid computing log(0)
    '''

    def __init__(self, distribution, loss_boundaries, epsilon=0.000001):
        self.distribution = distribution
        self.loss_boundaries = loss_boundaries
        self.EPSILON = epsilon

    def _compute_F(self, pred, z):
        idx = torch.searchsorted(self.loss_boundaries,
                                 z.view(-1,1),
                                 right=True).view(-1)
        b_lb = self.loss_boundaries[idx-1]
        b_ub = self.loss_boundaries[idx]
        F_lb = self.distribution.cdf(pred, b_lb.view(-1,1))
        F_ub = self.distribution.cdf(pred, b_ub.view(-1,1))
        return F_lb, F_ub

    def _logarithmic(self, F_lb, F_ub, epsilon):
        return -torch.sum(torch.log(F_ub-F_lb+epsilon))

    def _logarithmic_censored(self, F_lb, F_ub, F_c, epsilon):
        w = ((F_ub - F_c) / (1.0 - F_c)).detach()
        temp = w * torch.log(F_ub-F_lb+epsilon)
        temp += (1.0-w) * torch.log(1.0-F_ub+epsilon)
        return -torch.sum(temp)

    def loss(self, pred, z, e=None):
        F_lb, F_ub = self._compute_F(pred, z)
        if e is None:
            F_lb_uncensored = F_lb
            F_ub_uncensored = F_ub
        else:
            uncensored = e.bool()
            F_lb_uncensored = F_lb[uncensored]
            F_ub_uncensored = F_ub[uncensored]

        loss = self._logarithmic(F_lb_uncensored,
                                 F_ub_uncensored,
                                 self.EPSILON)
        if e is not None:
            F_c = self.distribution.cdf(pred, z.view(-1,1), ~uncensored)
            loss += self._logarithmic_censored(F_lb[~uncensored],
                                               F_ub[~uncensored],
                                               F_c,
                                               self.EPSILON)
        return loss / pred.shape[0]

class CensoredNegativeLogLikelihood:
    def __init__(self, distribution, loss_boundaries, epsilon=0.000001):
        self.nll = NegativeLogLikelihood(distribution,
                                         loss_boundaries,
                                         epsilon)

    def loss(self, pred, z, e):
        return self.nll.loss(pred, z, e)

class Brier:
    '''
    Brier score
    self.distribution : distribution
    self.loss_boundaries : Tensor (dtype=torch.float)
        self.loss_boundaries.shape = [ # bin ]
    '''

    def __init__(self, distribution, loss_boundaries):
        self.distribution = distribution
        self.loss_boundaries = loss_boundaries

    def loss(self, pred, y, e=None):
        # compute Fs_lb and Fs_ub
        boundaries = torch.tile(self.loss_boundaries, (pred.shape[0],1))
        Fs_lb = self.distribution.cdf(pred, boundaries[:,:-1])
        Fs_ub = self.distribution.cdf(pred, boundaries[:,1:])

        # set coef as one-hot vector
        idx = torch.searchsorted(self.loss_boundaries,
                                 y.view(-1,1),
                                 right=True).view(-1)
        n_bin = len(self.loss_boundaries)-1
        one_hot = torch.nn.functional.one_hot(idx-1, num_classes=n_bin)
        coef = one_hot.to(torch.float)

        # update coef for censored data
        if e is not None:
            uncensored = e.bool()
            F_c = self.distribution.cdf(pred, y.view(-1,1), ~uncensored)
            alpha = (Fs_ub[~uncensored] - F_c) / (1.0 - F_c)
            coef[~uncensored] *= alpha
            upper_fill = np.tri(n_bin, n_bin, -1, dtype=np.float32)
            upper_fill = torch.from_numpy(upper_fill.T[idx[~uncensored]-1])
            beta = (Fs_ub[~uncensored] - Fs_lb[~uncensored]) * upper_fill
            coef[~uncensored] += beta / (1.0 - F_c)

        # delete gradients
        coef = coef.detach()

        # compute loss
        fi = Fs_ub - Fs_lb
        sq1 = (fi - 1.0) * (fi - 1.0)
        sq0 = fi * fi
        return torch.sum(coef*sq1 + (1.0-coef)*sq0) / pred.shape[0]

class CensoredBrier:
    def __init__(self, distribution, loss_boundaries):
        self.brier = Brier(distribution, loss_boundaries)

    def loss(self, pred, z, e):
        return self.brier.loss(pred, z, e)

class RankedProbabilityScore:
    def __init__(self, distribution, loss_boundaries):
        self.distribution = distribution
        self.loss_boundaries = loss_boundaries

    def loss(self, pred, y, e=None):
        # compute Fs
        boundaries = torch.tile(self.loss_boundaries, (pred.shape[0],1))
        Fs = self.distribution.cdf(pred, boundaries)

        # compute coef
        idx = torch.searchsorted(self.loss_boundaries, y.view(-1,1),
                                 right=True).view(-1)
        n_bin = len(self.loss_boundaries)-1
        lower_fill = np.tri(n_bin, n_bin, -1, dtype=np.float32)
        lower_fill = torch.from_numpy(lower_fill[idx-1])
        coef = 1.0 - lower_fill[:,:-1]

        # update coef of censored data points
        if e is not None:
            uncensored = e.bool()
            F_c = self.distribution.cdf(pred, y.view(-1,1), ~uncensored)
            coef[~uncensored] *= (Fs[~uncensored,1:-1] - F_c) / (1.0 - F_c)

        # delete gradients
        coef = coef.detach()

        # compute loss
        sq1 = (Fs[:,1:-1] - 1.0) * (Fs[:,1:-1] - 1.0)
        sq0 = Fs[:,1:-1] * Fs[:,1:-1]
        return torch.sum((1.0-coef)*sq0 + coef*sq1) / pred.shape[0]

class CensoredRankedProbabilityScore:
    def __init__(self, distribution, loss_boundaries):
        self.rps = RankedProbabilityScore(distribution, loss_boundaries)

    def loss(self, pred, z, e):
        return self.rps.loss(pred, z, e)

class Pinball:
    def __init__(self, distribution, loss_boundaries):
        self.distribution = distribution
        self.loss_boundaries = loss_boundaries

    def _pinball_loss(self, y, y_pred):
        taus = self.loss_boundaries[1:-1]
        diff = y - y_pred
        w = (diff >= 0.0).float()
        loss = w * (diff * taus)
        loss += (1.0-w)*(diff * (taus-1.0))
        return loss

    def loss(self, pred, y, e=None):
        taus = self.loss_boundaries[1:-1]
        boundaries = torch.tile(taus, (pred.shape[0],1))
        y_pred = self.distribution.icdf(pred, boundaries)
        y = y.view(-1,1)

        # compute loss for uncensored data points
        if e is None:
            y_uncensored = y
            y_pred_uncensored = y_pred
        else:
            uncensored = e.bool()
            y_uncensored = y[uncensored]
            y_pred_uncensored = y_pred[uncensored]
        loss = torch.sum(self._pinball_loss(y_uncensored, y_pred_uncensored))

        # compute loss for censored data points
        if e is not None:
            # compute parameters
            c = y[~uncensored]
            c_pred = y_pred[~uncensored]
            tau_c = self.distribution.cdf(c_pred, c)
            w = ((taus - tau_c) / (1 - tau_c))
            w = torch.clamp(w, min=0.0)

            # delete gradients
            w = w.detach()

            # compute loss
            loss += torch.sum(w * self._pinball_loss(c, c_pred))
            c_max = self.distribution.boundaries[-1]
            c_inf = c_max * torch.ones_like(c_pred)
            loss += torch.sum((1.0-w) * self._pinball_loss(c_inf, c_pred))

        return loss / pred.shape[0]

class Portnoy:
    def __init__(self, distribution, loss_boundaries):
        self.pinball = Pinball(distribution, loss_boundaries)

    def loss(self, pred, z, e):
        return self.pinball.loss(pred, z, e)
