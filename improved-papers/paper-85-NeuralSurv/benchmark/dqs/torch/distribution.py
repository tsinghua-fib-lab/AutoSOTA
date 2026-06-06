import torch

class DistributionLinear:
    '''
    This class represents probability distribution.
    Key points of cumulative distribution functions are stored as boundaries,
    and they are connected by using linear interpolation.
    '''

    def __init__(self, boundaries, axis='target'):
        '''
        axis should be 'target' or 'quantile'
        '''
        self.boundaries = boundaries
        if axis=='target':
            self.axis_is_target = True
        elif axis=='quantile':
            self.axis_is_target = False
        else:
            raise ValueError('Unknown axis value'+axis)

    def _interpolate(self, pred, y, mask):
        if mask is not None:
            pred = pred[mask]
            y = y[mask]
        cum_pred = torch.cumsum(pred, dim=1)
        F_pred = torch.cat([torch.zeros(pred.shape[0],1), cum_pred], 1)

        # compute idx and ratio
        idx = torch.searchsorted(self.boundaries, y, right=True)
        b_lb = self.boundaries[idx-1]
        mask_of = (idx < len(self.boundaries))
        ratio = torch.zeros_like(b_lb)
        b_ub = self.boundaries[idx[mask_of]]
        ratio[mask_of] = (y[mask_of]-b_lb[mask_of]) / (b_ub-b_lb[mask_of])
        idx[~mask_of] -= 1
        ratio[~mask_of] = 1.0

        # linear interpolation
        left = torch.gather(F_pred, 1, idx-1)
        right = torch.gather(F_pred, 1, idx)
        return torch.lerp(left, right, ratio)

    def _interpolate_inv(self, pred, quantiles, mask):
        if mask is not None:
            pred = pred[mask]
            quantiles = quantiles[mask]
        cum_pred = torch.cumsum(pred, dim=1)
        F_pred = torch.cat([torch.zeros(pred.shape[0],1), cum_pred], 1)

        # compute idx and ratio
        idx = torch.searchsorted(F_pred, quantiles, right=True)
        Fs_lb = torch.gather(F_pred, 1, idx-1)
        mask_of = (idx < len(self.boundaries)).view(-1)
        Fs_ub_mask = torch.gather(F_pred[mask_of], 1, idx[mask_of])
        ratio = torch.zeros_like(Fs_lb)
        ratio_numerator = quantiles[mask_of] - Fs_lb[mask_of]
        ratio[mask_of] =  ratio_numerator / (Fs_ub_mask - Fs_lb[mask_of])
        idx[~mask_of] -= 1
        ratio[~mask_of] = 1.0

        # linear interpolation
        left = self.boundaries[idx-1]
        right = self.boundaries[idx]
        return torch.lerp(left, right, ratio)

    def cdf(self, pred, y, mask=None):
        '''
        Cumulative distribution function.

        Parameters
        ----------
        pred : Tensor
            Each row represents a probability distribution.
            The sum of each row must be equal to one.
            Tensor shape is [batch size, n_bin+1].
        y : Tensor
            Compute CDF of y
            Tensor shape is [batch size, col_size].
        mask : Tensor
            Mask rows of pred and y.
            Tensor shape is [batch size].

        Returns
        -------
        quantiles : Tensor
            Computed quantiles of y.
            Tensor shape is equal to the shape of y.
        '''
        if self.axis_is_target:
            return self._interpolate(pred, y, mask)
        else:
            return self._interpolate_inv(pred, y, mask)

    def icdf(self, pred, quantile, mask=None):
        '''
        Inverse of cumulative distribution function.

        Parameters
        ----------
        pred : Tensor
            Piecewise-linear CDF with n_bin+1 endpoints.
            Each row corresponds to a CDF.
            pred[:,0] = 0.0 and pred[:,-1] = 1.0
            Tensor shape is [batch size, n_bin+1].
        quantile : Tensor
            Quantiles
            Tensor shape is [batch size, col_size].
        mask : Tensor
            Mask rows of pred and y.
            Tensor shape is [batch size].

        Returns
        -------
        y : Tensor
            Compute y.
            Tensor shape is equal to the shape of quantile.
        '''
        if self.axis_is_target:
            return self._interpolate_inv(pred, quantile, mask)
        else:
            return self._interpolate(pred, quantile, mask)
