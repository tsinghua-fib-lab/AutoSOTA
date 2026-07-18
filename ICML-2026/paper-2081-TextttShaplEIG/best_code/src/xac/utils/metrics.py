import math

import gpytorch
import torch
from torch.nn.functional import binary_cross_entropy


def compute_mae(prop_post, prop_gt):
    if prop_post.covariance_matrix.ndim == 3:
        return torch.abs(
            prop_post.mean.mean(dim=0) - prop_gt.squeeze()
            if prop_gt.ndim == 2
            else prop_gt
        ).mean()

    else:
        return torch.abs(
            prop_post.mean - prop_gt.squeeze() if prop_gt.ndim == 2 else prop_gt
        ).mean()

def compute_mse(prop_post, prop_gt, normalized: bool =False):
    if prop_post.covariance_matrix.ndim == 3:
        if normalized:
            raise NotImplementedError("Normalized MSE is not implemented for Gaussian mixture property posteriors.")

        return torch.square(
            prop_post.mean.mean(dim=0) - prop_gt.squeeze()
            if prop_gt.ndim == 2
            else prop_gt
        ).mean()

    else:
        if normalized:
            _mean= prop_post.mean / prop_gt.sum(axis=0)
            _gt= prop_gt.squeeze() / prop_gt.sum(axis=0) if prop_gt.ndim == 2 else prop_gt / prop_gt.sum(axis=0)
        
        else:
            _mean= prop_post.mean
            _gt= prop_gt.squeeze() if prop_gt.ndim == 2 else prop_gt

        return torch.square(
            _mean - _gt
        ).mean()


def compute_nlpd(prop_post, prop_gt):
    # Each dim of property posterior (M) is seen as one test observation
    if prop_post.covariance_matrix.ndim == 3:
        # NLPD for Gaussian mixture
        log_prob = -gpytorch.metrics.negative_log_predictive_density(
            prop_post, prop_gt.squeeze(dim=1) if prop_gt.ndim == 2 else prop_gt
        )
        return -(
            torch.logsumexp(log_prob, dim=0) - math.log(log_prob.numel())
        )  # Apply log-sum exp trick

    else:
        # Vanilla NLPD
        return gpytorch.metrics.negative_log_predictive_density(
            prop_post, prop_gt.squeeze(dim=1) if prop_gt.ndim == 2 else prop_gt
        )
    

def compute_msll(test_post, test_y, train_y):
    if test_post.covariance_matrix.ndim == 3:
        raise NotImplementedError(
            "MSLL is not implemented for Gaussian mixture property posteriors."
        )
    else:
        return gpytorch.metrics.mean_standardized_log_loss(
            test_post, 
            test_y.squeeze(dim=1) if test_y.ndim == 2 else test_y, 
            train_y.squeeze(dim=1) if train_y.ndim == 2 else train_y
        )
    
def compute_calibration(test_post, test_y, calibration_level):
    if test_post[1].ndim == 3:
        #test_post.covariance_matrix.ndim == 3:
        raise NotImplementedError(
            "Calibration is not implemented for Gaussian mixture property posteriors."
        )
    else:
        #Compute the calibration given a calibration level (e.g. 0.95)
        # test_post_mean = test_post.mean #Shape: (n_test)
        # test_post_var= test_post.covariance_matrix.diag().clamp(min=1e-6) #Shape: (n_test)
        #New version, tuple and not MVN anymore
        test_post_mean = test_post[0] #Shape: (n_test)
        test_post_var= test_post[1].clamp(min=1e-6) #Shape: (n_test)

        # Compute the quantile corresponding to the desired confidence level
        quantile = 0.5 * (1 + calibration_level)
        z_value = torch.distributions.Normal(0, 1).icdf(torch.tensor([quantile]))

        # Compute the predictive intervals
        lower_bound = test_post_mean - torch.sqrt(test_post_var) * torch.tensor(z_value)
        upper_bound = test_post_mean + torch.sqrt(test_post_var) * torch.tensor(z_value)

        # Check if the true values fall within the predictive intervals
        within_interval = (test_y.squeeze() >= lower_bound) & (test_y.squeeze() <= upper_bound)

        #Compute the calibration as the proportion of true values within the intervals        
        calibration = within_interval.double().mean()

        return calibration


def compute_ce_loss(predicted_prob, prop_gt_binary):
    return binary_cross_entropy(
        input=predicted_prob,
        target=prop_gt_binary,
        reduction="none",
    )


def compute_accuracy(predicted_prob, prop_gt_binary):
    return ((predicted_prob > 0.5).double() == prop_gt_binary).double()
