from tqdm import tqdm

import torch
import torch.nn as nn
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import gaussian_kde
import numpy as np

from .base_postprocessor import BasePostprocessor


class AdaScalePostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(AdaScalePostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.percentile = self.args.percentile
        self.k1 = self.args.k1
        self.k2 = self.args.k2
        self.lmbda = self.args.lmbda
        self.o = self.args.o
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.setup_flag = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        self.net = net
        if not self.setup_flag:
            feature_log = []
            feature_perturbed_log = []
            feature_shift_log = []
            net.eval()
            self.feature_dim = net.backbone.feature_size
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['val'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data = batch['data'].cuda()
                    data = data.float()
                    with torch.enable_grad():
                        data.requires_grad = True
                        output, feature = net(data, return_feature=True)
                        labels = output.detach().argmax(dim=1)
                        net.zero_grad()
                        score = output[torch.arange(len(labels)), labels]
                        score.backward(torch.ones_like(labels))
                        grad = data.grad.data.detach()
                    feature_log.append(feature.data.cpu())
                    data_perturbed = self.perturb(data, grad)
                    _, feature_perturbed = net(data_perturbed,
                                               return_feature=True)
                    feature_shift = abs(feature - feature_perturbed)
                    feature_shift_log.append(feature_shift.data.cpu())
                    feature_perturbed_log.append(feature_perturbed.data.cpu())
            all_features = torch.cat(feature_log, axis=0)
            all_perturbed = torch.cat(feature_perturbed_log, axis=0)
            all_shifts = torch.cat(feature_shift_log, axis=0)

            total_samples = all_features.size(0)
            num_samples = self.args.num_samples if hasattr(
                self.args, 'num_samples') else total_samples
            indices = torch.randperm(total_samples)[:num_samples]

            self.feature_log = all_features[indices]
            self.feature_perturbed_log = all_perturbed[indices]
            self.feature_shift_log = all_shifts[indices]
            self.setup_flag = True
            # Extract BN running variance for per-channel weighting (ALGO-04)
            try:
                bn_var = net.backbone.layer4[-1].bn3.running_var
                self.bn_weight = 1.0 / torch.sqrt(bn_var + 1e-5)
                self.bn_weight = self.bn_weight / self.bn_weight.mean()
            except Exception:
                self.bn_weight = None
        else:
            pass

    @torch.no_grad()
    def get_percentile(self, feature, feature_perturbed, feature_shift):
        topk_indices = torch.topk(feature, dim=1, k=self.k1_)[1]
        topk_feature_perturbed = torch.gather(
            torch.relu(feature_perturbed), 1,
            topk_indices)  # correction term C_o
        topk_indices = torch.topk(feature, dim=1, k=self.k2_)[1]
        topk_feature_shift = torch.gather(feature_shift, 1, topk_indices)  # Q
        # BN-calibrated channel weighting (ALGO-04)
        if hasattr(self, 'bn_weight') and self.bn_weight is not None:
            bn_w = self.bn_weight.to(feature.device)
            shift_w = bn_w[topk_indices]
            topk_feature_shift_weighted = topk_feature_shift * shift_w
            q_term = topk_feature_shift_weighted.sum(dim=1)
        else:
            q_term = topk_feature_shift.sum(dim=1)
        topk_norm = topk_feature_perturbed.sum(
            dim=1) + self.lmbda * q_term  # Q^{\prime}
        topk_norm_cpu = topk_norm.cpu().numpy()
        if hasattr(self, 'kde') and self.kde is not None:
            percent = 1 - np.interp(topk_norm_cpu, self.kde_cdf_xs, self.kde_cdf_ys)
        else:
            percent = 1 - self.ecdf(topk_norm_cpu)
        percentile = self.min_percentile + percent * (self.max_percentile -
                                                      self.min_percentile)
        return torch.from_numpy(percentile)

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data):
        with torch.enable_grad():
            data.requires_grad = True
            output, feature = net(data, return_feature=True)
            labels = output.detach().argmax(dim=1)
            net.zero_grad()
            score = output[torch.arange(len(labels)), labels]
            score.backward(torch.ones_like(labels))
            grad = data.grad.data.detach()
            data.requires_grad = False
        data_perturbed = self.perturb(data, grad)
        _, feature_perturbed = net(data_perturbed, return_feature=True)
        feature_shift = abs(feature - feature_perturbed)
        percentile = self.get_percentile(feature, feature_perturbed,
                                         feature_shift)
        output = net.forward_threshold(feature, percentile)
        _, pred = torch.max(output, dim=1)
        conf = torch.logsumexp(output, dim=1)
        return pred, conf

    @torch.no_grad()
    def perturb(self, data, grad):
        batch_size, channels, height, width = data.shape
        n_pixels = int(channels * height * width * self.o)
        abs_grad = abs(grad).view(batch_size, channels * height * width)
        _, topk_indices = torch.topk(abs_grad, n_pixels, dim=1, largest=False)
        mask = torch.zeros_like(abs_grad, dtype=torch.uint8)
        mask.scatter_(1, topk_indices, 1)
        mask = mask.view(batch_size, channels, height, width)
        data_ood = data + grad.sign() * mask * 0.5
        return data_ood

    def set_hyperparam(self, hyperparam: list):
        self.percentile = hyperparam[0]
        self.min_percentile, self.max_percentile = self.percentile[
            0], self.percentile[1]
        self.k1 = hyperparam[1]
        self.k2 = hyperparam[2]
        self.lmbda = hyperparam[3]
        self.o = hyperparam[4]
        self.k1_ = int(self.feature_dim * self.k1 / 100)
        self.k2_ = int(self.feature_dim * self.k2 / 100)
        topk_indices = torch.topk(self.feature_log, k=self.k1_, dim=1)[1]
        topk_feature_perturbed = torch.gather(
            torch.relu(self.feature_perturbed_log), 1, topk_indices)
        topk_indices = torch.topk(self.feature_log, k=self.k2_, dim=1)[1]
        topk_feature_shift_log = torch.gather(self.feature_shift_log, 1,
                                              topk_indices)
        sum_log = topk_feature_perturbed.sum(
            dim=1) + self.lmbda * topk_feature_shift_log.sum(dim=1)
        self.ecdf = ECDF(sum_log)
        # KDE-smoothed CDF for smoother percentile mapping
        try:
            self.kde = gaussian_kde(sum_log.numpy())
            # Pre-compute CDF grid for fast interpolation
            xs = np.linspace(sum_log.min(), sum_log.max(), 1000)
            self.kde_cdf_xs = xs
            self.kde_cdf_ys = np.array([self.kde.integrate_box_1d(-np.inf, x) for x in xs])
        except Exception:
            self.kde = None

    def get_hyperparam(self):
        return [self.percentile, self.k1, self.k2, self.lmbda, self.o]
