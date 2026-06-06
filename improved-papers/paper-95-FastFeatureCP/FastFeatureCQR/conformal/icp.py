from __future__ import division
from collections import defaultdict
from functools import partial

import abc
import ipdb
from tqdm import tqdm
import numpy as np
import sklearn.base
from sklearn.base import BaseEstimator
import torch
import torch.nn.functional as F
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

from .utils import compute_coverage, WeightedMSE, mse, default_loss


class RegressionErrFunc(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(RegressionErrFunc, self).__init__()

    @abc.abstractmethod
    def apply(self, prediction, y):  
        pass

    @abc.abstractmethod
    def apply_inverse(self, nc, significance):  # , norm=None, beta=0):
        pass


class AbsErrorErrFunc(RegressionErrFunc):
    def __init__(self):
        super(AbsErrorErrFunc, self).__init__()

    def apply(self, prediction, y):
        err = np.abs(prediction - y)
        if err.ndim > 1:
            err = np.linalg.norm(err, ord=np.inf, axis=1)
        return err

    def apply_inverse(self, nc, significance):
        nc = np.sort(nc)[::-1]
        border = int(np.floor(significance * (nc.size + 1))) - 1
        # TODO: should probably warn against too few calibration examples
        border = min(max(border, 0), nc.size - 1)
        return np.vstack([nc[border], nc[border]])


# CQR symmetric error function
class QuantileRegErrFunc(RegressionErrFunc):
    """Calculates conformalized quantile regression error.

    For each correct output in ``y``, nonconformity is defined as

    .. math::
        max{\hat{q}_low - y, y - \hat{q}_high}

    """

    def __init__(self):
        super(QuantileRegErrFunc, self).__init__()

    def apply(self, prediction, y):
        y_lower = prediction[:, 0]
        y_upper = prediction[:, -1]
        error_low = y_lower - y
        error_high = y - y_upper
        err = np.maximum(error_high, error_low)
        return err

    def apply_inverse(self, nc, significance):
        nc = np.sort(nc, 0)
        index = int(np.ceil((1 - significance) * (nc.shape[0] + 1))) - 1
        index = min(max(index, 0), nc.shape[0] - 1)
        return np.vstack([nc[index], nc[index]])


class FeatQuantileErrFunc(RegressionErrFunc):
    def __init__(self, feat_norm):
        super(FeatQuantileErrFunc, self).__init__()
        self.feat_norm = feat_norm

    def apply(self, prediction, z):
        # return np.mean(prediction - z, axis=1)
        ret = (prediction - z).norm(p=self.feat_norm, dim=1)  #.mean(dim=0)

        return ret

    def apply_inverse(self, nc, significance):
        nc = np.sort(nc)[::-1]
        border = int(np.floor(significance * (nc.size + 1))) - 1
        border = min(max(border, 0), nc.size - 1)
        return np.vstack([nc[border], nc[border]]) 


class BaseScorer(sklearn.base.BaseEstimator):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(BaseScorer, self).__init__()

    @abc.abstractmethod
    def fit(self, x, y):
        pass

    @abc.abstractmethod
    def score(self, x, y=None, if_compute_cubic=False):
        pass

    @abc.abstractmethod
    def score_batch(self, dataloader):
        pass


class BaseModelNc(BaseScorer):
    def __init__(self, model, err_func, normalizer=None, beta=1e-6):
        super(BaseModelNc, self).__init__()
        self.err_func = err_func
        self.model = model
        self.normalizer = normalizer
        self.beta = beta

        # If we use sklearn.base.clone (e.g., during cross-validation),
        # object references get jumbled, so we need to make sure that the
        # normalizer has a reference to the proper model adapter, if applicable.
        if (self.normalizer is not None and hasattr(self.normalizer, 'base_model')):
            self.normalizer.base_model = self.model

        self.last_x, self.last_y = None, None
        self.last_prediction = None
        self.clean = False

    def fit(self, x, y):
        self.model.fit(x, y)
        if self.normalizer is not None:
            self.normalizer.fit(x, y)
        self.clean = False

    def score(self, x, y=None, if_compute_cubic=False):
        n_test = x.shape[0]
        prediction = self.model.predict(x)
        if self.normalizer is not None:
            norm = self.normalizer.score(x) + self.beta
        else:
            norm = np.ones(n_test)

        if prediction.ndim > 1:
            ret_val = self.err_func.apply(prediction, y)
        else:
            ret_val = self.err_func.apply(prediction, y) / norm

        return ret_val

    def score_batch(self, dataloader):
        ret_val = []
        for x, _, y in tqdm(dataloader):
            prediction = self.model.predict(x)
            if self.normalizer is not None:
                norm = self.normalizer.score(x) + self.beta
            else:
                norm = np.ones(len(x))

            if prediction.ndim > 1:
                batch_ret_val = self.err_func.apply(prediction, y.detach().cpu().numpy())
            else:
                batch_ret_val = self.err_func.apply(prediction, y.detach().cpu().numpy()) / norm
            ret_val.append(batch_ret_val)
        ret_val = np.concatenate(ret_val, axis=0)
        return ret_val


class RegressorNc(BaseModelNc):
    def __init__(self, model, err_func=AbsErrorErrFunc(), normalizer=None, is_cqr=False, beta=1e-6):
        super(RegressorNc, self).__init__(model, err_func, normalizer, beta)
        self.is_cqr = is_cqr

    def predict(self, x, nc, significance=None):
        n_test = x.shape[0]
        prediction = self.model.predict(x)
        if self.normalizer is not None:
            norm = self.normalizer.score(x) + self.beta
        else:
            norm = np.ones(n_test)

        if significance:
            if self.is_cqr:
                intervals = np.zeros((x.shape[0], 1, 2))
            else:
                intervals = np.zeros((x.shape[0], self.model.model.out_shape, 2))
            err_dist = self.err_func.apply_inverse(nc, significance)  # (2, y_dim)
            err_dist = np.stack([err_dist] * n_test)  # (B, 2, y_dim)
            if prediction.ndim > 1 and self.is_cqr:  # CQR
                # print(err_dist[:, 0].shape)
                intervals[..., 0] = prediction[:, 0, None] - err_dist[:, 0]
                intervals[..., 1] = prediction[:, 1, None] + err_dist[:, 1]
            elif prediction.ndim > 1:
                intervals[..., 0] = prediction - err_dist[:, 0]
                intervals[..., 1] = prediction + err_dist[:, 1]
            else:  # regular conformal prediction
                err_dist *= norm[:, None, None]
                intervals[..., 0] = prediction[:, None] - err_dist[:, 0]
                intervals[..., 1] = prediction[:, None] + err_dist[:, 1]

            return intervals
        else:  # Not tested for CQR
            significance = np.arange(0.01, 1.0, 0.01)
            intervals = np.zeros((x.shape[0], 2, significance.size))

            for i, s in enumerate(significance):
                err_dist = self.err_func.apply_inverse(nc, s)
                err_dist = np.hstack([err_dist] * n_test)
                err_dist *= norm

                intervals[:, 0, i] = prediction - err_dist[0, :]
                intervals[:, 1, i] = prediction + err_dist[0, :]

            return intervals


class FeatErrorErrFunc(RegressionErrFunc):
    def __init__(self, feat_norm):
        super(FeatErrorErrFunc, self).__init__()
        self.feat_norm = feat_norm

    def apply(self, prediction, z):
        # return np.mean(prediction - z, axis=1)
        ret = (prediction - z).norm(p=self.feat_norm, dim=1)  #.mean(dim=0)
        return ret

    def apply_inverse(self, nc, significance):
        nc = np.sort(nc)[::-1]
        border = int(np.floor(significance * (nc.size + 1))) - 1
        border = min(max(border, 0), nc.size - 1)
        return np.vstack([nc[border], nc[border]]) 


class FeatRegressorNc(BaseModelNc):
    def __init__(self, model,
                 inv_lr, inv_step, criterion=default_loss, feat_norm=np.inf, certification_method=0, cert_optimizer='sgd',
                 normalizer=None, beta=1e-6, g_out_process=None):
        if feat_norm in ["inf", np.inf, float('inf')]:
            self.feat_norm = np.inf
        elif (type(feat_norm) == int or float):
            self.feat_norm = feat_norm
        else:
            raise NotImplementedError
        err_func = FeatErrorErrFunc(feat_norm=self.feat_norm)

        super(FeatRegressorNc, self).__init__(model, err_func, normalizer, beta)
        self.criterion = criterion
        self.inv_lr = inv_lr
        self.inv_step = inv_step
        self.certification_method = certification_method
        self.cmethod = ['IBP', 'IBP+backward', 'backward', 'CROWN-Optimized'][self.certification_method]
        print(f"Use {self.cmethod} method for certification")

        self.cert_optimizer = cert_optimizer
        # the function to post process the output of g, because FCN needs interpolate and reshape
        self.g_out_process = g_out_process
        self.feature_cubic = None
        self.output_cubic = None

    def inv_g(self, z0, y, step=None, record_each_step=False):

        z = z0.detach().clone()
        z = z.detach()
        z.requires_grad_()
        if self.cert_optimizer == "sgd":
            optimizer = torch.optim.SGD([z], lr=self.inv_lr)
        elif self.cert_optimizer == "adam":
            optimizer = torch.optim.Adam([z], lr=self.inv_lr)

        self.model.model.eval()
        each_step_z = []
        for _ in range(step):
            pred = self.model.model.g(z)
            if self.g_out_process is not None:
                pred = self.g_out_process(pred)

            loss = self.criterion(pred.squeeze(), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if record_each_step:
                each_step_z.append(z.detach().cpu().clone())

        if record_each_step:
            return each_step_z
        else:
            return z.detach().cpu()

    def get_each_step_err_dist(self, x, y, z_pred, steps):
        each_step_z_true = self.inv_g(z_pred, y, step=steps, record_each_step=True)

        if self.normalizer is not None:
            raise NotImplementedError
        else:
            norm = np.ones(len(x))

        err_dist_list = []
        for i, step_z_true in enumerate(each_step_z_true):
            err_dist = self.err_func.apply(z_pred.detach().cpu(), step_z_true.detach().cpu()).numpy() / norm
            err_dist_list.append(err_dist)
        return err_dist_list

    def coverage_loose(self, x, y, z_pred, steps, val_significance):
        z_pred_detach = z_pred.detach().clone()

        idx = torch.randperm(len(z_pred_detach))
        n_val = int(np.floor(len(z_pred_detach) / 5))
        val_idx, cal_idx = idx[:n_val], idx[n_val:]

        cal_x, val_x = x[cal_idx], x[val_idx]
        cal_y, val_y = y[cal_idx], y[val_idx]
        cal_z_pred, val_z_pred = z_pred_detach[cal_idx], z_pred_detach[val_idx]

        cal_score_list = self.get_each_step_err_dist(cal_x, cal_y, cal_z_pred, steps=steps)

        val_coverage_list = []
        for i, step_cal_score in enumerate(cal_score_list):
            val_predictions = self.predict(x=val_x.detach().cpu().numpy(), nc=step_cal_score,
                                           significance=val_significance)
            val_y_lower, val_y_upper = val_predictions[..., 0], val_predictions[..., 1]
            val_coverage, _, _ = compute_coverage(val_y.detach().cpu().numpy(), val_y_lower, val_y_upper, val_significance,
                                               name="{}-th step's validation".format(i))
            val_coverage_list.append(val_coverage)
        return val_coverage_list, len(val_x)

    def coverage_tight(self, x, y, z_pred,  steps, val_significance):
        z_pred_detach = z_pred.detach().clone()

        idx = torch.randperm(len(z_pred_detach))
        n_val = int(np.floor(len(z_pred_detach) / 5))
        val_idx, cal_idx = idx[:n_val], idx[n_val:]

        cal_x, val_x = x[cal_idx], x[val_idx]
        cal_y, val_y = y[cal_idx], y[val_idx]
        cal_z_pred, val_z_pred = z_pred_detach[cal_idx], z_pred_detach[val_idx]

        cal_score_list = self.get_each_step_err_dist(cal_x, cal_y, cal_z_pred, steps=steps)
        val_score_list = self.get_each_step_err_dist(val_x, val_y, val_z_pred, steps=steps)

        val_coverage_list = []
        for i, (cal_score, val_score) in enumerate(zip(cal_score_list, val_score_list)):
            err_dist_threshold = self.err_func.apply_inverse(nc=cal_score, significance=val_significance)[0][0]
            val_coverage = np.sum(val_score < err_dist_threshold) * 100 / len(val_score)
            val_coverage_list.append(val_coverage)
        return val_coverage_list, len(val_x)

    def find_best_step_num(self, x, y, z_pred):
        max_inv_steps = 200
        val_significance = 0.1

        each_step_val_coverage, val_num = self.coverage_loose(x, y, z_pred, steps=max_inv_steps, val_significance=val_significance)

        tolerance = 1
        count = 0
        final_coverage, best_step = None, None
        for i, val_coverage in enumerate(each_step_val_coverage):
            print("{}-th step's validation coverage is {}".format(i, val_coverage))
            if val_coverage > (1 - val_significance) * 100 and final_coverage is None:
                count += 1
                if count == tolerance:
                    final_coverage = val_coverage
                    best_step = i
            elif val_coverage <= (1 - val_significance) * 100 and count > 0:
                count = 0

        if final_coverage is None or best_step is None:
            raise ValueError(
                "does not find a good step to make the coverage higher than {}".format(1 - val_significance))
        print("The best inv_step is {}, which gets {} coverage on val set".format(best_step + 1, final_coverage))
        return best_step + 1

    def find_best_step_num_batch(self, dataloader):
        max_inv_steps = 200
        val_significance = 0.1

        accumulate_val_coverage = np.zeros(max_inv_steps)
        accumulate_val_num = 0
        print("begin to find the best step number")
        for x, _, y in tqdm(dataloader):
            x, y = x.to(self.model.device), y.to(self.model.device)
            z_pred = self.model.model.encoder(x)
            batch_each_step_val_coverage, val_num = self.coverage_tight(x, y, z_pred, steps=max_inv_steps, val_significance=val_significance)  # length: max_inv_steps
            accumulate_val_coverage += np.array(batch_each_step_val_coverage) * val_num
            accumulate_val_num += val_num

        each_step_val_coverage = accumulate_val_coverage / accumulate_val_num

        tolerance = 3
        count = 0
        final_coverage, best_step = None, None
        for i, val_coverage in enumerate(each_step_val_coverage):
            print("{}-th step's validation tight coverage is {}".format(i, val_coverage))
            if val_coverage > (1 - val_significance) * 100 and final_coverage is None:
                count += 1
                if count == tolerance:
                    final_coverage = val_coverage
                    best_step = i
            elif val_coverage <= (1 - val_significance) * 100 and count > 0:
                count = 0

        if final_coverage is None or best_step is None:
            raise ValueError(
                "does not find a good step to make the coverage higher than {}".format(1 - val_significance))
        print("The best inv_step is {}, which gets {} coverage on val set".format(best_step + 1, final_coverage))
        return best_step + 1

    def score(self, x, y=None, if_compute_cubic=False):  # overwrite BaseModelNc.score()
        self.model.model.eval()
        n_test = x.shape[0]
        x, y = torch.from_numpy(x).to(self.model.device), torch.from_numpy(y).to(self.model.device)
        z_pred = self.model.model.encoder(x)

        if self.inv_step is None:
            self.inv_step = self.find_best_step_num(x, y, z_pred)

        z_true = self.inv_g(z_pred, y, step=self.inv_step)

        if self.normalizer is not None:
            raise NotImplementedError
        else:
            norm = np.ones(n_test)

        ret_val = self.err_func.apply(z_pred.detach().cpu(), z_true.detach().cpu())  # || z_pred - z_true ||
        ret_val = ret_val.numpy() / norm
        if if_compute_cubic:
            self.compute_cubic(x, y, z_pred.detach(), ret_val)
        return ret_val

    def compute_cubic(self, x, y, z_pred, scores):
        lipschiz = (self.model.model.g[0].weight.norm() * self.model.model.g[3].weight.norm()).cpu().item()

        z_quantile = self.err_func.apply_inverse(scores, significance=0.9)
        lengths = torch.zeros(len(scores)).to(self.model.device)
        for i, (z_pred_i, score_i) in enumerate(zip(z_pred, scores)):
            lirpa_model = BoundedModule(self.model.model.g, torch.empty_like(z_pred_i))
            ptb = PerturbationLpNorm(norm=self.feat_norm, eps=score_i)  # feat_err_dist=[[0.122, 0.122]]
            my_input = BoundedTensor(z_pred_i, ptb)

            # cmethod = "CROWN-IBP"
            if 'Optimized' in self.cmethod:
                lirpa_model.set_bound_opts(
                    {'optimize_bound_args': {'ob_iteration': 20, 'ob_lr': 0.1, 'ob_verbose': 0}})
            lb, ub = lirpa_model.compute_bounds(x=(my_input,), method=self.cmethod)  # (bs, 1), (bs, 1)
            lengths[i] = ub - lb

        length_quantile = self.err_func.apply_inverse(lengths.detach().cpu().numpy(), significance=0.9)

        feature_cubic = lipschiz * np.mean(np.abs(scores - z_quantile[0][0]))
        output_cubic = np.mean(lengths.detach().cpu().numpy() - length_quantile[0][0])
        # print("z_metric", z_metric)
        # print("length_metric", length_metric)
        self.feature_cubic = feature_cubic
        self.output_cubic = output_cubic

    def score_batch(self, dataloader):
        self.model.model.eval()
        if self.inv_step is None:
            self.inv_step = self.find_best_step_num_batch(dataloader)

        print('calculating score:')
        ret_val = []
        for x, _, y in tqdm(dataloader):
            x, y = x.to(self.model.device), y.to(self.model.device)

            if self.normalizer is not None:
                raise NotImplementedError
            else:
                norm = np.ones(len(x))

            z_pred = self.model.model.encoder(x)
            z_true = self.inv_g(z_pred, y, step=self.inv_step)
            batch_ret_val = self.err_func.apply(z_pred.detach().cpu(), z_true.detach().cpu())
            batch_ret_val = batch_ret_val.detach().cpu().numpy() / norm
            ret_val.append(batch_ret_val)
        ret_val = np.concatenate(ret_val, axis=0)
        return ret_val

    def predict(self, x, nc, significance=None):
        n_test = x.shape[0]
        prediction = self.model.predict(x)

        if self.normalizer is not None:
            norm = self.normalizer.score(x) + self.beta
        else:
            norm = np.ones(n_test)

        if significance:
            intervals = np.zeros((x.shape[0], self.model.model.out_shape, 2))
            feat_err_dist = self.err_func.apply_inverse(nc, significance) 

            if prediction.ndim > 1:  # CQR
                if isinstance(x, torch.Tensor):
                    x = x.to(self.model.device)
                else:
                    x = torch.from_numpy(x).to(self.model.device)
                z = self.model.model.encoder(x).detach()

                # certification
                lirpa_model = BoundedModule(self.model.model.g, torch.empty_like(z))
                ptb = PerturbationLpNorm(norm=self.feat_norm, eps=feat_err_dist[0][0])
                my_input = BoundedTensor(z, ptb)

                if 'Optimized' in self.cmethod:
                    lirpa_model.set_bound_opts(
                        {'optimize_bound_args': {'ob_iteration': 20, 'ob_lr': 0.1, 'ob_verbose': 0}})
                lb, ub = lirpa_model.compute_bounds(x=(my_input,), method=self.cmethod)
                if self.g_out_process is not None:
                    lb = self.g_out_process(lb)
                    ub = self.g_out_process(ub)
                lb, ub = lb.detach().cpu().numpy(), ub.detach().cpu().numpy()

                intervals[..., 0] = lb
                intervals[..., 1] = ub
            else:  # regular conformal prediction
                # err_dist *= norm
                if not isinstance(x, torch.Tensor):
                    x = torch.from_numpy(x).to(self.model.device)
                z = self.model.model.encoder(x).detach()

                # certification
                lirpa_model = BoundedModule(self.model.model.g, torch.empty_like(z))
                ptb = PerturbationLpNorm(norm=self.feat_norm, eps=feat_err_dist[0][0]) 
                my_input = BoundedTensor(z, ptb)

                if 'Optimized' in self.cmethod:
                    lirpa_model.set_bound_opts({'optimize_bound_args': {'ob_iteration': 20, 'ob_lr': 0.1, 'ob_verbose': 0}})
                lb, ub = lirpa_model.compute_bounds(x=(my_input,), method=self.cmethod)  # (bs, 1), (bs, 1)
                if self.g_out_process is not None:
                    lb = self.g_out_process(lb)
                    ub = self.g_out_process(ub)
                lb, ub = lb.detach().cpu().numpy(), ub.detach().cpu().numpy()

                intervals[..., 0] = lb
                intervals[..., 1] = ub

            return intervals

        else:  # Not tested for CQR
            raise NotImplementedError

            significance = np.arange(0.01, 1.0, 0.01)
            intervals = np.zeros((x.shape[0], 2, significance.size))

            for i, s in enumerate(significance):
                err_dist = self.err_func.apply_inverse(nc, s)
                err_dist = np.hstack([err_dist] * n_test)
                err_dist *= norm

                intervals[:, 0, i] = prediction - err_dist[0, :]
                intervals[:, 1, i] = prediction + err_dist[0, :]

            return intervals


class CqrFeatRegressorNc(FeatRegressorNc):
    def __init__(self, model,
                 # err_func=FeatErrorErrFunc(),
                 inv_lr, inv_step, criterion=default_loss, feat_norm=np.inf, certification_method=0, cert_optimizer='sgd',
                 normalizer=None, beta=1e-6, g_out_process=None):
        super(CqrFeatRegressorNc, self).__init__(model, inv_lr, inv_step, criterion, feat_norm, certification_method,
                                                 cert_optimizer, normalizer, beta, g_out_process)

    def inv_g(self, z0, y, target_dim, step=None, record_each_step=False):
        z = z0.detach().clone()
        z = z.detach()
        z.requires_grad_()
        if self.cert_optimizer == "sgd":
            optimizer = torch.optim.SGD([z], lr=self.inv_lr)
        elif self.cert_optimizer == "adam":
            optimizer = torch.optim.Adam([z], lr=self.inv_lr)

        self.model.model.eval()
        each_step_z = []
        for _ in range(step):
            pred = self.model.model.g(z)
            if self.g_out_process is not None:
                pred = self.g_out_process(pred)

            loss = self.criterion(pred.squeeze()[:, target_dim], y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if record_each_step:
                each_step_z.append(z.detach().cpu().clone())

        if record_each_step:
            return each_step_z
        else:
            return z.detach().cpu()

    def score(self, x, y=None, if_compute_metric=True):
        self.model.model.eval()
        n_test = x.shape[0]
        x, y = torch.from_numpy(x).to(self.model.device), torch.from_numpy(y).to(self.model.device)
        z_pred = self.model.model.encoder(x)
        y_pred = self.model.model.g(z_pred)

        lower_z_true = self.inv_g(z_pred, y, target_dim=0, step=self.inv_step)
        upper_z_true = self.inv_g(z_pred, y, target_dim=1, step=self.inv_step)


        if self.normalizer is not None:
            raise NotImplementedError
        else:
            norm = np.ones(n_test)

        lower_flag = (y_pred[:, 0] < y).cpu().numpy().astype(np.float64) * 2 - 1
        upper_flag = (y_pred[:, 1] > y).cpu().numpy().astype(np.float64) * 2 - 1

        lower_ret_val = self.err_func.apply(z_pred.detach().cpu(), lower_z_true.detach().cpu())
        lower_ret_val = (lower_ret_val.numpy() / norm) * lower_flag

        upper_ret_val = self.err_func.apply(z_pred.detach().cpu(), upper_z_true.detach().cpu())
        upper_ret_val = (upper_ret_val.numpy() / norm) * upper_flag
        return lower_ret_val, upper_ret_val

    def get_lirpa_estimation(self, z, ptb_eps):
        # certification
        lirpa_model = BoundedModule(self.model.model.g, torch.empty_like(z))
        ptb = PerturbationLpNorm(norm=self.feat_norm, eps=ptb_eps)
        my_input = BoundedTensor(z, ptb)

        if 'Optimized' in self.cmethod:
            lirpa_model.set_bound_opts(
                {'optimize_bound_args': {'ob_iteration': 20, 'ob_lr': 0.1, 'ob_verbose': 0}})
        lb, ub = lirpa_model.compute_bounds(x=(my_input,), method=self.cmethod)  # (bs, 1), (bs, 1)
        if self.g_out_process is not None:
            lb = self.g_out_process(lb)
            ub = self.g_out_process(ub)
        lb, ub = lb.detach().cpu().numpy(), ub.detach().cpu().numpy()
        return lb, ub

    def predict(self, x, lower_nc, upper_nc, significance=None):
        n_test = x.shape[0]
        prediction = self.model.predict(x)

        if self.normalizer is not None:
            norm = self.normalizer.score(x) + self.beta
        else:
            norm = np.ones(n_test)

        intervals = np.zeros((x.shape[0], 1, 2))
        lower_feat_err_dist = self.err_func.apply_inverse(lower_nc, significance / 2) 
        upper_feat_err_dist = self.err_func.apply_inverse(upper_nc, significance / 2)

        if isinstance(x, torch.Tensor):
            x = x.to(self.model.device)
        else:
            x = torch.from_numpy(x).to(self.model.device)
        z = self.model.model.encoder(x).detach()

        lower_lb, lower_ub = self.get_lirpa_estimation(z, abs(lower_feat_err_dist[0][0]))
        upper_lb, upper_ub = self.get_lirpa_estimation(z, abs(upper_feat_err_dist[0][0]))

        if lower_feat_err_dist[0][0] < 0:
            intervals[..., 0, 0] = lower_lb[:, 0]
        else:
            intervals[..., 0, 0] = lower_ub[:, 0]

        if upper_feat_err_dist[0][0] > 0:
            intervals[..., 0, 1] = upper_lb[:, 1]
        else:
            intervals[..., 0, 1] = upper_ub[:, 1]
        return intervals

class CQR_FFCPErrorErrFunc(RegressionErrFunc):
    def __init__(self):
        super(CQR_FFCPErrorErrFunc, self).__init__()

    def apply_grad(self,prediction, y, grad):

        y_lo = prediction[:, 0]
        y_up = prediction[:, -1]
        grad_lo = grad[:, :, 0]
        grad_up = grad[:, :, -1]

        error_lo = y_lo - y
        error_up = y - y_up

        grad_lo = np.linalg.norm(grad_lo, axis=1)
        grad_up = np.linalg.norm(grad_up, axis=1)
        grad_lo += 1e-6
        grad_up += 1e-6
        error_lo = error_lo / grad_lo
        error_up = error_up / grad_up

        err = np.maximum(error_up, error_lo)

        return err

    def apply_inverse_grad(self, nc, significance):
        nc = np.sort(nc, 0)
        index = int(np.ceil((1 - significance) * (nc.shape[0] + 1))) - 1
        index = min(max(index, 0), nc.shape[0] - 1)
        return np.vstack([nc[index], nc[index]])


class CQR_FFCP_RegressorNc(BaseModelNc):
    def __init__(self, model, err_func=CQR_FFCPErrorErrFunc(), normalizer=None, beta=1e-6):
        super(CQR_FFCP_RegressorNc, self).__init__(model, err_func, normalizer, beta)
        err_func = CQR_FFCPErrorErrFunc()
        self.is_cqr = 1


    def predict(self, model, x_cal, y_cal, x_test, y_test, layer_index, significance=None):
        n_test = x_test.shape[0]

        prediction_cal, grads_intermediate_cal_lo, _ = model.predict_with_grads(x_cal, layer_index, 0)
        prediction_cal, grads_intermediate_cal_up, _ = model.predict_with_grads(x_cal, layer_index, 1)
        
        grads_intermediate_cal = np.stack((grads_intermediate_cal_lo, grads_intermediate_cal_up), axis=2)

        nc = self.err_func.apply_grad(prediction_cal, y_cal, grads_intermediate_cal)

        prediction, grads_intermediate_lo, _ = model.predict_with_grads(x_test, layer_index, 0)
        prediction, grads_intermediate_up, _ = model.predict_with_grads(x_test, layer_index, 1)
        pre_dim = prediction

        grad_norms_lo= np.array([np.linalg.norm(g) for g in grads_intermediate_lo])
        grad_norms_up= np.array([np.linalg.norm(g) for g in grads_intermediate_up])

        norm = np.ones(n_test)
        grad_norms_lo = grad_norms_lo.reshape(-1, 1)
        grad_norms_up = grad_norms_up.reshape(-1, 1)

        grad_norms_lo += 1e-6
        grad_norms_up += 1e-6

        grad_norms_shape = grad_norms_up

        if significance:
            if self.is_cqr:
                intervals = np.zeros((grad_norms_shape.shape[0], 1, 2))
            else:
                intervals = np.zeros((grad_norms_shape.shape[0], self.model.model.out_shape, 2))

            err_dist = self.err_func.apply_inverse_grad(nc, significance)  # (2, y_dim)
            err_dist = np.stack([err_dist] * n_test)  # (B, 2, y_dim)

            if pre_dim.ndim > 1 and self.is_cqr:  # CQR
                intervals[..., 0] = prediction[:, 0, None] - err_dist[:, 0] * grad_norms_lo
                intervals[..., 1] = prediction[:, 1, None] + err_dist[:, 1] * grad_norms_up
            elif pre_dim.ndim > 1:
                intervals[..., 0] = prediction - err_dist[:, 0] * grad_norms_lo
                intervals[..., 1] = prediction + err_dist[:, 1] * grad_norms_up
            else:  # regular conformal prediction
                err_dist *= norm[:, None, None]
                intervals[..., 0] = prediction[:, None] - err_dist[:, 0] * grad_norms_lo
                intervals[..., 1] = prediction[:, None] + err_dist[:, 1] * grad_norms_up

            return intervals
        else:  # Not tested for CQR

            significance = np.arange(0.01, 1.0, 0.01)
            intervals = np.zeros((grad_norms_shape.shape[0], 2, significance.size))

            for i, s in enumerate(significance):
                err_dist = self.err_func.apply_inverse(nc, s)
                err_dist = np.hstack([err_dist] * n_test)
                err_dist *= norm

                intervals[:, 0, i] = prediction - err_dist[0, :] * grad_norms_lo
                intervals[:, 1, i] = prediction + err_dist[0, :] * grad_norms_up

            return intervals

class BaseIcp(BaseEstimator):
    def __init__(self, nc_function, condition=None):
        self.cal_x, self.cal_y = None, None
        self.nc_function = nc_function

        # Check if condition-parameter is the default function (i.e.,
        # lambda x: 0). This is so we can safely clone the object without
        # the clone accidentally having self.conditional = True.
        default_condition = lambda x: 0
        is_default = (callable(condition) and
                      (condition.__code__.co_code ==
                       default_condition.__code__.co_code))

        if is_default:
            self.condition = condition
            self.conditional = False
        elif callable(condition):
            self.condition = condition
            self.conditional = True
        else:
            self.condition = lambda x: 0
            self.conditional = False

    @classmethod
    def get_problem_type(cls):
        return 'regression'

    def fit(self, x, y):
        self.nc_function.fit(x, y)

    def calibrate(self, x, y, increment=False, if_compute_cubic=False):
        self._calibrate_hook(x, y, increment)
        self._update_calibration_set(x, y, increment)

        if self.conditional:
            category_map = np.array([self.condition((x[i, :], y[i])) for i in range(y.size)])
            self.categories = np.unique(category_map)
            # defaultdict never raises a KeyError. It provides a default value for the key that does not exists.
            self.cal_scores = defaultdict(partial(np.ndarray, 0))

            for cond in self.categories:
                idx = category_map == cond
                cal_scores = self.nc_function.score(self.cal_x[idx, :], self.cal_y[idx], if_compute_cubic=if_compute_cubic)
                self.cal_scores[cond] = np.sort(cal_scores, 0)[::-1]
        else:
            self.categories = np.array([0])
            cal_scores = self.nc_function.score(self.cal_x, self.cal_y, if_compute_cubic=if_compute_cubic)
            self.cal_scores = {0: np.sort(cal_scores, 0)[::-1]}

    def calibrate_batch(self, dataloader):
        if self.conditional:
            raise NotImplementedError

        else:
            self.categories = np.array([0])
            cal_scores = self.nc_function.score_batch(dataloader)
            self.cal_scores = {0: np.sort(cal_scores, 0)[::-1]}

    def _calibrate_hook(self, x, y, increment):
        pass

    def _update_calibration_set(self, x, y, increment):
        if increment and self.cal_x is not None and self.cal_y is not None:
            self.cal_x = np.vstack([self.cal_x, x])
            self.cal_y = np.hstack([self.cal_y, y])
        else:
            self.cal_x, self.cal_y = x, y


class IcpRegressor(BaseIcp):
    def __init__(self, nc_function, condition=None):
        super(IcpRegressor, self).__init__(nc_function, condition)

    def predict(self, x, significance=None):
        self.nc_function.model.model.eval()

        n_significance = (99 if significance is None
                          else np.array(significance).size)

        if n_significance > 1:
            prediction = np.zeros((x.shape[0], self.nc_function.model.model.out_shape, 2, n_significance))
        else:
            prediction = np.zeros((x.shape[0], self.nc_function.model.model.out_shape, 2))

        condition_map = np.array([self.condition((x[i, :], None))
                                  for i in range(x.shape[0])])

        for condition in self.categories:
            idx = condition_map == condition
            if np.sum(idx) > 0:
                p = self.nc_function.predict(x[idx, :], self.cal_scores[condition], significance)  # return an interval
                if n_significance > 1:
                    prediction[idx, :, :] = p
                else:
                    prediction[idx, :] = p

        return prediction

    def if_in_coverage(self, x, y, significance):
        self.nc_function.model.model.eval()
        condition_map = np.array([self.condition((x[i, :], None))
                                  for i in range(x.shape[0])])
        result_array = np.zeros(len(x)).astype(bool)
        for condition in self.categories:
            idx = condition_map == condition
            if np.sum(idx) > 0:
                err_dist = self.nc_function.score(x[idx, :], y[idx])
                err_dist_threshold = self.nc_function.err_func.apply_inverse(self.cal_scores[condition], significance)[0][0]
                result_array[idx] = (err_dist < err_dist_threshold)
        return result_array

    def if_in_coverage_batch(self, dataloader, significance):
        self.nc_function.model.model.eval()
        err_dist = self.nc_function.score_batch(dataloader)
        err_dist_threshold = self.nc_function.err_func.apply_inverse(self.cal_scores[0], significance)[0][0]
        result_array = (err_dist < err_dist_threshold)
        return result_array


class CqrIcpRegressor(IcpRegressor):
    def __init__(self, nc_function, condition=None):
        assert isinstance(nc_function, CqrFeatRegressorNc)
        super(IcpRegressor, self).__init__(nc_function, condition)

    def calibrate(self, x, y, increment=False):
        self._calibrate_hook(x, y, increment)
        self._update_calibration_set(x, y, increment)

        self.categories = np.array([0])
        cal_lower_scores, cal_upper_scores = self.nc_function.score(self.cal_x, self.cal_y)
        self.cal_lower_scores = {0: np.sort(cal_lower_scores, 0)[::-1]}
        self.cal_upper_scores = {0: np.sort(cal_upper_scores, 0)[::-1]}

    def predict(self, x, significance=None):
        self.nc_function.model.model.eval()

        n_significance = (99 if significance is None
                          else np.array(significance).size)

        if n_significance > 1:
            prediction = np.zeros((x.shape[0], 1, 2, n_significance))
        else:
            prediction = np.zeros((x.shape[0], 1, 2))

        condition_map = np.array([self.condition((x[i, :], None))
                                  for i in range(x.shape[0])])

        for condition in self.categories:
            idx = condition_map == condition
            if np.sum(idx) > 0:
                p = self.nc_function.predict(x[idx, :], self.cal_lower_scores[condition], self.cal_upper_scores[condition], significance)  # return an interval
                if n_significance > 1:
                    prediction[idx, :, :] = p
                else:
                    prediction[idx, :] = p

        return prediction

    def if_in_coverage(self, x, y, significance):
        self.nc_function.model.model.eval()
        condition_map = np.array([self.condition((x[i, :], None))
                                  for i in range(x.shape[0])])
        result_array = np.zeros(len(x)).astype(bool)
        for condition in self.categories:
            idx = condition_map == condition
            if np.sum(idx) > 0:
                lower_err_dist, upper_err_dist = self.nc_function.score(x[idx, :], y[idx])
                lower_err_dist_threshold = self.nc_function.err_func.apply_inverse(self.cal_lower_scores[condition], significance / 2)[0][0]
                upper_err_dist_threshold = self.nc_function.err_func.apply_inverse(self.cal_upper_scores[condition], significance / 2)[0][0]
                result_array = (lower_err_dist < lower_err_dist_threshold) & (upper_err_dist < upper_err_dist_threshold)
        return result_array


def calc_p(ncal, ngt, neq, smoothing=False):
    if smoothing:
        return (ngt + (neq + 1) * np.random.uniform(0, 1)) / (ncal + 1)
    else:
        return (ngt + neq + 1) / (ncal + 1)
