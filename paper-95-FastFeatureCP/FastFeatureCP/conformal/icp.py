from __future__ import division
from collections import defaultdict
from functools import partial
import matplotlib.pyplot as plt
import abc
from tqdm import tqdm
import numpy as np
import sklearn.base
from sklearn.base import BaseEstimator
import torch
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import pandas as pd
from .utils import compute_coverage, default_loss


class RegressionErrFunc(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(RegressionErrFunc, self).__init__()

    @abc.abstractmethod
    def apply(self, prediction, y):
        pass

    @abc.abstractmethod
    def apply_inverse(self, nc, significance):
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
    def score(self, x, y=None):
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

    def score(self, x, y=None):
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
    def __init__(self, model, err_func=AbsErrorErrFunc(), normalizer=None, beta=1e-6):
        super(RegressorNc, self).__init__(model, err_func, normalizer, beta)

    def predict(self, x, nc, significance=None):
        n_test = x.shape[0]
        prediction = self.model.predict(x)
        if self.normalizer is not None:
            norm = self.normalizer.score(x) + self.beta
        else:
            norm = np.ones(n_test)

        if significance:
            intervals = np.zeros((x.shape[0], self.model.model.out_shape, 2))
            err_dist = self.err_func.apply_inverse(nc, significance)  # (2, y_dim)
            err_dist = np.stack([err_dist] * n_test)  # (B, 2, y_dim)
            if prediction.ndim > 1:  # CQR
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

        ret = (prediction - z).norm(p=self.feat_norm, dim=1)

        return ret


    def apply_inverse(self, nc, significance):
        nc = np.sort(nc)[::-1]
        border = int(np.floor(significance * (nc.size + 1))) - 1
        border = min(max(border, 0), nc.size - 1)

        return np.vstack([nc[border], nc[border]])


class FeatRegressorNc(BaseModelNc):
    def __init__(self, model,
                 # err_func=FeatErrorErrFunc(),
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
            val_coverage, _ = compute_coverage(val_y.detach().cpu().numpy(), val_y_lower, val_y_upper, val_significance,
                                               name="{}-th step's validation".format(i), verbose=False)
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
        # each_step_val_coverage, val_num = self.coverage_tight(x, y, z_pred, steps=max_inv_steps, val_significance=val_significance)

        tolerance = 1
        count = 0
        final_coverage, best_step = None, None
        for i, val_coverage in enumerate(each_step_val_coverage):
            # print("{}-th step's validation coverage is {}".format(i, val_coverage))
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
            # batch_each_step_val_coverage, val_num = self.coverage_loose(x, y, z_pred, steps=max_inv_steps, val_significance=val_significance)  # length: max_inv_steps
            batch_each_step_val_coverage, val_num = self.coverage_tight(x, y, z_pred, steps=max_inv_steps, val_significance=val_significance)  # length: max_inv_steps
            accumulate_val_coverage += np.array(batch_each_step_val_coverage) * val_num
            accumulate_val_num += val_num

        each_step_val_coverage = accumulate_val_coverage / accumulate_val_num

        tolerance = 3
        count = 0
        final_coverage, best_step = None, None
        for i, val_coverage in enumerate(each_step_val_coverage):
            # print("{}-th step's validation tight coverage is {}".format(i, val_coverage))
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

    def score(self, x, y=None):  # overwrite BaseModelNc.score()
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
        return ret_val

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

            if prediction.ndim > 1:
                if isinstance(x, torch.Tensor):
                    x = x.to(self.model.device)
                else:
                    x = torch.from_numpy(x).to(self.model.device)
                z = self.model.model.encoder(x).detach()

                lirpa_model = BoundedModule(self.model.model.g, torch.empty_like(z))
                ptb = PerturbationLpNorm(norm=self.feat_norm, eps=feat_err_dist[0][0])
                my_input = BoundedTensor(z, ptb)
                # print(my_input.type())
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
            else:
                if not isinstance(x, torch.Tensor):
                    x = torch.from_numpy(x).to(self.model.device)
                z = self.model.model.encoder(x).detach()

                lirpa_model = BoundedModule(self.model.model.g, torch.empty_like(z))
                lirpa_model.eval()                            
                _ = lirpa_model(torch.zeros_like(z))        
                ptb = PerturbationLpNorm(norm=self.feat_norm, eps=feat_err_dist[0][0])  # feat_err_dist=[[0.122, 0.122]]
                my_input = BoundedTensor(z, ptb)

                if 'Optimized' in self.cmethod:
                    lirpa_model.set_bound_opts({
                        'optimize_bound_args': {'ob_iteration': 20, 'ob_lr': 0.1, 'ob_verbose': 0}
                    })

                lb, ub = lirpa_model.compute_bounds(x=(my_input,), method=self.cmethod)  # (bs, 1), (bs, 1)
                if self.g_out_process is not None:
                    lb = self.g_out_process(lb)
                    ub = self.g_out_process(ub)
                lb, ub = lb.detach().cpu().numpy(), ub.detach().cpu().numpy()

                # lirpa_model = BoundedModule(self.model.model.g, torch.empty_like(z))
                # ptb = PerturbationLpNorm(norm=self.feat_norm, eps=feat_err_dist[0][0])  # feat_err_dist=[[0.122, 0.122]]
                # my_input = BoundedTensor(z, ptb)

                # if 'Optimized' in self.cmethod:
                #     lirpa_model.set_bound_opts({'optimize_bound_args': {'ob_iteration': 20, 'ob_lr': 0.1, 'ob_verbose': 0}})
                # lb, ub = lirpa_model.compute_bounds(x=(my_input,), method=self.cmethod)  # (bs, 1), (bs, 1)
                # if self.g_out_process is not None:
                #     lb = self.g_out_process(lb)
                #     ub = self.g_out_process(ub)
                # lb, ub = lb.detach().cpu().numpy(), ub.detach().cpu().numpy()

                intervals[..., 0] = lb
                intervals[..., 1] = ub

            return intervals

        else:
            raise NotImplementedError

class FFCPErrorErrFunc(RegressionErrFunc):
    def __init__(self):
        super(FFCPErrorErrFunc, self).__init__()

    def apply_grad(self,prediction, y, grad):
        err = np.abs(prediction.squeeze() - y)
        if err.ndim > 1:
            err = np.linalg.norm(err, ord=np.inf, axis=1)
        grad_norms = np.array([np.linalg.norm(g) for g in grad])
        grad_norms = np.maximum(grad_norms, 1e-6)  # prevent division by near-zero
        err = err / grad_norms
        return err

    def apply_inverse_grad(self, nc, significance):
        nc = np.sort(nc)[::-1]
        border = int(np.floor(significance * (nc.size + 1)))  # slightly less conservative (removed -1)
        border = min(max(border, 0), nc.size - 1)

        return np.vstack([nc[border], nc[border]])

class FFCP_RegressorNc(BaseModelNc):
    def __init__(self, model, err_func=FFCPErrorErrFunc(), normalizer=None, beta=1e-6):
        super(FFCP_RegressorNc, self).__init__(model, err_func, normalizer, beta)
        err_func = FFCPErrorErrFunc()


    def predict(self, model, x_cal, y_cal, x_test, y_test, layer_index, significance=None):
        n_test = x_test.shape[0]
        prediction_cal, grads_intermediate_cal, gradx_cal = model.predict_with_grads(x_cal, layer_index)

        # Compute calibration gradient norms for determining clip threshold
        grad_norms_cal = np.array([np.linalg.norm(g) for g in grads_intermediate_cal])
        grad_norms_cal = np.maximum(grad_norms_cal, 1e-6)
        clip_threshold = np.percentile(grad_norms_cal, 93)  # clip at 93rd pctile

        # Recompute nc with clipped calibration gradient norms
        err_cal = np.abs(prediction_cal.squeeze() - y_cal)
        if err_cal.ndim > 1:
            err_cal = np.linalg.norm(err_cal, ord=np.inf, axis=1)
        clipped_cal_norms = np.clip(grad_norms_cal, 1e-6, clip_threshold)
        nc = err_cal / clipped_cal_norms

        prediction, grads_intermediate, gradx = model.predict_with_grads(x_test, layer_index)

        grad_norms = np.array([np.linalg.norm(g) for g in grads_intermediate])
        grad_norms = np.maximum(grad_norms, 1e-6)  # prevent near-zero inflation
        grad_norms = np.clip(grad_norms, 1e-6, clip_threshold)  # clip at calibration 95th pctile
        norm = np.ones(n_test)
        grad_norms = grad_norms.reshape(-1, 1)

        if significance:
            if y_test.shape[0] == 1:
                intervals = np.zeros((x_test.shape[0], y_test.shape[0], 2))
            else:
                intervals = np.zeros((x_test.shape[0], 10, 2))
            err_dist = self.err_func.apply_inverse_grad(nc, significance)  # ABS
            err_dist = np.stack([err_dist] * n_test)  # (B, 2, y_dim)
            prediction=prediction.reshape(-1, 1)

            if prediction.ndim > 1:  # CQR
                intervals[..., 0] = prediction - err_dist[:, 0] * grad_norms
                intervals[..., 1] = prediction + err_dist[:, 1] * grad_norms

            else:  # regular conformal prediction
                err_dist *= norm[:, None, None]
                prediction = prediction.squeeze()
                err_dist = err_dist.squeeze()

                grad_norms = grad_norms.squeeze()
                a = np.divide(err_dist[:, 0], grad_norms, where=grad_norms != 0)
                intervals[..., 0] = prediction - err_dist[:, 0] * grad_norms
                intervals[..., 1] = prediction + err_dist[:, 1] * grad_norms

            return intervals

        else:  # Not tested for CQR
            significance = np.arange(0.01, 1.0, 0.01)
            intervals = np.zeros((x_test.shape[0], 1, significance.size))

            for i, s in enumerate(significance):
                err_dist = self.err_func.apply_inverse_grad(nc, s)
                err_dist = np.hstack([err_dist] * n_test)
                err_dist *= norm

                intervals[..., 0] = prediction - err_dist[:, 0] * grad_norms
                intervals[..., 1] = prediction + err_dist[:, 1] * grad_norms

            return intervals

class BaseIcp(BaseEstimator):
    def __init__(self, nc_function, condition=None):
        self.cal_x, self.cal_y = None, None
        self.nc_function = nc_function

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

    def calibrate(self, x, y, increment=False):
        self._calibrate_hook(x, y, increment)
        self._update_calibration_set(x, y, increment)

        if self.conditional:
            category_map = np.array([self.condition((x[i, :], y[i])) for i in range(y.size)])
            self.categories = np.unique(category_map)
            self.cal_scores = defaultdict(partial(np.ndarray, 0))

            for cond in self.categories:
                idx = category_map == cond
                cal_scores = self.nc_function.score(self.cal_x[idx, :], self.cal_y[idx])
                self.cal_scores[cond] = np.sort(cal_scores, 0)[::-1]
        else:
            self.categories = np.array([0])
            cal_scores = self.nc_function.score(self.cal_x, self.cal_y)
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
                p = self.nc_function.predict(x[idx, :], self.cal_scores[condition], significance)
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


def calc_p(ncal, ngt, neq, smoothing=False):
    if smoothing:
        return (ngt + (neq + 1) * np.random.uniform(0, 1)) / (ncal + 1)
    else:
        return (ngt + neq + 1) / (ncal + 1)
