## Code Snipptes from the Demo code notebook
##


import pandas as pd
import numpy as np
import math
import time as time

import wandb
#from wandb.integration.keras import WandbCallback

import enbPI.utils_quick as util
import torch
import torch.nn as nn
from skgarden import RandomForestQuantileRegressor
from numpy.lib.stride_tricks import sliding_window_view


#### Main Class ####
class prediction_interval_with_SPCI():
    '''
        Create prediction intervals assuming Y_t = f(X_t) + \sigma(X_t)\eps_t
        Currently, assume the regression function is by default MLP implemented with PyTorch, as it needs to estimate BOTH f(X_t) and \sigma(X_t), where the latter is impossible to estimate using scikit-learn modules

        Most things carry out, except that we need to have different estimators for f and \sigma.

        fit_func = None: use MLP above
    '''

    def __init__(self, X_train, X_predict, Y_train, Y_predict, fit_func=None):
        self.regressor = fit_func
        self.X_train = X_train
        self.X_predict = X_predict
        self.Y_train = Y_train
        self.Y_predict = Y_predict
        # Predicted training data centers by EnbPI
        self.Ensemble_train_interval_centers = []
        self.Ensemble_train_interval_sigma = []
        # Predicted test data centers by EnbPI
        self.Ensemble_pred_interval_centers = []
        self.Ensemble_pred_interval_sigma = []
        self.Ensemble_online_resid = []  # LOO scores
        self.beta_hat_bins = []

    def fit_bootstrap_models_online(self, B, device, miss_test_idx=[], fit_sigmaX=True):
        '''
          Adapted A.A.
          Return individual ensemble model predictions and bootstrap sample for logging purpose

          Original:
          Train B bootstrap estimators from subsets of (X_train, Y_train), compute aggregated predictors, and compute the residuals
          fit_sigmaX: If False, just avoid predicting \sigma(X_t) by defaulting it to 1
        '''
        n, d = self.X_train.shape
        n1 = len(self.X_predict)
        # hold indices of training data for each f^b
        boot_samples_idx = util.generate_bootstrap_samples(n, n, B)
        # hold predictions from each f^b for fX and sigma&b for sigma
        boot_predictionsFX = torch.zeros(B, n+n1).to(device)
        boot_predictionsSigmaX = torch.ones(B, n+n1).to(device)
        # for i^th column, it shows which f^b uses i in training (so exclude in aggregation)
        in_boot_sample = np.zeros((B, n), dtype=bool)
        out_sample_predictFX = torch.zeros(n, n1).to(device)
        out_sample_predictSigmaX = torch.ones(n, n1).to(device)
        start = time.time()
        Xfull = torch.vstack([self.X_train, self.X_predict])
        for b in range(B):
            Xboot, Yboot = self.X_train[boot_samples_idx[b], :], self.Y_train[boot_samples_idx[b], ]
            in_boot_sample[b, boot_samples_idx[b]] = True
            if self.regressor.__class__.__name__ == 'NoneType':
                start1 = time.time()
                model_f = MLP(d).to(device)
                wandb.watch(model_f, idx=b)
                optimizer_f = torch.optim.Adam(model_f.parameters(), lr=1e-3)
                if fit_sigmaX:
                    model_sigma = MLP(d, sigma=True).to(device)
                    wandb.watch(model_sigma, idx=B+b)
                    optimizer_sigma = torch.optim.Adam(model_sigma.parameters(), lr=2e-3)
                for epoch in range(300):
                    fXhat = model_f(Xboot)
                    sigmaXhat = torch.ones(len(fXhat)).to(device)
                    if fit_sigmaX:
                        sigmaXhat = model_sigma(Xboot)
                    loss = ((Yboot-fXhat)/sigmaXhat).pow(2).mean()/2
                    optimizer_f.zero_grad()
                    if fit_sigmaX:
                        optimizer_sigma.zero_grad()
                    loss.backward()
                    optimizer_f.step()
                    if fit_sigmaX:
                        optimizer_sigma.step()
                with torch.no_grad():
                    boot_predictionsFX[b] = model_f(Xfull).flatten()
                    if fit_sigmaX:
                        boot_predictionsSigmaX[b] = model_sigma(Xfull).flatten()
                print(f'Took {time.time()-start1} secs to finish the {b}th boostrap model')
            else:
                model = self.regressor
                model.fit(Xboot, Yboot)
                boot_predictionsFX[b] = torch.from_numpy(model.predict(Xfull).flatten()).to(device)
                # NOTE, NO sigma estimation because these methods by deFAULT are fitting Y, but we have no observation of errors
        print(f'Finish Fitting {B} Bootstrap models, took {time.time()-start} secs.')
        start = time.time()
        keep = []
        for i in range(n):
            b_keep = np.argwhere(~(in_boot_sample[:, i])).reshape(-1)
            pred_iFX = boot_predictionsFX[b_keep, i].mean()
            pred_iSigmaX = boot_predictionsSigmaX[b_keep, i].mean()
            pred_testFX = boot_predictionsFX[b_keep, n:].mean(0)
            pred_testSigmaX = boot_predictionsSigmaX[b_keep, n:].mean(0)
            if(len(b_keep) > 0):
                self.Ensemble_train_interval_centers.append(pred_iFX)
                self.Ensemble_train_interval_sigma.append(pred_iSigmaX)
                resid_LOO = (self.Y_train[i] - pred_iFX)/pred_iSigmaX
                out_sample_predictFX[i] = pred_testFX
                out_sample_predictSigmaX[i] = pred_testSigmaX
                keep = keep+[b_keep]
            self.Ensemble_online_resid.append(resid_LOO.item())
        sorted_out_sample_predictFX = out_sample_predictFX.mean(0)  # length n1
        sorted_out_sample_predictSigmaX = out_sample_predictSigmaX.mean(0)  # length n1
        resid_out_sample = (self.Y_predict-sorted_out_sample_predictFX)/sorted_out_sample_predictSigmaX
        if len(miss_test_idx) > 0:
            # Replace missing residuals with that from the immediate predecessor that is not missing, as
            # o/w we are not assuming prediction data are missing
            for idx in range(len(miss_test_idx)):
                i = miss_test_idx[idx]
                if i > 0:
                    j = i-1
                    while j in miss_test_idx[:idx]:
                        j -= 1
                    resid_out_sample[i] = resid_out_sample[j]

                else:
                    # The first Y during testing is missing, let it be the last of the training residuals
                    # note, training data already takes out missing values, so doing is is fine
                    resid_out_sample[0] = self.Ensemble_online_resid[-1]
        self.Ensemble_online_resid = np.append(self.Ensemble_online_resid, resid_out_sample.cpu().detach().numpy())
        # print(f'Finish Computing LOO residuals, took {time.time()-start} secs.')
        # print(f'Max LOO test residual is {np.max(self.Ensemble_online_resid[n:])}')
        # print(f'Min LOO test residual is {np.min(self.Ensemble_online_resid[n:])}')
        self.Ensemble_pred_interval_centers = sorted_out_sample_predictFX
        self.Ensemble_pred_interval_sigma = sorted_out_sample_predictSigmaX
        return boot_predictionsFX[:, n:], boot_predictionsSigmaX[:, n:], in_boot_sample

    def compute_PIs_Ensemble_online(self, alpha, stride=1, smallT=True, past_window=100, use_quantile_regr=False, quantile_regr='RF',
                                    quantile_reg_retrain=True, quantile_reg_param={'max_depth': 2, 'random_state': 0}, quantile_with_Xt=False):
        '''
            smallT: if True, we would only start with the last n number of LOO residuals, rather than use the full length T ones. Used in change detection
                NOTE: smallT can be important if time-series is very dynamic, in which case training MORE data may actaully be worse (because quantile longer)
                HOWEVER, if fit quantile regression, set it to be FALSE because we want to have many training pts for the quantile regressor
            use_quantile_regr: if True, we fit conditional quantile to compute the widths, rather than simply using empirical quantile
        '''
        n1 = len(self.X_train)
        if smallT:
            n1 = min(past_window, len(self.X_train))
        # Now f^b and LOO residuals have been constructed from earlier
        out_sample_predict = self.Ensemble_pred_interval_centers.cpu().detach().numpy()
        out_sample_predictSigmaX = self.Ensemble_pred_interval_sigma
        start = time.time()
        # Matrix, where each row is a UNIQUE slice of residuals with length stride.
        resid_strided = util.strided_app(self.Ensemble_online_resid[len(self.X_train)-n1:-1], n1, stride)
        print(f'Shape of slided residual lists is {resid_strided.shape}')
        num_unique_resid = resid_strided.shape[0]
        width_left = np.zeros(num_unique_resid)
        width_right = np.zeros(num_unique_resid)
        # # NEW, alpha becomes alpha_t. Uncomment things below if we decide to use this upgraded EnbPI
        # alpha_t = alpha
        # errs = []
        # gamma = 0.005
        # method = 'simple'  # 'simple' or 'complex'
        # self.alphas = []
        # NOTE: 'max_features='log2', max_depth=2' make the model "simpler", which improves performance in practice
        for i in range(num_unique_resid):
            # for p in range(stride):  # NEW for adaptive alpha
            past_resid = resid_strided[i, :]
            curr_SigmaX = out_sample_predictSigmaX[i].item()
            if use_quantile_regr:
                # New predicted conditional quntile
                # 1. Get "past_resid" into an auto-regressive fashion
                # This should be more carefully examined, b/c it depends on how long \hat{\eps}_t depends on the past
                # From practice, making it small make intervals wider
                n2 = past_window
                residX = sliding_window_view(past_resid, window_shape=n2)
                residY = past_resid[n2:]
                if quantile_with_Xt:
                    past_Xt = torch.concat((self.X_train, self.X_predict), dim=0)[i:i+n1, :]
                    window_Xt = sliding_window_view(past_Xt, window_shape=n2, axis=0)
                    window_Xt = window_Xt.reshape(window_Xt.shape[0], -1)
                else:
                    window_Xt = torch.empty((residX.shape[0], 0))
                # 2. Fit the model. Default quantile regressor is the quantile RF from
                # scikit-garden: https://scikit-garden.github.io/
                # NOTE, should NOT warm start, as it makes result poor, although training is longer
                if quantile_regr == 'RF':
                    if quantile_reg_retrain or i == 0:
                        rfqr = RandomForestQuantileRegressor(**quantile_reg_param)
                        rfqr.fit(np.concatenate((residX[:-1], window_Xt[:-1]), axis=1), residY)
                    # 3. Find best \hat{\beta} via evaluating many quantiles
                    beta_hat_bin = util.binning_use_RF_quantile_regr(rfqr, np.concatenate((residX[-1], window_Xt[-1])), alpha)
                    width_left[i] = curr_SigmaX*rfqr.predict(
                        np.concatenate((residX[-1].reshape(1, -1), window_Xt[-1].reshape(1, -1)), axis=1), math.ceil(100 * beta_hat_bin))
                    width_right[i] = curr_SigmaX*rfqr.predict(
                        np.concatenate((residX[-1].reshape(1, -1), window_Xt[-1].reshape(1, -1)), axis=1), math.ceil(100 * (1-alpha+beta_hat_bin)))
                # if quantile_regr == 'LR':
                #     start1 = time.time()
                #     wleft, wright = util.binning_use_linear_quantile_regr(
                #         residX, residY, alpha)
                #     if i == 0:
                #         print(
                #             f'100 Linear QRegr approx. takes {100*(time.time()-start1)} secs.')
                #     width_left[i] = curr_SigmaX*wleft
                #     width_right[i] = curr_SigmaX*wright
                if i % int(num_unique_resid/20) == 0:
                    print(f'Width at test {i} is {width_right[i]-width_left[i]}')
            else:
                # Naive empirical quantile
                # The number of bins will be determined INSIDE binning
                beta_hat_bin = util.binning(past_resid, alpha)
                # beta_hat_bin = util.binning(past_resid, alpha_t)
                self.beta_hat_bins.append(beta_hat_bin)
                width_left[i] = curr_SigmaX*np.percentile(past_resid, math.ceil(100*beta_hat_bin))
                width_right[i] = curr_SigmaX*np.percentile(past_resid, math.ceil(100*(1-alpha+beta_hat_bin)))
        print( f'Finish Computing {num_unique_resid} UNIQUE Prediction Intervals, took {time.time()-start} secs.')
        # This is because |width|=T1/stride.
        width_left = np.repeat(width_left, stride)
        # This is because |width|=T1/stride.
        width_right = np.repeat(width_right, stride)
        PIs_Ensemble = pd.DataFrame(np.c_[out_sample_predict+width_left,
                                          out_sample_predict+width_right], columns=['lower', 'upper'])
        self.PIs_Ensemble = PIs_Ensemble

    '''
        All together
    '''

    def get_results(self, true_Y_predict=[], method='Ensemble'):
        '''
            Adapted A.A:
            1) Do not build dataframe but just return values of coverage (mean) and width (mean)
            2) Directly build rolling window here

            Original:
            NOTE: I added a "true_Y_predict" option, which will be used for calibrating coverage under missing data
            In particular, this is needed when the Y_predict we use for training is NOT the same as true Y_predict
        '''
        if method == 'Ensemble':
            PI = self.PIs_Ensemble
        Ytest = self.Y_predict.cpu().detach().numpy()
        coverage = ((np.array(PI['lower']) <= Ytest) & (np.array(PI['upper']) >= Ytest))
        if len(true_Y_predict) > 0:
            coverage = ((np.array(PI['lower']) <= true_Y_predict) & (np.array(PI['upper']) >= true_Y_predict))
        cov_mean = coverage.mean()
        print(f'Average Coverage is {cov_mean}')
        width = (PI['upper'] - PI['lower'])
        width_mean = width.mean()
        print(f'Average Width is {width_mean}')

        return cov_mean, width_mean, coverage, width


def CP_LS(X, Y, x, alpha, weights=[], tags=[]):
    # Barber et al. 2022: Nex-CP
    # weights are used for computing quantiles for the prediction interval
    # tags are used as weights in weighted least squares regression
    n = len(Y)

    if(len(tags) == 0):
        tags = np.ones(n+1)

    if(len(weights) == 0):
        weights = np.ones(n+1)
    if(len(weights) == n):
        weights = np.r_[weights, 1]
    weights = weights / np.sum(weights)

    # randomly permute one weight for the regression
    random_ind = int(np.where(np.random.multinomial(1, weights, 1))[1])
    tags[np.c_[random_ind, n]] = tags[np.c_[n, random_ind]]

    XtX = (X.T*tags[:-1]).dot(X) + np.outer(x, x)*tags[-1]
    a = Y - X.dot(np.linalg.solve(XtX, (X.T*tags[:-1]).dot(Y)))
    b = -X.dot(np.linalg.solve(XtX, x))*tags[-1]
    a1 = -x.T.dot(np.linalg.solve(XtX, (X.T*tags[:-1]).dot(Y)))
    b1 = 1 - x.T.dot(np.linalg.solve(XtX, x))*tags[-1]
    # if we run weighted least squares on (X[1,],Y[1]),...(X[n,],Y[n]),(x,y)
    # then a + b*y = residuals of data points 1,..,n
    # and a1 + b1*y = residual of data point n+1

    y_knots = np.sort(
        np.unique(np.r_[((a-a1)/(b1-b))[b1-b != 0], ((-a-a1)/(b1+b))[b1+b != 0]]))
    y_inds_keep = np.where(((np.abs(np.outer(a1+b1*y_knots, np.ones(n)))
                             > np.abs(np.outer(np.ones(len(y_knots)), a)+np.outer(y_knots, b))) *
                            weights[:-1]).sum(1) <= 1-alpha)[0]
    y_PI = np.array([y_knots[y_inds_keep.min()], y_knots[y_inds_keep.max()]])
    if(weights[:-1].sum() <= 1-alpha):
        y_PI = np.array([-np.inf, np.inf])
    return y_PI

#### Model and data helper ####


class MLP(nn.Module):
    def __init__(self, d, sigma=False):
        super(MLP, self).__init__()
        H = 64
        layers = [nn.Linear(d, H), nn.ReLU(), nn.Linear(
            H, H), nn.ReLU(), nn.Linear(H, 1)]
        self.sigma = sigma
        if self.sigma:
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        perturb = 1e-3 if self.sigma else 0
        return self.layers(x)+perturb
