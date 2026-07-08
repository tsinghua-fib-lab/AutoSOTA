# from tensorflow.keras.optimizers import Adam
# import keras
import pandas as pd
import numpy as np
# from keras.models import clone_model
from sklearn.linear_model import LogisticRegression
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import math
import time as time
import enbPI.utils_EnbPI as util
import warnings
warnings.filterwarnings("ignore")


class prediction_interval():
    '''
        Create prediction intervals using different methods (i.e., EnbPI, J+aB ICP, Weighted, Time-series)
    '''

    def __init__(self, fit_func, X_train, X_predict, Y_train, Y_predict):
        self.regressor = fit_func
        self.X_train = X_train
        self.X_predict = X_predict
        self.Y_train = Y_train
        self.Y_predict = Y_predict
        # Predicted training data centers by EnbPI
        self.Ensemble_train_interval_centers = []
        # Predicted test data centers by EnbPI
        self.Ensemble_pred_interval_centers = []
        self.Ensemble_online_resid = np.array([])  # LOO scores
        self.Ensemble_pred_interval_ends = []  # Upper and lower end
        self.beta_hat_bins = []
        self.ICP_fitted_func = []  # it only store 1 fitted ICP func.
        self.ICP_resid = np.array([])
        self.WeightCP_online_resid = np.array([])
        self.JaB_boot_samples_idx = 0
        self.JaB_boot_predictions = 0

    def fit_bootstrap_models_online(self, B, miss_test_idx):
        '''
          Train B bootstrap estimators from subsets of (X_train, Y_train), compute aggregated predictors, and compute the residuals
        '''
        n = len(self.X_train)
        n1 = len(self.X_predict)
        # hold indices of training data for each f^b
        boot_samples_idx = util.generate_bootstrap_samples(n, n, B)
        # hold predictions from each f^b
        boot_predictions = np.zeros((B, (n+n1)), dtype=float)
        # for i^th column, it shows which f^b uses i in training (so exclude in aggregation)
        in_boot_sample = np.zeros((B, n), dtype=bool)
        out_sample_predict = np.zeros((n, n1))
        start = time.time()
        for b in range(B):
            model = self.regressor
            # NOTE: it is CRITICAL to clone the model, as o/w it will OVERFIT to the model across different iterations of bootstrap S_b.
            # I originally did not understand that doing so is necessary but now know it
            if self.regressor.__class__.__name__ == 'Sequential':
                start1 = time.time()
                model = clone_model(self.regressor)
                opt = Adam(5e-4)
                model.compile(loss='mean_squared_error', optimizer=opt)
                callback = keras.callbacks.EarlyStopping(
                    monitor='loss', patience=10)
                bsize = int(0.1*len(np.unique(boot_samples_idx[b])))  # Was 0.1
                if self.regressor.name == 'NeuralNet':
                    # verbose definition here: https://keras.io/api/models/model_training_apis/#fit-method. 0 means silent
                    # NOTE: I do NOT want epoches to be too large, as we then tend to be too close to the actual Y_t, NOT f(X_t).
                    # Epoch was 250
                    model.fit(self.X_train[boot_samples_idx[b], :], self.Y_train[boot_samples_idx[b], ],
                              epochs=250, batch_size=bsize, callbacks=[callback], verbose=0)
                else:
                    # This is RNN, mainly have different shape and decrease epochs for faster computation
                    model.fit(self.X_train[boot_samples_idx[b], :], self.Y_train[boot_samples_idx[b], ],
                              epochs=10, batch_size=bsize, callbacks=[callback], verbose=0)
                # NOTE, this multiplied by B tells us total estimation time
                print(
                    f'Took {time.time()-start1} secs to fit the {b}th boostrap model')
            else:
                model = model.fit(self.X_train[boot_samples_idx[b], :],
                                  self.Y_train[boot_samples_idx[b], ])
            boot_predictions[b] = model.predict(
                np.r_[self.X_train, self.X_predict]).flatten()
            in_boot_sample[b, boot_samples_idx[b]] = True
        print(
            f'Finish Fitting B Bootstrap models, took {time.time()-start} secs.')
        start = time.time()
        keep = []
        for i in range(n):
            b_keep = np.argwhere(~(in_boot_sample[:, i])).reshape(-1)
            if(len(b_keep) > 0):
                # NOTE: Append these training centers too see their magnitude
                # The reason is sometimes they are TOO close to actual Y.
                self.Ensemble_train_interval_centers.append(
                    boot_predictions[b_keep, i].mean())
                resid_LOO = self.Y_train[i] - \
                    boot_predictions[b_keep, i].mean()
                out_sample_predict[i] = boot_predictions[b_keep, n:].mean(0)
                keep = keep+[b_keep]
            else:  # if aggregating an empty set of models, predict zero everywhere
                resid_LOO = self.Y_train[i]
                out_sample_predict[i] = np.zeros(n1)
            self.Ensemble_online_resid = np.append(
                self.Ensemble_online_resid, resid_LOO)
            keep = keep+[]
        # print(f'Max LOO training residual is {np.max(self.Ensemble_online_resid)}')
        # print(f'Min LOO training residual is {np.min(self.Ensemble_online_resid)}')
        sorted_out_sample_predict = out_sample_predict.mean(
            axis=0)  # length n1
        resid_out_sample = self.Y_predict-sorted_out_sample_predict
        if len(miss_test_idx) > 0:
            # Replace missing residuals with that from the immediate predecessor that is not missing, as
            # o/w we are not assuming prediction data are missing
            for l in range(len(miss_test_idx)):
                i = miss_test_idx[l]
                if i > 0:
                    j = i-1
                    while j in miss_test_idx[:l]:
                        j -= 1
                    resid_out_sample[i] = resid_out_sample[j]

                else:
                    # The first Y during testing is missing, let it be the last of the training residuals
                    # note, training data already takes out missing values, so doing is is fine
                    resid_out_sample[0] = self.Ensemble_online_resid[-1]
        self.Ensemble_online_resid = np.append(
            self.Ensemble_online_resid, resid_out_sample)
        # print(f'Finish Computing LOO residuals, took {time.time()-start} secs.')
        # print(f'Max LOO test residual is {np.max(self.Ensemble_online_resid[n:])}')
        # print(f'Min LOO test residual is {np.min(self.Ensemble_online_resid[n:])}')
        self.Ensemble_pred_interval_centers = sorted_out_sample_predict

    def compute_PIs_Ensemble_online(self, alpha, stride, smallT=False):
        # If smallT, we would only start with the last n number of LOO residuals, rather than use the full length T ones. Used in change detection
        ''' NOTE: smallT can be important if time-series is very dynamic, in which case training MORE data may actaully be worse (because quantile longer)'''
        n = len(self.X_train)
        if smallT:
            past_window = 1000 if len(self.X_train) > 2000 else 100
            past_window = 500
            n = min(past_window, len(self.X_train))
        # Now f^b and LOO residuals have been constructed from earlier
        out_sample_predict = self.Ensemble_pred_interval_centers
        start = time.time()
        # Matrix, where each row is a UNIQUE slice of residuals with length stride.
        resid_strided = util.strided_app(
            self.Ensemble_online_resid[len(self.X_train)-n:-1], n, stride)
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
        for i in range(num_unique_resid):
            # for p in range(stride):  # NEW for adaptive alpha
            past_resid = resid_strided[i, :]
            # The number of bins will be determined INSIDE binning
            beta_hat_bin = util.binning(past_resid, alpha)
            # beta_hat_bin = util.binning(past_resid, alpha_t)
            self.beta_hat_bins.append(beta_hat_bin)
            width_left[i] = np.percentile(
                past_resid, math.ceil(100*beta_hat_bin))
            width_right[i] = np.percentile(
                past_resid, math.ceil(100*(1-alpha+beta_hat_bin)))
            # width_right[i] = np.percentile(past_resid, math.ceil(100*(1-alpha_t+beta_hat_bin)))
            # Y_t = self.Y_predict[i*stride+p]
            # Y_t_hat = out_sample_predict[i*stride+p]
            # lower_t, upper_t = Y_t_hat+width_left[i], Y_t_hat+width_right[i]
            # err = 1 if Y_t < lower_t or Y_t > upper_t else 0
            # errs.append(err)
            # alpha_t = util.adjust_alpha_t(alpha_t, alpha, errs, gamma, method)
            # if alpha_t > 1:
            #     alpha_t = 1
            # if alpha_t < 0:
            #     alpha_t = 0
            # self.alphas.append(alpha_t)
        print(
            f'Finish Computing {num_unique_resid} UNIQUE Prediction Intervals, took {time.time()-start} secs.')
        # This is because |width|=T1/stride.
        width_left = np.repeat(width_left, stride)
        # This is because |width|=T1/stride.
        width_right = np.repeat(width_right, stride)
        PIs_Ensemble = pd.DataFrame(np.c_[out_sample_predict+width_left,
                                          out_sample_predict+width_right], columns=['lower', 'upper'])
        self.Ensemble_pred_interval_ends = PIs_Ensemble
        # print(time.time()-start)
        return PIs_Ensemble

    '''
        Jackknife+-after-bootstrap (used in Figure 8)
    '''

    def fit_bootstrap_models(self, B):
        '''
          Train B bootstrap estimators and calculate LOO predictions on X_train and X_predict
        '''
        n = len(self.X_train)
        boot_samples_idx = util.generate_bootstrap_samples(n, n, B)
        n1 = len(np.r_[self.X_train, self.X_predict])
        # P holds the predictions from individual bootstrap estimators
        predictions = np.zeros((B, n1), dtype=float)
        for b in range(B):
            model = self.regressor
            if self.regressor.__class__.__name__ == 'Sequential':
                model = clone_model(self.regressor)
                opt = Adam(5e-4)
                model.compile(loss='mean_squared_error', optimizer=opt)
                callback = keras.callbacks.EarlyStopping(
                    monitor='loss', patience=10)
                if self.regressor.name == 'NeuralNet':
                    model.fit(self.X_train[boot_samples_idx[b], :], self.Y_train[boot_samples_idx[b], ],
                              epochs=1000, batch_size=100, callbacks=[callback], verbose=0)
                else:
                    # This is RNN, mainly have different shape and decrease epochs
                    model.fit(self.X_train[boot_samples_idx[b], :], self.Y_train[boot_samples_idx[b], ],
                              epochs=100, batch_size=100, callbacks=[callback], verbose=0)
            else:
                model = model.fit(self.X_train[boot_samples_idx[b], :],
                                  self.Y_train[boot_samples_idx[b], ])
            predictions[b] = model.predict(
                np.r_[self.X_train, self.X_predict]).flatten()
        self.JaB_boot_samples_idx = boot_samples_idx
        self.JaB_boot_predictions = predictions

    def compute_PIs_JaB(self, alpha):
        '''
        Using mean aggregation
        '''
        n = len(self.X_train)
        n1 = len(self.X_predict)
        boot_samples_idx = self.JaB_boot_samples_idx
        boot_predictions = self.JaB_boot_predictions
        B = len(boot_predictions)
        in_boot_sample = np.zeros((B, n), dtype=bool)
        for b in range(len(in_boot_sample)):
            in_boot_sample[b, boot_samples_idx[b]] = True
        resids_LOO = np.zeros(n)
        muh_LOO_vals_testpoint = np.zeros((n, n1))
        for i in range(n):
            b_keep = np.argwhere(~(in_boot_sample[:, i])).reshape(-1)
            if(len(b_keep) > 0):
                resids_LOO[i] = np.abs(
                    self.Y_train[i] - boot_predictions[b_keep, i].mean())
                muh_LOO_vals_testpoint[i] = boot_predictions[b_keep, n:].mean(
                    0)
            else:  # if aggregating an empty set of models, predict zero everywhere
                resids_LOO[i] = np.abs(self.Y_train[i])
                muh_LOO_vals_testpoint[i] = np.zeros(n1)
        ind_q = (np.ceil((1-alpha)*(n+1))).astype(int)
        return pd.DataFrame(
            np.c_[np.sort(muh_LOO_vals_testpoint.T - resids_LOO, axis=1).T[-ind_q],
                  np.sort(muh_LOO_vals_testpoint.T + resids_LOO, axis=1).T[ind_q-1]],
            columns=['lower', 'upper'])

    '''
        Inductive Conformal Prediction
    '''

    def compute_PIs_ICP(self, alpha, l):
        n = len(self.X_train)
        proper_train = np.random.choice(n, l, replace=False)
        X_train = self.X_train[proper_train, :]
        Y_train = self.Y_train[proper_train]
        X_calibrate = np.delete(self.X_train, proper_train, axis=0)
        Y_calibrate = np.delete(self.Y_train, proper_train)
        model = self.regressor
        if self.regressor.__class__.__name__ == 'Sequential':
            callback = keras.callbacks.EarlyStopping(
                monitor='loss', patience=10)
            bsize = int(0.1*X_train.shape[0])  # Was 0.1
            if self.regressor.name == 'NeuralNet':
                model.fit(self.X_train, self.Y_train,
                          epochs=250, batch_size=bsize, callbacks=[callback], verbose=0)
            else:
                # This is RNN, mainly have different epochs
                model.fit(X_train, Y_train,
                          epochs=10, batch_size=bsize, callbacks=[callback], verbose=0)
            self.ICP_fitted_func.append(model)
        else:
            self.ICP_fitted_func.append(self.regressor.fit(X_train, Y_train))
        predictions_calibrate = self.ICP_fitted_func[0].predict(
            X_calibrate).flatten()
        calibrate_residuals = np.abs(Y_calibrate-predictions_calibrate)
        out_sample_predict = self.ICP_fitted_func[0].predict(
            self.X_predict).flatten()
        self.ICP_resid = np.append(
            self.ICP_resid, calibrate_residuals)  # length n-l
        ind_q = math.ceil(100*(1-alpha))  # 1-alpha%
        width = np.abs(np.percentile(self.ICP_resid, ind_q, axis=-1).T)
        PIs_ICP = pd.DataFrame(np.c_[out_sample_predict-width,
                                     out_sample_predict+width], columns=['lower', 'upper'])
        # print(time.time()-start)
        return PIs_ICP

    def compute_PIs_ICP_online(self, alpha, l, stride=1):
        n = len(self.X_train)
        proper_train = np.random.choice(n, l, replace=False)
        X_train = self.X_train[proper_train, :]
        Y_train = self.Y_train[proper_train]
        X_calibrate = np.delete(self.X_train, proper_train, axis=0)
        Y_calibrate = np.delete(self.Y_train, proper_train)
        model = self.regressor
        if self.regressor.__class__.__name__ == 'Sequential':
            callback = keras.callbacks.EarlyStopping(
                monitor='loss', patience=10)
            bsize = int(0.1*X_train.shape[0])  # Was 0.1
            if self.regressor.name == 'NeuralNet':
                model.fit(self.X_train, self.Y_train,
                          epochs=250, batch_size=bsize, callbacks=[callback], verbose=0)
            else:
                # This is RNN, mainly have different epochs
                model.fit(X_train, Y_train,
                          epochs=10, batch_size=bsize, callbacks=[callback], verbose=0)
            self.ICP_fitted_func.append(model)
        else:
            self.ICP_fitted_func.append(self.regressor.fit(X_train, Y_train))
        predictions_calibrate = self.ICP_fitted_func[0].predict(
            X_calibrate).flatten()
        calibrate_residuals = np.abs(Y_calibrate-predictions_calibrate)
        out_sample_predict = self.ICP_fitted_func[0].predict(
            self.X_predict).flatten()
        out_sample_residuals = np.abs(self.Y_predict-out_sample_predict)
        self.ICP_resid = np.append(
            self.ICP_resid, calibrate_residuals)  # length n-l
        self.ICP_resid = np.append(
            self.ICP_resid, out_sample_residuals)  # length n-l
        resid_strided = util.strided_app(
            self.ICP_resid[:-1], n-l, stride)
        ind_q = math.ceil(100*(1-alpha))  # 1-alpha%
        print(f'Shape of slided residual lists is {resid_strided.shape}')
        num_unique_resid = resid_strided.shape[0]
        width = np.zeros(num_unique_resid)
        for i in range(num_unique_resid):
            past_resid = resid_strided[i, :]
            width[i] = np.percentile(past_resid, ind_q)
        width = np.repeat(width, stride)
        PIs_ICP = pd.DataFrame(np.c_[out_sample_predict-width,
                                     out_sample_predict+width], columns=['lower', 'upper'])
        return PIs_ICP

    '''
        Weighted Conformal Prediction
    '''

    def compute_PIs_Weighted_ICP(self, alpha, l):
        '''The residuals are weighted by fitting a logistic regression on
           (X_calibrate, C=0) \cup (X_predict, C=1'''
        n = len(self.X_train)
        n1 = len(self.X_predict)
        proper_train = np.random.choice(n, l, replace=False)
        X_train = self.X_train[proper_train, :]
        Y_train = self.Y_train[proper_train]
        X_calibrate = np.delete(self.X_train, proper_train, axis=0)
        Y_calibrate = np.delete(self.Y_train, proper_train)
        # Main difference from ICP
        C_calibrate = np.zeros(n-l)
        C_predict = np.ones(n1)
        X_weight = np.r_[X_calibrate, self.X_predict]
        C_weight = np.r_[C_calibrate, C_predict]
        if len(X_weight.shape) > 2:
            # Reshape for RNN
            tot, _, shap = X_weight.shape
            X_weight = X_weight.reshape((tot, shap))
        clf = LogisticRegression(random_state=0).fit(X_weight, C_weight)
        Prob = clf.predict_proba(X_weight)
        Weights = Prob[:, 1]/(1-Prob[:, 0])  # n-l+n1 in length
        model = self.regressor
        if self.regressor.__class__.__name__ == 'Sequential':
            callback = keras.callbacks.EarlyStopping(
                monitor='loss', patience=10)
            bsize = int(0.1*X_train.shape[0])  # Was 0.1
            if self.regressor.name == 'NeuralNet':
                model.fit(self.X_train, self.Y_train,
                          epochs=250, batch_size=bsize, callbacks=[callback], verbose=0)
            else:
                # This is RNN, mainly have different epochs
                model.fit(X_train, Y_train,
                          epochs=10, batch_size=bsize, callbacks=[callback], verbose=0)
            self.ICP_fitted_func.append(model)
        else:
            self.ICP_fitted_func.append(self.regressor.fit(X_train, Y_train))
        predictions_calibrate = self.ICP_fitted_func[0].predict(
            X_calibrate).flatten()
        calibrate_residuals = np.abs(Y_calibrate-predictions_calibrate)
        out_sample_predict = self.ICP_fitted_func[0].predict(
            self.X_predict).flatten()
        self.WeightCP_online_resid = np.append(
            self.WeightCP_online_resid, calibrate_residuals)  # length n-1
        width = np.abs(util.weighted_quantile(values=self.WeightCP_online_resid, quantiles=1-alpha,
                                              sample_weight=Weights[:n-l]))
        PIs_ICP = pd.DataFrame(np.c_[out_sample_predict-width,
                                     out_sample_predict+width], columns=['lower', 'upper'])
        # print(time.time()-start)
        return PIs_ICP

    def compute_PIs_Weighted_ICP_online(self, alpha, l, stride=1):
        '''The residuals are weighted by fitting a logistic regression on
           (X_calibrate, C=0) \cup (X_predict, C=1'''
        n = len(self.X_train)
        n1 = len(self.X_predict)
        proper_train = np.random.choice(n, l, replace=False)
        X_train = self.X_train[proper_train, :]
        Y_train = self.Y_train[proper_train]
        X_calibrate = np.delete(self.X_train, proper_train, axis=0)
        Y_calibrate = np.delete(self.Y_train, proper_train)
        # Main difference from ICP
        C_calibrate = np.zeros(n-l)
        C_predict = np.ones(n1)
        X_weight = np.r_[X_calibrate, self.X_predict]
        C_weight = np.r_[C_calibrate, C_predict]
        if len(X_weight.shape) > 2:
            # Reshape for RNN
            tot, _, shap = X_weight.shape
            X_weight = X_weight.reshape((tot, shap))
        clf = LogisticRegression(random_state=0).fit(X_weight, C_weight)
        Prob = clf.predict_proba(X_weight)
        Weights = Prob[:, 1]/(1-Prob[:, 0])  # n-l+n1 in length
        model = self.regressor
        if self.regressor.__class__.__name__ == 'Sequential':
            callback = keras.callbacks.EarlyStopping(
                monitor='loss', patience=10)
            bsize = int(0.1*X_train.shape[0])  # Was 0.1
            if self.regressor.name == 'NeuralNet':
                model.fit(self.X_train, self.Y_train,
                          epochs=250, batch_size=bsize, callbacks=[callback], verbose=0)
            else:
                # This is RNN, mainly have different epochs
                model.fit(X_train, Y_train,
                          epochs=10, batch_size=bsize, callbacks=[callback], verbose=0)
            self.ICP_fitted_func.append(model)
        else:
            self.ICP_fitted_func.append(self.regressor.fit(X_train, Y_train))
        predictions_calibrate = self.ICP_fitted_func[0].predict(
            X_calibrate).flatten()
        calibrate_residuals = np.abs(Y_calibrate-predictions_calibrate)
        out_sample_predict = self.ICP_fitted_func[0].predict(
            self.X_predict).flatten()
        out_sample_residuals = np.abs(self.Y_predict-out_sample_predict)
        self.WeightCP_online_resid = np.append(
            self.WeightCP_online_resid, calibrate_residuals)  # length n-1
        self.WeightCP_online_resid = np.append(
            self.WeightCP_online_resid, out_sample_residuals)  # length n-1
        resid_strided = util.strided_app(
            self.WeightCP_online_resid[:-1], n-l, stride)
        num_unique_resid = resid_strided.shape[0]
        weight_strided = util.strided_app(
            Weights[:-1], n-l, stride)
        width = np.zeros(num_unique_resid)
        for i in range(num_unique_resid):
            past_resid = resid_strided[i, :]
            past_weights = weight_strided[i, :]
            width[i] = np.abs(util.weighted_quantile(values=past_resid,
                                                     quantiles=1-alpha, sample_weight=past_weights))
        width = np.repeat(width, stride)
        PIs_WICP = pd.DataFrame(np.c_[out_sample_predict-width,
                                      out_sample_predict+width], columns=['lower', 'upper'])
        # print(time.time()-start)
        return PIs_WICP

    def compute_PIs_tseries_online(self, alpha, name):
        '''
            Use train_size to form model and the rest to be out-sample-prediction
        '''
        # Concatenate training and testing together
        data = pd.DataFrame(np.r_[self.Y_train, self.Y_predict])
        # Train model
        train_size = len(self.Y_train)
        if name == 'ARIMA(10,1,10)':
            training_mod = SARIMAX(data[:train_size], order=(10, 1, 10))
            mod = SARIMAX(data, order=(10, 1, 10))
        if name == 'ExpSmoothing':
            training_mod = ExponentialSmoothing(
                data[:train_size], trend=True, damped_trend=True, seasonal=24)
            mod = ExponentialSmoothing(
                data, trend=True, damped_trend=True, seasonal=24)
        if name == 'DynamicFactor':
            training_mod = DynamicFactorMQ(data[:train_size])
            mod = DynamicFactorMQ(data)
        print('training')
        training_res = training_mod.fit(disp=0)
        print('training done')
        # Use in full model
        res = mod.filter(training_res.params)
        # Get the insample prediction interval (which is outsample prediction interval)
        pred = res.get_prediction(
            start=data.index[train_size], end=data.index[-1])
        pred_int = pred.conf_int(alpha=alpha)  # prediction interval
        PIs_res = pd.DataFrame(
            np.c_[pred_int.iloc[:, 0], pred_int.iloc[:, 1]], columns=['lower', 'upper'])
        return(PIs_res)

    def Winkler_score(self, PIs_ls, data_name, methods_name, alpha):
        # Examine if each test point is in the intervals
        # If in, then score += width of intervals
        # If not,
        # If true y falls under lower end, score += width + 2*(lower end-true y)/alpha
        # If true y lies above upper end, score += width + 2*(true y-upper end)/alpha
        n1 = len(self.Y_predict)
        score_ls = []
        for i in range(len(methods_name)):
            score = 0
            for j in range(n1):
                upper = PIs_ls[i].loc[j, 'upper']
                lower = PIs_ls[i].loc[j, 'lower']
                width = upper-lower
                truth = self.Y_predict[j]
                if (truth >= lower) & (truth <= upper):
                    score += width
                elif truth < lower:
                    score += width + 2 * (lower-truth)/alpha
                else:
                    score += width + 2 * (truth-upper)/alpha
            score_ls.append(score)
        return(score_ls)

    '''
        All together
    '''

    def run_experiments(self, alpha, stride, data_name, itrial, true_Y_predict=[], get_plots=False, none_CP=False, methods=['Ensemble', 'ICP', 'Weighted_ICP'], smallT=False, non_EnbPI_online=False):
        '''
            NOTE: I added a "true_Y_predict" option, which will be used for calibrating coverage under missing data
            In particular, this is needed when the Y_predict we use for training is NOT the same as true Y_predict
            Update:
                smallT: bool, denotes whether we use ALL past LOO residuals or just a small set. Used in quickest detection (see Sec.6, Fig.7 in the paper)
        '''
        train_size = len(self.X_train)
        np.random.seed(98765+itrial)
        if none_CP:
            results = pd.DataFrame(columns=['itrial', 'dataname',
                                            'method', 'train_size', 'coverage', 'width'])
            print('Not using Conformal Prediction Methods')
            save_name = {'ARIMA(10,1,10)': 'ARIMA',
                         'ExpSmoothing': 'ExpSmoothing',
                         'DynamicFactor': 'DynamicFactor'}
            PIs = []
            for name in save_name.keys():
                print(f'Running {name}')
                PI_res = self.compute_PIs_tseries_online(alpha, name=name)
                if ('Solar' in data_name) | ('Wind' in data_name):
                    PI_res['lower'] = np.maximum(PI_res['lower'], 0)
                coverage_res = ((np.array(PI_res['lower']) <= self.Y_predict) & (
                    np.array(PI_res['upper']) >= self.Y_predict)).mean()
                print(f'Average Coverage is {coverage_res}')
                width_res = (PI_res['upper'] - PI_res['lower']).mean()
                print(f'Average Width is {width_res}')
                results.loc[len(results)] = [itrial, data_name, save_name[name],
                                             train_size, coverage_res, width_res]
                PIs.append(PI_res)
        else:
            results = pd.DataFrame(columns=['itrial', 'dataname', 'muh_fun',
                                            'method', 'train_size', 'coverage', 'width'])
            PIs = []
            for method in methods:
                print(f'Runnning {method}')
                if method == 'JaB':
                    PI = self.compute_PIs_JaB(alpha)
                elif method == 'Ensemble':
                    PI = eval(f'compute_PIs_{method}_online({alpha},{stride},{smallT})',
                              globals(), {k: getattr(self, k) for k in dir(self)})
                else:
                    l = math.ceil(0.5*len(self.X_train))
                    if non_EnbPI_online:
                        PI = eval(f'compute_PIs_{method}_online({alpha},{l},{stride})',
                                  globals(), {k: getattr(self, k) for k in dir(self)})
                    else:
                        PI = eval(f'compute_PIs_{method}({alpha},{l})',
                                  globals(), {k: getattr(self, k) for k in dir(self)})
                if ('Solar' in data_name) | ('Wind' in data_name):
                    PI['lower'] = np.maximum(PI['lower'], 0)
                PIs.append(PI)
                coverage = ((np.array(PI['lower']) <= self.Y_predict) & (
                    np.array(PI['upper']) >= self.Y_predict)).mean()
                if len(true_Y_predict) > 0:
                    coverage = ((np.array(PI['lower']) <= true_Y_predict) & (
                        np.array(PI['upper']) >= true_Y_predict)).mean()
                print(f'Average Coverage is {coverage}')
                width = (PI['upper'] - PI['lower']).mean()
                print(f'Average Width is {width}')
                results.loc[len(results)] = [itrial, data_name,
                                             self.regressor.__class__.__name__, method, train_size, coverage, width]
        if get_plots:
            PIs.append(results)
            return(PIs)
        else:
            return(results)


class QOOB_or_adaptive_CI():
    '''
        Implementation of the QOOB method (Gupta et al., 2021) or the adaptive CI (Gibbs et al., 2022)
    '''

    def __init__(self, fit_func, X_train, X_predict, Y_train, Y_predict):
        self.regressor = fit_func
        self.X_train = X_train
        self.X_predict = X_predict
        self.Y_train = Y_train
        self.Y_predict = Y_predict
    ##############################
    # First on QOOB

    def fit_bootstrap_agg_get_lower_upper(self, B, beta_quantiles):
        '''
          Train B bootstrap estimators from subsets of (X_train, Y_train), compute aggregated predictors, compute scors r_i(X_i,Y_i), and finally get the intervals [l_i(X_n+j),u_i(X_n+j)] for each LOO predictor and the jth prediction in test sample
        '''
        n = len(self.X_train)
        n1 = len(self.X_predict)
        # hold indices of training data for each f^b
        boot_samples_idx = util.generate_bootstrap_samples(n, n, B)
        # hold lower and upper quantile predictions from each f^b
        boot_predictions_lower = np.zeros((B, (n+n1)), dtype=float)
        boot_predictions_upper = np.zeros((B, (n+n1)), dtype=float)
        # for i^th column, it shows which f^b uses i in training (so exclude in aggregation)
        in_boot_sample = np.zeros((B, n), dtype=bool)
        out_sample_predict_lower = np.zeros((n, n1))
        out_sample_predict_upper = np.zeros((n, n1))
        start = time.time()
        for b in range(B):
            # Fit quantile random forests
            model = self.regressor
            model = model.fit(self.X_train[boot_samples_idx[b], :],
                              self.Y_train[boot_samples_idx[b], ])
            pred_boot = model.predict_quantiles(
                np.r_[self.X_train, self.X_predict], quantiles=beta_quantiles)
            boot_predictions_lower[b] = pred_boot[:, 0]
            boot_predictions_upper[b] = pred_boot[:, 1]
            in_boot_sample[b, boot_samples_idx[b]] = True
        print(
            f'Finish Fitting B Bootstrap models, took {time.time()-start} secs.')
        start = time.time()
        self.QOOB_rXY = []  # the non-conformity scores
        for i in range(n):
            b_keep = np.argwhere(~(in_boot_sample[:, i])).reshape(-1)
            if(len(b_keep) > 0):
                # NOTE: Append these training centers too see their magnitude
                # The reason is sometimes they are TOO close to actual Y.
                quantile_lower = boot_predictions_lower[b_keep, i].mean()
                quantile_upper = boot_predictions_upper[b_keep, i].mean()
                out_sample_predict_lower[i] = boot_predictions_lower[b_keep, n:].mean(
                    0)
                out_sample_predict_upper[i] = boot_predictions_upper[b_keep, n:].mean(
                    0)
            else:  # if aggregating an empty set of models, predict zero everywhere
                print(f'No bootstrap estimator for {i}th LOO estimator')
                quantile_lower = np.percentile(
                    self.Y_train, beta_quantiles[0]*100)
                quantile_upper = np.percentile(
                    self.Y_train, beta_quantiles[1]*100)
                out_sample_predict_lower[i] = np.repeat(quantile_lower, n1)
                out_sample_predict_upper[i] = np.repeat(quantile_upper, n1)
            self.QOOB_rXY.append(self.get_rXY(
                self.Y_train[i], quantile_lower, quantile_upper))
        # print('Finish Computing QOOB training' +
        #       r'$\{r_i(X_i,Y_i)\}_{i=1}^N$'+f', took {time.time()-start} secs.')
        # Finally, subtract/add the QOOB_rXY from the predictions
        self.QOOB_rXY = np.array(self.QOOB_rXY)
        out_sample_predict_lower = (
            out_sample_predict_lower.transpose()-self.QOOB_rXY).transpose()
        out_sample_predict_upper = (
            out_sample_predict_upper.transpose()+self.QOOB_rXY).transpose()
        F_minus_i_out_sample = np.r_[
            out_sample_predict_lower, out_sample_predict_upper]
        return F_minus_i_out_sample  # Matrix of shape 2n-by-n1

    def compute_QOOB_intervals(self, data_name, itrial, B, alpha=0.1, get_plots=False):
        results = pd.DataFrame(columns=['itrial', 'dataname', 'muh_fun',
                                        'method', 'train_size', 'coverage', 'width'])
        beta_quantiles = [alpha*2, 1-alpha*2]
        # beta_quantiles = [alpha/2, 1-alpha/2]  # Even make thresholds smaller, still not good
        F_minus_i_out_sample = self.fit_bootstrap_agg_get_lower_upper(
            B, beta_quantiles)
        n1 = F_minus_i_out_sample.shape[1]
        PIs = []
        for i in range(n1):
            curr_lower_upper = F_minus_i_out_sample[:, i]
            # print(f'Test point {i}')
            PIs.append(self.get_lower_upper_n_plus_i(curr_lower_upper, alpha))
        PIs = pd.DataFrame(PIs, columns=['lower', 'upper'])
        self.PIs = PIs
        if 'Solar' in data_name:
            PIs['lower'] = np.maximum(PIs['lower'], 0)
        coverage, width = util.ave_cov_width(PIs, self.Y_predict)
        results.loc[len(results)] = [itrial, data_name,
                                     self.regressor.__class__.__name__, 'QOOB', self.X_train.shape[0], coverage, width]
        if get_plots:
            return [PIs, results]
        else:
            return results
    # QOOB helpers

    def get_rXY(self, Ytrain_i, quantile_lower, quantile_upper):
        # Get r_i(X_i,Y_i) as in Eq. (2) of QOOB
        if Ytrain_i < quantile_lower:
            return quantile_lower-Ytrain_i
        elif Ytrain_i > quantile_upper:
            return Ytrain_i-quantile_upper  # There was a small error here
        else:
            return 0

    # AdaptCI helpers
    def get_Ei(self, Ytrain_i, quantile_lower, quantile_upper):
        return np.max([quantile_lower-Ytrain_i, Ytrain_i-quantile_upper])

    def get_lower_upper_n_plus_i(self, curr_lower_upper, alpha):
        # This implements Algorithm 1 of QOOB
        # See https://github.com/AIgen/QOOB/blob/master/MATLAB/methods/QOOB_interval.m for matlab implementation
        n2 = len(curr_lower_upper)
        n = int(n2/2)
        S_ls = np.r_[np.repeat(1, n), np.repeat(0, n)]
        idx_sort = np.argsort(curr_lower_upper)  # smallest to larget
        S_ls = S_ls[idx_sort]
        curr_lower_upper = curr_lower_upper[idx_sort]
        count = 0
        lower_i = np.inf
        upper_i = -np.inf
        threshold = alpha*(n+1)-1
        for i in range(n2):
            if S_ls[i] == 1:
                count += 1
                if count > threshold and count-1 <= threshold and lower_i == np.inf:
                    lower_i = curr_lower_upper[i]
                    # print(f'QOOB lower_end {lower_i}')
            else:
                if count > threshold and count-1 <= threshold and upper_i == -np.inf:
                    upper_i = curr_lower_upper[i]
                    # print(f'QOOB upper_end {upper_i}')
                count -= 1
        return [lower_i, upper_i]

    ##############################
    # Next on AdaptiveCI

    def compute_AdaptiveCI_intervals(self, data_name, itrial, l, alpha=0.1, get_plots=False):
        results = pd.DataFrame(columns=['itrial', 'dataname', 'muh_fun',
                                        'method', 'train_size', 'coverage', 'width'])
        n = len(self.X_train)
        proper_train = np.arange(l)
        X_train = self.X_train[proper_train, :]
        Y_train = self.Y_train[proper_train]
        X_calibrate = np.delete(self.X_train, proper_train, axis=0)
        Y_calibrate = np.delete(self.Y_train, proper_train)
        # NOTE: below works when the model can takes in MULTIPLE quantiles together (e.g., the RangerForest)
        model = self.regressor
        model = model.fit(X_train, Y_train)
        quantile_pred = model.predict_quantiles(
            np.r_[X_calibrate, self.X_predict], quantiles=[alpha/2, 1-alpha/2])
        # NOTE: below works for sklearn linear quantile: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html#sklearn.linear_model.QuantileRegressor
        # # In particular, it is much slower than the quantile RF with similar results
        # model_l, model_u = self.regressor
        # qpred_l, qpred_u = model_l.fit(X_train, Y_train).predict(np.r_[X_calibrate, self.X_predict]), model_u.fit(
        #     X_train, Y_train).predict(np.r_[X_calibrate, self.X_predict])
        # quantile_pred = np.c_[qpred_l, qpred_u]
        self.quantile_pred = quantile_pred
        Dcal_pred = quantile_pred[:n-l]
        Test_pred = quantile_pred[n-l:]
        # TODO: I guess I can use the QOOB idea, by using "get_rXY"
        Dcal_scores = np.array([self.get_Ei(Ycal, quantile_lower, quantile_upper) for Ycal,
                                quantile_lower, quantile_upper in zip(Y_calibrate, Dcal_pred[:, 0], Dcal_pred[:, 1])])
        self.Escore = Dcal_scores
        # Sequentially get the intervals with adaptive alpha
        alpha_t = alpha
        errs = []
        gamma = 0.005
        method = 'simple'  # 'simple' or 'complex'
        PIs = []
        self.alphas = [alpha_t]
        for t, preds in enumerate(Test_pred):
            lower_pred, upper_pred = preds
            width = np.percentile(Dcal_scores, 100*(1-alpha_t))
            # print(f'At test time {t}')
            # print(f'alpha={alpha_t} & width={width}')
            lower_t, upper_t = lower_pred-width, upper_pred+width
            PIs.append([lower_t, upper_t])
            # Check coverage and update alpha_t
            Y_t = self.Y_predict[t]
            err = 1 if Y_t < lower_t or Y_t > upper_t else 0
            errs.append(err)
            alpha_t = util.adjust_alpha_t(alpha_t, alpha, errs, gamma, method)
            if alpha_t > 1:
                alpha_t = 1
            if alpha_t < 0:
                alpha_t = 0
            self.alphas.append(alpha_t)
        PIs = pd.DataFrame(PIs, columns=['lower', 'upper'])
        if 'Solar' in data_name:
            PIs['lower'] = np.maximum(PIs['lower'], 0)
        self.errs = errs
        self.PIs = PIs
        coverage, width = util.ave_cov_width(PIs, self.Y_predict)
        results.loc[len(results)] = [itrial, data_name,
                                     self.regressor.__class__.__name__, 'Adaptive_CI', self.X_train.shape[0], coverage, width]
        if get_plots:
            return [PIs, results]
        else:
            return results
        # TODO: I guess I can use the QOOB idea, by using "get_rXY"
        Dcal_scores = np.array([self.get_Ei(Ycal, quantile_lower, quantile_upper) for Ycal,
                                quantile_lower, quantile_upper in zip(Y_calibrate, Dcal_pred[:, 0], Dcal_pred[:, 1])])
        self.Escore = Dcal_scores
        # Sequentially get the intervals with adaptive alpha
        alpha_t = alpha
        errs = []
        gamma = 0.005
        method = 'simple'  # 'simple' or 'complex'
        PIs = []
        self.alphas = [alpha_t]
        for t, preds in enumerate(Test_pred):
            lower_pred, upper_pred = preds
            width = np.percentile(Dcal_scores, 100*(1-alpha_t))
            # print(f'At test time {t}')
            # print(f'alpha={alpha_t} & width={width}')
            lower_t, upper_t = lower_pred-width, upper_pred+width
            PIs.append([lower_t, upper_t])
            # Check coverage and update alpha_t
            Y_t = self.Y_predict[t]
            err = 1 if Y_t < lower_t or Y_t > upper_t else 0
            errs.append(err)
            alpha_t = util.adjust_alpha_t(alpha_t, alpha, errs, gamma, method)
            if alpha_t > 1:
                alpha_t = 1
            if alpha_t < 0:
                alpha_t = 0
            self.alphas.append(alpha_t)
        PIs = pd.DataFrame(PIs, columns=['lower', 'upper'])
        if 'Solar' in data_name:
            PIs['lower'] = np.maximum(PIs['lower'], 0)
        self.errs = errs
        self.PIs = PIs
        coverage, width = util.ave_cov_width(PIs, self.Y_predict)
        results.loc[len(results)] = [itrial, data_name,
                                     self.regressor.__class__.__name__, 'Adaptive_CI', self.X_train.shape[0], coverage, width]
        if get_plots:
            return [PIs, results]
        else:
            return results
