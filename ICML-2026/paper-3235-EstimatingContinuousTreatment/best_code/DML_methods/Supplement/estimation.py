# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 21:04:04 2020
Last update Sunday Jan 27 11:30 am 2025

This file provides the main double debiased machine learning estimator for
continuous treatments. The class "DDMLCT" performs the estimation when the
.fit method is called.

DDMLCT is initialized by passing 2 models (such as sklearn models) which have
.fit and .predict methods. One for estimating the generalized propensity score (GPS)
and one for estimation gamma. model1 is used for estimating gamma and model2 is
used for estimation of the GPS

DDMLCT saves computation time by fitting models for each value of t once for
each fold of cross-fititng. Since the K Neural Network in Colangelo and Lee (2025)
uses a unique loss which depends on t, we had to define another class "NN_DDMLCT"
which accounts for this and fits the models for every t. Other than this adjustment,
both classes are nearly identical.

Comments on packages used:
    -copy is used to make copies of the models used to initiate DDMLCT.
    -pandas is used during rescaling in order to rescale non-dummies
    -scipy.stats.norm is used for the computation of the gaussian kernel
    -numpy is used for the storage of most of the data and attributes. If data is
     passed to the .fit method as a pandas dataframe it is converted to numpy arrays
     before being passed to the models to fit.

"""

import numpy as np
from scipy.stats import norm
import pandas as pd
import copy
import sklearn
from scipy.optimize import minimize
from scipy.stats import norm
import random
import tqdm


# This function evaluates the gaussian kernel wtih bandwidth h at point x
def gaussian_kernel(x, h):
    k = (1 / h) * norm.pdf((x / h), 0, 1)
    return k


# This function evaluates the epanechnikov kernel
def e_kernel(x, h):
    k = (1 / h) * (3 / 4) * (1 - ((x / h) ** 2))
    k = k * (abs(x / h) <= 1)
    return k


# The class which defines the proposed estimator.
class DDMLCT:
    """
    DDMLCT is initialized by passing 2 models (such as sklearn models) which have
    .fit and .predict methods. One for estimating the generalized propensity score (GPS)
    and one for estimation gamma. model1 is used for estimating gamma and model2 is
    used for estimation of the GPS

    Parameters
    ----------

    model1: Machine learning model for the estimation of gamma
    model2: Machine learning model for the estiamtion of the generalized propensity score

    Attributes
    ----------
    beta: An array containing all of the estimates for the dose-response function for each t
    std_errors: An array containing all the standard errors corresponding to beta

    Methods
    -------
    .fit: the only method that should be called by a user. Given covariates X,
    outcome Y, treatment T, list of values of t to evaluate the estimator at,
    number of sub-samples for cross-fitting L, bandwidth choice h, whether to
    add basis functions and whether to standardize.
    """

    def __init__(self, model1, model2):
        self.model1 = copy.deepcopy(model1)
        self.model2 = copy.deepcopy(model2)
        self.beta = np.array(())
        self.std_errors = np.array(())
        self.Vt = np.array(())
        self.Bt = np.array(())
        self.summary = None
        self.scaling = {"mean_Y": 0, "sd_Y": 1, "mean_T": 0, "sd_T": 1}
        self.naive_count = 0
        self.L = 5
        self.gamma_models = []

    # Reset is called everytime the fit method is called so that .fit can be
    # called multiple times on the same object with potentially overlapping
    # values of t.
    def reset(self):
        self.beta = np.array(())
        self.std_errors = np.array(())
        self.Vt = np.array(())
        self.Bt = np.array(())
        self.summary = None
        self.scaling = {"mean_Y": 0, "sd_Y": 1, "mean_T": 0, "sd_T": 1}
        self.naive_count = 0
        self.L = 5

    # Define the "Naive" machine learning estimator based only on the estimation
    # of gamma. This function is only fit once for each subsample, then the fitted
    # models are used repeatedly for the other values of t to increase computational
    # efficiency. The "naive_count" variable keeps track of this. Once it is incremented
    # to the level of L, then this indicates the model has already been fit for each
    # subsample.
    def naive(self, Xf, XT, Xt, Y, I, I_C, L):
        if self.naive_count < self.L:
            self.gamma_models.append(
                self.model1.fit(np.column_stack((XT[I_C], Xf[I_C])), Y[I_C])
            )
            self.naive_count += 1
        gamma = self.gamma_models[L].predict(np.column_stack((Xt[I], Xf[I])))
        return gamma

    # The function to estimate the GPS. g is the kernel smoothing function to
    # be modeled.
    def ipw(self, Xf, g, I, I_C):
        self.model2.fit(Xf[I_C], g[I_C])
        gps = self.model2.predict(Xf[I])

        return gps

    # For a given t, this function is called to estimate for each individual sub-sample
    # from cross-fitting. If L=5 then this function is called 5 times for each t.
    def fit_L(self, Xf, XT, Xt, Y, g, K, I, I_C, L):
        gamma = self.naive(Xf, XT, Xt, Y, I, I_C, L)
        gps = self.ipw(Xf, g, I, I_C)
        self.kept = np.concatenate((self.kept, I))

        # Compute the summand
        psi = np.mean(gamma) + np.mean(((Y[I] - gamma) * (K[I] / gps)))

        # Average over all indexes to get an estimate of beta hat
        beta_hat = np.mean(psi)

        return beta_hat, gamma, gps

    # This function is called for every t in the list of t given to the fit
    # function. trep is the value of t repeated n times. XT is the matrix of
    # X,T and the added basis function which may be functions of T. Xt is this
    # same matrix evaluated at the given value of t. Xf is the matrix to be used
    # for the GPS estimation. If added basis functions are used this is different
    # from X, otherwise Xf = X. It has to be given separately because otherwise
    # it can't be determined which variables depend on T and which don't.
    def fit_t(self, Xf, T, Y, trep, L, XT, Xt, trep_sdml=None, sdml=False):

        self.kept = np.array(
            (), dtype=int
        )  # used for trimming which is not currently implemented

        T_t = T - trep

        g = gaussian_kernel(T_t, self.h)
        K = e_kernel(T_t, self.h)
        gamma = np.zeros(self.n)
        gps = np.zeros(self.n)
        beta_hat = np.zeros(L)

        # Iterate over all L sub-samples. I_split was defined in the fit function
        # so that the same split is used for all choice of t.
        for i in range(L):
            if L == 1:
                I = self.I_split[0]
                I_C = self.I_split[0]
            else:
                I = self.I_split[i]
                # Define the complement as the union of all other sets
                I_C = [x for x in np.arange(self.n) if x not in I]

            beta_hat[i], gamma[I], gps[I] = self.fit_L(Xf, XT, Xt, Y, g, K, I, I_C, i)

        # We now average over all sub-samples to get our estimates and standard
        # errors.
        self.n = len(self.kept)
        beta_hat = np.mean(beta_hat)
        self.beta = np.append(self.beta, beta_hat)
        IF = (
            (K[self.kept] / gps[self.kept]) * (Y[self.kept] - gamma[self.kept])
            + gamma[self.kept]
            - beta_hat
        )
        std_error = np.sqrt((1 / ((self.n) ** 2)) * np.sum(IF**2))
        self.Bt = np.append(
            self.Bt,
            (1 / (self.n * (self.h**2)))
            * (
                np.sum(
                    (K[self.kept] / gps[self.kept]) * (Y[self.kept] - gamma[self.kept])
                )
            ),
        )
        self.Vt = np.append(self.Vt, (std_error**2) * (self.n * self.h))
        self.std_errors = np.append(self.std_errors, std_error)
        self.gps.loc[self.kept, str(trep[0])] = gps[self.kept]

    # The only function that a user should be calling. If basis functions are
    # used, or the data is asked to be standardized, then that is performed in this
    # function. sdml argument tells whether we wish to use the simulated
    # dml version of the estimator.
    def fit(
        self, X, T, Y, t_list, L=5, h=None, basis=False, standardize=False, sdml=False
    ):
        self.reset()
        self.naive_count = 0
        self.n = len(Y)
        self.t_list = np.array(t_list, ndmin=1)
        self.L = L
        self.I_split = np.array_split(np.array(range(self.n)), L)

        # If no bandwidth is specified, use rule of thumb
        if h == None:
            self.h = np.std(T) * (self.n**-0.2)
        else:
            self.h = h

        X, T, Y, t_list = self.reformat(X, T, Y, t_list, standardize)

        self.gps = pd.DataFrame(index=range(self.n))
        if basis == True:
            XT, Xf, ind = self.augment(X, T)
            if standardize == True:
                Xf = self.scale_non_dummies(Xf)[0]
                XT, scaler = self.scale_non_dummies(XT)
        else:
            XT = T
            Xf = X

        for t in tqdm.tqdm(np.array((t_list), ndmin=1)):
            self.n = len(Y)
            trep = np.repeat(t, self.n)
            trep_sdml = np.array(random.choices(np.array(T) * self.h + t, k=self.n))
            if sdml == True:
                if basis == True:
                    Xt = self.augment(X, trep_sdml, ind)[0]
                    if standardize == True:
                        Xt = self.scale_non_dummies(Xt, scaler)[0]
                else:
                    Xt = trep_sdml
            else:
                if basis == True:
                    Xt = self.augment(X, trep, ind)[0]
                    if standardize == True:
                        Xt = self.scale_non_dummies(Xt, scaler)[0]
                else:
                    Xt = trep
            self.fit_t(Xf, T, Y, trep, L, XT, Xt, trep_sdml, sdml)

        self.h_star = ((np.mean(self.Vt) / (4 * (np.mean(self.Bt**2)))) ** 0.2) * (
            self.n**-0.2
        )

        if standardize == True:
            self.descale()

        self.gps.columns = self.t_list

    # This function augments the data with added basis functions. Xf is the matrix
    # X with added basis functions that only depend on X, for use in estimation the
    # GPS. XT is the matrix of X and T and added basis functions which may depend
    # on T themselves. Xt is the matrix XT but evaluated at T=t.
    def augment(self, X, T, ind=None):
        T = T.reshape(len(T), 1)
        XT = np.column_stack((T, (T**2), (T**3), T * X))
        Xf = np.column_stack((X, X**2, X**3))
        Xf = np.unique(Xf, axis=1)
        if np.array_equal(ind, None):
            XT, ind = np.unique(XT, axis=1, return_index=True)
        else:
            XT = XT[:, ind]
        return XT, Xf, ind

    # This function is used to scale, but only non-dummy variables are
    # re-scaled.
    def scale_non_dummies(self, D, scaler=None):
        D = pd.DataFrame(D)
        if scaler == None:
            scaler = sklearn.preprocessing.StandardScaler()
            D[D.select_dtypes("float64").columns] = scaler.fit_transform(
                D.select_dtypes("float64")
            )
        else:
            D[D.select_dtypes("float64").columns] = (
                D[D.select_dtypes("float64").columns] - scaler.mean_
            ) / scaler.scale_
        return np.array(D), scaler

    # This function makes sure all the data and inputs are in the right format
    # before fitting. The data is scaled
    def reformat(self, X, T, Y, t_list, standardize):
        if standardize == True:
            df = pd.DataFrame(data=np.column_stack((Y, T, X)))
            self.scaling = {
                "mean_Y": np.mean(df[0]),
                "sd_Y": np.std(df[0]),
                "mean_T": np.mean(df[1]),
                "sd_T": np.std(df[1]),
            }
            # Only standardize float64 columns if they exist
            float64_cols = df.select_dtypes("float64").columns
            if len(float64_cols) > 0:
                df[float64_cols] = sklearn.preprocessing.StandardScaler().fit_transform(
                    df[float64_cols]
                )

            Y = df[0]
            T = df[1]
            X = df.loc[:, 2:]
            del df
            t_list = (t_list - self.scaling["mean_T"]) / self.scaling["sd_T"]
            self.h = self.h / self.scaling["sd_T"]
        X = np.array((X))
        T = np.array((T))
        Y = np.array((Y))
        return X, T, Y, t_list

    # This function is used at the end of estimation to convert estimates into
    # numbers that are interpretable based on the scale of the original data-set
    def descale(self):
        self.std_errors = self.std_errors * self.scaling["sd_Y"]
        self.h_star = self.h_star * self.scaling["sd_T"]
        self.beta = (self.beta * self.scaling["sd_Y"]) + self.scaling["mean_Y"]
        self.h = self.h * self.scaling["sd_T"]
        self.Vt = self.Vt * (self.scaling["sd_Y"] ** 2)


class NN_DDMLCT(DDMLCT):
    # Similar to DDMLCT "naive" function but adjusted to fit for every t.
    def naive(self, Xf, XT, Xt, Y, I, I_C, L, K):
        gamma_model = self.model1.fit(Xf[I_C], Y[I_C], K[I_C])
        gamma = gamma_model.predict(Xf[I])
        return gamma

    # The function to estimate the GPS. g is the kernel smoothing function to
    # be modeled.
    def ipw(self, Xf, g, I, I_C):
        self.model2.fit(Xf[I_C], g[I_C])
        gps = self.model2.predict(Xf[I])

        return gps

    # For a given t, this function is called to estimate for each individual sub-sample
    # from cross-fitting. If L=5 then this function is called 5 times for each t.
    def fit_L(self, Xf, XT, Xt, Y, g, K, I, I_C, L):
        gamma = self.naive(Xf, XT, Xt, Y, I, I_C, L, K)
        gps = self.ipw(Xf, g, I, I_C)
        self.kept = np.concatenate((self.kept, I))

        # Compute the summand
        psi = np.mean(gamma) + np.mean(((Y[I] - gamma) * (K[I] / gps)))

        # Average over all indexes to get an estimate of beta hat
        beta_hat = np.mean(psi)

        return beta_hat, gamma, gps

    # This function is called for every t in the list of t given to the fit
    # function. trep is the value of t repeated n times. XT is the matrix of
    # X,T and the added basis function which may be functions of T. Xt is this
    # same matrix evaluated at the given value of t. Xf is the matrix to be used
        # for the GPS estimation. If added basis functions are used this is different
    # from X, otherwise Xf = X. It has to be given separately because otherwise
    # it can't be determined which variables depend on T and which don't.
    def fit_t(self, Xf, T, Y, trep, L, XT, Xt, trep_sdml=None, sdml=False):
        self.kept = np.array(
            (), dtype=int
        )  # used for trimming which is not currently implemented

        T_t = T - trep
        g = gaussian_kernel(T_t, self.h)
        K = e_kernel(T_t, self.h)  #
        gamma = np.zeros(self.n)
        gps = np.zeros(self.n)
        beta_hat = np.zeros(L)

        # Iterate over all L sub-samples. I_split was defined in the fit function
        # so that the same split is used for all choice of t.
        for i in range(L):
            if L == 1:
                I = self.I_split[0]
                I_C = self.I_split[0]
            else:
                I = self.I_split[i]
                # Define the complement as the union of all other sets
                I_C = [x for x in np.arange(self.n) if x not in I]

            beta_hat[i], gamma[I], gps[I] = self.fit_L(Xf, XT, Xt, Y, g, K, I, I_C, i)

        # We now average over all sub-samples to get our estimates and standard
        # errors.
        self.n = len(self.kept)
        beta_hat = np.mean(beta_hat)
        self.beta = np.append(self.beta, beta_hat)
        IF = (
            (K[self.kept] / gps[self.kept]) * (Y[self.kept] - gamma[self.kept])
            + gamma[self.kept]
            - beta_hat
        )
        std_error = np.sqrt((1 / ((self.n) ** 2)) * np.sum(IF**2))
        self.Bt = np.append(
            self.Bt,
            (1 / (self.n * (self.h**2)))
            * (
                np.sum(
                    (K[self.kept] / gps[self.kept]) * (Y[self.kept] - gamma[self.kept])
                )
            ),
        )
        self.Vt = np.append(self.Vt, (std_error**2) * (self.n * self.h))
        self.std_errors = np.append(self.std_errors, std_error)
        self.gps.loc[self.kept, str(trep[0])] = gps[self.kept]


# This class is used to compute the DDMLCT estimator with alternative GPS estimation.
# we currently do not implement this for our numerical results due to computational
# infeasibility.
# optimization for later: Prevent computing of all g's every calling of ipw
# allow user input of t_grid and epsilon.
class DDMLCT_gps2(DDMLCT):
    def ipw(self, Xf, g, T, t, I, I_C):
        epsilon = 0.025
        t_grid = np.arange(t - 1.5 * self.h, t + 1.5 * self.h, self.h / 50)
        self.model2.fit(Xf[I_C], g[I_C])
        cdf_hat = self.model2.predict(Xf[I])
        cdf_hat = cdf_hat.reshape((len(cdf_hat), 1))
        cdf_hats = np.zeros((len(I), len(t_grid)))
        for i in range(len(t_grid)):
            trep = np.repeat(t_grid[i], self.n)
            t_T = trep - T
            g = norm.cdf(t_T / self.h)
            self.model2.fit(Xf[I_C], g[I_C])
            cdf_hats[:, i] = self.model2.predict(Xf[I])

        df = pd.DataFrame(cdf_hats)
        lower = df[df > (cdf_hat - epsilon)].T.apply(pd.Series.first_valid_index)

        lower = lower.fillna(0)
        lower = lower.astype(int)
        lower = lower.values.tolist()

        df = pd.DataFrame(cdf_hats)
        upper = df[df > (cdf_hat + epsilon)].T.apply(pd.Series.first_valid_index)

        upper = upper.fillna(len(t_grid) - 1)
        upper = upper.astype(int)
        upper = upper.values.tolist()

        t_matrix = np.repeat(np.array(t_grid, ndmin=2), len(I), axis=0)

        t_upper = t_matrix[np.arange(len(t_matrix)), upper]
        t_lower = t_matrix[np.arange(len(t_matrix)), lower]

        inverse_gps = (t_upper - t_lower) / (2 * epsilon)

        gps = 1 / inverse_gps
        return gps

    def fit_L(self, Xf, XT, Xt, Y, g, T, K, I, I_C, L, t):
        gamma = self.naive(Xf, XT, Xt, Y, I, I_C, L)
        gps = self.ipw(Xf, g, T, t, I, I_C)
        self.kept = np.concatenate((self.kept, I))

        # Compute the summand
        psi = np.mean(gamma) + np.mean(((Y[I] - gamma) * (K[I] / gps)))

        # Average over all indexes to get an estimate of beta hat
        beta_hat = np.mean(psi)

        return beta_hat, gamma, gps

    def fit_t(self, Xf, T, Y, trep, L, XT, Xt, trep_sdml=None, sdml=False):

        self.kept = np.array(
            (), dtype=int
        )  # used for trimming which is not currently implemented

        T_t = T - trep
        t_T = trep - T
        g = norm.cdf(t_T / self.h)
        K = e_kernel(T_t, self.h)
        gamma = np.zeros(self.n)
        gps = np.zeros(self.n)
        beta_hat = np.zeros(L)

        # Iterate over all L sub-samples. I_split was defined in the fit function
        # so that the same split is used for all choice of t.
        for i in range(L):
            if L == 1:
                I = self.I_split[0]
                I_C = self.I_split[0]
            else:
                I = self.I_split[i]
                # Define the complement as the union of all other sets
                I_C = [x for x in np.arange(self.n) if x not in I]

            beta_hat[i], gamma[I], gps[I] = self.fit_L(
                Xf, XT, Xt, Y, g, T, K, I, I_C, i, trep[0]
            )

        # We now average over all sub-samples to get our estimates and standard
        # errors.
        self.n = len(self.kept)
        beta_hat = np.mean(beta_hat)
        self.beta = np.append(self.beta, beta_hat)
        IF = (
            (K[self.kept] / gps[self.kept]) * (Y[self.kept] - gamma[self.kept])
            + gamma[self.kept]
            - beta_hat
        )
        std_error = np.sqrt((1 / ((self.n) ** 2)) * np.sum(IF**2))
        self.Bt = np.append(
            self.Bt,
            (1 / (self.n * (self.h**2)))
            * (
                np.sum(
                    (K[self.kept] / gps[self.kept]) * (Y[self.kept] - gamma[self.kept])
                )
            ),
        )
        self.Vt = np.append(self.Vt, (std_error**2) * (self.n * self.h))
        self.std_errors = np.append(self.std_errors, std_error)
        self.gps.loc[self.kept, str(trep[0])] = gps[self.kept]
