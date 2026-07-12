import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, check_is_fitted, clone
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from utils import bootstrap_var
from scipy.stats import wilcoxon
import random
from scipy.stats import binomtest

class Sobol_CPI(BaseEstimator):
    """
    Sobol Conditional Permutation Importance (CPI) algorithm.
    :footcite:t:`

    Parameters
    ----------
    estimator: scikit-learn compatible estimator
        The predictive model.
    imputation_model: scikit-learn compatible estimator or list of estimators
        The model(s) used to estimate the covariates. If a single estimator is
        provided, it will be cloned for each covariate group. Otherwise, a list of
        potentially different estimators can be provided, the length of the
        list must match the number of covariate groups.
    n_permutations: int, default=50
        Number of permutations to perform.
    loss: callable, default=root_mean_squared_error
        Loss function to evaluate the model performance.
    score_proba: bool, default=False
        Whether to use the predict_proba method of the estimator.
    random_state: int, default=None
        Random seed for the permutation.
    n_jobs: int, default=1
        Number of jobs to run in parallel.

    References
    ----------
    .. footbibliography::
    
    """

    def __init__(
        self,
        estimator,
        imputation_model,
        n_permutations: int = 1,
        loss: callable = mean_squared_error,
        score_proba: bool = False,
        random_state: int = None,
        n_jobs: int = 1,
    ):

        check_is_fitted(estimator)
        self.estimator = estimator
        self.imputation_model = imputation_model

        self.n_permutations = n_permutations
        self.random_state = random_state
        self.loss = loss
        self.score_proba = score_proba
        self.n_jobs = n_jobs
        self.rng = np.random.RandomState(random_state)

    def fit(self, X, y=None, groups=None):
        """
        Fit the covariate estimators to predict each group of covariates from
        the others.

        Parameters
        ----------
        X: np.ndarray of shape (n_samples, n_features)
            The input samples. If groups is provided, the columns must correspond to
            the values of the groups dictionary.
        y: np.ndarray of shape (n_samples,)
            The target values. Not used in the fitting of the covariate estimators.
        groups: dict, default=None
            Dictionary of groups for the covariates. The keys are the group names
            and the values are lists of covariate indices.
        """
        self.groups = groups
        if isinstance(self.imputation_model, list):
            self._list_imputation_models = self.imputation_model
        else:
            self._list_imputation_models = []

        if self.groups is None:
            self.n_groups = X.shape[1]
            self.groups = {j: [j] for j in range(self.n_groups)}
        else:
            self.n_groups = len(self.groups)
        # create a list of covariate estimators for each group if not provided
        if len(self._list_imputation_models) == 0:
            self._list_imputation_models = [
                clone(self.imputation_model) for _ in range(self.n_groups)
            ]

        def _joblib_fit_one_group(estimator, X, y, j):
            """
            Fit a single covariate estimator to predict a single group of
            covariates.
            """
            if isinstance(X, pd.DataFrame):
                X_j = X[self.groups[j]].copy().values
                X_minus_j = X.drop(columns=self.groups[j]).values
            else:
                X_j = X[:, self.groups[j]].copy()
                if X_j.shape[1]==1:
                    X_j = X_j.ravel()
                X_minus_j = np.delete(X, self.groups[j], axis=1)
            estimator.fit(X_minus_j, X_j)
            return estimator

        # Parallelize the fitting of the covariate estimators
        self._list_imputation_models = Parallel(n_jobs=self.n_jobs)(
            delayed(_joblib_fit_one_group)(estimator, X, y, j)
            for j, estimator in zip(self.groups.keys(), self._list_imputation_models)
        )

        return self

    def predict(self, X, y=None, n_cal=10):
        """
        Compute the prediction of the model with perturbed data for each group.
        For each group of covariates, the residuals are computed using the
        covariate estimators. The residuals are then permuted and the model is
        re-evaluated n_cal times. Then, the mean is taken over the predictions.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The input samples.
        y: array-like of shape (n_samples,)
            The target values.
        n_cal: int, default=10
            Number of perturbed predictions for each sample.

        Returns
        -------
        residual_permuted_y_pred: np.ndarray of shape (n_groups, n_permutations, n_samples)
            The predictions of the model averaged over n_cal conditional permutation for each group
        """
        if len(self._list_imputation_models) == 0:
            raise ValueError("fit must be called before predict")
        for m in self._list_imputation_models:
            check_is_fitted(m)

        def _joblib_predict_one_group(imputation_model, X, j, n_cal=10):
            """
            Compute n_cal predictions of the model with the permuted data for a
            single group of covariates and then averages them.
            """
            list_y_pred_perm = []
            if isinstance(X, pd.DataFrame):
                X_j = X[self.groups[j]].copy().values
                X_minus_j = X.drop(columns=self.groups[j]).values
                group_ids = [
                    i for i, col in enumerate(X.columns) if col in self.groups[j]
                ]
                non_group_ids = [
                    i for i, col in enumerate(X.columns) if col not in self.groups[j]
                ]
            else:
                X_j = X[:, self.groups[j]].copy()
                X_minus_j = np.delete(X, self.groups[j], axis=1)
                group_ids = self.groups[j]
                non_group_ids = np.delete(np.arange(X.shape[1]), group_ids)

            X_j_hat = imputation_model.predict(X_minus_j).reshape(X_j.shape)
            residual_j = X_j - X_j_hat

            for _ in range(self.n_permutations):
                y_pred_perm = np.zeros(X.shape[0])
                for _ in range(n_cal):
                    X_j_perm = X_j_hat + self.rng.permutation(residual_j)
                    X_perm = np.empty_like(X)
                    X_perm[:, non_group_ids] = X_minus_j
                    X_perm[:, group_ids] = X_j_perm
                    if isinstance(X, pd.DataFrame):
                        X_perm = pd.DataFrame(X_perm, columns=X.columns)

                    if self.score_proba:
                        y_pred_perm += self.estimator.predict_proba(X_perm)/n_cal
                    else:
                        y_pred_perm += self.estimator.predict(X_perm)/n_cal
                list_y_pred_perm.append(y_pred_perm)

            return np.array(list_y_pred_perm)

     # Parallelize the computation of the importance scores for each group
        out_list = Parallel(n_jobs=self.n_jobs)(
            delayed(_joblib_predict_one_group)(imputation_model, X, j, n_cal)
            for j, imputation_model in zip(
                self.groups.keys(), self._list_imputation_models
            )
        )

        residual_permuted_y_pred = np.stack(out_list, axis=0)
        return residual_permuted_y_pred

    def score(self, X, y, n_cal=10, p_val='corrected_n', bootstrap=False):
        """
        Compute the importance scores and p-values for each group of covariates.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The input samples.
        y: array-like of shape (n_samples,)
            The target values.
        n_cal: int, default=10
            Calibration set size.
        p_val: {'emp_var', 'corrected_sqrt', 'corrected_n', 'corrected_sqd'}, optional
            Method used to compute the p-value:
            - 'emp_var': not corrected term, and if divides by 0 then p-value=1.
            - 'corrected_sqrt': Corrects the variance by adding var(y)/n**0.5.
            - 'corrected_n': Corrects the variance by adding var(y)/n.
            - 'corrected_sqd': Corrects the variance by adding var(y)/n**2.
        bootstrap: bool, default=False
            If True, variance estimated with bootstrap, otherwise accross individuals.

        Returns
        -------
        out_dict: dict
            A dictionary containing the following keys:
            - 'loss_reference': the loss of the model with the original data.
            - 'loss_perm': a dictionary containing the loss of the model with
            the permuted data for each group.
            - 'importance': the importance scores for each group (estimated Total Sobol Index).
            - 'pval': the p-value.
            - 'std': standard deviation.
        """
        check_is_fitted(self.estimator)
        if len(self._list_imputation_models) == 0:
            raise ValueError("fit must be called before score")
        for m in self._list_imputation_models:
            check_is_fitted(m)

        out_dict = dict()

        if self.score_proba:
            y_pred = self.estimator.predict_proba(X)
        else:
            y_pred = self.estimator.predict(X)

        loss_reference = self.loss(y_true=y, y_pred=y_pred)
        out_dict["loss_reference"] = loss_reference #original loss of the model with original inputs
        loss_coord_by_coord=[]#original loss coordinate by coordinate
        for n_t in range(y.shape[0]):
            loss_coord_by_coord.append(self.loss(y_true=np.array([y[n_t]]), y_pred=np.array([y_pred[n_t]])))
        loss_coord_by_coord=np.array(loss_coord_by_coord)
        y_pred_perm = self.predict(X, y, n_cal=n_cal)

        out_dict["loss_perm"] = dict()
        for j, y_pred_j in enumerate(y_pred_perm):
            list_loss_perm = []
            for y_pred_perm_j in y_pred_j:
                list_loss_perm.append(self.loss(y_true=y, y_pred=y_pred_perm_j))
            out_dict["loss_perm"][j] = np.array(list_loss_perm)

        out_dict["loss_std"] = dict()
        loss_coord_j = np.zeros((X.shape[1],y.shape[0]))
        for j, y_pred_j in enumerate(y_pred_perm):
            list_std_perm = []
            loss_perm=[]
            for y_pred_j_perm in y_pred_j:
                inter_loss = []
                for n_t in range(y.shape[0]):
                    loss_coord_j[j, n_t]=self.loss(y_true=np.array([y[n_t]]), y_pred=np.array([y_pred_j_perm[n_t]]))
                    inter_loss.append((loss_coord_j[j, n_t]-loss_coord_by_coord[n_t])*n_cal/(n_cal+1))           
                if bootstrap:
                    list_std_perm.append(bootstrap_var(inter_loss, len(inter_loss), len(inter_loss)))
                else:
                    list_std_perm.append(np.std(inter_loss)/ np.sqrt(y.shape[0]))
                loss_perm.extend(inter_loss)
            out_dict["loss_std"][j] = np.array(list_std_perm)


        out_dict["importance"] = np.array(
            [
                (np.mean(out_dict["loss_perm"][j])- loss_reference)*n_cal/(n_cal+1) 
                for j in range(self.n_groups)
            ]
        )
        out_dict["std"] = np.array(
            [
                (np.mean(out_dict["loss_std"][j]))
                for j in range(self.n_groups)
            ]
        )
        if p_val=='emp_var':
            out_dict['pval']=norm.sf(out_dict["importance"] / (out_dict["std"]))
            out_dict["pval"][np.isnan(out_dict["pval"])] = 1.0
        elif p_val=='corrected_n':
            out_dict["std"] += np.std(y)/y.shape[0]
            out_dict['pval']=norm.sf(out_dict["importance"] / (out_dict["std"]))
        elif p_val=='corrected_sqrt':
            out_dict["std"] += np.std(y)/np.sqrt(y.shape[0])
            out_dict['pval']=norm.sf(out_dict["importance"] / (out_dict["std"]))
        elif p_val == 'corrected_sqd':
            out_dict["std"] += np.std(y)/(y.shape[0]**2)
            out_dict['pval']=norm.sf(out_dict["importance"] / (out_dict["std"]))
        elif p_val=='wilcox':
            p_value_=np.zeros(X.shape[1])
            for j in range(X.shape[1]):
                _, p_value_[j] = wilcoxon(loss_coord_j[j], loss_coord_by_coord, alternative="greater")
            out_dict['pval']=p_value_
        elif p_val=='sign_test':
            p_value_=np.zeros(X.shape[1])
            for j in range(X.shape[1]):
                stat_sign_test = binomtest(np.sum(loss_coord_j[j] > loss_coord_by_coord), y.shape[0], alternative="greater")
                p_value_[j]=stat_sign_test.pvalue
            out_dict['pval']=p_value_

        return out_dict