import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, check_is_fitted, clone
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from utils import knockoff_threshold
from scipy.stats import wilcoxon
import random
from scipy.stats import binomtest

class Semi_KO(BaseEstimator):
    """
    Semi-Knockoffs algorithm.
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
        imputation_model_y,
        loss: callable = mean_squared_error,
        score_proba: bool = False,
        random_state: int = None,
        n_jobs: int = 1,
    ):

        check_is_fitted(estimator)
        self.estimator = estimator
        self.imputation_model = imputation_model
        self.imputation_model_y = imputation_model_y
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
        if isinstance(self.imputation_model_y, list):
            self._list_imputation_models_y = self.imputation_model_y
        else:
            self._list_imputation_models_y = []

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
        if len(self._list_imputation_models_y) == 0:
            self._list_imputation_models_y = [
                clone(self.imputation_model_y) for _ in range(self.n_groups)
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
        
        def _joblib_fit_one_group_y(estimator, X, y, j):
            """
            Fit a single covariate estimator to predict a single group of
            covariates.
            """
            if isinstance(X, pd.DataFrame):
                X_j = X[self.groups[j]].copy().values
                X_minus_j = X.drop(columns=self.groups[j]).copy()
                # append y as a new column
                X_minus_j["y"] = y
                X_minus_j = X_minus_j.values
            else:
                X_j = X[:, self.groups[j]].copy()
                if X_j.shape[1] == 1:
                    X_j = X_j.ravel()
                X_minus_j = np.delete(X, self.groups[j], axis=1)
                # append y as last column
                X_minus_j = np.column_stack((X_minus_j, y))
            estimator.fit(X_minus_j, X_j)
            return estimator
        

        # Parallelize the fitting of the covariate estimators
        self._list_imputation_models = Parallel(n_jobs=self.n_jobs)(
            delayed(_joblib_fit_one_group)(estimator, X, y, j)
            for j, estimator in zip(self.groups.keys(), self._list_imputation_models)
        )

        self._list_imputation_models_y = Parallel(n_jobs=self.n_jobs)(
            delayed(_joblib_fit_one_group_y)(estimator, X, y, j)
            for j, estimator in zip(self.groups.keys(), self._list_imputation_models_y)
        )

        return self

    def predict(self, X, y, n_perm):
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

        def _joblib_predict_one_group(imputation_model, X, j, n_perm):
            """
            Compute n_perm predictions of the model with the permuted data for a
            single group of covariates.
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

            for _ in range(n_perm):
                y_pred_perm = np.zeros(X.shape[0])
                X_j_perm = X_j_hat + self.rng.permutation(residual_j)
                X_perm = np.empty_like(X)
                X_perm[:, non_group_ids] = X_minus_j
                X_perm[:, group_ids] = X_j_perm
                if isinstance(X, pd.DataFrame):
                    X_perm = pd.DataFrame(X_perm, columns=X.columns)

                if self.score_proba:
                    y_pred_perm += self.estimator.predict_proba(X_perm)
                else:
                    y_pred_perm += self.estimator.predict(X_perm)
                list_y_pred_perm.append(y_pred_perm)

            return np.array(list_y_pred_perm)
        
        def _joblib_predict_one_group_y(imputation_model, X, y, j, n_perm):
            """
            Compute n_perm predictions of the model with the permuted data for a
            single group of covariates.
            """
            list_y_pred_perm = []

            if isinstance(X, pd.DataFrame):
                X_j = X[self.groups[j]].copy().values
                X_minus_j = X.drop(columns=self.groups[j]).copy()
                X_minus_j["y"] = y
                X_minus_j = X_minus_j.values
                group_ids = [
                    i for i, col in enumerate(X.columns) if col in self.groups[j]
                ]
                non_group_ids = [
                    i for i, col in enumerate(X.columns) if col not in self.groups[j]
                ]
            else:
                X_j = X[:, self.groups[j]].copy()
                X_minus_j = np.delete(X, self.groups[j], axis=1)
                X_minus_j_y = np.column_stack((X_minus_j, y))
                group_ids = self.groups[j]
                non_group_ids = np.delete(np.arange(X.shape[1]), group_ids)
            X_j_hat = imputation_model.predict(X_minus_j_y).reshape(X_j.shape)
            residual_j = X_j - X_j_hat

            for _ in range(n_perm):
                y_pred_perm = np.zeros(X.shape[0])
                X_j_perm = X_j_hat + self.rng.permutation(residual_j)
                X_perm = np.empty_like(X)
                X_perm[:, non_group_ids] = X_minus_j
                X_perm[:, group_ids] = X_j_perm
                if isinstance(X, pd.DataFrame):
                    X_perm = pd.DataFrame(X_perm, columns=X.columns)

                if self.score_proba:
                    y_pred_perm += self.estimator.predict_proba(X_perm)
                else:
                    y_pred_perm += self.estimator.predict(X_perm)
                list_y_pred_perm.append(y_pred_perm)

            return np.array(list_y_pred_perm)


     # Parallelize the computation of the importance scores for each group
        out_list = Parallel(n_jobs=self.n_jobs)(
            delayed(_joblib_predict_one_group)(imputation_model, X, j, n_perm)
            for j, imputation_model in zip(
                self.groups.keys(), self._list_imputation_models
            )
        )

        residual_permuted_y_pred = np.stack(out_list, axis=0)

        out_list_y = Parallel(n_jobs=self.n_jobs)(
            delayed(_joblib_predict_one_group_y)(imputation_model, X, y, j, n_perm)
            for j, imputation_model in zip(
                self.groups.keys(), self._list_imputation_models_y
            )
        )

        residual_permuted_y_pred_y = np.stack(out_list_y, axis=0)

        return residual_permuted_y_pred, residual_permuted_y_pred_y

    def score(self, X, y, p_val='wilcox', n_perm=1, fdr=0.1):
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


        y_pred_perm, y_pred_perm_y = self.predict(X, y, n_perm)

        out_dict["loss_perm"] = dict()
        out_dict["loss_perm_y"] = dict()
        for j, (y_pred_j, y_pred_y_j) in enumerate(zip(y_pred_perm, y_pred_perm_y)):
            # Vectorized loss computation: (n_perm, n_samples) - (n_samples,) -> (n_perm, n_samples)
            # For MSE loss: (y_pred - y)^2, flattened to (n_perm * n_samples,)
            if self.loss is mean_squared_error:
                loss_perm_j = (y_pred_j - y) ** 2
                loss_perm_y_j = (y_pred_y_j - y) ** 2
                out_dict["loss_perm"][j] = loss_perm_j.ravel()
                out_dict["loss_perm_y"][j] = loss_perm_y_j.ravel()
            else:
                list_loss_perm = []
                list_loss_perm_y = []
                for y_pred_perm_j, y_pred_perm_y_j in zip(y_pred_j, y_pred_y_j):
                    for n_t in range(y_pred_perm_j.shape[0]):
                        list_loss_perm.append(self.loss(y_true=np.array([y[n_t]]), y_pred=np.array([y_pred_perm_j[n_t]])))
                        list_loss_perm_y.append(self.loss(y_true=np.array([y[n_t]]), y_pred=np.array([y_pred_perm_y_j[n_t]])))
                out_dict["loss_perm"][j] = np.array(list_loss_perm)
                out_dict["loss_perm_y"][j] = np.array(list_loss_perm_y)



        out_dict["importance"] = np.array(
            [
                (np.mean(out_dict["loss_perm"][j])-np.mean(out_dict["loss_perm_y"][j]))
                for j in range(self.n_groups)
            ]
        )
       
        if p_val=='wilcox':
            p_value_=np.zeros(X.shape[1])
            for j in range(X.shape[1]):
                _, p_value_[j] = wilcoxon(out_dict["loss_perm"][j], out_dict["loss_perm_y"][j], alternative="greater")
            out_dict['pval']=p_value_
        elif p_val=='sign_test':
            p_value_=np.zeros(X.shape[1])
            for j in range(X.shape[1]):
                stat_sign_test = binomtest(np.sum( out_dict["loss_perm"][j] > out_dict["loss_perm_y"][j]), out_dict["loss_perm_y"][j].shape[0], alternative="greater")
                p_value_[j]=stat_sign_test.pvalue
            out_dict['pval']=p_value_
        elif p_val=='FDR':
            ko_thr = knockoff_threshold(out_dict["importance"], fdr=fdr)
            out_dict['selected']= np.where(out_dict["importance"] >= ko_thr)[0]
            

        return out_dict