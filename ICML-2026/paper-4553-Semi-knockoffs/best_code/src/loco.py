import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, check_is_fitted, clone
from sklearn.metrics import root_mean_squared_error
from scipy.stats import norm
from utils import bootstrap_var
from scipy.stats import wilcoxon
from scipy.stats import binomtest

class LOCO(BaseEstimator):
    """
    Leave-One-Covariate-Out (LOCO) algorithm as described in
    :footcite:t:


    Parameters
    ----------
    estimator: scikit-learn compatible estimator
        The predictive model.
    loss: callable, default=root_mean_squared_error
        Loss function to evaluate the model performance.
    method: str, default='predict'
        Method to use for predicting values that will be used to compute
        the loss and the importance scores. The method must be implemented by the
        estimator. Supported methods are 'predict', 'predict_proba',
        'decision_function' and 'transform'.
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
        loss: callable = root_mean_squared_error,
        method: str = "predict",
        random_state: int = None,
        n_jobs: int = 1,
    ):

        check_is_fitted(estimator)
        self.estimator = estimator
        self.random_state = random_state
        self.loss = loss
        self.method = method
        self.n_jobs = n_jobs
        self.rng = np.random.RandomState(random_state)
        self._list_estimators = []

    def fit(self, X, y, groups=None):
        """
        Fit the estimators on each subset of covariates.

        Parameters
        ----------
        X: np.ndarray of shape (n_samples, n_features)
            The input samples. If groups is provided, the columns must correspond to
            the values of the groups dictionary.
        y: np.ndarray of shape (n_samples,)
            The target values used to fit the sub-models.
        groups: dict, default=None
            Dictionary of groups for the covariates. The keys are the group names
            and the values are lists of covariate indices.
        """
        self.groups = groups
        if self.groups is None:
            self.n_groups = X.shape[1]
            self.groups = {j: [j] for j in range(self.n_groups)}
        else:
            self.n_groups = len(self.groups)
        # create a list of covariate estimators for each group if not provided

        self._list_estimators = [clone(self.estimator) for _ in range(self.n_groups)]

        def _joblib_fit_one_group(estimator, X, y, j):
            """
            Fit a single model on a subset of covariates.
            """
            if isinstance(X, pd.DataFrame):
                X_minus_j = X.drop(columns=self.groups[j])
            else:
                X_minus_j = np.delete(X, self.groups[j], axis=1)
            estimator.fit(X_minus_j, y)
            return estimator

        # Parallelize the fitting of the covariate estimators
        self._list_estimators = Parallel(n_jobs=self.n_jobs)(
            delayed(_joblib_fit_one_group)(estimator, X, y, j)
            for j, estimator in zip(self.groups.keys(), self._list_estimators)
        )

        return self

    def predict(self, X, y):
        """
        Compute the prediction from each subset of covariates using the fitted
        sub-models.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The input samples.
        y: array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        out_list: array-like of shape (n_samples, n_features)
            Predicted outputs without the covariate.
        """
        check_is_fitted(self.estimator)
        if len(self._list_estimators) == 0:
            raise ValueError("fit must be called before predict")
        for m in self._list_estimators:
            check_is_fitted(m)


        def _joblib_predict_one_group(estimator_j, X, y, j):
            """
            Compute the importance score for a single group of covariates
            removed.
            """
            if isinstance(X, pd.DataFrame):
                X_minus_j = X.drop(columns=self.groups[j])
            else:
                X_minus_j = np.delete(X, self.groups[j], axis=1)

            y_pred_loco = getattr(estimator_j, self.method)(X_minus_j)

            return y_pred_loco

        # Parallelize the computation of the importance scores for each group
        out_list = Parallel(n_jobs=self.n_jobs)(
            delayed(_joblib_predict_one_group)(estimator_j, X, y, j)
            for j, estimator_j in zip(self.groups.keys(), self._list_estimators)
        )

        return np.stack(out_list, axis=0)

    def score(self, X, y, p_val='corrected_sqrt', bootstrap=False):
        """
        Compute the importance scores for each group of covariates.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The input samples.
        y: array-like of shape (n_samples,)
            The target values.
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
            - 'importance': the importance scores for each group.
            - 'std': the standard deviation of the LOCO estimate.
            - 'pval': p-values
        """
        check_is_fitted(self.estimator)
        if len(self._list_estimators) == 0:
            raise ValueError("fit must be called before predict")
        for m in self._list_estimators:
            check_is_fitted(m)

        out_dict = dict()
        y_pred = getattr(self.estimator, self.method)(X)

        loss_reference = self.loss(y_true=y, y_pred=y_pred)
        out_dict["loss_reference"] = loss_reference
        loss_coord_by_coord=[]
        for n_t in range(y.shape[0]):
            loss_coord_by_coord.append(self.loss(y_true=np.array([y[n_t]]), y_pred=np.array([y_pred[n_t]])))
        loss_coord_by_coord=np.array(loss_coord_by_coord)
        y_pred_loco = self.predict(X, y)

        out_dict["loss_loco"] = np.array(
            [self.loss(y_true=y, y_pred=y_pred_loco[j]) for j in range(self.n_groups)]
        )

        out_dict["importance"] = out_dict["loss_loco"] - loss_reference
        loss_coord_j = np.zeros((X.shape[1],y.shape[0]))

        out_dict["loss_std"] = dict()
        for j, y_pred_j in enumerate(y_pred_loco):
            inter_loss = []
            for n_t in range(y.shape[0]):
                loss_coord_j[j, n_t]=self.loss(y_true=np.array([y[n_t]]), y_pred=np.array([y_pred_j[n_t]]))
                inter_loss.append(loss_coord_j[j, n_t]-loss_coord_by_coord[n_t]) 
            if bootstrap:
                out_dict["loss_std"][j] = (bootstrap_var(inter_loss, len(inter_loss), len(inter_loss)))
            else:
                out_dict["loss_std"][j]=(np.std(inter_loss)/ np.sqrt(y.shape[0]))

        out_dict["std"] = np.array(
            [
                (out_dict["loss_std"][j])
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