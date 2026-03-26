import numpy as np
import pandas as pd

try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
except ImportError:
    print("rpy2 not installed, skipping GRF baseline.")

from .base import BaselineModel


class GRFBaseline(BaselineModel):
    def __init__(self, hpo: bool = True):
        super().__init__(hpo)

        # Activate automatic conversion between pandas and R dataframes
        pandas2ri.activate()

        # Import necessary R packages
        self.base = importr("base")
        self.grf = importr("grf")
        self.stats = importr("stats")

    def estimate_cate(
        self,
        X_train: np.ndarray,
        t_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ):
        """
        Estimate CATE using R's grf package directly from Python using rpy2.

        Args:
            X_train: Training features
            t_train: Binary treatment indicators for training data
            y_train: Outcome values for training data
            X_test: Test features for which to predict CATE
        Returns:
            cate_estimates: Estimated CATE values
        """
        # Convert numpy arrays to pandas DataFrames
        X_train = pd.DataFrame(X_train, columns=[f"X{i+1}" for i in range(X_train.shape[1])])
        X_test = pd.DataFrame(X_test, columns=[f"X{i+1}" for i in range(X_test.shape[1])])

        # standardize the features
        X_mean, X_std = X_train.mean(), X_train.std() + 1e-8
        X_train = (X_train - X_mean) / X_std
        X_test = (X_test - X_mean) / X_std

        # standardize the outcomes
        y_mean, y_std = y_train.mean(), y_train.std() + 1e-8
        y_train = (y_train - y_mean) / y_std

        # Convert pandas DataFrames to R dataframes
        r_X_train = pandas2ri.py2rpy(X_train)
        r_X_test = pandas2ri.py2rpy(X_test)

        # Convert treatment and outcome to R vectors
        r_W_train = ro.FloatVector(t_train)
        r_Y_train = ro.FloatVector(y_train)

        # Set tuning parameters
        r_tune_params = ro.StrVector(["all"]) if self.hpo else ro.StrVector(["none"])
        r_num_trees = ro.IntVector([4000])
        r_honesty = ro.BoolVector([True])
        r_seed = ro.IntVector([42])

        # Call R's causal_forest
        r_cf = self.grf.causal_forest(
            X=r_X_train,
            Y=r_Y_train,
            W=r_W_train,
            num_trees=r_num_trees,
            tune_parameters=r_tune_params,
            honesty=r_honesty,
            seed=r_seed,
        )

        # Predict CATE on test set
        r_predictions = self.grf.predict_causal_forest(r_cf, r_X_test)

        # Convert R predictions to numpy arrays
        cate_estimates = np.array(r_predictions.rx2("predictions"))

        return cate_estimates * y_std

    def estimate_ate(
        self,
        X: np.ndarray,
        t: np.ndarray,
        y: np.ndarray,
    ):
        """
        Estimate ATE using R's grf package directly from Python using rpy2.
        Args:
            X: Features
            t: Binary treatment indicators
            y: Outcome values
        Returns:
            ate_estimate: Estimated ATE value
        """
        # Convert numpy arrays to pandas DataFrames
        X = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(X.shape[1])])

        # standardize the features
        X_mean, X_std = X.mean(), X.std() + 1e-8
        X = (X - X_mean) / X_std

        # standardize the outcomes
        y_mean, y_std = y.mean(), y.std() + 1e-8
        y = (y - y_mean) / y_std

        # Convert pandas DataFrames to R dataframes
        r_X_train = pandas2ri.py2rpy(X)

        # Convert treatment and outcome to R vectors
        r_W_train = ro.FloatVector(t)
        r_Y_train = ro.FloatVector(y)

        # Set tuning parameters
        r_tune_params = ro.StrVector(["all"])
        r_num_trees = ro.IntVector([4000])
        r_honesty = ro.BoolVector([True])
        r_seed = ro.IntVector([42])

        # Call R's causal_forest
        r_cf = self.grf.causal_forest(
            X=r_X_train,
            Y=r_Y_train,
            W=r_W_train,
            num_trees=r_num_trees,
            tune_parameters=r_tune_params,
            honesty=r_honesty,
            seed=r_seed,
        )

        r_ate = self.grf.average_treatment_effect(r_cf)
        ate_estimate = np.array(r_ate)[0]

        return ate_estimate * y_std
