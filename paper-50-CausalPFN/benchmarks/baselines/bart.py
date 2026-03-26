import numpy as np
import pandas as pd

try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr

except ImportError:
    print("rpy2 not installed, skipping BART baseline.")

from .base import BaselineModel


class BartBaseline(BaselineModel):
    def __init__(self, hpo: bool = True):
        super().__init__(hpo)

        # Activate automatic conversion between pandas and R dataframes
        pandas2ri.activate()

        # Import necessary R packages
        self.base = importr("base")
        self.stats = importr("stats")
        self.bartCause = importr("bartCause")
        if hpo:
            raise ValueError("Hyperparameter optimization is not supported for BART.")

    def estimate_cate(
        self,
        X_train: np.ndarray,
        t_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ):
        """
        Estimate CATE using R's BART package directly from Python using rpy2.

        Args:
            X_train: Training features
            t_train: Binary treatment indicators for training data
            y_train: Outcome values for training data
            X_test: Test features for which to predict CATE
        Returns:
            cate_estimates: Estimated CATE values
        """
        X_train = pd.DataFrame(X_train, columns=[f"X{i+1}" for i in range(X_train.shape[1])])
        X_test = pd.DataFrame(X_test, columns=[f"X{i+1}" for i in range(X_test.shape[1])])

        # standardize the features
        X_mean, X_std = X_train.mean(), X_train.std() + 1e-8
        X_train = (X_train - X_mean) / X_std
        X_test = (X_test - X_mean) / X_std

        y_mean, y_std = y_train.mean(), y_train.std() + 1e-8
        y_train = (y_train - y_mean) / y_std

        # Convert data
        r_X_train = pandas2ri.py2rpy(X_train)
        r_treatment = ro.FloatVector(t_train)
        r_outcome = ro.FloatVector(y_train)
        r_X_test = pandas2ri.py2rpy(X_test)

        # Set seed for reproducibility
        ro.r(f"set.seed(1234)")

        # Fit BART model
        bartc_fit = self.bartCause.bartc(
            response=r_outcome,
            treatment=r_treatment,
            confounders=r_X_train,
            method_rsp=ro.StrVector(["bart"]),
            method_trt=ro.StrVector(["bart"]),
            estimand=ro.StrVector(["ate"]),
            p_scoreAsCovariate=ro.BoolVector([True]),  # includes propensity score as a covariate
            commonSup_rule=ro.StrVector(["sd"]),  # to deal with lack of common support
            args_rsp=ro.ListVector(
                {
                    "n.chains": ro.IntVector([10]),
                    "n.trees": ro.IntVector([400]),  # default is 200
                    "keepTrees": ro.BoolVector([True]),
                }
            ),
            args_trt=ro.ListVector(
                {
                    "n.chains": ro.IntVector([10]),
                    "n.trees": ro.IntVector([200]),  # default is 200
                    "keepTrees": ro.BoolVector([True]),
                }
            ),
            verbose=ro.BoolVector([False]),
            seed=ro.IntVector([42]),
            crossvalidate=ro.BoolVector([False]),  # NOTE: Setting this to True makes training extremely slow
        )

        # Create test data frame
        r_test_data = ro.r["data.frame"](r_X_test)

        # Predict on test data
        predictions = ro.r["predict"](bartc_fit, r_test_data, type=ro.StrVector(["icate"]))

        return np.array(predictions) * y_std

    def estimate_ate(
        self,
        X: np.ndarray,
        t: np.ndarray,
        y: np.ndarray,
    ):
        """
        Estimate ATE using R's BART package directly from Python using rpy2.
        Args:
            X: Features
            t: Binary treatment indicators
            y: Outcome values
        Returns:
            ate_estimate: Estimated ATE value
        """
        X = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(X.shape[1])])

        # standardize the features
        X_mean, X_std = X.mean(), X.std() + 1e-8
        X = (X - X_mean) / X_std

        y_mean, y_std = y.mean(), y.std() + 1e-8
        y = (y - y_mean) / y_std

        # Convert data
        r_X_train = pandas2ri.py2rpy(X)
        r_treatment = ro.FloatVector(t)
        r_outcome = ro.FloatVector(y)

        # Set seed for reproducibility
        ro.r(f"set.seed(1234)")

        # Fit BART model
        bartc_fit = self.bartCause.bartc(
            response=r_outcome,
            treatment=r_treatment,
            confounders=r_X_train,
            method_rsp=ro.StrVector(["bart"]),
            method_trt=ro.StrVector(["bart"]),
            estimand=ro.StrVector(["ate"]),
            p_scoreAsCovariate=ro.BoolVector([True]),  # includes propensity score as a covariate
            commonSup_rule=ro.StrVector(["sd"]),  # to deal with lack of common support
            args_rsp=ro.ListVector(
                {
                    "n.chains": ro.IntVector([10]),
                    "n.trees": ro.IntVector([400]),  # default is 200
                    "keepTrees": ro.BoolVector([True]),
                }
            ),
            args_trt=ro.ListVector(
                {
                    "n.chains": ro.IntVector([10]),
                    "n.trees": ro.IntVector([200]),  # default is 200
                    "keepTrees": ro.BoolVector([True]),
                }
            ),
            verbose=ro.BoolVector([False]),
            seed=ro.IntVector([42]),
            crossvalidate=ro.BoolVector([False]),  # NOTE: Setting this to True makes training extremely slow
        )

        # Predict on test data
        ate = np.mean(ro.r["extract"](bartc_fit, type=ro.StrVector(["pate"])))

        return ate * y_std
