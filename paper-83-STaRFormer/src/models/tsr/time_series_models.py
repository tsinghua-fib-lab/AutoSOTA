import numpy as np


class TimeSeriesRegressor:
    """
    This is a super class for time series regressors
    """

    def __init__(self,
                 output_directory: str):
        """
        Initialise the regression model
        """
        self.output_directory = output_directory
        self.train_duration = None
        self.name = "TimeSeriesRegressor"
        pass

    def fit(self,
            x_train: np.array,
            y_train: np.array,
            x_val: np.array = None,
            y_val: np.array = None):
        """
        Fit the regression model
        """
        pass

    def predict(self, x: np.array):
        """
        Do prediction using the regression model on x
        """
        pass
