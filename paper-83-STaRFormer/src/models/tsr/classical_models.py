import time
import numpy as np

from scipy.stats import loguniform
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from utils.tools import save_test_duration, save_train_duration


class Regressor:
    """
    This is a super class for general machine learning regressors.
    It takes features extracted from the time series as input and passed to a regression model.
    """

    def __init__(self, output_directory: str):
        """
        Initialise the regression model
        """
        self.name = "Regressor"
        self.output_directory = output_directory
        self.train_duration = None
        self.model = None
        self.params = None

    def summary(self):
        """
        Provide a summary of the model
        """
        print("{}()".format(self.name, self.params))

    def fit(self,
            x_train: np.array,
            y_train: np.array):
        """
        Fit the regression model

        Inputs:
            x_train: training data (num_examples, num_timestep, num_channels) or (num_examples, num_features)
            y_train: training target
        """
        print("[{}] Fitting the regressor with data of {}".format(self.name, x_train.shape))

        start_time = time.perf_counter()

        if len(x_train.shape) == 3:
            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])

        self.cv(x_train, y_train)

        self.model.fit(x_train, y_train)

        self.train_duration = time.perf_counter() - start_time

        save_train_duration(self.output_directory + 'train_duration.csv', self.train_duration)

        print("[{}] Fitting completed, took {}s".format(self.name, self.train_duration))

    def predict(self, x: np.array):
        """
        Do prediction using the regression model on x

        Inputs:
            x: data for prediction (num_examples, num_timestep, num_channels) or (num_examples, num_features)
        """
        print("[{}] Predicting data of {}".format(self.name, x.shape))

        start_time = time.perf_counter()

        if len(x.shape) == 3:
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])

        y_pred = self.model.predict(x)

        test_duration = time.perf_counter() - start_time

        save_test_duration(self.output_directory + 'test_duration.csv', test_duration)

        print("[{}] Predicting completed, took {}s".format(self.name, test_duration))

        return y_pred

    def cv(self, x_train: np.array, y_train: np.array):
        """
        Do cross validation to optimise hyperparameters of the model
        """
        pass


class SVRRegressor(Regressor):
    """
    This is a wrapper for SupportVectorRegression (SVR) model.
    """

    def __init__(self,
                 output_directory: str,
                 verbose: int = 0,
                 model_params=None):
        """
        Initialise the SVR model

        Inputs:
            output_directory: path to store results/models
            verbose: verbosity
            model_params: parameters for SVR in dictionary format
        """

        super().__init__(output_directory)
        self.name = "SVR"

        if model_params is None:
            model_params = {
                "kernel": "rbf",
                "gamma": "scale",
                "verbose": verbose
            }
        self.params = model_params
        self.build_model(**model_params)

        return

    def build_model(self, **model_params):
        """
        Build the SVR model

        Inputs:
            model_params: parameters for SVR in dictionary format
        Outputs:
            model: initialised model
        """
        from sklearn.svm import SVR

        self.model = SVR(**model_params)
        return self.model

    def cv(self,
           x_train: np.array,
           y_train: np.array,
           x_val: np.array = None,
           y_val: np.array = None,
           hpo: str = "random"):
        """
        Cross validation on SVR

        Inputs:
            x_train: training data (num_examples, num_features)
            y_train: training target
            x_val: validation data (num_examples, num_features)
            y_val: validation target
            hpo: hyper parameter optimisation option
        """
        from sklearn.svm import SVR

        params = self.params
        params["verbose"] = 0

        best_param = None
        best_param_score = None
        if hpo == "grid":
            search_params = [{"kernel": ["rbf"],
                              "gamma": [0.001, 0.01, 0.1, 1],
                              "C": [0.1, 1, 10, 100]},
                             {"kernel": ["sigmoid"],
                              "gamma": [0.001, 0.01, 0.1, 1],
                              "C": [0.1, 1, 10, 100]}]

            search = GridSearchCV(SVR(**params), search_params, n_jobs=-1, cv=3,
                                  scoring="neg_mean_squared_error", verbose=1)
            search.fit(x_train, y_train)

            best_param = search.best_params_
            best_param_score = search.best_score_
        elif hpo == "random":
            search_params = {"kernel": ["rbf", "sigmoid", "linear"],
                             "gamma": loguniform(0.001, 1),
                             "C": loguniform(0.1, 100)}
            search = RandomizedSearchCV(SVR(**params), search_params, n_jobs=-1, cv=3,
                                        random_state=1234,
                                        scoring="neg_mean_squared_error", verbose=1)
            search.fit(x_train, y_train)
            best_param = search.best_params_
            best_param_score = search.best_score_

        if best_param is not None:
            print("Best Param: {}, with scores: {}".format(best_param, best_param_score))
            self.params.update(best_param)
            self.build_model(**self.params)


class RFRegressor(Regressor):
    """
    This is a wrapper for RandomForest model.
    """

    def __init__(self,
                 output_directory: str,
                 verbose: int = 0,
                 model_params=None):
        """
        Initialise the RF model

        Inputs:
            output_directory: path to store results/models
            verbose: verbosity
            model_params: parameters for SVR in dictionary format
        """

        super().__init__(output_directory)
        self.name = "RandomForest"

        if model_params is None:
            model_params = {
                "n_estimators": 100,
                "n_jobs": -1,
                "verbose": verbose
            }

        self.params = model_params
        self.build_model(**model_params)

        return

    def build_model(self, **model_params):
        """
        Build the RF model

        Inputs:
            model_params: parameters for SVR in dictionary format
        Outputs:
            model: initialised model
        """

        from sklearn.ensemble import RandomForestRegressor

        self.model = RandomForestRegressor(**model_params)
        return self.model

    def cv(self,
           x_train: np.array,
           y_train: np.array,
           x_val: np.array = None,
           y_val: np.array = None,
           hpo: str = "random"):
        """
        Cross validation on RF

        Inputs:
            x_train: training data (num_examples, num_features)
            y_train: training target
            x_val: validation data (num_examples, num_features)
            y_val: validation target
            hpo: hyper parameter optimisation option
        """
        from sklearn.ensemble import RandomForestRegressor

        params = self.params
        params["verbose"] = 0

        best_param = None
        best_param_score = None
        if hpo == "grid":
            search_params = {
                "n_estimators": [100, 500, 1000],
                "max_depth": [5, 10, 15, 20],
                "min_samples_leaf": [1, 5, 10, 15]
            }
            search = GridSearchCV(RandomForestRegressor(**params), search_params,
                                  n_jobs=1, cv=3,
                                  scoring="neg_mean_squared_error", verbose=True)
            search.fit(x_train, y_train)
            best_param = search.best_params_
            best_param_score = search.best_score_
        elif hpo == "random":
            search_params = {
                "n_estimators": [100, 500, 1000],
                "max_depth": [5, 10, 15, 20],
                "min_samples_leaf": [1, 5, 10, 15]
            }
            search = RandomizedSearchCV(RandomForestRegressor(**params), search_params,
                                        n_jobs=1, cv=3,
                                        random_state=1234,
                                        scoring="neg_mean_squared_error", verbose=True)
            search.fit(x_train, y_train)
            best_param = search.best_params_
            best_param_score = search.best_score_

        if best_param is not None:
            print("Best Param: {}, with scores: {}".format(best_param, best_param_score))
            self.params.update(best_param)
            self.params["n_jobs"] = -1
            self.build_model(**self.params)


class XGBoostRegressor(Regressor):
    """
    This is a wrapper for XGBoost.
    """

    def __init__(self, output_directory, verbose=0, model_params=None):
        """
        Initialise the XGBoost model

        Inputs:
            output_directory: path to store results/models
            verbose: verbosity
            model_params: parameters for SVR in dictionary format
        """
        super().__init__(output_directory)
        self.name = "XGBoost"

        if model_params is None:
            model_params = {
                "n_estimators": 100,
                "n_jobs": 0,
                "verbosity": verbose
            }
        self.params = model_params
        self.build_model(**model_params)

        return

    def cv(self,
           x_train: np.array,
           y_train: np.array,
           x_val: np.array = None,
           y_val: np.array = None,
           hpo: str = "random"):
        """
        Cross validation on XGBoost

        Inputs:
            x_train: training data (num_examples, num_features)
            y_train: training target
            x_val: validation data (num_examples, num_features)
            y_val: validation target
            hpo: hyper parameter optimisation option
        """
        from xgboost import XGBRegressor

        params = self.params
        params["verbosity"] = 0
        if x_val is not None:
            fit_params = {"early_stopping_rounds": 20,
                          "eval_set": [(x_val, y_val)],
                          "verbose": False}
        else:
            fit_params = {"verbose": False}

        best_param = None
        best_param_score = None
        if hpo == "grid":
            search_params = {
                "n_estimators": [100, 500, 1000],
                "max_depth": [5, 10, 15, 20],
                "learning_rate": [0.1, 0.05, 0.01]
            }
            search = GridSearchCV(XGBRegressor(**params), search_params, n_jobs=1, cv=3,
                                  scoring="neg_mean_squared_error", verbose=True)
            search.fit(x_train, y_train, **fit_params)
            best_param = search.best_params_
            best_param_score = search.best_score_
        elif hpo == "random":
            search_params = {
                "n_estimators": [100, 500, 1000],
                "max_depth": [5, 10, 15, 20],
                "learning_rate": loguniform(loc=0.01, scale=0.1)
            }
            search = RandomizedSearchCV(XGBRegressor(**params), search_params, n_jobs=1, cv=3,
                                        random_state=1234,
                                        scoring="neg_mean_squared_error", verbose=True)
            search.fit(x_train, y_train, **fit_params)
            best_param = search.best_params_
            best_param_score = search.best_score_

        if best_param is not None:
            print("Best Param: {}, with scores: {}".format(best_param, best_param_score))
            self.params.update(best_param)
            self.params["n_jobs"] = 0
            self.build_model(**self.params)

    def build_model(self, **model_params):
        """
        Build the XGBoost model

        Inputs:
            model_params: parameters for SVR in dictionary format
        Outputs:
            model: initialised model
        """
        from xgboost import XGBRegressor

        self.model = XGBRegressor(**model_params)

        return self.model


class LinearRegressor(Regressor):
    """
    This is a wrapper for Linear models.
    """

    def __init__(self,
                 output_directory: str,
                 model_params,
                 type: str = "lr"):
        """
        Initialise the Linear model

        Inputs:
            output_directory: path to store results/models
            model_params: parameters for SVR in dictionary format
            type: type of linear model, lr or ridge
        """

        super().__init__(output_directory)
        self.name = type.upper()
        self.params = model_params
        self.build_model(**model_params)

        return

    def build_model(self, **model_params):
        """
        Build the Linear model

        Inputs:
            model_params: parameters for SVR in dictionary format
        Outputs:
            model: initialised model
        """
        if self.name == "RIDGE":
            from sklearn.linear_model import RidgeCV
            self.model = RidgeCV(**model_params)
        else:
            from sklearn.linear_model import LinearRegression
            self.model = LinearRegression(**model_params)
        return self.model

    def fit(self,
            x_train: np.array,
            y_train: np.array):
        """
        Fit Linear model

        Inputs:
            x_train: training data (num_examples, num_timestep, num_channels) or (num_examples, num_features)
            y_train: training target
        """
        print("[{}] Fitting the regressor with data of {}".format(self.name, x_train.shape))

        start_time = time.perf_counter()

        if len(x_train.shape) == 3:
            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])

        self.cv(x_train, y_train)

        self.model.fit(x_train, y_train)

        self.train_duration = time.perf_counter() - start_time

        save_train_duration(self.output_directory + 'train_duration.csv', self.train_duration)

        print("[{}] Fitting completed, took {}s".format(self.name, self.train_duration))

    def predict(self, x: np.array):
        """
        Do prediction with Rocket

        Inputs:
            x: data for prediction (num_examples, num_timestep, num_channels) or (num_examples, num_features)
        Outputs:
            y_pred: prediction
        """

        print("[{}] Predicting data of {}".format(self.name, x.shape))

        start_time = time.perf_counter()

        if len(x.shape) == 3:
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])

        y_pred = self.model.predict(x)

        test_duration = time.perf_counter() - start_time

        save_test_duration(self.output_directory + 'test_duration.csv', test_duration)

        print("[{}] Predicting completed, took {}s".format(self.name, test_duration))

        return y_pred
