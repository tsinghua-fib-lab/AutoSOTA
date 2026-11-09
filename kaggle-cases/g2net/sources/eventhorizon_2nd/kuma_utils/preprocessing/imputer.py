import numpy as np
import pandas as pd
import lightgbm as lgb

from .utils import analyze_column


class LGBMImputer:
    '''
    Regression imputer using LightGBM
    '''

    def __init__(self, cat_features=[], n_iter=100, verbose=False):
        self.n_iter = n_iter
        self.cat_features = cat_features
        self.verbose = verbose
        self.n_features = None
        self.feature_names = None
        self.feature_with_missing = None
        self.imputers = {}
        self.offsets = {}
        self.objectives = {}
        
    def fit(self, X, y=None):
        self.n_features = X.shape[1]
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = [f'f{i}' for i in range(self.n_features)]
            X = pd.DataFrame(X, columns=self.feature_names)
        self.feature_with_missing = [
            col for col in self.feature_names if X[col].isnull().sum() > 0]

        for icol, col in enumerate(self.feature_with_missing):
            if icol in self.cat_features:
                nuni = X[col].dropna().nunique()
                if nuni == 2:
                    params = {
                        'objective': 'binary'
                    }
                elif nuni > 2:
                    params = {
                        'objective': 'multiclass',
                        'num_class': nuni + 1
                    }
            else:  # automatic analyze column
                if analyze_column(X[col]) == 'numeric':
                    params = {
                        'objective': 'regression'
                    }
                else:
                    nuni = X[col].dropna().nunique()
                    if nuni == 2:
                        params = {
                            'objective': 'binary'
                        }
                    elif nuni > 2:
                        params = {
                            'objective': 'multiclass',
                            'num_class': nuni + 1
                        }
                    else:
                        print(f'column {col} has only one unique value.')
                        continue

            params['verbosity'] = -1

            null_idx = X[col].isnull()
            x_train = X.loc[~null_idx].drop(col, axis=1)
            x_test = X.loc[null_idx].drop(col, axis=1)
            y_offset = X[col].min()
            y_train = X.loc[~null_idx, col].astype(int) - y_offset
            dtrain = lgb.Dataset(
                data=x_train,
                label=y_train
            )

            early_stopping_rounds = int(self.n_iter/10)
            model = lgb.train(
                params, dtrain, valid_sets=[dtrain],
                num_boost_round=self.n_iter,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=0,
            )

            self.imputers[col] = model
            self.offsets[col] = y_offset
            self.objectives[col] = params['objective']
            if self.verbose:
                print(
                    f'{col}:\t{self.objectives[col]}...iter{model.best_iteration}/{self.n_iter}')

    def transform(self, X):
        output_X = X.copy()

        for icol, col in enumerate(self.feature_with_missing):
            model = self.imputers[col]
            y_offset = self.offsets[col]
            objective = self.objectives[col]

            null_idx = X[col].isnull()
            x_test = X.loc[null_idx].drop(col, axis=1)

            y_test = model.predict(x_test)
            if objective == 'multiclass':
                y_test = np.argmax(y_test, axis=1).astype(float)
            elif objective == 'binary':
                y_test = (y_test > 0.5).astype(float)
            y_test += y_offset
            output_X.loc[null_idx, col] = y_test
            if objective in ['multiclass', 'binary']:
                output_X[col] = output_X[col].astype(int)

        return output_X
        
    def fit_transform(self, X, y=None):
        output_X = X.copy()
        self.n_features = X.shape[1]
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = [f'f{i}' for i in range(self.n_features)]
            X = pd.DataFrame(X, columns=self.feature_names)
        self.feature_with_missing = [col for col in self.feature_names if X[col].isnull().sum() > 0]

        for icol, col in enumerate(self.feature_with_missing):
            if icol in self.cat_features:
                nuni = X[col].dropna().nunique()
                if nuni == 2:
                    params = {
                        'objective': 'binary'
                    }
                elif nuni > 2:
                    params = {
                        'objective': 'multiclass',
                        'num_class': nuni + 1
                    }
            else: # automatic analyze column
                if analyze_column(X[col]) == 'numeric':
                    params = {
                        'objective': 'regression'
                    }
                else:
                    nuni = X[col].dropna().nunique()
                    if nuni == 2:
                        params = {
                            'objective': 'binary'
                        }
                    elif nuni > 2:
                        params = {
                            'objective': 'multiclass',
                            'num_class': nuni + 1
                        }
                    else:
                        print(f'column {col} has only one unique value.')
                        continue
          
            params['verbosity'] = -1
            
            null_idx = X[col].isnull()
            x_train = X.loc[~null_idx].drop(col, axis=1)
            x_test = X.loc[null_idx].drop(col, axis=1)
            y_offset = X[col].min()
            y_train = X.loc[~null_idx, col].astype(int) - y_offset
            dtrain = lgb.Dataset(
                data=x_train,
                label=y_train
            )

            early_stopping_rounds = int(self.n_iter/10)
            model = lgb.train(
                params, dtrain, valid_sets=[dtrain], 
                num_boost_round=self.n_iter,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=0,
            )

            y_test = model.predict(x_test)
            if params['objective'] == 'multiclass':
                y_test = np.argmax(y_test, axis=1).astype(float)
            elif params['objective'] == 'binary':
                y_test = (y_test > 0.5).astype(float)
            y_test += y_offset
            output_X.loc[null_idx, col] = y_test
            if params['objective'] in ['multiclass', 'binary']:
                output_X[col] = output_X[col].astype(int)
            self.imputers[col] = model
            self.offsets[col] = y_offset
            self.objectives[col] = params['objective']
            if self.verbose:
                print(f'{col}:\t{self.objectives[col]}...iter{model.best_iteration}/{self.n_iter}')
        
        return output_X
