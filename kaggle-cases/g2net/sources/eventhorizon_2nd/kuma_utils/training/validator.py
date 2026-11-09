import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.calibration import calibration_curve
import optuna

import matplotlib.pyplot as plt
import matplotlib.cm as colormap
import seaborn as sns
try:
    import japanize_matplotlib
except:
    pass

from .trainer import Trainer, MODEL_ZOO
from .logger import LGBMLogger

from pathlib import Path
import pickle


class CrossValidator:
    '''
    Amazing cross validation wrapper for sklearn models

    Some useful features:
    - Automated parameter tuning using Optuna
    - Most features of Trainer are inherited
    '''
    
    SNAPSHOT_ITEMS = [
        'serial', 'models', 'is_trained',
        'fold_indices', 'scores', 'best_score', 'outoffold'
    ]

    def __init__(self, model=None, path=None, serial='cv0'):
        self.serial = serial
        if model is not None:
            self.model = model
            self.models = []
            self.fold_indices = None
            self.scores = []
            self.iterations = []
            self.outoffold = None
            self.prediction = None
            self.best_score = None
            self.is_trained = False
        elif path is not None:
            self.load(path)
        else:
            raise ValueError('either model or path must be given.')

    def _data_check(self, data):
        assert isinstance(data, (list, tuple))
        assert len(data) >= 2
        for i, d in enumerate(data):
            assert isinstance(d, (pd.Series, pd.DataFrame, np.ndarray))
            if i == 0:
                dshape = len(d)
            assert dshape == len(d)
            dshape = len(d)

    def _split_data(self, data, idx):
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.iloc[idx]
        else:
            return data[idx]

    def train(self,
              # Dataset
              data, cat_features=None, groups=None, folds=KFold(n_splits=5), 
              # Model
              params={}, fit_params={},
              # Optuna
              tune_model=False, optuna_params=None, maximize=True,
              eval_metric=None, n_trials=None, timeout=None, 
              lgbm_n_trials=[7, 20, 10, 6, 20], 
              # Misc
              logger=None, n_jobs=-1):

        self._data_check(data)
        self.models = []

        if isinstance(logger, (str, Path)):
            logger = LGBMLogger(logger, stdout=True, file=True)
        elif logger is None:
            logger = LGBMLogger(logger, stdout=True, file=False)
        assert isinstance(logger, LGBMLogger)

        _fit_params = fit_params.copy()
        if 'verbose_eval' in fit_params.keys():
            _fit_params.update({'verbose_eval': False})

        if callable(getattr(folds, 'split', None)):
            # splitter
            fold_iter = enumerate(folds.split(X=data[0], y=data[1], groups=groups))
            self.nfold = folds.get_n_splits()

        else:
            # index
            fold_iter = enumerate(folds)
            self.nfold = len(folds)

        self.fold_indices = []
        self.scores = []
        self.iterations = []
        if not isinstance(eval_metric, (list, tuple)):
            eval_metric = [eval_metric]

        for fold_i, (train_idx, valid_idx) in fold_iter:
            
            logger(f'[{self.serial}] Starting fold {fold_i}')

            self.fold_indices.append([train_idx, valid_idx])
            
            train_data = [self._split_data(d, train_idx) for d in data]
            valid_data = [self._split_data(d, valid_idx) for d in data]

            trn = Trainer(self.model, serial=f'{self.serial}_fold{fold_i}')
            trn.train(
                train_data=train_data, valid_data=valid_data, cat_features=cat_features,
                params=params, fit_params=_fit_params,
                tune_model=tune_model, optuna_params=optuna_params, 
                lgbm_n_trials=lgbm_n_trials, 
                maximize=maximize, eval_metric=eval_metric[0],
                n_trials=n_trials, timeout=timeout, 
                logger=logger, n_jobs=n_jobs
            )

            best_score = trn.get_best_score()
            best_iter = trn.get_best_iteration()

            all_metrics = [best_score] + [
                m(trn.get_model(), valid_data) for m in eval_metric if m is not None]
            log_str = f'[{self.serial}] Fold {fold_i}: '
            for i in range(len(all_metrics)):
                if i == 0:
                    name_metric = 'eval'
                else:
                    name_metric = f'monitor{i-1}'
                log_str += f'{name_metric}={all_metrics[i]:.6f} '
            log_str += f'(iter={best_iter})'
            logger(log_str)
            
            if fold_i == 0:
                _outoffold = trn.smart_predict(valid_data[0])
                self.outoffold = np.empty((data[0].shape[0], *_outoffold.shape[1:]), dtype=np.float16)
                self.outoffold[valid_idx] = _outoffold
            else:
                self.outoffold[valid_idx] = trn.smart_predict(valid_data[0])

            self.scores.append(best_score)
            self.iterations.append(best_iter)
            self.models.append(trn)

        mean_score = np.mean(self.scores)
        se_score = np.std(self.scores)
        self.best_score = [mean_score, se_score]
        logger(f'[{self.serial}] Overall metric: {mean_score:.6f} + {se_score:.6f}')

        self.is_trained = True

    fit = train

    def predict(self, X, **kwargs):
        assert self.is_trained
        self.prediction = []
        for trn in self.models:
            self.prediction.append(trn.predict(X, **kwargs))
        return self.prediction

    def predict_proba(self, X, **kwargs):
        assert self.is_trained
        self.prediction = []
        for trn in self.models:
            self.prediction.append(trn.predict_proba(X, **kwargs))
        return self.prediction

    def smart_predict(self, X, **kwargs):
        assert self.is_trained
        self.prediction = []
        for trn in self.models:
            self.prediction.append(trn.smart_predict(X, **kwargs))
        return self.prediction

    def get_model(self):
        return self.models
    
    def get_feature_importance(self, importance_type='auto', normalize=True, fit_params=None,
                               as_pandas='auto'):
        imps = []
        for trn in self.models:
            imps.append(trn.get_feature_importance(
                importance_type, normalize, fit_params, as_pandas=False))
        if as_pandas in ['auto', True]:
            return pd.DataFrame(imps)
        else:
            return imps

    def plot_feature_importance(self, importance_type='auto', normalize=True, fit_params=None,
                                sort=True, size=5, save_to=None):
        imp_df = self.get_feature_importance(
            importance_type, normalize, fit_params, as_pandas=True)
        plt.figure(figsize=(size, imp_df.shape[1]/3))
        order = imp_df.mean().sort_values(ascending=False).index.tolist() \
            if sort else None
        sns.barplot(data=imp_df, orient='h', ci='sd',
                    order=order, palette="coolwarm")
        if save_to is not None:
             plt.savefig(save_to)
        plt.show()

    def plot_calibration_curve(self, data, predict_params={}, size=4, save_to=None):
        X, y = data[0], data[1]
        approx = self.smart_predict(X, **predict_params)
        if isinstance(approx, list):
            approx = np.stack(approx).mean(0)
        fig = plt.figure(figsize=(size, size*1.5), tight_layout=True)
        gs = fig.add_gridspec(3, 1)
        ax1 = fig.add_subplot(gs[0:2, 0])
        ax2 = fig.add_subplot(gs[2, 0])
        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y, approx, n_bins=10)
        ax1.plot([0, 1], [0, 1], color='gray')
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-")
        ax1.set_xlabel('Fraction of positives')
        ax1.set_ylabel('Mean of prediction values')
        ax1.grid()
        ax1.set_xlim([0.0, 1.0])
        sns.distplot(approx, bins=10, ax=ax2)
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylabel('Count')
        if save_to is not None:
            plt.savefig(save_to)
        plt.show()

    def save(self, path):
        with open(path, 'wb') as f:
            snapshot = tuple([getattr(self, item)
                              for item in self.SNAPSHOT_ITEMS])
            pickle.dump(snapshot, f)

    def load(self, path):
        with open(path, 'rb') as f:
            snapshot = pickle.load(f)
        for i, item in enumerate(self.SNAPSHOT_ITEMS):
            setattr(self, item, snapshot[i])

    def __repr__(self):
        desc = f'CrossValidator: {self.serial}\n'
        items = ['models', 'is_trained', 'best_score']
        for i in items:
            desc += f'{i}: {getattr(self, i)}\n'
        return desc

    def info(self):
        print(self.__repr__())
