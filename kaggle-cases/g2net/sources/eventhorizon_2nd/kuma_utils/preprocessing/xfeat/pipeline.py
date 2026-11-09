'''
xfeat (https://github.com/pfnet-research/xfeat) Mod
'''

import numpy as np
import pandas as pd
from sklearn.utils.validation import column_or_1d, check_is_fitted
from typing import List, Optional
import optuna
from xfeat.base import TransformerMixin, OptunaSelectorMixin
from xfeat.types import XDataFrame, XSeries, CSeries, XNDArray


class Pipeline(TransformerMixin):
    '''
    Enhanced Pipeline
    What's new:
    - specify target column in advance
    - work with other packages such as catgory_encoders
    '''

    def __init__(self, transforms, target_col=None):
        self._transforms = transforms
        self._target_col = target_col

    def fit(self, input_df: XDataFrame) -> None:
        raise RuntimeError(
            "Pipeline doesnt support fit(). Use fit_transform().")

    def from_trial(self, trial: optuna.trial.FrozenTrial):
        for transform in self._transforms:
            if isinstance(transform, OptunaSelectorMixin):
                transform.from_trial(trial)

    def set_trial(self, trial):
        for transform in self._transforms:
            if isinstance(transform, OptunaSelectorMixin):
                transform.set_trial(trial)

    def fit_transform(self, input_df: XDataFrame) -> XDataFrame:
        if self._target_col is not None:
            y = input_df[self._target_col].copy()
        else:
            y = None
        
        for transform in self._transforms:
            if y is not None:
                try:
                    input_df = transform.fit_transform(input_df, y)
                except:
                    input_df = transform.fit_transform(input_df)
            else:
                input_df = transform.fit_transform(input_df)
        return input_df

    def transform(self, input_df: XDataFrame) -> XDataFrame:
        for transform in self._transforms:
            input_df = transform.transform(input_df)
        return input_df

    def get_selected_cols(self) -> Optional[List[str]]:
        for i, transform in enumerate(self._transforms, start=1):
            if isinstance(transform, OptunaSelectorMixin):
                if i == len(self._transforms):
                    logger.warning("Optuna selector is not last component.")

                return transform.get_selected_cols()

        return None
