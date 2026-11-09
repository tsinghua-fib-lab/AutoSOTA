'''
xfeat
https://github.com/pfnet-research/xfeat
Modified
'''

import numpy as np
import pandas as pd
from sklearn.utils.validation import column_or_1d, check_is_fitted
import xfeat
from xfeat.types import XDataFrame, XSeries, CSeries, XNDArray
from xfeat.cat_encoder._target_encoder import _TargetEncoder


class TargetEncoder(xfeat.TargetEncoder):
    '''
    Enhanced Multi-Fold Target Encoder
    What's new:
    - take second arguement in .fit() and .fit_transform()
    - add random noise
    '''

    def __init__(self, noise_level=0.0, random_state=None, **kwargs):
        super().__init__(**kwargs)
        self.noise_level = noise_level
        self.random_state = random_state

    def fit(self, input_df: XDataFrame, y: XSeries = None) -> None:
        input_cols = self._input_cols
        if not input_cols:
            input_cols = input_df.columns.tolist()
            self._input_cols = input_cols

        # Remove `target_col` from `self._input_cols`.
        if self._target_col in self._input_cols:
            self._input_cols.remove(self._target_col)

        for col in self._input_cols:
            target_encoder = _TargetEncoder(self.fold)
            self._target_encoders[col] = target_encoder
            if y is None:
                y = input_df[self._target_col]
            target_encoder.fit(input_df[col], y)

    def transform(self, input_df: XDataFrame) -> XDataFrame:
        out_df = input_df.copy()

        for col in self._input_cols:
            out_col = self._output_prefix + col + self._output_suffix
            if isinstance(input_df[col], pd.Series):
                X = column_or_1d(input_df[col], warn=True)
            elif cudf and isinstance(input_df[col], cudf.Series):
                X = input_df[col]
            else:
                raise TypeError

            out_df[out_col] = self._target_encoders[col].transform(X)

        if self.noise_level > 0:
            np.random.seed(self.random_state)
            out_df += np.random.normal(0, self.noise_level, out_df.shape)

        return out_df

    def fit_transform(self, input_df: XDataFrame, y: XSeries = None) -> XDataFrame:
        out_df = input_df.copy()

        input_cols = self._input_cols
        if not input_cols:
            input_cols = input_df.columns.tolist()
            self._input_cols = input_cols

        # Remove `target_col` from `self._input_cols`.
        if self._target_col in self._input_cols:
            self._input_cols.remove(self._target_col)

        for col in self._input_cols:
            out_col = self._output_prefix + col + self._output_suffix
            target_encoder = _TargetEncoder(self.fold)
            self._target_encoders[col] = target_encoder

            if isinstance(input_df[col], pd.Series):
                X = column_or_1d(input_df[col], warn=True)
                if y is None:
                    y = column_or_1d(input_df[self._target_col], warn=True)
                else:
                    y = column_or_1d(y, warn=True)
            elif cudf and isinstance(input_df[col], cudf.Series):
                X = input_df[col]
                if y is None:
                    y = input_df[self._target_col]
            else:
                raise TypeError

            out_df[out_col] = target_encoder.fit_transform(X, y).copy()

        if self.noise_level > 0:
            np.random.seed(self.random_state)
            out_df += np.random.normal(0, self.noise_level, out_df.shape)

        return out_df
