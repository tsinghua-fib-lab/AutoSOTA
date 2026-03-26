# ---------------------------------------------------------------------------------
# Portions of this file are derived from ForecastPFN
# - Source: https://github.com/abacusai/ForecastPFN
# - Paper: ForecastPFN: Synthetically-Trained Zero-Shot Forecasting
# - License: Apache License 2.0

# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------


import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from keras import backend
from sklearn.preprocessing import StandardScaler

from probts.model.forecaster import Forecaster


def smape(y_true, y_pred):
    """ Calculate Armstrong's original definition of sMAPE between `y_true` & `y_pred`.
        `loss = 200 * mean(abs((y_true - y_pred) / (y_true + y_pred), axis=-1)`
        Args:
        y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
        Returns:
        Symmetric mean absolute percentage error values. shape = `[batch_size, d0, ..
        dN-1]`.
        """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    diff = tf.abs(
        (y_true - y_pred) /
        backend.maximum(y_true + y_pred, backend.epsilon())
    )
    return 200.0 * backend.mean(diff, axis=-1)


class ForecastPFN(Forecaster):
    def __init__(
        self,
        label_len: int = 48,
        ckpt_path: str = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.no_training = True

        self.label_len = label_len
        
        self.model = tf.keras.models.load_model(ckpt_path, custom_objects={'smape': smape})


    def _ForecastPFN_time_features(self, x_mark_enc: np.ndarray, x_mark_dec: np.ndarray):
        def extract_time_features(ts):
            original_shape = ts.shape
            ts = ts.reshape(-1)  # Flatten the array
            if type(ts[0]) == datetime.datetime:
                year = np.array([x.year for x in ts])
                month = np.array([x.month for x in ts])
                day = np.array([x.day for x in ts])
                day_of_week = np.array([x.weekday() + 1 for x in ts])
                day_of_year = np.array([x.timetuple().tm_yday for x in ts])
            else:
                ts = pd.to_datetime(ts)
                year = ts.year.values
                month = ts.month.values
                day = ts.day.values
                day_of_week = ts.day_of_week.values + 1
                day_of_year = ts.day_of_year.values
            
            features = np.stack([year, month, day, day_of_week, day_of_year], axis=-1)
            return features.reshape(*original_shape, -1).squeeze()

        # Process the encoder and decoder inputs
        x_mark_enc_features = extract_time_features(x_mark_enc)
        x_mark_dec_features = extract_time_features(x_mark_dec)

        return x_mark_enc_features, x_mark_dec_features

    def _process_tuple(self, x, x_mark, y_mark, horizon):
        """
        x: tensor of shape (n, 1)
        x_mark: tensor of shape (n, d)
        y_mark: tensor of shape (horizon, d)

        where
        n       is the input sequence length
        horizon is the output sequence length
        d is the dimensionality of the time_stamp (5 for ForecastPFN)
        """
        if tf.reduce_all(x == x[0]):
            x = tf.concat([x[:-1], x[-1:] + 1], axis=0)
        
        history = x.numpy()
        scaler = StandardScaler()
        scaler.fit(history)
        history = scaler.transform(history)
        
        history_mean = np.nanmean(history[-6:])
        history_std = np.nanstd(history[-6:])
        local_scale = history_mean + history_std + 1e-4
        
        history = np.clip(history / local_scale, a_min=0, a_max=1)
        
        if x.shape[0] != 100:
            if x.shape[0] > 100:
                target = x_mark[-100:, :]
                history = history[-100:, :]
            else:
                target = tf.pad(x_mark, [[100 - x.shape[0], 0], [0, 0]])
                history = tf.pad(history, [[100 - x.shape[0], 0], [0, 0]])
            
            history = tf.repeat(tf.expand_dims(history, axis=0), horizon, axis=0)[:, :, 0]
            ts = tf.repeat(tf.expand_dims(target, axis=0), horizon, axis=0)
        else:
            ts = tf.repeat(tf.expand_dims(x_mark, axis=0), horizon, axis=0)
            history = tf.convert_to_tensor(history, dtype=tf.float32)
        
        task = tf.fill([horizon], 1)
        y_mark_tensor = tf.convert_to_tensor(y_mark[-horizon:, :], dtype=tf.int64)
        target_ts = tf.expand_dims(y_mark_tensor, axis=1)
        
        model_input = {'ts': ts, 'history': history, 'target_ts': target_ts, 'task': task}
        pred_vals = self.model(model_input)
        
        scaled_vals = pred_vals['result'].numpy().T.reshape(-1) * pred_vals['scale'].numpy().reshape(-1)
        scaled_vals = scaler.inverse_transform([scaled_vals])
        return scaled_vals
    
    
    def _process_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        preds = []
        for idx, (x, y, x_mark, y_mark) in enumerate(zip(batch_x, batch_y, batch_x_mark, batch_y_mark)):
            pred = self._process_tuple(x, x_mark, y_mark, self.prediction_length)
            preds.append(pred)
        return preds


    def forecast(self, batch_data, num_samples=None):
        # For now, we only support batch_size=1
        B, _, K = batch_data.past_target_cdf.shape
        inputs = batch_data.past_target_cdf[:, -self.context_length:, ...].cpu()
        x_mark_enc = batch_data.past_time_feat[:, -self.context_length:, ...].cpu().numpy().astype('datetime64[s]')
        x_mark_dec = batch_data.future_time_feat.cpu().numpy().astype('datetime64[s]')
        x_mark_enc, x_mark_dec = self._ForecastPFN_time_features(x_mark_enc, x_mark_dec)

        x_mark_dec = tf.concat([x_mark_enc[:, -self.label_len:, :], x_mark_dec], axis=1)
        
        inputs = tf.reshape(inputs, [-1, self.context_length, 1])
        x_mark_enc = tf.repeat(x_mark_enc, repeats=K, axis=0)
        x_mark_dec = tf.repeat(x_mark_dec, repeats=K, axis=0)
        
        dec_inp = tf.zeros_like(inputs[:, -self.prediction_length:, :])
        dec_inp = tf.concat([inputs[:, -self.label_len:, :], dec_inp], axis=1)
        x_mark_enc = tf.cast(x_mark_enc, tf.int64)
        x_mark_dec = tf.cast(x_mark_dec, tf.int64)
        
        outputs = self._process_batch(inputs, dec_inp, x_mark_enc, x_mark_dec)
        outputs = tf.concat(outputs, axis=0)
        outputs = tf.reshape(outputs, [B, -1, K])
        outputs = outputs[:, :self.prediction_length, :].numpy()
        outputs = torch.tensor(outputs)
        return outputs.unsqueeze(1)