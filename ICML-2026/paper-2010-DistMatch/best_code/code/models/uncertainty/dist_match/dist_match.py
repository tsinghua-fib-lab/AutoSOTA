import os
from pathlib import Path
from typing import Optional, List, Union

from torch import nn
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from models.forcast.forcast_base import PredictionOutputType, FCPredictionData
from models.uncertainty.pi_base import (
    PIModel,
    PIModelPrediction,
    PIPredictionStepData,
    PICalibData,
    PICalibArtifacts,
)
from models.uncertainty.dist_match.utils import (
    match_ks_stat,
    match_ks_p_val,
    match_rand,
    match_mi,
    match_kl,
    match_wd,
)
from models.uncertainty.dist_match.tree import DistMatchQRF
from models.uncertainty.dist_match.mask_cache import MaskCacheManager

from utils.calc_np import calc_residuals


class DistMatch(PIModel):
    def __init__(self, **kwargs):
        super(DistMatch, self).__init__(
            use_dedicated_calibration=True,
            fc_prediction_out_modes=(PredictionOutputType.POINT,),
        )
        self.qrf: DistMatchQRF = None
        self._qrf_upd_steps = kwargs.get("qrf_upd_steps", None)
        self._past_window_len = kwargs.get("past_window_len", 100)
        self._match_threshold = kwargs.get("match_threshold", 0.8)
        self._qrf_param = kwargs.get("qrf_param", dict())
        self._beta_calc_bins = kwargs.get("beta_calc_bins", 5)
        self._matcher_param = kwargs.get("matcher_param", dict())
        self._matcher: callable | nn.Module
        self._matcher_lower_is_match = True  # KS/MI: lower=similar; WD/KL: lower=dissimilar
        self._set_matcher(kwargs.get("match_method", "ks"))
        self._matcher_trainable = False
        self._n_train_samples = None
        self._data_mode = kwargs.get("data_mode", "error")
        self._input_mode = kwargs.get("input_mode", "normal")
        self._auto_calibrate_threshold = kwargs.get("auto_calibrate_threshold", False)
        self._auto_calibrate_percentile = kwargs.get("auto_calibrate_percentile", 50)
        self._adaptive_window = kwargs.get("adaptive_window", False)
        self._window_candidates = kwargs.get("window_candidates", [50, 75, 100, 125, 150, 200])

    @property
    def _base_dir(self):
        return os.path.join(
            self._forcast_service._experiment_config.base_proj_dir, "models_save/uc/"
        )

    def _compare_with_threshold(
        self, data: Union[np.ndarray, float]
    ) -> Union[np.ndarray, bool]:
        if self._matcher_lower_is_match:
            return data < self._match_threshold
        else:
            return data >= self._match_threshold

    def _match(self, x1, x2) -> bool:
        return self._compare_with_threshold(self._matcher(x1, x2))

    def _set_matcher(self, method: str) -> float:
        self._matcher = None
        self._matcher_method = method
        # Default: lower value = more similar (match)
        self._matcher_lower_is_match = True
        match self._matcher_method:
            case "ks_stat":
                self._matcher = match_ks_stat
            case "ks":
                self._matcher = match_ks_p_val
            case "mi":
                self._matcher = match_mi
            case "rand":
                self._matcher = match_rand
            case "wd":
                self._matcher = match_wd
                self._matcher_lower_is_match = False  # WD returns 1-distance: higher=similar
            case "kl":
                self._matcher = match_kl
                self._matcher_lower_is_match = False  # KL returns 1-divergence: higher=similar
            case _:
                raise NotImplemented(f"Matcher {method} is not implemented")

    def _calibrate(
        self, calib_data: [PICalibData], alphas, **kwargs
    ) -> [PICalibArtifacts]:
        pass

    def calibrate_individual(
        self,
        calib_data: PICalibData,
        alpha,
        calib_artifact: Optional[PICalibArtifacts],
        mix_calib_data: Optional[List[PICalibData]],
        mix_calib_artifact: Optional[List[PICalibArtifacts]],
    ) -> PICalibArtifacts:
        return self._train_qrf_from_inputs(
            ts_id=calib_data.ts_id,
            X_past=calib_data.X_pre_calib,
            Y_past=calib_data.Y_pre_calib,
            X_reg_train=calib_data.X_calib,
            Y_reg_train=calib_data.Y_calib,
            step_offset=calib_data.step_offset,
        )

    def _train_qrf_from_inputs(
        self,
        ts_id,
        X_past,
        Y_past,
        X_reg_train,
        Y_reg_train,
        step_offset,
    ):
        calib_artifacts = PICalibArtifacts()

        Y_hat = self._forcast_service.predict(
            FCPredictionData(
                ts_id=ts_id,
                X_past=X_past,
                Y_past=Y_past,
                X_step=X_reg_train,
                step_offset=step_offset,
            ),
            retrieve_tensor=False,
        ).point

        # Adapt to any shape of data
        eps_reg_train = calc_residuals(
            y_hat=Y_hat.squeeze(), y=Y_reg_train.numpy().squeeze()
        )[:, None]
        calib_artifacts.fc_Y_hat = Y_hat
        calib_artifacts.eps = eps_reg_train
        self._calib_eps_last = eps_reg_train

        # Adaptive window selection via in-sample Winkler evaluation
        if self._adaptive_window and len(eps_reg_train) >= max(self._window_candidates) + self._past_window_len:
            import logging
            _logger = logging.getLogger(__name__)
            candidates = self._window_candidates
            best_window = self._past_window_len
            best_score = float("inf")

            for w in candidates:
                orig_window = self._past_window_len
                self._past_window_len = w
                try:
                    # Train QRF on full calibration set (uses _train_qrf which respects _past_window_len)
                    tmp_qrf = self._train_qrf(X_reg_train, eps_reg_train)
                    # Build eval windows matching the QRF input format
                    # _train_qrf produces x_in of shape (n-w-1, w, 1) after preprocessing
                    x_raw = self._get_inputs_by_mode(X_reg_train, eps_reg_train)
                    x_win = np.lib.stride_tricks.sliding_window_view(
                        x_raw, window_shape=w, axis=-2
                    ).swapaxes(-1, -2)[:-1]  # shape (n-w-1, w, 1)
                    x_win = self._preprocess_inputs(x_win)
                    # Use last 1/3 for evaluation
                    n_eval = max(5, len(x_win) // 4)
                    x_eval = x_win[-n_eval:]  # shape (n_eval, w, 1)
                    if len(x_eval) < 3:
                        self._past_window_len = orig_window
                        continue
                    widths_cv = tmp_qrf.predict(x_eval)
                    # Ground truth: residuals corresponding to eval windows
                    y_eval_start = -(n_eval + 1)
                    y_eval = eps_reg_train[y_eval_start:].squeeze()
                    y_eval = y_eval[-len(widths_cv):]
                    if len(y_eval) != len(widths_cv):
                        y_eval = eps_reg_train[-len(widths_cv):].squeeze()
                    y_in_pi = ((y_eval >= widths_cv[:, 0]) & (y_eval <= widths_cv[:, 1]))
                    coverage = y_in_pi.mean() if len(y_in_pi) > 0 else 1.0
                    avg_width = (widths_cv[:, 1] - widths_cv[:, 0]).mean()
                    if coverage >= 0.90:
                        score = avg_width
                    else:
                        miscoverage_penalty = (0.90 - coverage) * avg_width * 10
                        score = avg_width + miscoverage_penalty
                    _logger.info(f"[AdaptiveWindow] w={w}: n_eval={n_eval}, coverage={coverage:.4f}, avg_width={avg_width:.2f}, score={score:.2f}")
                    if score < best_score:
                        best_score = score
                        best_window = w
                except Exception as e:
                    _logger.warning(f"[AdaptiveWindow] w={w} failed: {e}")
                finally:
                    self._past_window_len = orig_window

            _logger.info(f"[AdaptiveWindow] Selected w={best_window} (score={best_score:.2f}), original w={orig_window}")
            self._past_window_len = best_window

        # Auto-calibrate match_threshold from calibration residual distribution
        if self._auto_calibrate_threshold and self._matcher_method == "ks_stat":
            eps_flat = eps_reg_train.squeeze()
            if len(eps_flat) > 2 * self._past_window_len:
                windows = np.lib.stride_tricks.sliding_window_view(
                    eps_flat, window_shape=self._past_window_len
                )
                n_windows = len(windows)
                # Compute KS stats between non-overlapping adjacent blocks
                # (windows separated by past_window_len, so no overlap)
                block_step = self._past_window_len
                ks_block = []
                for i in range(0, n_windows - block_step, block_step):
                    ks_block.append(match_ks_stat(windows[i], windows[i + block_step]))
                # Also random pairs for reference
                n_rand = min(200, n_windows * (n_windows - 1) // 2)
                rng = np.random.RandomState(42)
                ks_random = []
                for _ in range(n_rand):
                    i, j = rng.randint(0, n_windows, 2)
                    if i == j:
                        j = (j + 1) % n_windows
                    ks_random.append(match_ks_stat(windows[i], windows[j]))
                # Use specified percentile of non-overlapping block KS stats
                calibrated_threshold = float(np.percentile(ks_block, self._auto_calibrate_percentile))
                # Clamp to reasonable range
                calibrated_threshold = float(np.clip(calibrated_threshold, 0.001, 0.5))
                print(f"[AutoCalib] {n_windows} windows: block-adj KS median={np.median(ks_block):.5f}, " +
                      f"random KS median={np.median(ks_random):.5f}, " +
                      f"p{self._auto_calibrate_percentile}={calibrated_threshold:.5f}, " +
                      f"(config threshold={self._match_threshold:.5f})")
                self._match_threshold = calibrated_threshold
            else:
                print(f"[AutoCalib] Not enough calibration data ({len(eps_flat)} pts) for 2x window length {self._past_window_len}")

        # DO NOT USE CACHE
        # self.qrf = self._train_qrf(X_reg_train, eps_reg_train, ts_id)
        self.qrf = self._train_qrf(X_reg_train, eps_reg_train)

        self._n_train_samples = len(X_reg_train)

        forecast_model = self._forcast_service._model_config.model
        self.qrf.save(
            f"{self._base_dir}/qrf_{self._matcher_method}<{self._match_threshold}_{self._data_mode}_{self._input_mode}_{ts_id}_{forecast_model}|{self._past_window_len}.pkl"
        )

        return calib_artifacts

    def _get_inputs_by_mode(self, inputs: np.ndarray, resids: np.ndarray) -> np.ndarray:
        return resids if self._data_mode == "error" else inputs

    def _train_qrf(
        self, inputs: np.ndarray, resids: np.ndarray, ts_id: Optional[str] = None
    ):
        assert len(inputs) == len(resids)
        x = self._get_inputs_by_mode(inputs, resids)
        y = resids

        x_in = sliding_window_view(x, window_shape=self._past_window_len, axis=-2)
        x_in = x_in.swapaxes(-1, -2)

        x_in = x_in[:-1]
        y_in = y[self._past_window_len :].squeeze()

        x_in = self._preprocess_inputs(x_in)
        match_mask = (
            self._load_cached_mask(ts_id, x_in)
            if ts_id is not None
            else self._compute_mask(x_in)
        )

        qrf = DistMatchQRF(
            **self._qrf_param,
            alpha=0.1,
            n_quantile_bins=self._beta_calc_bins,
            feature_dim=-1,
            matcher=self._match,
            match_mask=match_mask,
            relevance_matcher=self._matcher,
        )
        qrf.fit(x_in, y_in, preserve_match_mask=True)

        return qrf

    def pre_predict(self, **kwargs):
        super().pre_predict(**kwargs)
        self.qrf.set_alpha(kwargs["alpha"], n_quantile_bins=self._beta_calc_bins)
        self.qrf.reset_updates(self._n_train_samples)

    def _predict_step(
        self, pred_data: PIPredictionStepData, **kwargs
    ) -> PIModelPrediction:
        # Retrieve data
        alpha, x_step, x_past, y_past, eps_past, cur_step = (
            pred_data.alpha,
            pred_data.X_step,
            pred_data.X_past,
            pred_data.Y_past,
            pred_data.eps_past,
            pred_data.step_offset_prediction,
        )

        # Calculate y_hat and prediction interval for current step
        Y_hat = self._forcast_service.predict(
            FCPredictionData(
                ts_id=pred_data.ts_id,
                X_past=x_past,
                Y_past=y_past,
                X_step=x_step,
                step_offset=pred_data.step_offset_overall,
            )
        ).point

        eps_reg = np.concatenate(
            [
                self._calib_eps_last,
                np.array(eps_past).reshape(-1, 1),
            ]
        )
        # Fix: use median fill instead of zero for NaN residuals to avoid masking signal
        eps_reg = np.nan_to_num(eps_reg, nan=np.nanmedian(eps_reg), posinf=0.0, neginf=0.0)

        x_reg = np.concatenate([x_past, x_step])
        x_reg = x_reg[-len(eps_reg) :, ...]

        if (
            self._qrf_upd_steps is not None
            and cur_step > 0
            and self._qrf_upd_steps > 0
            and cur_step % self._qrf_upd_steps == 0
        ):
            self.qrf = self._train_qrf(x_reg, eps_reg)

        x_all = self._get_inputs_by_mode(x_reg, eps_reg)
        x_all = x_all[None, ...]
        x_prev_test = self._preprocess_inputs(
            x_all[:, -self._past_window_len - 1 : -1, ...]
        )
        x_test = self._preprocess_inputs(x_all[:, -self._past_window_len :, ...])

        y_prev_test = eps_reg[-1:, 0]
        id_prev_test = np.array([len(eps_reg)])

        self.qrf.predict_with_update(x_prev_test, y_prev_test, id_prev_test)
        widths = self.qrf.predict(x_test)

        width_low = widths[0][0]
        width_high = widths[0][1]

        pred_int = Y_hat + width_low, Y_hat + width_high
        return PIModelPrediction(pred_interval=pred_int, fc_Y_hat=Y_hat)

    def _load_cached_mask(self, ts_id: int, data: np.ndarray):
        forecast_model = self._forcast_service._model_config.model
        path = f"{self._base_dir}/cache_{self._matcher_method}_{self._data_mode}_{self._input_mode}_{ts_id}_{forecast_model}|{self._past_window_len}.npy"
        manager = self._get_mask_manager(path)
        if self._matcher_param.get("retrain", False):
            manager.put(data, data)
        mask = manager.get(data, data)
        mask = self._compare_with_threshold(mask)
        return mask

    def _compute_mask(self, data: np.ndarray) -> np.ndarray:
        manager = self._get_mask_manager()
        mask = manager.compute(data, data)
        mask = self._compare_with_threshold(mask)
        return mask

    def _get_mask_manager(self, path: Optional[str] = None):
        return MaskCacheManager(
            matcher=self._matcher,
            path=path,
            batch_size=self._qrf_param.get("batch_size", None),
        )

    def _preprocess_inputs(self, inputs: np.ndarray):
        match self._input_mode:
            case "delta":
                return np.diff(inputs, 1, axis=-2)
            case "residual":
                return inputs[..., 1:, :] - inputs[..., :1, :]
        return inputs

    def model_ready(self):
        return True

    def can_handle_different_alpha(self):
        return True
