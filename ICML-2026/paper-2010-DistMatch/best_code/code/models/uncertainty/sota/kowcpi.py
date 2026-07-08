import math
from typing import Tuple, Dict, Optional, List

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import torch
from torch import nn, Tensor

from models.forcast.forcast_base import PredictionOutputType, FCPredictionData
from models.uncertainty.pi_base import (
    PIModel,
    PIModelPrediction,
    PIPredictionStepData,
    PICalibData,
    PICalibArtifacts,
)
from utils.calc_np import calc_residuals


def patch_data(
    data: np.ndarray,
    patch_len: int,
) -> np.ndarray:
    return np.lib.stride_tricks.sliding_window_view(data, patch_len)


def get_residuals_dataset(
    residuals: np.ndarray,
    patch_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    residuals = np.ravel(residuals)
    patches = patch_data(residuals, patch_len)
    inputs = patches[:-1]
    targets = patches[1:, -1]
    return inputs, targets


def epanechnikov_kernel(u: Tensor):
    zeros = torch.zeros_like(u)
    return (0.75 * (1 - u.pow(2))).where(abs(u) <= 1, zeros)


def compute_kernel(kernel_fn: callable, u: Tensor, h: float, p: int) -> Tensor:
    u_norm = u.norm(p=2, dim=-1, keepdim=True)
    return kernel_fn(u_norm / h) / math.pow(h, p)


def compute_data_relationship(
    target: Tensor,
    data: Tensor,
    lbd: Tensor,
    kernel_fn: callable,
    bandwidth: float,
    n_channels: int = 1,
) -> Tuple[Tensor, Tensor]:
    diff = data - target
    kernel = compute_kernel(kernel_fn, diff, bandwidth, n_channels)
    relationship = 1 + lbd * diff[..., :1] * kernel
    return relationship, kernel


def compute_lambda_loss(
    target: Tensor,
    data: Tensor,
    lbd: Tensor,
    kernel_fn: callable,
    bandwidth: float,
    n_channels: int = 1,
):
    data_relationship, _ = compute_data_relationship(
        target, data, lbd, kernel_fn, bandwidth, n_channels
    )
    loss = -data_relationship.log().sum()

    return loss


def get_optimal_lambda(
    target: Tensor,
    data: Tensor,
    kernel_fn: callable,
    bandwidth: float = 1.0,
    n_channels: int = 1,
    lr: float = 1e-1,
    patience: int = 3,
    n_epochs: int = 200,
):
    lbd = nn.Parameter(torch.tensor(1.0)) # TODO: ADD NOISE [Optional]
    opt = torch.optim.Adam([lbd], lr)

    best_loss = np.inf
    best_lbd = 0.0
    cur_patience = patience

    for _ in range(n_epochs):
        opt.zero_grad()
        loss = compute_lambda_loss(target, data, lbd, kernel_fn, bandwidth, n_channels)
        loss.backward()
        opt.step()
        loss_item = loss.item()
        lbd_item = lbd.item()

        if loss_item < best_loss:
            best_loss = loss_item
            cur_patience = patience
            best_lbd = lbd_item
        else:
            cur_patience -= 1

        # print(
        #     f"[{cur_patience}/{patience}] Loss: {loss:.6f}; Lambda: {lbd_item:.6f} {best_lbd:.6f}"
        # )
        if cur_patience <= 0:
            break

    return best_lbd


def get_adj_weights(
    target: Tensor,
    data: Tensor,
    lbd: float,
    kernel_fn: callable,
    bandwidth: float,
    n_channels: int,
) -> Tensor:
    data_relationship, kernel = compute_data_relationship(
        target, data, lbd, kernel_fn, bandwidth, n_channels
    )
    probabilities = 1 / (len(data) * data_relationship)
    weights = probabilities * kernel
    weights = weights / weights.sum()
    return weights


def get_quantiles(
    weights: Tensor,
    ys: Tensor,
    alphas: Tensor,
):
    sorted_idx = torch.argsort(ys)
    sorted_ys = ys[sorted_idx]
    sorted_weights = weights[sorted_idx]
    cumsum = torch.cumsum(sorted_weights, dim=0)
    
    quantiles = []
    for alpha in alphas:
        idx = torch.argmax((cumsum >= alpha).float())
        quantiles.append(sorted_ys[idx])
    return torch.tensor(quantiles)


def get_bounds(quantiles: np.ndarray) -> np.ndarray:
    n_quantiles = len(quantiles)
    n_ids = n_quantiles // 2
    upper_ids = list(range(n_ids))
    lower_ids = list(range(n_quantiles - n_ids, n_quantiles))
    widths = quantiles[upper_ids] - quantiles[lower_ids]
    min_width_id = np.argmin(widths)
    return np.array(
        [quantiles[upper_ids[min_width_id]], quantiles[lower_ids[min_width_id]]]
    )


class KowCPIModel(PIModel):
    def __init__(self, **kwargs):
        super().__init__(
            use_dedicated_calibration=True,
            fc_prediction_out_modes=(PredictionOutputType.POINT,),
        )
        self._past_window_len = kwargs["past_window_len"]
        self._beta_calc_bins = kwargs.get("beta_calc_bins", 2)
        self._bandwitdth = kwargs.get("bandwidth", 1)
        self._alpha_t = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        Y_hat = self._forcast_service.predict(
            FCPredictionData(
                ts_id=calib_data.ts_id,
                X_past=calib_data.X_pre_calib,
                Y_past=calib_data.Y_pre_calib,
                X_step=calib_data.X_calib,
                step_offset=calib_data.step_offset,
            ),
            retrieve_tensor=False,
        ).point
        eps_reg_train = calc_residuals(y_hat=Y_hat, y=calib_data.Y_calib.numpy())
        self._calib_eps = eps_reg_train

        calib_artifacts = PICalibArtifacts()
        calib_artifacts.fc_Y_hat = Y_hat
        calib_artifacts.eps = eps_reg_train
        return calib_artifacts

    def pre_predict(self, **kwargs):
        super().pre_predict(**kwargs)
        self._alpha_t = kwargs["alpha"]  # Reset

    def _predict_step(
        self, pred_data: PIPredictionStepData, **kwargs
    ) -> PIModelPrediction:
        # Retrieve data
        _, X_step, X_past, Y_past, eps_past = (
            pred_data.alpha,
            pred_data.X_step,
            pred_data.X_past,
            pred_data.Y_past,
            pred_data.eps_past[-self._past_window_len :],
        )

        widths = self._get_regions(
            eps_reg_train=np.concatenate((self._calib_eps, pred_data.eps_past.numpy())),
            eps_test=eps_past,
        )

        # Calculate y_hat and prediction interval for current step
        Y_hat = self._forcast_service.predict(
            FCPredictionData(
                ts_id=pred_data.ts_id,
                X_past=X_past,
                Y_past=Y_past,
                X_step=X_step,
                step_offset=pred_data.step_offset_overall,
            )
        ).point

        width_high, width_low = widths
        # width_low = self._curr_SigmaX * self.model.predict(X_reg, beta)[1][0][0]
        # width_high = self._curr_SigmaX * self.model.predict(X_reg, (1 - alpha + beta))[1][0][1]
        pred_int = Y_hat + width_low, Y_hat + width_high
        return PIModelPrediction(pred_interval=pred_int, fc_Y_hat=Y_hat)

    @property
    def _beta_bins(self) -> np.ndarray:
        high_quantiles = np.linspace(start=0, stop=self._alpha_t, num=self._beta_calc_bins + 1)[:-1]
        return np.concatenate([1 - self._alpha_t + high_quantiles, high_quantiles])

    def _get_regions(self, eps_reg_train: np.ndarray, eps_test: Tensor):
        kernel_fn = epanechnikov_kernel
        bandwidth = self._bandwitdth
        n_channels = self._past_window_len
        alphas = torch.tensor(self._beta_bins).double().to(self._device)
        calib_xs, calib_ys = get_residuals_dataset(eps_reg_train, self._past_window_len)

        test_tensor = eps_test.ravel().double().to(self._device)
        res_x_tensor, res_y_tensor = (
            torch.from_numpy(res_arr).double().to(self._device)
            for res_arr in (calib_xs, calib_ys)
        )

        lbd = get_optimal_lambda(
            test_tensor, res_x_tensor, kernel_fn, bandwidth, n_channels
        )
        adj_weights = get_adj_weights(
            test_tensor, res_x_tensor, lbd, kernel_fn, bandwidth, n_channels
        )

        quantiles = get_quantiles(adj_weights.squeeze(-1), res_y_tensor, alphas)
        quantiles = quantiles.cpu().numpy()
        return get_bounds(quantiles)

    def model_ready(self):
        return True

    def required_past_len(self) -> Tuple[int, int]:
        fc_required_len = super().required_past_len()
        return max(fc_required_len[0], self._past_window_len), max(
            fc_required_len[1], self._past_window_len
        )

    def _check_pred_data(self, pred_data: PIPredictionStepData):
        assert pred_data.alpha is not None
        assert pred_data.eps_past is not None

    @property
    def can_handle_different_alpha(self):
        return True
