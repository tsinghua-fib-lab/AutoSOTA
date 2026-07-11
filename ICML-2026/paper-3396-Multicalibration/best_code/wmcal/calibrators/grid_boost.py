# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from math import ceil
from typing import Iterator, Literal, NamedTuple, Optional, Tuple, override

import numpy as np
from tqdm import tqdm, trange

from ..data import Dataset
from ..data.datasets.synthetic import SyntheticDataset
from ..predictors import Predictor
from ..utils import (
    get_experiment_config,
    get_logger,
    get_rng,
    xover,
)
from ..utils.grid_utils import create_grid_sampled
from . import Calibrator, CalibratorConfig

logger = get_logger(__name__)


class DeltaWeight(NamedTuple):
    g: Optional[int]  # The index of the weight in the grid to update
    sign: int  # +1 for increase, -1 for decrease
    source: Literal["check1", "check2"]  # Which check triggered this update


@dataclass
class GridBoostCalibratorConfig(CalibratorConfig):
    output_dim: int
    # Optimization parameters
    eps: float
    # Grid parameters
    grid_iter_size: int
    grid_resolution: float
    grid_size: int
    # Training parameters
    batch_size: int
    max_iter: int
    early_stop: bool
    # Check priority
    check2_prob: float
    eps_start: float | None = None  # If set, eps anneals from eps_start to eps over max_iter

    def __post_init__(self) -> None:
        assert self.check2_prob >= 0 and self.check2_prob <= 1, "check2_prob must be in [0, 1]"


class GridBoostCalibrator(Calibrator):
    def __init__(
        self,
        config: GridBoostCalibratorConfig,
        predictor: Predictor,
        dataset: Dataset,
    ) -> None:
        super().__init__(config, predictor, dataset)
        self.eps_start = config.eps_start
        self.eps_end = config.eps
        if self.eps_start is not None and self.eps_start != self.eps_end:
            self.eps = self.eps_start
            self.eps_decay = (self.eps_end / self.eps_start) ** (1.0 / config.max_iter)
            self._anneal = True
        else:
            self.eps = config.eps
            self._anneal = False
        self.alpha = self.eps / (2 * dataset.scale())
        self.lr = self.alpha / 2
        self.output_dim = config.output_dim
        self.grid_resolution = config.grid_resolution
        self.grid_iter_size = config.grid_iter_size  # Gb
        self.batch_size = config.batch_size
        self.max_iter = config.max_iter
        self.early_stop = config.early_stop
        self.check2_prob = config.check2_prob

        top_k = getattr(self.dataset, "top_k", None)
        top_k = top_k if top_k == 1 else None
        experiment_config = get_experiment_config()
        assert experiment_config is not None, "Experiment config must be set before initializing GridBoostCalibrator"
        seed = experiment_config.seed
        self.grid = create_grid_sampled(
            self.output_dim,
            self.grid_resolution,
            config.grid_size,
            top_k,
            seed=seed,
        )

        self._init_state()

    def _init_state(self) -> None:
        self.t = 0
        self.delta_weights = []

    def _grid_iter(self, y_pred: np.ndarray) -> Iterator[Tuple[int, int, np.ndarray]]:
        for g in range(0, self.grid.shape[0], self.grid_iter_size):
            weights = self.grid[g : g + self.grid_iter_size]  # (Gb, D)
            g_end = g + weights.shape[0]

            # (Gb, 1, D) * (1, B, D) -> (Gb, B, D)
            wy_pred = weights[:, np.newaxis, :] * y_pred[np.newaxis, :, :]
            y_ohe = self.dataset.decision_function(wy_pred)
            yield g, g_end, y_ohe

    def _step(self, y_pred: np.ndarray, y_base: np.ndarray, dw: DeltaWeight) -> np.ndarray:
        if dw.source == "check1":
            delta = self.dataset.decision_function(y_base * self.grid[dw.g][np.newaxis, :]) * dw.sign
        else:
            delta = self.dataset.decision_function(y_pred) * dw.sign

        return np.clip(y_pred + self.lr * delta, 0.0, 1.0)

    @override
    def predict(self, X: np.ndarray, y_base: Optional[np.ndarray] = None) -> np.ndarray:
        if y_base is None:
            y_base = self.predictor.predict(X)

        y_pred = y_base.copy()
        for delta_weight in self.delta_weights:
            y_pred = self._step(y_pred, y_base, delta_weight)

        return y_pred

    def _check1(self, y: np.ndarray, y_base: np.ndarray, y_pred: np.ndarray) -> DeltaWeight | None:
        diff = y - y_pred  # (b, d)

        for g_start, g_end, y_ohe in self._grid_iter(y_base):
            # y_ohe: (g, b, d)
            errs = 2 * (diff[np.newaxis, :] * y_ohe).sum(axis=-1).mean(axis=-1)
            crossing_idx = xover(np.abs(errs), self.eps)

            if crossing_idx == -1:
                continue

            err = errs[crossing_idx]
            sign = int(np.sign(err))
            g_star = g_start + crossing_idx  # The violating weight index
            return DeltaWeight(g=g_star, sign=sign, source="check1")

        return None

    def _check2(self, y: np.ndarray, y_pred: np.ndarray) -> DeltaWeight | None:
        diff = y - y_pred
        ohe = self.dataset.decision_function(y_pred)  # (b, d)

        err = 2 * (diff * ohe).sum(axis=-1).mean(axis=-1)  # Average bias
        if np.abs(err) < self.eps:
            return None

        sign = int(np.sign(err))
        return DeltaWeight(g=None, sign=sign, source="check2")  # g=None indicates EW update

    def _test_pre(self) -> None:
        config = get_experiment_config()
        assert config is not None, "Experiment config must be set before running _test_pre"

        X, y_true = self.dataset.load_test()
        y_base = self.predictor.predict(X)

        # Pre-calibration MSE
        mse = float(np.mean((y_true - y_base) ** 2))
        logger.log_metric("mse", mse, t=self.t)

        max_util = (y_true * self.dataset.decision_function(y_true)).sum(axis=1).mean()
        self._max_util = max_util  # Store for post-calibration logging
        logger.log_metric("util", max_util, type="oracle")

        best_util, best_g = -1, -1
        grid_iter = tqdm(
            self._grid_iter(y_base),
            desc=" " * 17 + "Evaluating grid",
            total=ceil(len(self.grid) / self.grid_iter_size),
        )

        for g_start, g_end, y_ohe in grid_iter:
            # y_ohe: (Gb, B, D)
            for g in range(g_start, g_end):
                # Compute utility
                util = float((y_true * y_ohe[g - g_start]).sum() / y_true.shape[0])
                logger.log_metric("util", util, type="grid", g=g)

                if util >= best_util:
                    best_util, best_g = util, g

        logger.info(f"Best g={best_g}: {best_util:.3f} -> {best_util / max_util:.3%} of oracle ({max_util:.3f})")

        # Evaluate EW
        y_ohe = self.dataset.decision_function(y_base)
        util = float((y_true * y_ohe).sum() / y_true.shape[0])
        logger.log_metric("util", util, type="ew", t=self.t)
        logger.info(f"Pre-calibration EW: {util:.3f} -> {util / max_util:.3%} of oracle")

    @override
    def fit(self) -> None:
        """Fit GridBoostCalibrator to dataset.

        If use_fresh_samples=True, generates new synthetic samples at each iteration.
        Otherwise, samples from calibration dataset.
        """

        config = get_experiment_config()
        assert config is not None, "Experiment config must be set before fitting GridBoostCalibrator"
        rng = get_rng()
        self._test_pre()

        self._init_state()
        tab = " " * 17
        n_updates = 0

        pbar = trange(1, self.max_iter + 1, desc=tab + "Training GridBoostCalibrator")

        X_test, y_test = self.dataset.load_test()
        y_test_base = self.predictor.predict(X_test)
        y_test_pred = self.predict(X_test, y_test_base)

        for self.t in pbar:
            # Epsilon annealing: decay eps from eps_start to eps_end
            if self._anneal:
                self.eps = self.eps_start * (self.eps_decay ** self.t)
                self.alpha = self.eps / (2 * self.dataset.scale())
                self.lr = self.alpha / 2

            if isinstance(self.dataset, SyntheticDataset):
                X_batch, y_batch = self.dataset.synth(self.batch_size)
            else:
                raise NotImplementedError("GridBoostCalibrator currently only supports SyntheticDataset")

            y_base = self.predictor.predict(X_batch)
            y_pred = self.predict(X_batch, y_base)

            if rng.random() < self.check2_prob:
                dw = self._check2(y_batch, y_pred)
                dw = self._check1(y_batch, y_base, y_pred) if dw is None else dw
            else:
                dw = self._check1(y_batch, y_base, y_pred)
                dw = self._check2(y_batch, y_pred) if dw is None else dw

            if dw is None:
                if self.early_stop:
                    break
            else:
                self.delta_weights.append(dw)
                n_updates += 1

            # Compute and log metrics on test data
            if dw is not None:
                y_test_pred = self._step(y_test_pred, y_test_base, dw)
            elif self.t != self.max_iter:
                # Skip test data evaluation if not the last iteration
                continue

            y_test_ohe = self.dataset.decision_function(y_test_pred)

            # Compute utility
            util = float((y_test * y_test_ohe).sum() / y_test.shape[0])
            logger.log_metric("util", util, type="ew", t=self.t)
            util_pct = util / self._max_util * 100

            if self.t == self.max_iter:
                logger.info(f"Post-calibration EW: {util:.3f} -> {util / self._max_util:.3%} of oracle")

            # Compute MSE
            mse = float(np.mean((y_test - y_test_pred) ** 2))
            logger.log_metric("mse", mse, t=self.t)
            pbar.set_postfix(updates=n_updates, mse=f"{mse:.3f}", util=f"{util_pct:.1f}%")
