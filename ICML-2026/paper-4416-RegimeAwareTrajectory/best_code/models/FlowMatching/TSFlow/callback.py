# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import numpy as np
import torch
from gluonts.evaluation import (
    Evaluator,
    MultivariateEvaluator,
    make_evaluation_predictions,
)
from pytorch_lightning import Callback
from tqdm.auto import tqdm

import sys
sys.path.extend(['./', '../', '../../'])
from models.FlowMatching.TSFlow.utils import Setting
from models.FlowMatching.TSFlow.utils.plots import plot_figures
from models.FlowMatching.TSFlow.utils.util import create_splitter


class EvaluateCallback(Callback):
    def __init__(
        self,
        context_length,
        prediction_length,
        num_samples,
        model,
        datasets,
        logdir,
        eval_every=1,
        **kwargs,
    ):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.model = model
        self.datasets = datasets
        self.logdir = logdir
        self.eval_every = eval_every
        self.log_metrics = (
            {
                "CRPS",
                "ND",
                "NRMSE",
                "m_sum_CRPS",
            }
            if model.setting == Setting.MULTIVARIATE
            else {"CRPS", "ND", "NRMSE"}
        )

    def on_train_epoch_end(self, trainer, pl_module):
        if (pl_module.current_epoch + 1) % self.eval_every == 0:
            device = pl_module.device

            pl_module.eval()
            assert pl_module.training is False
            prediction_splitter = create_splitter(
                max(
                    self.context_length + max(self.model.lags_seq),
                    self.model.prior_context_length,
                ),
                self.prediction_length,
                mode="test",
            )
            for tag, dataset in self.datasets.items():
                pl_module.num_samples = self.num_samples

                if pl_module.setting == Setting.UNIVARIATE:
                    batch_size = 1024 // self.num_samples
                    evaluator = Evaluator(num_workers=1)
                elif pl_module.setting == Setting.MULTIVARIATE:
                    batch_size = 1
                    evaluator = MultivariateEvaluator(target_agg_funcs={"sum": np.sum})

                predictor_pytorch = pl_module.get_predictor(
                    prediction_splitter,
                    batch_size=batch_size,
                    device=device,
                )
                forecast_it, ts_it = make_evaluation_predictions(
                    dataset=dataset,
                    predictor=predictor_pytorch,
                    num_samples=self.num_samples,
                )

                forecasts_pytorch = list(tqdm(forecast_it, total=len(dataset)))
                tss_pytorch = list(ts_it)
                metrics_pytorch, per_ts = evaluator(tss_pytorch, forecasts_pytorch)
                metrics_pytorch["CRPS"] = metrics_pytorch["mean_wQuantileLoss"]
                if pl_module.setting == Setting.UNIVARIATE:
                    plot_figures(
                        tss_pytorch[0:4],
                        forecasts_pytorch[0:4],
                        pl_module.context_length,
                        pl_module.prediction_length,
                        trainer,
                        set=f"{tag}",
                    )
                else:
                    metrics_pytorch["m_sum_CRPS"] = metrics_pytorch[
                        "m_sum_mean_wQuantileLoss"
                    ]
                if metrics_pytorch["CRPS"] < pl_module.best_crps and tag == "val":
                    pl_module.best_crps = metrics_pytorch["CRPS"]
                    ckpt_path = Path(self.logdir) / "best_checkpoint.ckpt"
                    torch.save(
                        pl_module.state_dict(),
                        ckpt_path,
                    )
                pl_module.log_dict(
                    {
                        f"{tag}_{metric}": metrics_pytorch[metric]
                        for metric in self.log_metrics
                    }
                )
            pl_module.train()
