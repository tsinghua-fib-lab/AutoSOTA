import pandas as pd
import jax.numpy as jnp
import numpy as np
import torch
from joblib import Parallel, delayed
from SurvivalEVAL.Evaluator import LifelinesEvaluator
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.metrics.brier_score import BrierScore
from torchsurv.stats.ipcw import get_ipcw


class EvaluationMetrics:

    def __init__(
        self,
        risk,
        survival,
        event_train,
        time_train,
        event_test,
        time_test,
        method="bayesian",
    ):

        self.method = method
        self.time_array = np.array(time_test)
        self.event_array = np.array(event_test)
        self.time_train_array = np.array(time_train)
        self.event_train_array = np.array(event_train)

        # Convert to tensors
        self.risk = torch.tensor(risk.tolist(), dtype=torch.float32)
        if survival is not None:
            self.survival = torch.tensor(survival.tolist(), dtype=torch.float32)
            self.survival_array = self.survival
        else:
            self.survival = None
        self.time = torch.tensor(time_test.tolist(), dtype=torch.float32)
        self.event = torch.tensor(event_test.tolist(), dtype=torch.bool)
        time_train_tensor = torch.tensor(time_train.tolist(), dtype=torch.float32)
        event_train_tensor = torch.tensor(event_train.tolist(), dtype=torch.bool)

        # Get IPCW at test time using estimated censoring distribution on train
        ipcw = get_ipcw(event_train_tensor, time_train_tensor, self.time)

        # Initiate empty dir
        self.metrics = {}

        # Get C-index
        self.get_c_index("harrell")
        self.get_c_index("antolini")

        # Get IPCW C-index
        self.get_c_index("harrell", ipcw)
        self.get_c_index("antolini", ipcw)

        # Get IBS
        self.get_integrated_brier_score()

        # Get IPCW IBS
        self.get_integrated_brier_score(ipcw)

        # Get D-calibration and KM-calibration
        self.get_D_KM_calibration()

    def get_D_KM_calibration(self):
        if self.survival is None:
            d_calibration = None
            km_calibration = None
        else:
            # Get D and KM calibration
            surv_test_pd = pd.DataFrame(self.survival_array.T)
            surv_test_pd.index = self.time_array

            if surv_test_pd.isna().any().any():
                surv_test_pd = surv_test_pd.fillna(1.0)

            evl = LifelinesEvaluator(
                surv_test_pd,
                self.time_array,
                self.event_array,
                self.time_train_array,
                self.event_train_array,
            )
            d_calibration = evl.d_calibration()[0]
            km_calibration = evl.km_calibration()

        self.metrics["d_calibration"] = {
            "value": d_calibration,
        }
        self.metrics["km_calibration"] = {
            "value": km_calibration,
        }

    def get_c_index(self, name="harrell", ipcw=None):

        if name == "harrell":
            self.estimate = self.risk
        elif name == "antolini":
            if self.survival is not None:
                self.estimate = -self.survival
            else:
                self.estimate = None

        if ipcw is not None:
            weight = ipcw
            metric_name = "ipcw_c_index_" + name
        else:
            weight = torch.ones_like(self.time)
            metric_name = "c_index_" + name

        if self.estimate is None:
            cindex_eval = None
        elif self.method == "frequentist":
            cindex = ConcordanceIndex()
            cindex_eval = jnp.array(
                cindex(self.estimate, self.event, self.time, weight)
            )
        elif self.method == "bayesian":

            def compute_cindex(i):
                cindex = ConcordanceIndex()
                return cindex(self.estimate[:, :, i], self.event, self.time, weight)

            cindex_samples = Parallel(n_jobs=-1)(
                delayed(compute_cindex)(i) for i in range(self.estimate.shape[2])
            )

            # Stack into one tensor
            cindex_samples = torch.stack(cindex_samples)

            # Summarize C-index
            cindex_eval = torch.nanmedian(cindex_samples).numpy()

        self.metrics[metric_name] = {
            "value": cindex_eval,
        }

    def get_integrated_brier_score(self, ipcw=None):

        ibs_samples = []

        if ipcw is not None:
            weight = ipcw
            metric_name = "ipcw_ibs"
        else:
            weight = torch.ones_like(self.time)
            metric_name = "ibs"

        if self.survival is None:
            bs = None

        elif self.method == "frequentist":
            brier_score = BrierScore()
            brier_score(self.survival, self.event, self.time, weight=weight)
            brier_score.brier_score = torch.where(
                torch.isnan(brier_score.brier_score), 0.0, brier_score.brier_score
            )
            bs = jnp.array(brier_score.integral())

        elif self.method == "bayesian":

            def compute_brierscore(i):
                brier_score = BrierScore()
                return brier_score(
                    self.survival[:, :, i], self.event, self.time, weight=weight
                )

            ibs_samples = Parallel(n_jobs=-1)(
                delayed(compute_brierscore)(i) for i in range(self.survival.shape[2])
            )

            # Stack into one tensor
            ibs_samples = torch.stack(ibs_samples)

            # Summarize C-index
            bs = torch.nanmedian(ibs_samples).numpy()

        self.metrics[metric_name] = {
            "value": bs,
        }
