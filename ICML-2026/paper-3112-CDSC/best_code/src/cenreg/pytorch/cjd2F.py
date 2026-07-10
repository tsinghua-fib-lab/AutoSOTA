import itertools
import os
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim


class MseModel(nn.Module):
    def __init__(
        self,
        jd_pred: np.ndarray,
        copula,
        learning_rate: float = 0.01,
        focal_risk: int = -1,
        init_f: np.ndarray | None = None,
        optimizer=None,
    ):
        super().__init__()
        if init_f is not None:
            if len(init_f.shape) != 3:
                raise ValueError("init_f must be a 3D array")
        if len(jd_pred.shape) != 3:
            raise ValueError("jd_pred must be a 3D array")

        self.jd_pred = torch.tensor(jd_pred, dtype=torch.float32).detach()
        self.focal_risk = focal_risk
        self.fc = nn.Linear(1, jd_pred.size, bias=False)
        self.shape = init_f.shape if init_f is not None else jd_pred.shape
        self.copula = copula
        self.learning_rate = learning_rate
        if init_f is not None:
            logf = np.log(init_f + 1e-9).reshape(jd_pred.size, 1)
            self.fc.weight.data = torch.tensor(logf, dtype=torch.float32)
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = optimizer

    def forward(self):
        # compute f_pred
        x = torch.ones(1)
        x = self.fc(x).view(self.shape)
        f_pred = torch.softmax(x, dim=2)

        # convert from f_pred to F_pred
        F_pred = torch.cumsum(f_pred, dim=2)
        zeros = torch.zeros((f_pred.shape[0], F_pred.shape[1], 1), device=f_pred.device)
        F_pred = torch.cat((zeros, F_pred), dim=2)
        return F_pred

    def loss(self):
        F_pred = self.forward()
        loss = self._mean_squared_error(F_pred, self.jd_pred, self.copula, self.focal_risk)
        return loss

    def configure_optimizers(self):
        return self.optimizer

    def _copula_sum_sub(
        self,
        F_pred: torch.Tensor,
        c,
        idx_list: list[list[int]],
        i: int,
        k: int,
        idx_list_use_Ft: Sequence[int],
    ):
        if i == k:
            idx = torch.tensor(idx_list, dtype=torch.long)
            idx = idx.view(1, idx.shape[0], idx.shape[1])
            idx_expand = idx.expand(F_pred.shape[0], idx.shape[1], idx.shape[2])
            return c.cdf(torch.gather(F_pred, 2, idx_expand))

        if i in idx_list_use_Ft:
            idx_list.append([j + 1 for j in range(F_pred.shape[2] - 1)])
        else:
            idx_list.append([F_pred.shape[2] - 1 for j in range(F_pred.shape[2] - 1)])
        temp1 = self._copula_sum_sub(F_pred, c, idx_list, i + 1, k, idx_list_use_Ft)
        idx_list.pop()
        idx_list.append(list(range(F_pred.shape[2] - 1)))
        temp2 = self._copula_sum_sub(F_pred, c, idx_list, i + 1, k, idx_list_use_Ft)
        idx_list.pop()
        return temp1 - temp2

    def _Fpred2jdpred(
        self,
        F_pred: torch.Tensor,
        copula,
        w: torch.Tensor,
        num_risks: int,
        k: int,
        list_k: list[int],
    ):
        """
        Convert F_pred into jd_pred.

        Parameters
        ----------
        F_pred: Tensor of shape [batch_size, num_risks, num_bin_predictions+1]

        copula: function

        w: Tensor of shape [num_risks].
            The sum of w must be equal to 1.0.

        num_risks: int
            The number of events.

        k: int
            The index of the risk.

        list_k: list of int
            The list of indices of the risks.

        Returns
        -------
        q: Tensor of shape [batch_size, num_bin_predictions]
        """

        q = self._copula_sum_sub(F_pred, copula, [], 0, num_risks, [k])
        sign = 1.0
        for i in range(num_risks + 1):
            if i < 2 or w[i] == 0.0:
                continue
            for v in itertools.combinations(list_k, i):
                if k in v:
                    q -= sign * w[i] * self._copula_sum_sub(F_pred, copula, [], 0, num_risks, v)
            sign *= -1.0
        return q

    def _mean_squared_error(self, F_pred: torch.Tensor, jd_pred: torch.Tensor, copula, focal_risk: int = -1):
        """
        Loss function to estimate marginal distribution from joint distribution.

        Parameters
        ----------
        F_pred: Tensor of shape [batch_size, num_risks, num_bin_predictions+1]

        jd_pred: Tensor of shape [batch_size, num_risks, num_bin_predictions]

        copula: function

        focal_risk: int

        Returns
        -------
        loss : Tensor of shape [batch_size]
        """

        num_risks = F_pred.shape[1]

        # initialize w0 and w1
        if focal_risk >= 0:
            w0 = torch.zeros(num_risks + 1, device=F_pred.device)
            w1 = torch.ones(num_risks + 1, device=F_pred.device)
        else:
            w0 = torch.tensor([0.0] + [1 / i for i in range(1, num_risks + 1)])
            w1 = torch.tensor([0.0] + [1 / i for i in range(1, num_risks + 1)])

        # compute q
        list_K = list(range(num_risks))
        loss = 0.0
        for k in range(num_risks):
            if k == focal_risk:
                q = self._Fpred2jdpred(F_pred, copula, w0, num_risks, k, list_K)
            else:
                q = self._Fpred2jdpred(F_pred, copula, w1, num_risks, k, list_K)
            loss += F.mse_loss(q, jd_pred[:, k, :]) / num_risks
        return loss


# @TODO to be removed
def minimize_mse(model, num_epochs: int) -> np.ndarray:
    """
    Estimate marginal distribution from joint distribution.

    Parameters
    ----------
    model: pytorch model

    num_epochs: int
        Number of epochs.

    Returns
    -------
    F_pred: estimated CDF.
        np.ndarray of shape [batch_size, num_risks, num_bin_predictions+1]
    """
    if num_epochs <= 0:
        raise ValueError("num_epochs must be greater than 0")

    best_epoch = -1
    best_loss = float("inf")
    path = None
    prev_path = None
    os.makedirs("tb_logs/weibull_postprocess", exist_ok=True)
    optimizer = model.configure_optimizers()

    with torch.enable_grad():
        model.train()
        optimizer.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()

            loss = model.loss().mean()

            if loss < best_loss:
                best_loss = loss
                best_epoch = epoch
                prev_path = path
                path = "tb_logs/weibull_postprocess/epoch" + str(best_epoch) + ".ckpt"
                # print("Saving model to", path)
                try:
                    torch.save(
                        {
                            "epoch": best_epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        },
                        path,
                    )
                except OSError:
                    print("Failed to save model to", path)
                if prev_path is not None:
                    try:
                        # print("Removing", prev_path)
                        os.remove(prev_path)
                    except OSError:
                        print("Failed to remove", prev_path)

            if epoch % 50 == 0:
                print(
                    f"epoch={epoch}, mean loss={loss.item():.9f} (best_epoch={best_epoch}, best loss={best_loss:.9f})"
                )
                print("model.theta", model.theta)
            loss.backward()
            optimizer.step()

    if path is None:
        raise ValueError("path must be set to a valid checkpoint path")
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    optimizer.eval()
    with torch.no_grad():
        F_pred = model.forward()
    return F_pred.detach().numpy()
