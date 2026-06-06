# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""

import torch as t
import torch.nn as nn
import numpy as np
import pdb


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = 0.0
    result[result == np.inf] = 0.0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(
        self,
        insample: t.Tensor,
        freq: int,
        forecast: t.Tensor,
        target: t.Tensor,
        mask: t.Tensor,
    ) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(
        self,
        insample: t.Tensor,
        freq: int,
        forecast: t.Tensor,
        target: t.Tensor,
        mask: t.Tensor,
    ) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(
            divide_no_nan(
                t.abs(forecast - target), t.abs(forecast.data) + t.abs(target.data)
            )
            * mask
        )


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(
        self,
        insample: t.Tensor,
        freq: int,
        forecast: t.Tensor,
        target: t.Tensor,
        mask: t.Tensor,
    ) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)


def id_contrastive_loss(z1, z2, id):
    id = id.cpu().detach().numpy()
    str_pid = [str(i) for i in id]
    str_pid = np.array(str_pid, dtype=object)
    pid1, pid2 = np.meshgrid(str_pid, str_pid)
    pid_matrix = pid1 + "-" + pid2
    pids_of_interest = np.unique(
        str_pid + "-" + str_pid
    )  # unique combinations of pids of interest i.e. matching
    bool_matrix_of_interest = np.zeros((len(str_pid), len(str_pid)))
    for pid in pids_of_interest:
        bool_matrix_of_interest += pid_matrix == pid
    rows1, cols1 = np.where(
        np.triu(bool_matrix_of_interest, 1)
    )  # upper triangle same patient combs
    rows2, cols2 = np.where(
        np.tril(bool_matrix_of_interest, -1)
    )  # down triangle same patient combs

    B, H = z1.size(0), z1.size(1)
    loss = 0
    z1 = t.nn.functional.normalize(z1, dim=1)
    z2 = t.nn.functional.normalize(z2, dim=1)
    # B x H
    view1_array = z1
    view2_array = z2
    norm1_vector = view1_array.norm(dim=1).unsqueeze(0)
    norm2_vector = view2_array.norm(dim=1).unsqueeze(0)
    sim_matrix = t.mm(view1_array, view2_array.transpose(0, 1))
    norm_matrix = t.mm(norm1_vector.transpose(0, 1), norm2_vector)
    temperature = 0.1
    argument = sim_matrix / (norm_matrix * temperature)
    sim_matrix_exp = t.exp(argument)

    # diag_elements = t.diag(sim_matrix_exp)

    triu_sum = t.sum(sim_matrix_exp, 1)  # add column
    tril_sum = t.sum(sim_matrix_exp, 0)  # add row

    # loss_diag1 = -t.mean(t.log(diag_elements/triu_sum))
    # loss_diag2 = -t.mean(t.log(diag_elements/tril_sum))

    # loss = loss_diag1 + loss_diag2
    # loss_terms = 2
    loss_terms = 0

    # upper triangle same patient combs exist
    if len(rows1) > 0:
        triu_elements = sim_matrix_exp[
            rows1, cols1
        ]  # row and column for upper triangle same patient combinations
        loss_triu = -t.mean(t.log(triu_elements / triu_sum[rows1]))
        loss += loss_triu  # technically need to add 1 more term for symmetry
        loss_terms += 1

    # down triangle same patient combs exist
    if len(rows2) > 0:
        tril_elements = sim_matrix_exp[
            rows2, cols2
        ]  # row and column for down triangle same patient combinations
        loss_tril = -t.mean(t.log(tril_elements / tril_sum[cols2]))
        loss += loss_tril  # technically need to add 1 more term for symmetry
        loss_terms += 1

    if loss_terms == 0:
        return 0
    else:
        loss = loss / loss_terms
        return loss
