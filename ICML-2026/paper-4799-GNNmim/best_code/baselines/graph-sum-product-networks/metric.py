"""
Graph-Induced Sum-Product Networks

Files: metric.py

Authors:  Federico Errica (federico.errica@neclab.eu)
     Mathias Niepert (mathias.niepert@ki.uni-stuttgart.de)

NEC Laboratories Europe GmbH, Copyright (c) 2024, All rights reserved.

THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.

PROPRIETARY INFORMATION ---

SOFTWARE LICENSE AGREEMENT

ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY

BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
DOWNLOAD THE SOFTWARE.

This is a license agreement ("Agreement") between your academic institution
or non-profit organization or self (called "Licensee" or "You" in this
Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
Agreement).  All rights not specifically granted to you in this Agreement
are reserved for Licensor.

RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
ownership of any copy of the Software (as defined below) licensed under this
Agreement and hereby grants to Licensee a personal, non-exclusive,
non-transferable license to use the Software for noncommercial research
purposes, without the right to sublicense, pursuant to the terms and
conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
Agreement, the term "Software" means (i) the actual copy of all or any
portion of code for program routines made accessible to Licensee by Licensor
pursuant to this Agreement, inclusive of backups, updates, and/or merged
copies permitted hereunder or subsequently supplied by Licensor,  including
all or any file structures, programming instructions, user interfaces and
screen formats and sequences as well as any and all documentation and
instructions related to it, and (ii) all or any derivatives and/or
modifications created or made by You to any of the items specified in (i).

CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
proprietary to Licensor, and as such, Licensee agrees to receive all such
materials and to use the Software only in accordance with the terms of this
Agreement.  Licensee agrees to use reasonable effort to protect the Software
from unauthorized use, reproduction, distribution, or publication. All
publication materials mentioning features or use of this software must
explicitly include an acknowledgement the software was developed by NEC
Laboratories Europe GmbH.

COPYRIGHT: The Software is owned by Licensor.

PERMITTED USES:  The Software may be used for your own noncommercial
internal research purposes. You understand and agree that Licensor is not
obligated to implement any suggestions and/or feedback you might provide
regarding the Software, but to the extent Licensor does so, you are not
entitled to any compensation related thereto.

DERIVATIVES: You may create derivatives of or make modifications to the
Software, however, You agree that all and any such derivatives and
modifications will be owned by Licensor and become a part of the Software
licensed to You under this Agreement.  You may only use such derivatives and
modifications for your own noncommercial internal research purposes, and you
may not otherwise use, distribute or copy such derivatives and modifications
in violation of this Agreement.

BACKUPS:  If Licensee is an organization, it may make that number of copies
of the Software necessary for internal noncommercial use at a single site
within its organization provided that all information appearing in or on the
original labels, including the copyright and trademark notices are copied
onto the labels of the copies.

USES NOT PERMITTED:  You may not distribute, copy or use the Software except
as explicitly permitted herein. Licensee has not been granted any trademark
license as part of this Agreement.  Neither the name of NEC Laboratories
Europe GmbH nor the names of its contributors may be used to endorse or
promote products derived from this Software without specific prior written
permission.

You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
whole or in part, or provide third parties access to prior or present
versions (or any parts thereof) of the Software.

ASSIGNMENT: You may not assign this Agreement or your rights hereunder
without the prior written consent of Licensor. Any attempted assignment
without such consent shall be null and void.

TERM: The term of the license granted by this Agreement is from Licensee's
acceptance of this Agreement by downloading the Software or by using the
Software until terminated as provided below.

The Agreement automatically terminates without notice if you fail to comply
with any provision of this Agreement.  Licensee may terminate this Agreement
by ceasing using the Software.  Upon any termination of this Agreement,
Licensee will delete any and all copies of the Software. You agree that all
provisions which operate to protect the proprietary rights of Licensor shall
remain in force should breach occur and that the obligation of
confidentiality described in this Agreement is binding in perpetuity and, as
such, survives the term of the Agreement.

FEE: Provided Licensee abides completely by the terms and conditions of this
Agreement, there is no fee due to Licensor for Licensee's use of the
Software in accordance with this Agreement.

DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
RELATED MATERIALS.

SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
provided as part of this Agreement.

EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
permitted under applicable law, Licensor shall not be liable for direct,
indirect, special, incidental, or consequential damages or lost profits
related to Licensee's use of and/or inability to use the Software, even if
Licensor is advised of the possibility of such damage.

EXPORT REGULATION: Licensee agrees to comply with any and all applicable
export control laws, regulations, and/or other laws related to embargoes and
sanction programs administered by law.

SEVERABILITY: If any provision(s) of this Agreement shall be held to be
invalid, illegal, or unenforceable by a court or other tribunal of competent
jurisdiction, the validity, legality and enforceability of the remaining
provisions shall not in any way be affected or impaired thereby.

NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
or remedy under this Agreement shall be construed as a waiver of any future
or other exercise of such right or remedy by Licensor.

GOVERNING LAW: This Agreement shall be construed and enforced in accordance
with the laws of Germany without reference to conflict of laws principles.
You consent to the personal jurisdiction of the courts of this country and
waive their rights to venue outside of Germany.

ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
entire agreement between Licensee and Licensor as to the matter set forth
herein and supersedes any previous agreements, understandings, and
arrangements between the parties relating hereto.

THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
"""
from random import shuffle
from typing import List, Tuple

import numpy as np
import torch
from ogb import graphproppred
from pydgn.training.callback.metric import Metric, MulticlassAccuracy, Classification
from pydgn.training.event.state import State
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch import Tensor
from torch.nn.functional import binary_cross_entropy_with_logits

"""
#
#  NOTE: these metrics take the mean of the mini-batch scores, rather than an aggregated mean over all samples in the
#  dataset. This can be considered OK since we have enough data points (i.e., nodes) in the datasets
#
"""

class GSPNNodeLogLikelihood(Metric):

    @property
    def name(self) -> str:
        return "GSPN Node Negative Log Likelihood"

    def get_predictions_and_targets(self, targets: torch.Tensor, *outputs: List[torch.Tensor]) -> Tuple[
        torch.Tensor, torch.Tensor]:
        log_likelihood = outputs[2][0]
        non_masked_nodes = outputs[2][6]
        x = outputs[2][3]

        if non_masked_nodes is None:
            non_masked_nodes = torch.ones_like(x)

        return log_likelihood, non_masked_nodes

    def compute_metric(self, targets: torch.Tensor, predictions: torch.Tensor) -> torch.tensor:
        non_masked_nodes = targets
        log_likelihood = predictions


        if not torch.all(non_masked_nodes == 1.):
            num_non_masked_features = non_masked_nodes.sum(1)
            num_non_masked_features[num_non_masked_features == 0] = 1  # avoid division by zero

            # assert not torch.any(torch.isnan(-log_likelihood/num_non_masked_features))
            return (-log_likelihood/num_non_masked_features).mean()
        else:
            # print(log_likelihood.shape, log_likelihood.mean().item())
            return -log_likelihood.mean()


class MissingFeaturesMSE(Metric):

    @property
    def name(self) -> str:
        return "GSPN Missing Node Features MSE"

    def get_predictions_and_targets(self, targets: torch.Tensor, *outputs: List[torch.Tensor]) -> Tuple[
        torch.Tensor, torch.Tensor]:
        x, imputed_values = outputs[2][3], outputs[2][4]
        masked_nodes = outputs[2][5]

        return imputed_values[masked_nodes], x[masked_nodes]


    def compute_metric(self, targets: torch.Tensor, predictions: torch.Tensor) -> torch.tensor:
        return torch.nn.functional.mse_loss(predictions, targets)


class ConditionalMeanImputationLikelihood(Metric):
    @property
    def name(self) -> str:
        return "GSPN Conditional Mean Imputation Likelihood"

    def get_predictions_and_targets(self, targets: torch.Tensor, *outputs: List[torch.Tensor]) -> Tuple[
        torch.Tensor, torch.Tensor]:
        log_likelihood_missing_v = outputs[2][9]
        masked_nodes = outputs[2][5]
        num_masked_features = masked_nodes.sum(1, keepdim=True)

        return log_likelihood_missing_v, num_masked_features

    def compute_metric(self, targets: torch.Tensor, predictions: torch.Tensor) -> torch.tensor:
        num_masked_features = targets
        log_likelihood_missing_v = predictions

        # Filter out nodes for which there are no missing features
        # log_likelihood_missing_v = log_likelihood_missing_v[num_masked_features.squeeze() != 0]
        # num_masked_features = num_masked_features[num_masked_features.squeeze() != 0, :]
        # return (-log_likelihood_missing_v/num_masked_features.squeeze()).mean()

        return (-log_likelihood_missing_v).mean()


class FakeLoss(Metric):
    @property
    def name(self) -> str:
        return "Fake Loss"

    def get_predictions_and_targets(self, targets: torch.Tensor, *outputs: List[torch.Tensor]) -> Tuple[
        torch.Tensor, torch.Tensor]:
        return torch.zeros(1), torch.zeros(1)

    def compute_metric(self, targets: torch.Tensor, predictions: torch.Tensor) -> torch.tensor:
        return torch.zeros(1)

    def on_backward(self, state: State):
        pass


class BCEWithLogits(Classification):
    @property
    def name(self) -> str:
        return "BCE With Logits"

    def get_predictions_and_targets(self, targets: torch.Tensor, *outputs: List[torch.Tensor]) -> Tuple[
        torch.Tensor, torch.Tensor]:

        pred = outputs[0]

        if len(targets.shape) == 2:
            targets = targets.squeeze(dim=1)

        not_nan_mask = torch.logical_not(torch.isnan(targets))
        targets = targets[not_nan_mask]
        pred = pred[not_nan_mask]
        return pred, targets

    def compute_metric(self, targets: torch.Tensor, predictions: torch.Tensor) -> torch.tensor:
        bce = binary_cross_entropy_with_logits(predictions, targets)
        return bce


class OGBGROCAUC(MulticlassAccuracy):

    def __init__(self, use_as_loss: bool=False, reduction: str='mean',
                 accumulate_over_epoch: bool=True, force_cpu: bool=True):
        super().__init__(use_as_loss, reduction, accumulate_over_epoch, force_cpu)
        self.evaluator_name = 'ogbg-molpcba'
        self.evaluator = graphproppred.Evaluator(name=self.evaluator_name)

    @property
    def name(self) -> str:
        return "ROC-AUC"

    def get_predictions_and_targets(self, targets: torch.Tensor, *outputs: List[torch.Tensor]) -> Tuple[
        torch.Tensor, torch.Tensor]:

        pred = outputs[0]

        if len(targets.shape) == 2:
            targets = targets.squeeze(dim=1)

        targets = targets.detach().cpu()
        pred = pred.detach().cpu()

        return pred, targets

    def compute_metric(self, targets: torch.Tensor, predictions: torch.Tensor) -> torch.tensor:
        rocauc_list = []

        # adapted from ogb repository to use torch.sum in place of np.sum

        for i in range(targets.shape[1]):
            #AUC is only defined when there is at least one positive data.
            if torch.sum(targets[:,i] == 1) > 0 and torch.sum(targets[:,i] == 0) > 0:
                # ignore nan values
                is_labeled = targets[:,i] == targets[:,i]
                rocauc_list.append(roc_auc_score(targets[is_labeled,i], predictions[is_labeled,i]))

        if len(rocauc_list) == 0:
            raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

        return sum(rocauc_list)/len(rocauc_list)


class OGBGAP(MulticlassAccuracy):

    def __init__(self, use_as_loss: bool=False, reduction: str='mean',
                 accumulate_over_epoch: bool=True, force_cpu: bool=True):
        super().__init__(use_as_loss, reduction, accumulate_over_epoch, force_cpu)
        self.evaluator_name = 'ogbg-molpcba'
        self.evaluator = graphproppred.Evaluator(name=self.evaluator_name)

    @property
    def name(self) -> str:
        return "AP"

    def get_predictions_and_targets(self, targets: torch.Tensor, *outputs: List[torch.Tensor]) -> Tuple[
        torch.Tensor, torch.Tensor]:

        pred = outputs[0]

        if len(targets.shape) == 2:
            targets = targets.squeeze(dim=1)

        targets = targets.detach().cpu()
        pred = pred.detach().cpu()

        return pred, targets

    def compute_metric(self, targets: torch.Tensor, predictions: torch.Tensor) -> torch.tensor:
        ap_list = []

        # adapted from ogb repository to use torch.sum in place of np.sum

        for i in range(targets.shape[1]):
            #AP is only defined when there is at least one positive data.
            if torch.sum(targets[:,i] == 1) > 0 and torch.sum(targets[:,i] == 0) > 0:
                # ignore nan values
                is_labeled = targets[:,i] == targets[:,i]
                ap_list.append(average_precision_score(targets[is_labeled,i], predictions[is_labeled,i]))

        if len(ap_list) == 0:
            raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

        return sum(ap_list)/len(ap_list)


class OGBGrahPropPredEvaluator(Metric):
    @property
    def name(self) -> str:
        return f"OGB GrahPropPred Evaluator: {self.evaluator_name}"

    def __init__(self, use_as_loss: bool=False, reduction: str='mean',
                 use_nodes_batch_size: bool=False,
                 accumulate_over_time_steps: bool=False,
                 evaluator_name: str=None,
                 score_name: str=None):
        super().__init__(use_as_loss, reduction, use_nodes_batch_size, accumulate_over_time_steps)
        self.evaluator_name = evaluator_name
        self.score_name = score_name
        self.evaluator = graphproppred.Evaluator(name=evaluator_name)

    def get_predictions_and_targets(self, targets: torch.Tensor, *outputs: List[torch.Tensor]) -> Tuple[
        torch.Tensor, torch.Tensor]:

        pred = outputs[0]

        if len(targets.shape) == 2:
            targets = targets.squeeze(dim=1)

        # print(self.evaluator.expected_input_format)
        # print(self.evaluator.expected_output_format)

        targets = targets.detach().cpu()
        pred = pred.detach().cpu()

        return pred, targets

    def compute_metric(self, targets: torch.Tensor, predictions: torch.Tensor) -> torch.tensor:
        targets = targets.detach().cpu()
        predictions = predictions.detach().cpu()

        input_dict = {"y_true": targets, "y_pred": predictions}
        result_dict = self.evaluator.eval(input_dict)  # E.g., {"rocauc": 0.7321}

        return torch.tensor([result_dict[self.score_name]])


class DotProductLink(Metric):
    """
    Implements a dot product link prediction metric,
    as defined in https://arxiv.org/abs/1611.07308.
    """

    @property
    def name(self) -> str:
        """
        The name of the loss to be used in configuration files and displayed
        on Tensorboard
        """
        return "Dot Product Link Prediction"

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Uses node embeddings (outputs[1]) aand positive/negative edges
        (contained in targets by means of
        e.g.,
        a :obj:`~pydgn.data.provider.LinkPredictionSingleGraphDataProvider`)
        to return logits and target labels of an edge classification task.
        Args:
            targets (:class:`torch.Tensor`): ground truth
            outputs (List[:class:`torch.Tensor`]): outputs of the model
        Returns:
            A tuple of tensors (predicted_values, target_values)
        """
        node_embs = outputs[1]
        pos_edges, neg_edges = outputs[2][0], outputs[2][1]

        loss_edge_index = torch.cat((pos_edges, neg_edges), dim=1)
        loss_target = torch.cat(
            (torch.ones(pos_edges.shape[1]), torch.zeros(neg_edges.shape[1]))
        )

        # Taken from
        # rusty1s/pytorch_geometric/blob/master/examples/link_pred.py
        x_j = torch.index_select(node_embs, 0, loss_edge_index[0])
        x_i = torch.index_select(node_embs, 0, loss_edge_index[1])
        link_logits = torch.einsum("ef,ef->e", x_i, x_j)

        return link_logits, loss_target.to(link_logits.device)

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.tensor:
        """
        Applies BCEWithLogits to link logits and targets.
        Args:
            targets (:class:`torch.Tensor`): tensor of ground truth values
            predictions (:class:`torch.Tensor`):
                tensor of predictions of the model
        Returns:
            A tensor with the metric value
        """
        metric = torch.nn.functional.binary_cross_entropy_with_logits(
            predictions, targets
        )
        return metric


class DotProductAccuracy(Metric):
    @property
    def name(self) -> str:
        """
        The name of the loss to be used in configuration files and displayed
        on Tensorboard
        """
        return "Dot Product Link Prediction Accuracy"

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Uses node embeddings (outputs[1]) aand positive/negative edges
        (contained in targets by means of
        e.g.,
        a :obj:`~pydgn.data.provider.LinkPredictionSingleGraphDataProvider`)
        to return logits and target labels of an edge classification task.
        Args:
            targets (:class:`torch.Tensor`): ground truth
            outputs (List[:class:`torch.Tensor`]): outputs of the model
        Returns:
            A tuple of tensors (predicted_values, target_values)
        """
        node_embs = outputs[1]
        pos_edges, neg_edges = outputs[2][0], outputs[2][1]

        loss_edge_index = torch.cat((pos_edges, neg_edges), dim=1)
        loss_target = torch.cat(
            (torch.ones(pos_edges.shape[1]), torch.zeros(neg_edges.shape[1]))
        )

        # Taken from
        # rusty1s/pytorch_geometric/blob/master/examples/link_pred.py
        x_j = torch.index_select(node_embs, 0, loss_edge_index[0])
        x_i = torch.index_select(node_embs, 0, loss_edge_index[1])
        link_logits = torch.einsum("ef,ef->e", x_i, x_j)

        return torch.sigmoid(link_logits), loss_target.to(link_logits.device)

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.tensor:
        """
        Applies BCEWithLogits to link logits and targets.
        Args:
            targets (:class:`torch.Tensor`): tensor of ground truth values
            predictions (:class:`torch.Tensor`):
                tensor of predictions of the model
        Returns:
            A tensor with the metric value
        """

        if len(targets.shape) == 2:
            targets = targets.squeeze(dim=1)
        pred = (predictions >= 0.5).float()

        metric = (
            100.0 * (pred == targets).sum().float() / targets.size(0)
        )
        return metric


class DGILoss(Metric):

    @property
    def name(self) -> str:
        return "DGI Unsupervised Loss"

    def discriminate(self,
                     z: Tensor,
                     summary: Tensor,
                     sigmoid: bool = True) -> Tensor:
        r"""Given the patch-summary pair :obj:`z` and :obj:`summary`, computes
        the probability scores assigned to this patch-summary pair.

        Args:
            z (Tensor): The latent space.
            summary (Tensor): The summary vector.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        summary = summary.t() if summary.dim() > 1 else summary
        value = torch.matmul(z, summary)
        return torch.sigmoid(value) if sigmoid else value

    def get_predictions_and_targets(self, targets: torch.Tensor, *outputs: List[torch.Tensor]) -> Tuple[
        torch.Tensor, torch.Tensor]:
        pos_z, neg_z, summary = outputs[2][0], outputs[2][1], outputs[2][2]

        EPS = 1e-15

        pos_loss = -torch.log(
            self.discriminate(pos_z, summary, sigmoid=True) + EPS).mean()

        neg_loss = -torch.log(1 -
                              self.discriminate(neg_z, summary, sigmoid=True) +
                              EPS).mean()

        loss = (pos_loss + neg_loss).unsqueeze(0)
        return loss, loss
        # framework needs at least two vectors to pass to compute metric
        # we are bypassing that mechanism here

    def compute_metric(self, targets: torch.Tensor, predictions: torch.Tensor) -> torch.tensor:
        return predictions.mean(0)
