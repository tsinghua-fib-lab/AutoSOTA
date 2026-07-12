"""
Graph-Induced Sum-Product Networks

Files: baseline_mask.py

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
from typing import Tuple, Optional, List, Any

import torch
import torch.nn as nn
from pydgn.model.interface import ModelInterface
from torch.distributions import (
    Bernoulli, Normal, Independent,
)
from torch.nn import Linear, ReLU, Sequential, Identity, Sigmoid, BatchNorm1d
from torch.nn.functional import softplus, dropout
from torch.nn.parameter import Parameter
from torch_geometric.data import Batch
from pydgn.experiment.util import s2c
from torch_geometric.nn import global_add_pool, global_mean_pool, GINConv
from torch_geometric.utils import degree
from torch_scatter import scatter_mean, scatter_sum


class GaussianEmission(nn.Module):

    @staticmethod
    def log_likelihood(x, parameters):
        n = Normal(loc=parameters[:,:,0], scale=softplus(parameters[:,:,1]) + 1e-8)
        log_prob = n.log_prob(x).sum(-1) # sum logarithms along axis 1 == consider features as independent
        return log_prob

    def impute(self, params):
        return params[:,:,0], params


class MeanAggregation(ModelInterface):

    def __init__(
        self, dim_node_features, dim_edge_features, dim_target, readout_class, config
    ):
        super().__init__(
            dim_node_features, dim_edge_features, dim_target, readout_class, config
        )
        self.dummy_param = Parameter(torch.ones(1))
        self.mean_values = None

    def forward(self, data: Batch) -> Tuple[None, Any, List[Optional[Any]]]:
        """

        :param data:
        :return:
        """
        # extract data
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        node_mask = data.mask
        masked_nodes = torch.logical_not(node_mask)
        non_masked_nodes = node_mask

        x_original = x
        x_imputed = x.clone()

        x = x.clone()
        x_imputed[masked_nodes] = torch.nan

        if self.training and self.mean_values is None:
            self.mean_values = torch.nanmean(x_imputed, dim=0)

        mean_values = self.mean_values.unsqueeze(0).repeat(x_imputed.shape[0], 1)

        x[masked_nodes] = 0.

        # use sum because so that nan values (converted to 0) are not counted
        # then divide this according to the number of neighbors with feature != 0 for each feature
        x_sum_aggr = scatter_sum(x[edge_index[0]], edge_index[1], dim=0)

        v_degree = degree(edge_index[1], num_nodes=x_sum_aggr.shape[0])
        v_degree[v_degree == 0.] = 1.

        no_useful_neighboring_info = x_sum_aggr==0.

        # find features for which I have missing values AND my neighbors can give no contribution
        # and replace with mean values
        and_mask = torch.logical_and(masked_nodes, no_useful_neighboring_info)

        # Replace features for which we have no information from neighbors with mean values
        x_sum_aggr[and_mask] = mean_values[and_mask]
        imputed_values = x_sum_aggr/v_degree.unsqueeze(1)


        # substitute the computed values in the missing spots
        x_imputed = imputed_values

        return (
            None,
            x_imputed,
            [None, None, x_original, x_original, x_imputed, masked_nodes, non_masked_nodes, None, None, None],
        )


class GAE(ModelInterface):
    """
    Modified GAE that reconstructs node features rather than structure as in the original paper
    """
    def __init__(self, dim_node_features, dim_edge_features, dim_target, readout_class, config):
        super().__init__(dim_node_features, dim_edge_features, dim_target, readout_class, config)

        self.config = config
        self.dropout = config['dropout']
        self.num_layers = config['num_layers']
        self.embeddings_dim = [config['dim_embedding'] for _ in range(self.num_layers)]
        self.first_h = []
        self.nns = []
        self.convs = []
        self.linears = []
        self.concat_out_across_layers = config['concat_out_across_layers']

        train_eps = config['train_eps']

        for layer, out_emb_dim in enumerate(self.embeddings_dim):

            if layer == 0:
                self.first_h = Sequential(Linear(dim_node_features, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU(),
                                    Linear(out_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU())

                # We want to reconstruct the node features
                self.linears.append(Linear(out_emb_dim, dim_node_features*2))
            else:
                input_emb_dim = self.embeddings_dim[layer-1]
                self.nns.append(Sequential(Linear(input_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU(),
                                      Linear(out_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU()))
                self.convs.append(GINConv(self.nns[-1], train_eps=train_eps))  # Eq. 4.2

                # We want to reconstruct the node features
                self.linears.append(Linear(out_emb_dim, dim_node_features*2))

        self.nns = torch.nn.ModuleList(self.nns)
        self.convs = torch.nn.ModuleList(self.convs)
        self.linears = torch.nn.ModuleList(self.linears)  # has got one more for initial input

        self.mean_values = None


    def forward(self, data: Batch) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:

        # extract data
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        node_mask = data.mask
        masked_nodes = torch.logical_not(node_mask)
        non_masked_nodes = node_mask

        x_original = x
        x_imputed = x.clone()

        x_imputed[masked_nodes] = torch.nan

        if self.training and self.mean_values is None:
            self.mean_values = torch.nanmean(x_imputed, dim=0)

        mean_values = self.mean_values.unsqueeze(0).repeat(x_imputed.shape[0], 1)

        x_imputed[masked_nodes] = mean_values[masked_nodes]

        out = 0

        for layer in range(self.num_layers):
            if layer == 0:
                h = self.first_h(x_imputed)
                if self.concat_out_across_layers:
                    out += dropout(self.linears[layer](h), p=self.dropout)
            else:
                # Layer l ("convolution" layer)
                h = self.convs[layer-1](h, edge_index)
                if self.concat_out_across_layers:
                    out += dropout(self.linears[layer](h), p=self.dropout, training=self.training)

        if not self.concat_out_across_layers:
            out += dropout(self.linears[layer](h), p=self.dropout, training=self.training)

        # substitute the computed values in the missing spots

        x_imputed = out[:, :self.dim_node_features]

        dist = Normal(loc=out[:, :self.dim_node_features],
                      scale=softplus(out[:, self.dim_node_features:]))

        objective_v = dist.log_prob(x_original)
        objective_v_missing = objective_v[masked_nodes]

        objective_v = objective_v.squeeze()

        return (
            out,
            x_imputed,
            [objective_v, None, x_original, x_original, x_imputed, masked_nodes, non_masked_nodes, None, None, objective_v_missing],
        )
