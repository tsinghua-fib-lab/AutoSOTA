"""
Graph-Induced Sum-Product Networks

Files: readout.py

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
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
from pydgn.experiment.util import s2c
from pydgn.model.interface import ReadoutInterface, ModelInterface
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from model import exp_normalize_trick


class LinearGraphClassifier_GlobalReadout(ModelInterface):
    """
    This MLP computes a prediction starting from node embeddings
    """
    def __init__(self, dim_node_features, dim_edge_features, dim_target, readout_class, config):
        super().__init__(dim_node_features, dim_edge_features, dim_target, readout_class, config)

        if config['global_pooling'] == 'sum':
            self.global_pooling = global_add_pool
        elif config['global_pooling'] == 'mean':
            self.global_pooling = global_mean_pool
        elif config['global_pooling'] == 'max':
            self.global_pooling = global_max_pool
        else:
            raise NotImplementedError("Global pooling operator not recognized")

        self.out = torch.nn.Linear(dim_node_features, dim_target)

    def forward(self, data, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:
        node_embeddings, batch = data.x, data.batch
        graph_embeddings = self.global_pooling(node_embeddings, batch)
        out = self.out(graph_embeddings)
        return out, graph_embeddings


class MLPGraphClassifier_GlobalReadout(ModelInterface):
    """
    This MLP computes a prediction starting from node embeddings
    """
    def __init__(self, dim_node_features, dim_edge_features, dim_target, readout_class, config):
        super().__init__(dim_node_features, dim_edge_features, dim_target, readout_class, config)

        if config['global_pooling'] == 'sum':
            self.global_pooling = global_add_pool
        elif config['global_pooling'] == 'mean':
            self.global_pooling = global_mean_pool
        elif config['global_pooling'] == 'max':
            self.global_pooling = global_max_pool
        else:
            raise NotImplementedError("Global pooling operator not recognized")

        hidden_units = config['hidden_units']
        self.fc_local = torch.nn.Linear(dim_node_features, hidden_units)
        self.fc_global = torch.nn.Linear(hidden_units, hidden_units)
        self.out = torch.nn.Linear(hidden_units, dim_target)

    def forward(self, data, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:
        node_embeddings, batch = data.x, data.batch

        l = torch.relu(self.fc_local(node_embeddings))
        g = self.global_pooling(l, batch)
        o = self.fc_global(g)
        out = self.out(F.relu(o))
        return out, g


class MLPGraphClassifier_GraphEmbedding(ModelInterface):
    """
    This MLP computes a prediction starting from graph embeddings, without transforming the node embeddings
    """
    def __init__(self, dim_node_features, dim_edge_features, dim_target, readout_class, config):
        super().__init__(dim_node_features, dim_edge_features, dim_target, readout_class, config)

        if config['global_pooling'] == 'sum':
            self.global_pooling = global_add_pool
        elif config['global_pooling'] == 'mean':
            self.global_pooling = global_mean_pool
        elif config['global_pooling'] == 'max':
            self.global_pooling = global_max_pool
        else:
            raise NotImplementedError("Global pooling operator not recognized")

        hidden_units = config['hidden_units']
        self.fc_global = torch.nn.Linear(dim_node_features, hidden_units)
        self.out = torch.nn.Linear(hidden_units, dim_target)

    def forward(self, data, **kwargs) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:
        node_embeddings, batch = data.x, data.batch

        g = self.global_pooling(node_embeddings, batch)
        o = self.fc_global(g)
        out = self.out(F.relu(o))
        return out, g


class ProbabilisticGraphReadout(ReadoutInterface):
    """
    This is a probabilistic readout for predicting graph-related targets
    """
    def __init__(self, dim_node_features, dim_edge_features, dim_target, config):
        super().__init__(dim_node_features, dim_edge_features, dim_target, config)

        node_embedding_dim = config['num_mixtures']
        num_graph_mixtures = config['num_mixtures']
        num_layers = config['num_layers']
        self.num_hidden_neurons = config['num_hidden_neurons']  # same number of hidden neurons for all MLPs involved

        # P(Q_u = c)
        prob_vec = torch.nn.init.uniform_(torch.empty(num_layers, dtype=torch.float32))
        self.Lg = Parameter(prob_vec / prob_vec.sum(), requires_grad=True)

        self.node_transform = torch.nn.Linear(node_embedding_dim*num_layers, num_graph_mixtures*num_layers)
        self.graph_transform = torch.nn.Linear(num_graph_mixtures*num_layers, num_graph_mixtures*num_layers)

        if config['global_pooling'] == 'sum':
            self.global_pooling = global_add_pool
        elif config['global_pooling'] == 'mean':
            self.global_pooling = global_mean_pool

        self.emission_class = s2c(config['graph_emission_class'])
        self.emission = self.emission_class(dim_target,
                                            num_graph_mixtures*num_layers,
                                            self.num_hidden_neurons)

    def forward(self, node_embeddings: torch.tensor, batch: torch.Tensor, **kwargs) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:
        targets = kwargs['targets']

        Lg_norm = exp_normalize_trick(self.Lg, dim=0).unsqueeze(0).unsqueeze(2)

        node_tmp = self.node_transform((Lg_norm*node_embeddings).reshape((node_embeddings.shape[0], -1)))

        # The exp-normalize tick here is VERY IMPORTANT, as it acts like a non-linearity and
        # makes a real difference in the results.
        # It also has a probabilistic interpretation since we would like to aggregate
        # node information, weighted layer wise and then concatenated, represented as a probability
        graph_tmp = self.graph_transform(self.global_pooling(exp_normalize_trick(node_tmp, dim=1), batch))

        mixture_weights_g = exp_normalize_trick(graph_tmp, dim=1)

        params_g, log_likelihood_g, log_likelihood_g_comp = self.emission.forward(targets, mixture_weights_g)
        preds_g = self.emission.infer(mixture_weights_g, params_g)

        return mixture_weights_g, params_g, log_likelihood_g, log_likelihood_g_comp, preds_g


class ProbabilisticGraphReadoutNoLayerAttention(ProbabilisticGraphReadout):

    def forward(self, node_embeddings: torch.tensor, batch: torch.Tensor, **kwargs) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:
        targets = kwargs['targets']

        node_tmp = self.node_transform((node_embeddings).reshape((node_embeddings.shape[0], -1)))

        # The exp-normalize tick here is VERY IMPORTANT, as it acts like a non-linearity and
        # makes a real difference in the results.
        # It also has a probabilistic interpretation since we would like to aggregate
        # node information, weighted layer wise and then concatenated, represented as a probability
        graph_tmp = self.graph_transform(self.global_pooling(exp_normalize_trick(node_tmp, dim=1), batch))

        mixture_weights_g = exp_normalize_trick(graph_tmp, dim=1)

        params_g, log_likelihood_g, log_likelihood_g_comp = self.emission.forward(targets, mixture_weights_g)
        preds_g = self.emission.infer(mixture_weights_g, params_g)

        return mixture_weights_g, params_g, log_likelihood_g, log_likelihood_g_comp, preds_g


class ProbabilisticGraphReadoutNoLayerAttentionMLP(ProbabilisticGraphReadout):

    def __init__(self, dim_node_features, dim_edge_features, dim_target, config):
        super().__init__(dim_node_features, dim_edge_features, dim_target, config)
        num_graph_mixtures = config['num_mixtures']
        num_layers = config['num_layers']
        self.out = torch.nn.Linear(num_graph_mixtures*num_layers, dim_target)

    def forward(self, node_embeddings: torch.tensor, batch: torch.Tensor, **kwargs) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:
        targets = kwargs['targets']

        l = torch.relu(self.node_transform(node_embeddings.reshape((node_embeddings.shape[0], -1))))
        g = self.global_pooling(l, batch)
        o = self.graph_transform(g)
        out = self.out(F.relu(o))

        log_likelihood_g = - torch.nn.functional.cross_entropy(out, targets, reduction='none')

        return None, None, log_likelihood_g, None, out


class ProbabilisticGraphReadoutNoLayerAttentionMLPVersion2(ProbabilisticGraphReadout):

    def forward(self, node_embeddings: torch.tensor, batch: torch.Tensor, **kwargs) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:
        targets = kwargs['targets']

        node_tmp = torch.relu(self.node_transform((node_embeddings).reshape((node_embeddings.shape[0], -1))))

        # The exp-normalize tick here is VERY IMPORTANT, as it acts like a non-linearity and
        # makes a real difference in the results.
        # It also has a probabilistic interpretation since we would like to aggregate
        # node information, weighted layer wise and then concatenated, represented as a probability
        graph_tmp = self.graph_transform(self.global_pooling(node_tmp, batch))

        mixture_weights_g = exp_normalize_trick(graph_tmp, dim=1)

        params_g, log_likelihood_g, log_likelihood_g_comp = self.emission.forward(targets, mixture_weights_g)
        preds_g = self.emission.infer(mixture_weights_g, params_g)

        return mixture_weights_g, params_g, log_likelihood_g, log_likelihood_g_comp, preds_g
