"""
Graph-Induced Sum-Product Networks

Files: model.py

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
from shutil import make_archive
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
from pydgn.model.interface import ModelInterface
from sklearn.cluster import KMeans
from torch.nn.functional import gumbel_softmax, softmax
from torch.nn.parameter import Parameter
from torch.distributions import (
    Categorical,
    Multinomial,
    Independent,
    MixtureSameFamily,
    Normal,
)
from torch_geometric.data import Batch
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
from torch_geometric.utils import add_self_loops

from pydgn.experiment.util import s2c


def exp_normalize_trick(m, dim):
    """
    Exp-normalize trick: subtract the maximum value
    :param m: matrix (N,...,D,...) with unnormalized probabilities
    :param dim: the dimension over which to normalize
    :return: a new matrix (N,...,D,...) with normalized probability scores alongside dimension dim
    """
    max_vals, _ = torch.max(m, dim=dim, keepdim=True)
    m_minus_max = m - max_vals
    m_softmax = softmax(m_minus_max, dim=dim).clamp(1e-8, 1.0)
    return m_softmax


class GSPNBaseConv(MessagePassing):
    def __init__(self, dim_edge_features, num_mixtures, num_hidden_neurons, use_prior):
        """
        Initializes the probabilistic GSPN convolution mechanism following CGMM's transition distribution
        CGMM: https://www.jmlr.org/papers/volume21/19-470/19-470.pdf
        :param dim_edge_features: #todo not used at the moment
        :param num_mixtures:
        :param num_hidden_neurons:
        :param use_prior: if True, no neighboring states are available, so we learn the prior distribution
        """
        super().__init__(aggr="mean")  # it MUST stay mean
        self.dim_edge_features = dim_edge_features
        self.num_mixtures = num_mixtures
        self.num_hidden_neurons = num_hidden_neurons
        self.use_prior = use_prior

        if self.use_prior:
            # P(Q_u = c)
            prob_vec = torch.nn.init.uniform_(
                torch.empty(self.num_mixtures, dtype=torch.float32)
            )
            self.transition_table = Parameter(
                prob_vec / prob_vec.sum(), requires_grad=True
            )
        else:
            # P(Q_u = c | q_v = c')
            prob_vec = torch.nn.init.uniform_(
                torch.empty((self.num_mixtures, self.num_mixtures), dtype=torch.float32)
            )
            self.transition_table = Parameter(
                prob_vec / prob_vec.sum(dim=0, keepdim=True), requires_grad=True
            )

    def forward(self, edge_index, edge_attr, h_v, batch_size):
        """
        :param edge_index:
        :param edge_attr:
        :param h_v: a tensor of size NxC or None, depending on the layer
        :param batch_size:
        :return: the per-vertex (possibly unnormalized) parameters of a Gumbel-Softmax distribution
        """
        # Deal with isolated nodes by adding a self-loop
        edge_index, _ = add_self_loops(edge_index, num_nodes=batch_size)

        # Normalize parameters to obtain probabilities
        transition_table_norm = exp_normalize_trick(self.transition_table, dim=0)
        expand_transition = transition_table_norm.unsqueeze(
            dim=0
        )  # 1xC or 1xCxC, depending on the layer

        n_m = self.num_mixtures

        if self.use_prior:
            assert h_v is None
            return expand_transition.repeat(batch_size, 1)
        else:
            expand_h_v = h_v.unsqueeze(dim=1)  # Nx1xC
            # each node will bring a specific contribution to state i of its neighbors
            p_Q_weighted = expand_transition * expand_h_v  # NxCxC
            return self.propagate(edge_index, x=p_Q_weighted.reshape((-1, n_m * n_m)))


class GSPNEmission(nn.Module):
    @staticmethod
    def average_parameters(parameters_per_layer):
        """
        Combines the parameters of the same feature's emission computed at different levels to
        embed favorable inductive bias when imputing missing attributes.
        :param parameters_per_layer:
        :return: averaged parameters to be used by the log_likelihood function
        """
        return NotImplementedError("You should use a subclass of GSPNEmission")

    @staticmethod
    def log_likelihood(x, mixture_weights, parameters, masked_nodes=None):
        """
        Computes the log-likelihood for a vector of N samples associated with the mixture of Categoricals
        :param x: vector of N categories, one per sample (i.e., node)
        :param mixture_weights:
        :param parameters: vector of normalized probabilities for the categorical distribution
        :param masked_nodes: boolean matrix (N,F) containing the features to be retained (as 1) for each node 
        :return: log_likelihood vector of size N and a log_likelihood matrix of size N x num_components
        """
        return NotImplementedError("You should use a subclass of GSPNEmission")

    def __init__(self, dim_observable, num_mixtures, num_hidden_neurons):
        """
        Initializes the GSPN emission distribution, acts as a mixture of distributions
        :param dim_observable:
        :param num_mixtures:
        :param num_hidden_neurons:
        """
        super().__init__()
        self.dim_observable = dim_observable
        self.num_mixtures = num_mixtures
        self.num_hidden_neurons = num_hidden_neurons

    def forward(self, x, mixture_weights, masked_nodes=None):
        """
        :param x:
        :param mixture_weights:
        :param masked_nodes: boolean matrix (N,F) containing the features to be retained (as 1) for each node 
        :return: the per-vertex emission parameters and the per-vertex log-likelihood scores
        """
        return NotImplementedError("You should use a subclass of GSPNEmission")


class GSPNCategoricalEmission(GSPNEmission):
    @staticmethod
    def average_parameters(parameters_per_layer):
        stacked_average_params_v = torch.stack(parameters_per_layer, dim=1)
        return stacked_average_params_v.mean(dim=1)

    @staticmethod
    def log_likelihood(x, mixture_weights, parameters, masked_nodes=None):
        assert len(x.shape) == 1 or (len(x.shape) == 2), x.shape
        if len(x.shape) == 2:
            # remove second dimension to flatten Nx1 vector)
            x = x.argmax(dim=1)

        mix = Categorical(probs=mixture_weights)
        comp = Categorical(probs=parameters)
        mm = MixtureSameFamily(mix, comp)

       
        
        # if masked_nodes is not None or torch.sum(masked_nodes).item() != 0:
        #     # x[masked_nodes] = 0  # o qualsiasi valore valido
        #     log_prob = mm.log_prob(x)
        #     log_prob[masked_nodes.any(dim=1)] = 0.0
        #     return log_prob, comp.log_prob(x.unsqueeze(1))
        # else:
        #     return mm.log_prob(x), comp.log_prob(x.unsqueeze(1))

        if masked_nodes is not None:
            raise NotImplementedError("Not implemented!")
            # x[masked_nodes] = 0.  # just to make sure it's something computable
            
            # if mm._validate_args:
            #     mm._validate_sample(x)
            # x = mm._pad(x)
            # comp_log_prob_x = comp..base_dist.log_prob(x)  # [Samples, Components, Features]
            
            # # Since the features are independent, I will sum in log space alond the last dimension. 
            # # To make the gradient 0, multiply by 0 the masked features
            # masked_nodes_unsqueezed = masked_nodes.unsqueeze(1).repeat(1, comp_log_prob_x.shape[1], 1)
            # comp_log_prob_x[masked_nodes_unsqueezed] = comp_log_prob_x[masked_nodes_unsqueezed]*0

            # comp_log_prob_x = comp_log_prob_x.sum(dim=2)
            # log_prob = torch.logsumexp(comp_log_prob_x + mixture_weights.log(), dim=-1)  # [Samples, Components]
            # return log_prob, comp_log_prob_x
        else:
            return mm.log_prob(x), comp.log_prob(
                x.unsqueeze(1)
            )  # todo a bit redundant but clearer for now

    def __init__(self, dim_observable, num_mixtures, num_hidden_neurons):
        super().__init__(dim_observable, num_mixtures, num_hidden_neurons)
        self.num_categories = dim_observable
        self.categorical_probs = torch.nn.parameter.Parameter(torch.nn.init.uniform_(torch.empty(num_mixtures, dim_observable)),
                                                     requires_grad=True)



    def forward(self, x, mixture_weights, masked_nodes=None):
        categorical_probs = exp_normalize_trick(self.categorical_probs.unsqueeze(0), dim=2)

        imputed_probs = self.impute(categorical_probs, mixture_weights)

        (
            per_vertex_log_likelihood,
            per_vertex_log_likelihood_components,
        ) = self.log_likelihood(x, mixture_weights, categorical_probs, masked_nodes)

        return (
            categorical_probs,
            per_vertex_log_likelihood,
            per_vertex_log_likelihood_components,
            imputed_probs
        )

    def impute(self, params, mixture_weights):

        # do imputation for missing nodes
        imputed_probs = (params * mixture_weights.unsqueeze(2)).sum(dim=1)

        return imputed_probs


class GSPNMultiCategoricalEmission(GSPNEmission):

    @staticmethod
    def average_parameters(parameters_per_layer):
        stacked_average_params_v = torch.stack(parameters_per_layer, dim=1)
        return stacked_average_params_v.mean(dim=1)

    def log_likelihood(self, x, mixture_weights, parameters, masked_nodes=None):
        assert len(x.shape) == 2, x.shape

        log_prob = 0.
        comp_log_prob_x = 0.

        params_start = 0

        for i,e in enumerate(self.emissions):

            params = parameters[:, params_start:params_start+self.dim_categorical_features[i]]
            params_start += self.dim_categorical_features[i]

            e_log_prob, e_comp_log_prob_x = e.log_likelihood(x[:,i], mixture_weights.log(), params, masked_nodes)
            log_prob = log_prob + e_log_prob
            comp_log_prob_x = comp_log_prob_x + e_comp_log_prob_x

        return log_prob, comp_log_prob_x

    def __init__(self, dim_observable, num_mixtures, num_hidden_neurons, dim_categorical_features):
        super().__init__(dim_observable, num_mixtures, num_hidden_neurons)
        self.emissions = nn.ModuleList()
        self.dim_categorical_features = dim_categorical_features

        for d in dim_categorical_features:
            self.emissions.append(GSPNCategoricalEmission(d, self.num_mixtures, self.num_hidden_neurons))

    def forward(self, x, mixture_weights, masked_nodes=None):
        params = []
        per_vertex_log_likelihood = 0.
        per_vertex_log_likelihood_components = 0.

        for i,e in enumerate(self.emissions):
            e_params, e_per_vertex_log_likelihood, e_per_vertex_log_likelihood_components, _ = e.forward(x[:,i], mixture_weights, masked_nodes)
            params.append(e_params)
            per_vertex_log_likelihood = per_vertex_log_likelihood + e_per_vertex_log_likelihood
            per_vertex_log_likelihood_components = per_vertex_log_likelihood_components + e_per_vertex_log_likelihood_components

        #print([p.shape for p in params]); exit()

        if len(params[0].shape) == 3 and params[0].shape[0] == 1:
            params = [p.squeeze(0) for p in params]

        # concatenate along x axis
        params = torch.cat(params, dim=1)

        return (
            params,
            per_vertex_log_likelihood,
            per_vertex_log_likelihood_components,
            None
        )

    def impute(self, params, posterior_log):
        return None
        #raise NotImplementedError('To be implemented!')


class GSPNGaussianEmission(GSPNEmission):
    @staticmethod
    def average_parameters(parameters_per_layer):
        stacked_average_params_v = torch.stack(parameters_per_layer, dim=1)
        return stacked_average_params_v.mean(dim=1)

    @staticmethod
    def log_likelihood(x, mixture_weights, parameters, masked_nodes=None):
        assert len(x.shape) == 2, x.shape

        mix = Categorical(probs=mixture_weights)
        comp = Independent(
            Normal(loc=parameters[:, :, :, 0], scale=parameters[:, :, :, 1]), 1
        )
        mm = MixtureSameFamily(mix, comp)
        
        if masked_nodes is not None:
            x[masked_nodes] = 0.  # just to make sure it's something computable

            if mm._validate_args:
                mm._validate_sample(x)
            x = mm._pad(x)
            comp_log_prob_x = comp.base_dist.log_prob(x)  # [Samples, Components, Features]
            
            masked_nodes_unsqueezed = masked_nodes.unsqueeze(1).repeat(1, comp_log_prob_x.shape[1], 1)
            comp_log_prob_x[masked_nodes_unsqueezed] = comp_log_prob_x[masked_nodes_unsqueezed]*0

            comp_log_prob_x = comp_log_prob_x.sum(dim=2)
            log_prob = torch.logsumexp(comp_log_prob_x + mixture_weights.log(), dim=-1)  # [Samples, Components]

            return log_prob, comp_log_prob_x

        else:
            return mm.log_prob(x), comp.log_prob(
                x.unsqueeze(1)
            )  # todo a bit redundant but clearer for now

    def initialize_means(self, cluster_centers):
        self.normal_params.data[:, :, 0] = cluster_centers

    def __init__(self, dim_observable, num_mixtures, num_hidden_neurons):
        super().__init__(dim_observable, num_mixtures, num_hidden_neurons)
        self.normal_params = torch.nn.parameter.Parameter(torch.nn.init.uniform_(torch.empty(num_mixtures, dim_observable, 2)),
                                                   requires_grad=True)

    def forward(self, x, mixture_weights, masked_nodes=None):
        normal_params = self.normal_params.reshape(
            (-1, self.num_mixtures, self.dim_observable, 2)
        )
        # keep scale positive
        loc = normal_params[:, :, :, 0].unsqueeze(-1)
        scale = (
            torch.nn.functional.softplus(normal_params[:, :, :, 1]).unsqueeze(-1) + 1e-8
        )
        normal_params = torch.cat((loc, scale), dim=-1)

        imputed_means = self.impute(normal_params, mixture_weights)

        (
            per_vertex_log_likelihood,
            per_vertex_log_likelihood_components,
        ) = self.log_likelihood(x, mixture_weights, normal_params, masked_nodes)
        return (
            normal_params,
            per_vertex_log_likelihood,
            per_vertex_log_likelihood_components,
            imputed_means
        )

    def impute(self, params, mixture_weights):
        loc = params[:, :, :, 0]

        # do imputation for missing nodes
        imputed_means = (loc * mixture_weights.unsqueeze(2)).sum(dim=1)

        return imputed_means


class GSPN(ModelInterface):
    """
    Graph Sum-Product Network (unsupervised)
    """

    def __init__(
        self, dim_node_features, dim_edge_features, dim_target, readout_class, config
    ):
        """
        Initializes the Graph Sum-Product Network
        :param dim_node_features:
        :param dim_edge_features:
        :param dim_target:
        :param readout_class:
        :param config:
        """
        super().__init__(
            dim_node_features, dim_edge_features, dim_target, readout_class, config
        )

        self.num_layers = config["num_layers"]
        self.num_mixtures = config["num_mixtures"]
        self.num_hidden_neurons = config[
            "num_hidden_neurons"
        ]  # -- NOT USED RIGHT NOW -- same number of hidden neurons for all MLPs involved 
        self.convolution_class = s2c(config["convolution_class"])
        self.emission_class = s2c(config["emission_class"])
        self.emissions = nn.ModuleList()
        self.transitions = nn.ModuleList()
        self.avg_parameters_across_layers = config.get('avg_parameters_across_layers', True)
        self.use_kmeans = config.get('init_kmeans', False)
        self.add_self_loops = config.get('add_self_loops', False)
        self.initialized = Parameter(torch.tensor(False), requires_grad=False)
        self.kmeans = KMeans(n_clusters=self.num_mixtures, random_state=config['seed'])
        self.dim_categorical_features = config.get('dim_categorical_features', None)
        if self.dim_categorical_features is not None:
            self.dim_categorical_features = list(self.dim_categorical_features.values())

        for id_layer in range(self.num_layers):
            if self.emission_class == GSPNMultiCategoricalEmission:
                self.emissions.append(
                    self.emission_class(self.dim_node_features,
                                        self.num_mixtures,
                                        self.num_hidden_neurons,
                                        self.dim_categorical_features))
            else:
                self.emissions.append(
                    self.emission_class(
                        self.dim_node_features,
                        self.num_mixtures,
                        self.num_hidden_neurons
                    )
                )

            self.transitions.append(
                self.convolution_class(
                    dim_edge_features,
                    self.num_mixtures,
                    self.num_hidden_neurons,
                    use_prior=id_layer == 0,
                )
            )

        if self.readout_class is not None:
            """
            THIS IS THE SUPERVISED VERSION OF THE MODEL
            """
            self.readout = self.readout_class(
                dim_node_features, dim_edge_features, dim_target, config
            )
        self.classifier = torch.nn.Linear(self.num_layers * self.num_mixtures, dim_target)
    def forward(
        self, data: Batch, seed
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:
        """

        :param data:
        :return:
        """
        # extract data
        x, edge_index, edge_attr, batch = (
            data.masks[seed]['X_incomp'], #data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        x_imputed = x.clone()
        node_mask = ~data.masks[seed]['mask']
        masked_nodes = torch.logical_not(node_mask)
        x_imputed[masked_nodes] = torch.nan

        # Calcola la media ignorando i NaN
        mean_values = torch.nanmean(x_imputed, dim=0)

        # Trova le colonne dove la media è NaN (cioè dove tutta la colonna è NaN)
        only_nan_columns = torch.isnan(mean_values)

        # Sostituisci solo quelle colonne con 0
        mean_values[only_nan_columns] = 0.0

        # Applica le mean values solo sui nodi mascherati
        mean_values = mean_values.repeat(x.shape[0], 1)
        x_imputed[masked_nodes] = mean_values[masked_nodes]
        # if hasattr(data, 'mask'):
        #     # MASK is a boolean (nodes, features) matrix that is TRUE if the node HAS a specific feature
        #     node_mask = data.mask
        #     masked_nodes = torch.logical_not(node_mask)
        #     x_imputed[masked_nodes] = torch.nan
        #     # Replacing nan with mean values to run kmeans initialization
        #     mean_values = torch.nanmean(x_imputed, dim=0).repeat(x.shape[0], 1)
        #     x_imputed[masked_nodes] = mean_values[masked_nodes]
        # else:
            # masked_nodes = None


        if self.add_self_loops:
            edge_index = add_self_loops(edge_index)[0]
        
        # if self.training and self.use_kmeans:
        #     if not self.initialized:
        #         print("Initializing kmeans...")
        #         self.kmeans.fit(x.detach().cpu().numpy())
        #         clusters = torch.tensor(self.kmeans.cluster_centers_)
        #         mu = (clusters + torch.randn_like(clusters)).to(x.device)
        #         for id_layer in range(self.num_layers):
        #             self.emissions[id_layer].initialize_means(mu)
        #         self.initialized.data = torch.tensor(True)
        #         print("Done.")
        # if 100 * torch.isnan(x_imputed.detach().cpu()).sum().item() / x_imputed.detach().cpu().numel() >0.0:
        #     print("perc: ", 100 * torch.isnan(x_imputed.detach().cpu()).sum().item() / x_imputed.detach().cpu().numel(), "%")
        #     print(mean_values)
        if self.training and self.use_kmeans:
            if not self.initialized:
                # print("Initializing kmeans...")
                self.kmeans.fit(x_imputed.detach().cpu().numpy()) 
                clusters = torch.tensor(self.kmeans.cluster_centers_)
                mu = (clusters + torch.randn_like(clusters)).to(x.device)
                for id_layer in range(self.num_layers):
                    self.emissions[id_layer].initialize_means(mu)
                self.initialized.data = torch.tensor(True)
                # print("Done.")
        x_imputed[masked_nodes] = torch.nan
        # if hasattr(data, 'mask'):
        #     # Restore nan into masked positions, just to be sure
        #     x_imputed[masked_nodes] = torch.nan

        node_posterior_layer = []  # list of one node embedding matrix per layer
        params_v_layer = []

        h_v = None
        for id_layer in range(self.num_layers):
            mixture_weights = self.transitions[id_layer].forward(
                edge_index, edge_attr, h_v=h_v, batch_size=x.shape[0]
            )

            if id_layer > 0:
                mixture_weights_j = mixture_weights.reshape(
                    (-1, self.num_mixtures, self.num_mixtures)
                )
                # i.e., \sum_j P(i|j)*q_v(j), where u is v's neighboring node. Note: this does not depend on node v.
                mixture_weights = mixture_weights_j.sum(2)

            params_v, log_likelihood_v, log_likelihood_v_comp, imputed_values = self.emissions[
                id_layer
            ].forward(x_imputed, mixture_weights, masked_nodes)
            params_v_layer.append(params_v)

            if id_layer == 0:
                # Compute the node "posterior"
                unnormalized_posterior = (
                    mixture_weights * log_likelihood_v_comp.exp() + 1e-8
                )
                node_posterior = unnormalized_posterior / unnormalized_posterior.sum(
                    1, keepdim=True
                )

                # Pass "posterior" to next layers
                h_v = node_posterior
                node_posterior_layer.append(h_v)
                assert not torch.any(torch.isnan(log_likelihood_v_comp))
                assert not torch.any(torch.isnan(mixture_weights))
                assert not torch.any(torch.isnan(unnormalized_posterior))
                assert not torch.any(torch.isnan(node_posterior))

                avg_params_across_layers = params_v_layer[0]

            else:
                # We "generate" using the last mixing weights, but the emission parameters have been avg. across layers
                if id_layer == self.num_layers - 1:

                    if self.avg_parameters_across_layers:
                        # The individual layers jointly cooperate towards the generation of node features.
                        avg_params_across_layers = self.emissions[
                            id_layer
                        ].average_parameters(params_v_layer)

                        log_likelihood_v, log_likelihood_v_comp = self.emissions[
                            id_layer
                        ].log_likelihood(x_imputed, mixture_weights, avg_params_across_layers, masked_nodes)

                        imputed_values = self.emissions[id_layer].impute(avg_params_across_layers, mixture_weights)

                    else:
                        avg_params_across_layers = params_v_layer[-1]

                        log_likelihood_v, log_likelihood_v_comp = self.emissions[
                            id_layer
                        ].log_likelihood(x_imputed, mixture_weights, params_v, masked_nodes)

                # Compute the deterministic "posterior" for unsupervised node embeddings
                log_likelihood_v_comp_unsqueezed = log_likelihood_v_comp.unsqueeze(2)
                unnormalized_posterior = (
                    mixture_weights_j * log_likelihood_v_comp_unsqueezed.exp() + 1e-8
                )
                node_posterior = unnormalized_posterior / unnormalized_posterior.sum(
                    (1, 2), keepdim=True
                )

                # Pass "posterior" to next layers
                h_v = node_posterior.sum(2)
                node_posterior_layer.append(h_v)

        objective_v = log_likelihood_v  # this refers to the last layer, in which we may have averaged all parameters
        
        # node posteriors of all layers, interpreted as node embeddings
        node_posterior_layer = torch.stack(node_posterior_layer, dim=1)
        
        node_embeddings = node_posterior_layer.reshape((node_posterior_layer.shape[0],node_posterior_layer.shape[1] * node_posterior_layer.shape[2]))
        out = self.classifier(node_embeddings)
        return out
    
        preds_g, objective_g = None, None
        if self.readout_class is not None:
            readout_output = self.readout.forward(
                node_posterior_layer, batch, **{"targets": data.y}
            )
            (
                mixture_weights_g,
                params_g,
                log_likelihood_g,
                log_likelihood_g_comp,
                preds_g,
            ) = readout_output
            objective_g = log_likelihood_g

        return (
            preds_g,
            node_posterior_layer.reshape(
                (
                    node_posterior_layer.shape[0],
                    node_posterior_layer.shape[1] * node_posterior_layer.shape[2],
                )
            ),
            [objective_v, objective_g, x, x, imputed_values, None, None, mixture_weights, avg_params_across_layers],
        )
