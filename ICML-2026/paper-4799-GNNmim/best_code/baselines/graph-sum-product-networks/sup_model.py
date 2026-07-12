"""
Graph-Induced Sum-Product Networks

Files: sup_model.py

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
from pydgn.model.interface import ModelInterface
from torch import softmax
from torch.nn import Linear
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool, global_add_pool

from pydgn.experiment.util import s2c

class SupGSPN(ModelInterface):
    """
    Graph Sum-Product Network (supervised)
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
        self.num_graph_mixtures = config.get("num_graph_mixtures", None)
        self.num_hidden_neurons = config[
            "num_hidden_neurons"
        ]  # same number of hidden neurons for all MLPs involved
        self.convolution_class = s2c(config["convolution_class"])
        self.emission_class = s2c(config["emission_class"])
        self.emissions = nn.ModuleList()
        self.transitions = nn.ModuleList()
        self.avg_parameters_across_layers = config.get('avg_parameters_across_layers', True)
        self.use_kmeans = config.get('init_kmeans', False)

        self.global_readout = config['global_readout']

        for id_layer in range(self.num_layers):
            self.emissions.append(
                self.emission_class(
                    self.dim_node_features, self.num_mixtures, self.num_hidden_neurons
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

        if self.num_graph_mixtures is not None:
            """
            THIS IS THE SUPERVISED VERSION OF THE MODEL
            """
            self.readout_node = torch.nn.Parameter(torch.rand(self.num_mixtures*self.num_layers,
                                                              self.num_graph_mixtures),
                                                   requires_grad=True)
            self.readout_graph = Linear(self.num_graph_mixtures,
                                             dim_target, bias=False)

    def forward(
        self, data: Batch
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:
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

        # if self.add_self_loops:
        #     edge_index = add_self_loops(edge_index)[0]
        #
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
            ].forward(data.x, mixture_weights)
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
                        ].log_likelihood(x, mixture_weights, avg_params_across_layers)

                        imputed_values = self.emissions[id_layer].impute(avg_params_across_layers, mixture_weights)

                    else:
                        avg_params_across_layers = params_v_layer[-1]

                        log_likelihood_v, log_likelihood_v_comp = self.emissions[
                            id_layer
                        ].log_likelihood(x, mixture_weights, params_v)

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

        objective_v = log_likelihood_v  # this refers to the last layer, in which we average all parameters

        # node posteriors of all layers, interpreted as node embeddings
        node_posterior_layer = torch.stack(node_posterior_layer, dim=1)

        preds_g, objective_g = None, None
        if self.num_graph_mixtures is not None:
            """
            SUPERVISED VERSION
            """
            norm_table = softmax(self.readout_node + 1e-8, dim=1)
            node_embs = (node_posterior_layer.reshape(-1, self.num_layers * self.num_mixtures))
            tmp_node = torch.matmul(node_embs, norm_table)

            if self.global_readout == 'sum':
                tmp_graph = softmax(global_add_pool(tmp_node, batch) + 1e-8, dim=1)
            else:
                tmp_graph = global_mean_pool(tmp_node, batch)/self.num_layers


            preds_g = self.readout_graph(tmp_graph)
            objective_g = None

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
