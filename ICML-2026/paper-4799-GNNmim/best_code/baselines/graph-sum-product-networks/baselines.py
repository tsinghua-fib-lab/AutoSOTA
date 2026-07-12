"""
Graph-Induced Sum-Product Networks

Files: baselines.py

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
from ogb.graphproppred.mol_encoder import AtomEncoder
from pydgn.model.interface import ModelInterface
from torch import Tensor
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU, PReLU
from torch.nn.functional import dropout
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool, global_mean_pool, GINConv, \
    SAGEConv
from torch_geometric.nn.inits import reset, uniform


class GIN(ModelInterface):
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
        self.atom_encoder = None
        if 'atom_encoder_dim' in config:
            # works with ogbg-molpbca only
            atom_emb_dim = config['atom_encoder_dim']
            self.atom_encoder = AtomEncoder(emb_dim=atom_emb_dim)
            dim_node_features = atom_emb_dim

        train_eps = config['train_eps']
        if config['global_aggregation'] == 'sum':
            self.pooling = global_add_pool
        elif config['global_aggregation'] == 'mean':
            self.pooling = global_mean_pool

        for layer, out_emb_dim in enumerate(self.embeddings_dim):

            if layer == 0:
                self.first_h = Sequential(Linear(dim_node_features, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU(),
                                    Linear(out_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU())
                self.linears.append(Linear(out_emb_dim, dim_target))
            else:
                input_emb_dim = self.embeddings_dim[layer-1]
                self.nns.append(Sequential(Linear(input_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU(),
                                      Linear(out_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU()))
                self.convs.append(GINConv(self.nns[-1], train_eps=train_eps))  # Eq. 4.2

                self.linears.append(Linear(out_emb_dim, dim_target))

        self.nns = torch.nn.ModuleList(self.nns)
        self.convs = torch.nn.ModuleList(self.convs)
        self.linears = torch.nn.ModuleList(self.linears)  # has got one more for initial input


    def forward(self, data: Batch) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if self.atom_encoder is not None:
            x = self.atom_encoder(x)
        else:
            x = x.float()

        out = 0

        for layer in range(self.num_layers):
            if layer == 0:
                x = self.first_h(x)
                out += dropout(self.pooling(self.linears[layer](x), batch), p=self.dropout)
            else:
                # Layer l ("convolution" layer)
                x = self.convs[layer-1](x, edge_index)
                out += dropout(self.linears[layer](self.pooling(x, batch)), p=self.dropout, training=self.training)

        return out, x


class GAE_Adj(ModelInterface):
    """
    Original GAE that reconstructs the adjacency matrix to produce latent node representations
    """
    def __init__(self, dim_node_features, dim_edge_features, dim_target, readout_class, config):
        super().__init__(dim_node_features, dim_edge_features, dim_target, readout_class, config)

        self.config = config
        self.num_layers = config['num_layers']
        self.embeddings_dim = [config['dim_embedding'] for _ in range(self.num_layers)]
        self.first_h = []
        self.nns = []
        self.convs = []
        self.concat_out_across_layers = config['concat_out_across_layers']
        train_eps = config['train_eps']
        self.atom_encoder = None
        if 'atom_encoder_dim' in config:
            # works with ogbg-molpbca only
            atom_emb_dim = config['atom_encoder_dim']
            self.atom_encoder = AtomEncoder(emb_dim=atom_emb_dim)
            dim_node_features = atom_emb_dim

        for layer, out_emb_dim in enumerate(self.embeddings_dim):

            if layer == 0:
                self.first_h = Sequential(Linear(dim_node_features, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU(),
                                    Linear(out_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU())
            else:
                input_emb_dim = self.embeddings_dim[layer-1]
                self.nns.append(Sequential(Linear(input_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU(),
                                      Linear(out_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU()))
                self.convs.append(GINConv(self.nns[-1], train_eps=train_eps))  # Eq. 4.2

        self.nns = torch.nn.ModuleList(self.nns)
        self.convs = torch.nn.ModuleList(self.convs)


    def forward(self, data: Batch) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:

        # extract data
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        node_embeddings = []

        if self.atom_encoder is not None:
            x = self.atom_encoder(x)
        else:
            x = x.float()

        for layer in range(self.num_layers):
            if layer == 0:
                h = self.first_h(x.float())
                if self.concat_out_across_layers:
                    node_embeddings.append(h)
            else:
                # Layer l ("convolution" layer)
                h = self.convs[layer-1](h, edge_index)
                if self.concat_out_across_layers:
                    node_embeddings.append(h)

        if not self.concat_out_across_layers:
            node_embeddings = h
        else:
            node_embeddings = torch.cat(node_embeddings, dim=1)

        return (
            None,
            node_embeddings,
            [data.edge_index, data.neg_edge_index],
        )


class DGI(ModelInterface):
    """
    Original DGI. Code taken and adapted from PyG
    """
    def __init__(self, dim_node_features, dim_edge_features, dim_target, readout_class, config):
        super().__init__(dim_node_features, dim_edge_features, dim_target, readout_class, config)

        self.config = config
        self.num_layers = config['num_layers']
        self.embeddings_dim = [config['dim_embedding'] for _ in range(self.num_layers)]
        self.convs = []
        self.activations = []
        self.atom_encoder = None
        if 'atom_encoder_dim' in config:
            # works with ogbg-molpbca only
            atom_emb_dim = config['atom_encoder_dim']
            self.atom_encoder = AtomEncoder(emb_dim=atom_emb_dim)
            dim_node_features = atom_emb_dim

        # this is the encoder
        for layer, out_emb_dim in enumerate(self.embeddings_dim):

            if layer == 0:
                self.convs.append(SAGEConv(dim_node_features,
                                           out_emb_dim))
            else:
                input_emb_dim = self.embeddings_dim[layer-1]
                self.convs.append(SAGEConv(input_emb_dim,
                                           out_emb_dim))
            self.activations.append(PReLU())

        self.convs = torch.nn.ModuleList(self.convs)
        self.activations = torch.nn.ModuleList(self.activations)

        self.summary = lambda z, batch: torch.sigmoid(global_mean_pool(z,
                                                               batch))
        self.corruption = None  # implemented as transform in PyDGN

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.convs)
        reset(self.summary)


    def forward(self, data: Batch) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:

        # extract data
        x, x_corrupted, edge_index, edge_attr, batch = (
            data.x,
            data.x_corrupted,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        if self.atom_encoder is not None:
            x = self.atom_encoder(x)
            x_corrupted = self.atom_encoder(x_corrupted)
        else:
            x = x.float()
            x_corrupted = x_corrupted.float()

        """Returns the latent space for the input arguments, their
        corruptions and their summary representation."""

        # Create positive Z
        for layer in range(self.num_layers):
            # Layer l ("convolution" layer)

            if layer == 0:
                h_pos = x

            h_pos = self.convs[layer](h_pos, edge_index)
            h_pos = self.activations[layer](h_pos)

        pos_z = h_pos
        summary = self.summary(pos_z, batch)
        summary_node_broadcast = summary[batch]

        # Create negative Z
        for layer in range(self.num_layers):
            # Layer l ("convolution" layer)

            if layer == 0:
                h_neg = x_corrupted

            h_neg = self.convs[layer](h_neg, edge_index)
            h_neg = self.activations[layer](h_neg)

        neg_z = h_neg

        return (
            None,
            pos_z,
            [pos_z, neg_z, summary_node_broadcast],
        )
