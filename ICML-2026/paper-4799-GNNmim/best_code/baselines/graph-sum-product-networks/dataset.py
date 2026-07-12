"""
Graph-Induced Sum-Product Networks

Files: dataset.py

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
import copy
import os
from typing import Union, List, Tuple

import torch
from pydgn.data.dataset import InMemoryDataset, OGBGDatasetInterface, TUDatasetInterface
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset

class SyntheticDataset(InMemoryDataset):

    def __init__(self, root, name, raw_dir, per_community_weight, structure_weight,
                 transform=None, pre_transform=None, **kwargs):
        self.per_community_weight = per_community_weight
        self.structure_weight = structure_weight
        self.name = f'{name}_{per_community_weight}_{structure_weight}'
        self._raw_dir = raw_dir
        super().__init__(root=root, transform=transform, pre_transform=pre_transform, **kwargs)
        self._data_list = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return self._raw_dir

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return [f'data_list_{(i+1)*100}.pt' for i in range(1)]

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return [f'data.pt']

    @property
    def processed_dir(self) -> str:
       return os.path.join(self.root, self.name, 'processed')

    @property
    def raw_paths(self) -> List[str]:
        r"""The absolute filepaths that must be present in order to skip
        downloading."""
        files = self.raw_file_names
        return [os.path.join(self.raw_dir, f) for f in files]

    @property
    def processed_paths(self) -> List[str]:
        r"""The absolute filepaths that must be present in order to skip
        processing."""
        files = self.processed_file_names
        return [os.path.join(self.processed_dir, f) for f in files]

    def download(self):
        pass

    def get(self, idx: int) -> Data:
        return copy.copy(self._data_list[idx])

    def process(self):

        data_list = []

        for raw_path in self.raw_paths:

            partial_data_list = torch.load(raw_path)

            if self.pre_transform is not None:
                partial_data_list = [self.pre_transform(data) for data in partial_data_list]

            data_list.extend(partial_data_list)

        print(len(data_list))
        # TUDataset expects data and slices, we directly store the data list
        torch.save(data_list, self.processed_paths[0])

    @property
    def dim_node_features(self):
        return self._data_list[0].x.shape[1]

    @property
    def dim_edge_features(self):
        return 0

    @property
    def dim_target(self):
        return 0

    def len(self) -> int:
        return len(self._data_list)

    def __len__(self) -> int:
        return len(self._data_list)

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'


class OGBGmolpcbaFeatureMap(OGBGDatasetInterface):

    def __init__(self, root, name, transform=None,
                 pre_transform=None, pre_filter=None, meta_dict=None, **kwargs):
        super().__init__(root, name, transform, pre_transform, pre_filter, meta_dict)

        num_features = self.data.x.shape[1]
        for f in range(num_features):
            unique_values = torch.sort(torch.unique(self.data.x[:,f]), descending=False)[0]
            id = 0
            for v in unique_values.tolist():
                assert id <= v
                self.data.x[:, f][self.data.x[:, f] == v] = id
                id += 1

    def download(self):
        super().download()

    def process(self):
        super().process()


class TUDatasetInterfaceRegression(TUDatasetInterface):

    @property
    def dim_target(self):
        return self.data.y.shape[1] if len(self.data.y.shape) > 1 else 1

    def download(self):
        super().download()

    def process(self):
        super().process()


class TUDatasetInterfaceMissingData(TUDataset):

    def __init__(
        self,
        root,
        name,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        **kwargs,
    ):
        self.name = name
        # Do not call DatasetInterface __init__ method in this case, because
        # otherwise it will break
        super().__init__(
            root=root,
            name=name,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            use_node_attr=True
        )


    @property
    def dim_target(self):
        return self.data.y.shape[1] if len(self.data.y.shape) > 1 else 1

    def download(self):
        super().download()

    def process(self):
        super().process()

    @property
    def dim_node_features(self) -> int:
        return self.num_node_features

    @property
    def dim_edge_features(self) -> int:
        return self.num_edge_features
