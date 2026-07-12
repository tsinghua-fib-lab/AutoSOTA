"""
Graph-Induced Sum-Product Networks

Files: unsupervised_embedding_classification.py

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
import math
import os

import torch
from pydgn.data.sampler import RandomSampler
from pydgn.experiment.experiment import Experiment
from pydgn.static import LOSS, SCORE
from torch.utils.data import Subset

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader


class ClassificationTask(Experiment):

    def run_valid(self, dataset_getter, logger):
        """
        This function returns the training and validation or test accuracy
        :return: (training accuracy, validation/test accuracy)
        """
        unsupervised_config = self.model_config.unsupervised_config
        supervised_config = self.model_config.supervised_config

        embeddings_folder = unsupervised_config['embeddings_folder']
        batch_size = unsupervised_config['batch_size']
        num_layers = unsupervised_config['num_layers']
        if 'num_mixtures' in unsupervised_config:
            num_mixtures = unsupervised_config['num_mixtures']
        else:
            num_mixtures = unsupervised_config['dim_embedding']
        num_hidden_neurons = unsupervised_config['num_hidden_neurons']
        avg_parameters_across_layers = unsupervised_config.get('avg_parameters_across_layers', True)

        learning_rate = unsupervised_config['optimizer']['args']['lr']

        weak_supervision_percentage = self.model_config.get('weak_supervision_percentage', 1.)

        splitter = dataset_getter._get_splitter()
        train_indices = splitter.inner_folds[dataset_getter.outer_k][dataset_getter.inner_k].train_idxs
        num_train_indices = math.floor(weak_supervision_percentage*len(train_indices))
        val_indices = splitter.inner_folds[dataset_getter.outer_k][dataset_getter.inner_k].val_idxs
        num_val_indices = math.floor(weak_supervision_percentage*len(val_indices))

        if not os.path.exists(os.path.join(embeddings_folder, dataset_getter.dataset_name)):
            os.makedirs(os.path.join(embeddings_folder, dataset_getter.dataset_name))
        base_path = os.path.join(embeddings_folder,
                                 dataset_getter.dataset_name,
                                 f'{batch_size}_{num_layers}_{num_mixtures}_{num_hidden_neurons}_'
                                 f'{avg_parameters_across_layers}_'
                                 f'{learning_rate}_'
                                 f'{dataset_getter.outer_k + 1}_'
                                 f'{dataset_getter.inner_k + 1}')

        # The order of the Data objects in these lists is the same as the samples of the original split
        # This is because the list of generated embeddings has been reordered according to the permutation of the data
        # loader (see `unsupervised_embedding_generation.py` to know more)
        train_data_list = torch.load(base_path + '_train.torch')[:num_train_indices]
        val_data_list = torch.load(base_path + '_val.torch')[:num_val_indices]

        # Get the embedding size from the first graph in the dataset
        embedding_size = train_data_list[0].x.shape[1]

        # Get information from the supervised configuration
        batch_size = supervised_config['batch_size']
        shuffle = supervised_config['shuffle'] \
            if 'shuffle' in supervised_config else True
        collate_fn = lambda data_list: Batch.from_data_list(data_list)

        # Instantiate the Embedding Dataset for supervised learning
        train_loader = DataLoader(train_data_list, batch_size, shuffle, collate_fn=collate_fn)
        val_loader = DataLoader(val_data_list, batch_size, shuffle, collate_fn=collate_fn)

        dim_target = train_data_list[0].y.shape[1] if len(train_data_list[0].y.shape) > 1 else 1

        # Instantiate the Model
        model = self.create_supervised_model(embedding_size, dim_edge_features=0, dim_target=dim_target)

        # Instantiate the engine (it handles the training loop and the inference phase by abstracting the specifics)
        supervised_training_engine = self.create_supervised_engine(model)

        train_loss, train_score, _, \
        val_loss, val_score, _, \
        _, _, _ = supervised_training_engine.train(train_loader=train_loader,
                                                    validation_loader=val_loader,
                                                    test_loader=None,
                                                    max_epochs=supervised_config['epochs'],
                                                    logger=logger)

        train_res = {LOSS: train_loss, SCORE: train_score}
        val_res = {LOSS: val_loss, SCORE: val_score}
        return train_res, val_res

    def run_test(self, dataset_getter, logger):
        """
        This function returns the training and test accuracy. DO NOT USE THE TEST FOR ANY REASON
        :return: (training accuracy, test accuracy)
        """

        unsupervised_config = self.model_config.unsupervised_config
        supervised_config = self.model_config.supervised_config

        embeddings_folder = unsupervised_config['embeddings_folder']
        batch_size = unsupervised_config['batch_size']
        num_layers = unsupervised_config['num_layers']
        if 'num_mixtures' in unsupervised_config:
            num_mixtures = unsupervised_config['num_mixtures']
        else:
            num_mixtures = unsupervised_config['dim_embedding']
        num_hidden_neurons = unsupervised_config['num_hidden_neurons']
        avg_parameters_across_layers = unsupervised_config.get('avg_parameters_across_layers', True)
        learning_rate = unsupervised_config['optimizer']['args']['lr']

        weak_supervision_percentage = self.model_config.get('weak_supervision_percentage', 1.)

        outer_k = dataset_getter.outer_k
        inner_k = dataset_getter.inner_k
        # workaround the "safety" procedure of evaluation protocol, but we will not do anything wrong.
        # This is just to recover the stored embeddings (reusing the internal validation set to do early stopping
        if inner_k is None:
            dataset_getter.set_inner_k(0)
            inner_k = 0  # pick the split of the first (and only) inner fold
        # ====

        splitter = dataset_getter._get_splitter()
        train_indices = splitter.outer_folds[dataset_getter.outer_k].train_idxs
        num_train_indices = math.floor(weak_supervision_percentage*len(train_indices))
        val_indices = splitter.outer_folds[dataset_getter.outer_k].val_idxs
        num_val_indices = math.floor(weak_supervision_percentage*len(val_indices))

        if not os.path.exists(os.path.join(embeddings_folder, dataset_getter.dataset_name)):
            os.makedirs(os.path.join(embeddings_folder, dataset_getter.dataset_name))
        base_path = os.path.join(embeddings_folder,
                                 dataset_getter.dataset_name,
                                 f'{batch_size}_{num_layers}_{num_mixtures}_{num_hidden_neurons}_'
                                 f'{avg_parameters_across_layers}_'
                                 f'{learning_rate}_'
                                 f'{outer_k + 1}_'
                                 f'{inner_k + 1}')

        # The order of the Data objects in these lists is the same as the samples of the original split
        # This is because the list of generated embeddings has been reordered according to the permutation of the data
        # loader (see `unsupervised_embedding_generation.py` to know more)
        train_data_list = torch.load(base_path + '_train.torch')[:num_train_indices]
        val_data_list = torch.load(base_path + '_val.torch')[:num_val_indices]
        # We want to evaluate on the entire test set
        test_data_list = torch.load(base_path + '_test.torch')

        # Get the embedding size from the first graph in the dataset
        embedding_size = train_data_list[0].x.shape[1]

        # Get information from the supervised configuration
        batch_size = supervised_config['batch_size']
        shuffle = supervised_config['shuffle'] \
            if 'shuffle' in supervised_config else True
        collate_fn = lambda data_list: Batch.from_data_list(data_list)

        # Instantiate the Embedding Dataset for supervised learning
        train_loader = DataLoader(train_data_list, batch_size, shuffle, collate_fn=collate_fn)
        val_loader = DataLoader(val_data_list, batch_size, shuffle, collate_fn=collate_fn)
        test_loader = DataLoader(test_data_list, batch_size, shuffle, collate_fn=collate_fn)

        dim_target = train_data_list[0].y.shape[1] if len(train_data_list[0].y.shape) > 1 else 1

        # Instantiate the Model
        model = self.create_supervised_model(embedding_size, dim_edge_features=0, dim_target=dim_target)

        # Instantiate the engine (it handles the training loop and the inference phase by abstracting the specifics)
        supervised_training_engine = self.create_supervised_engine(model)

        train_loss, train_score, _, \
        val_loss, val_score, _, \
        test_loss, test_score, _ = supervised_training_engine.train(train_loader=train_loader,
                                                                     validation_loader=val_loader,
                                                                     test_loader=test_loader,
                                                                     max_epochs=supervised_config['epochs'],
                                                                     logger=logger)

        train_res = {LOSS: train_loss, SCORE: train_score}
        val_res = {LOSS: val_loss, SCORE: val_score}
        test_res = {LOSS: test_loss, SCORE: test_score}
        return train_res, val_res, test_res
