"""
Graph-Induced Sum-Product Networks

Files: unsupervised_embedding_generation.py

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
import os
import torch
from pydgn.experiment.experiment import Experiment
from pydgn.static import LOSS, SCORE


def reorder(obj, perm):
    assert len(obj) == len(perm) and len(obj) > 0
    return [y for (x, y) in sorted(zip(perm, obj))]


class EmbeddingTask(Experiment):

    def run_valid(self, dataset_getter, logger):
        unsupervised_config = self.model_config.unsupervised_config

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

        if not os.path.exists(os.path.join(embeddings_folder, dataset_getter.dataset_name)):
            os.makedirs(os.path.join(embeddings_folder, dataset_getter.dataset_name))
        base_path = os.path.join(embeddings_folder,
                                 dataset_getter.dataset_name,
                                 f'{batch_size}_{num_layers}_{num_mixtures}_{num_hidden_neurons}_'
                                 f'{avg_parameters_across_layers}_'
                                 f'{learning_rate}_'
                                 f'{dataset_getter.outer_k + 1}_'
                                 f'{dataset_getter.inner_k + 1}')

        batch_size = self.model_config.unsupervised_config['batch_size']
        shuffle = self.model_config.unsupervised_config['shuffle'] \
            if 'shuffle' in self.model_config.unsupervised_config else True

        train_loader = dataset_getter.get_inner_train(batch_size=batch_size, shuffle=shuffle)
        val_loader = dataset_getter.get_inner_val(batch_size=batch_size, shuffle=shuffle)
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        #
        # NOTE: since the model is unsupervised, we already compute and store on disk the test embeddings
        #       WITHOUT TRAINING ON IT (we borrow this idea from CGMM's implementation on PyDGN)
        # NOTE2: the outer kth TR set is the union of these internal TR and VL sets, so we can reuse the computed
        #        embeddings for the subsequent classifiers, see `unsup_model_classifier_experiment.py`
        #
        # NOTE3: storing unsup. embeddings allows to reuse them with different classifiers' configurations, which is
        #        very convenient in terms of computation, as we do not retrain an unsup. model each time.
        #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        test_loader = dataset_getter.get_outer_test(batch_size=batch_size, shuffle=shuffle)

        # Instantiate the Dataset
        dim_node_features = dataset_getter.get_dim_node_features()
        dim_edge_features = dataset_getter.get_dim_edge_features()
        dim_target = dataset_getter.get_dim_target()

        # Instantiate the Model
        model = self.create_unsupervised_model(dim_node_features, dim_edge_features, dim_target)

        # Instantiate the engine (it handles the training loop and the inference phase by abstracting the specifics)
        unsupervised_training_engine = self.create_unsupervised_engine(model)

        train_loss, train_score, train_data_list, \
        val_loss, val_score, val_data_list, \
        test_loss, test_score, test_data_list = unsupervised_training_engine.train(
            train_loader=train_loader,
            validation_loader=val_loader,
            test_loader=test_loader,
            max_epochs=self.model_config.unsupervised_config['epochs'],
            logger=logger)

        # This is fixed in PyDGN 1.3.0
        # # If samples (already shuffled because of the data split in Subset) have been also shuffled according to a
        # # permutation, i.e., in the last inference phase of the engine, use PyDGN RandomSampler to recover the
        # # permutation and store the embedding data list as in the original split.
        # # This way we recover the order of the [train/val/test]_idxs list in the split file.
        # # This is useful for subsequent experiments about weak supervision, and it is better in general to avoid
        # # confusion.
        # if shuffle:
        #     train_data_list = reorder(train_data_list, train_loader.sampler.permutation)
        #     val_data_list = reorder(val_data_list, val_loader.sampler.permutation)
        #     test_data_list = reorder(test_data_list, test_loader.sampler.permutation)

        torch.save(train_data_list, base_path + '_train.torch')
        torch.save(val_data_list, base_path + '_val.torch')
        torch.save(test_data_list, base_path + '_test.torch')

        train_res = {LOSS: train_loss, SCORE: train_score}
        val_res = {LOSS: val_loss, SCORE: val_score}
        return train_res, val_res

    def run_test(self, dataset_getter, logger):
        # Dummy values for the library
        tr_res = {LOSS: {'main_loss': torch.zeros(1)}, SCORE: {'main_score': torch.zeros(1)}}
        vl_res = {LOSS: {'main_loss': torch.zeros(1)}, SCORE: {'main_score': torch.zeros(1)}}
        te_res = {LOSS: {'main_loss': torch.zeros(1)}, SCORE: {'main_score': torch.zeros(1)}}
        return tr_res, vl_res, te_res
