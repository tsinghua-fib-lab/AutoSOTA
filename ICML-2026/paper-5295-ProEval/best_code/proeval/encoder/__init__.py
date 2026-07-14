# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ProEval encoder — neural encoder utilities for BQ priors.

Shared module used by both sampler and generator for encoder-based
prompt feature experiments (learned prompt embeddings).

Public API::

    from proeval.encoder import QuestionEncoder, load_encoder
    from proeval.encoder import get_phi_embeddings, compute_kernel_matrix
    from proeval.encoder import EncoderTrainer
"""

from proeval.encoder.nn_utils import (
    QuestionEncoder,
    compute_gp_loss,
    compute_gp_loss_with_reg,
    compute_kernel_matrix,
    compute_kl_loss,
    get_phi_embeddings,
    get_phi_embeddings_batch,
    load_encoder,
    save_encoder,
)
from proeval.encoder.trainer import EncoderTrainer

__all__ = [
    "QuestionEncoder",
    "EncoderTrainer",
    "compute_gp_loss",
    "compute_gp_loss_with_reg",
    "compute_kernel_matrix",
    "compute_kl_loss",
    "get_phi_embeddings",
    "get_phi_embeddings_batch",
    "load_encoder",
    "save_encoder",
]

