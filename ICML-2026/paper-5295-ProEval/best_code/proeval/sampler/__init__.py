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

"""ProEval sampler — Bayesian Quadrature active sampling.

Public API::

    from proeval.sampler import BQPriorSampler, SamplingResult

    sampler = BQPriorSampler(noise_variance=0.3)
    result = sampler.sample(predictions="svamp", target_model=0, budget=20)

``BQEncoderSampler`` is always importable, but invoking it requires the
``[encoder]`` extra (PyTorch). Encoder utilities (``QuestionEncoder``,
``compute_kernel_matrix``, ...) are re-exported only when torch is
installed.
"""

from proeval.sampler.bq import (
    BQEncoderSampler,
    BQPriorSampler,
    BQSampler,
    SamplingResult,
)
from proeval.sampler.data import (
    extract_model_predictions,
    load_embeddings,
    load_predictions,
    setup_train_test_split,
)
from proeval.sampler.pretrain_selector import (
    get_reference_benchmarks,
    select_pretrain_models_gmm,
)

__all__ = [
    "BQEncoderSampler",
    "BQPriorSampler",
    "BQSampler",
    "SamplingResult",
    "load_predictions",
    "load_embeddings",
    "extract_model_predictions",
    "setup_train_test_split",
    "select_pretrain_models_gmm",
    "get_reference_benchmarks",
]

try:
    from proeval.encoder import (  # noqa: F401
        QuestionEncoder,
        compute_gp_loss,
        compute_gp_loss_with_reg,
        compute_kernel_matrix,
        get_phi_embeddings,
        get_phi_embeddings_batch,
        load_encoder,
        save_encoder,
    )
except ImportError:
    pass
else:
    __all__ += [
        "QuestionEncoder",
        "compute_kernel_matrix",
        "compute_gp_loss",
        "compute_gp_loss_with_reg",
        "get_phi_embeddings",
        "get_phi_embeddings_batch",
        "save_encoder",
        "load_encoder",
    ]
