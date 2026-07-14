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

"""ProEval utilities — Dataset wrapper, metrics.

Public API::

    from proeval.utils import Dataset
    from proeval.utils import topic_entropy, embedding_coverage, failure_rate
    from proeval.utils import compute_samples_to_threshold, print_results_table
    from proeval.utils import MODEL_NAME_MAP
"""

from proeval.utils.dataset import Dataset
from proeval.utils.metrics import (
    compute_all_metrics,
    compute_samples_to_threshold,
    embedding_coverage,
    failure_rate,
    get_question_embeddings,
    overall_diversity,
    print_results_table,
    topic_entropy,
)

from proeval.utils.model_names import MODEL_NAME_MAP

__all__ = [
    "Dataset",
    "topic_entropy",
    "embedding_coverage",
    "overall_diversity",
    "failure_rate",
    "compute_all_metrics",
    "get_question_embeddings",
    "compute_samples_to_threshold",
    "print_results_table",
    "MODEL_NAME_MAP",
]
