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

"""ProEval generator — topic-aware test case generation.

Public API::

    from proeval.generator import TopicAwareGenerator
    from proeval.generator import random_generation, random_topic_generation
    from proeval.generator import setup_encoder_prior, get_posterior_embedding
"""

from proeval.generator.baselines import (
    random_generation,
    random_topic_generation,
)
from proeval.generator.core import (
    TopicAwareGenerator,
    extract_topics_bertopic,
    get_posterior_embedding,
    select_hard_problems_bq,
    setup_encoder_prior,
    ss_acquisition,
    ss_acquisition_batch,
)

__all__ = [
    "TopicAwareGenerator",
    "extract_topics_bertopic",
    "get_posterior_embedding",
    "select_hard_problems_bq",
    "setup_encoder_prior",
    "ss_acquisition",
    "ss_acquisition_batch",
    "random_generation",
    "random_topic_generation",
]

