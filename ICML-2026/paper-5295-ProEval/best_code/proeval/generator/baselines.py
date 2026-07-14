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

"""Generation baselines for BQ active evaluation.

Provides random generation baseline methods for comparison with
BQ-guided generation (SS-Gen, TSS) in :mod:`proeval.generator.core`.

Baselines:

- **Rand-Gen** — Pure random generation with no topic or anchor guidance.
- **Rand-T-Gen** — Random topic injection but no BQ-guided anchors.
"""

import random
from typing import Dict, List

from proeval.generator.core import TopicAwareGenerator


def random_generation(
    generator: TopicAwareGenerator,
    n_iterations: int,
    k_examples: int = 3,
) -> List[Dict]:
    """Pure random generation baseline (no topic, no anchors).

    Args:
        generator: TopicAwareGenerator instance (dataset-aware).
        n_iterations: Number of problems to generate.
        k_examples: Ignored (no anchors used).

    Returns:
        List of generated test cases.
    """
    results = []
    for _ in range(n_iterations):
        case = generator.generate(strategy="pure_random", k_examples=0)
        results.append(case)
    return results


def random_topic_generation(
    generator: TopicAwareGenerator,
    n_iterations: int,
    k_examples: int = 3,
) -> List[Dict]:
    """Random topic generation baseline (topic injected, no BQ anchors).

    Args:
        generator: TopicAwareGenerator instance (dataset-aware).
        n_iterations: Number of problems to generate.
        k_examples: Ignored (no anchors used).

    Returns:
        List of generated test cases.
    """
    results = []
    for _ in range(n_iterations):
        case = generator.generate(strategy="random_topic", k_examples=0)
        results.append(case)
    return results
