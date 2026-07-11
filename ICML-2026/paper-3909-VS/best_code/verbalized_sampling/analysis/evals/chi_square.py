# Copyright 2025 CHATS-Lab. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
from typing import List

import numpy as np
from scipy.stats import chi2_contingency


def evaluate_chi_square(responses: List[float], expected_dist: List[float] = None):
    """Evaluate responses using chi-square test."""
    if expected_dist is None:
        # Use uniform distribution as default
        # Legacy code
        # expected_dist = [1/10] * 10  # Assuming numbers are in [0, 9]
        random.seed(42)
        number_selection = random.choices(range(10), size=len(responses), replace=True)
        unique_numbers, counts = np.unique(number_selection, return_counts=True)
        expected_dist = np.zeros(10)
        expected_dist[unique_numbers] = counts

    # Create observed distribution
    observed, _ = np.histogram(responses, bins=len(expected_dist), range=(0, len(expected_dist)))
    observed = observed / sum(observed)  # Normalize

    # Perform chi-square test
    chi2, p_value = chi2_contingency([observed, expected_dist])[0:2]

    return {
        "chi2": chi2,
        "p_value": p_value,
        "observed": observed.tolist(),
        "expected": expected_dist,
    }
