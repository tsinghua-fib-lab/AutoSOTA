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

"""One-click sample usage of ProEval.

Run from the repo root::

    python experiment/sample_usage.py

Estimates a target model's error rate on SVAMP from ~50 active samples
(out of ~1k) and reports the MAE against the true error rate.
"""

import numpy as np

from proeval import BQPriorSampler
from proeval.sampler import extract_model_predictions, load_predictions


def main() -> None:
    sampler = BQPriorSampler(noise_variance=0.3)
    result = sampler.sample(
        predictions="svamp",
        target_model="gemini25_flash",
        budget=50,
    )

    df = load_predictions("svamp")
    pred_matrix, model_names = extract_model_predictions(df)
    true_mean = float(np.mean(pred_matrix[:, model_names.index("gemini25_flash")]))

    print(f"Estimated error rate: {result.estimates[-1]:.4f}")
    print(f"True error rate:      {true_mean:.4f}")
    print(f"MAE:                  {result.mae(true_mean):.4f}")
    print(f"Samples used:         {len(result.selected_indices)} / {len(pred_matrix)}")


if __name__ == "__main__":
    main()
