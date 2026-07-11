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

import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


def plot_histograms(target_dir: Path, output: Path, sizes: List[int]):
    """Plot histograms for different sampling sizes."""
    plt.figure(figsize=(15, 10))

    for i, size in enumerate(sizes, 1):
        plt.subplot(len(sizes), 1, i)

        # Read responses
        responses = []
        for file in target_dir.glob(f"*_size_{size}_*.jsonl"):
            with open(file) as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if isinstance(data, list):
                            responses.extend(data)
                        else:
                            responses.append(data)
                    except json.JSONDecodeError:
                        continue

        if not responses:
            continue

        # Convert responses to numbers
        numbers = []
        for resp in responses:
            if isinstance(resp, (int, float)):
                numbers.append(resp)
            elif isinstance(resp, dict) and "number" in resp:
                numbers.append(resp["number"])
            elif isinstance(resp, list):
                numbers.extend([x for x in resp if isinstance(x, (int, float))])

        if not numbers:
            continue

        # Plot histogram
        plt.hist(numbers, bins=50, alpha=0.7, label=f"Size {size}")
        plt.title(f"Distribution for Size {size}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.legend()

    plt.tight_layout()
    plt.savefig(output)
    plt.close()
