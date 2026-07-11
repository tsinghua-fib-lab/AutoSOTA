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

RC_PARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 15,
    "xtick.labelsize": 17,
    "ytick.labelsize": 18,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#666666",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 2.0,
    "lines.markersize": 8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}

COLORS = {
    "Direct": "#B8E0F5",  # Medium blue (baseline) - swapped with Sequence
    "CoT": "#7CC7EA",  # Light blue (baseline)
    "Sequence": "#6BB6FF",  # Distinct blue (baseline) - swapped with Direct
    "Multi-turn": "#4A90E2",  # Distinct blue (baseline)
    "VS-Standard": "#FFCCCB",  # Light red (our method)
    "VS-CoT": "#FF9999",  # Medium red (our method)
    "VS-Multi": "#FF6B6B",  # Distinct red (our method)
}

EDGE_COLORS = {
    "Direct": "#4A90E2",
    "CoT": "#4A90E2",
    "Sequence": "#4A90E2",
    "Multi-turn": "#4A90E2",
    "VS-Standard": "#FF6B6B",
    "VS-CoT": "#FF6B6B",
    "VS-Multi": "#FF6B6B",
}
