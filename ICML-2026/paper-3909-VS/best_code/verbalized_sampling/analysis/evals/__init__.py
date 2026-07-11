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

# from .safety import SafetyEvaluator  # TODO: Missing file
# from .accuracy import AccuracyEvaluator  # TODO: Missing file
# Import plotting functionality from the top-level plots module
from ...plots import ComparisonPlotter, plot_comparison_chart, plot_evaluation_comparison
from .base import BaseEvaluator
from .creative_writing_v3 import CreativeWritingV3Evaluator
from .creativity_index import CreativityIndexEvaluator
from .data_quality import SyntheticDataQualityEvaluator
from .diversity import DiversityEvaluator
from .factuality import FactualityEvaluator
from .joke_quality import JokeQualityEvaluator
from .length import LengthEvaluator
from .ngram import NgramEvaluator
from .quality import TTCTEvaluator
from .response_count import ResponseCountEvaluator

# Dialogue evaluators (integrated from persuasion_simulation)
try:
    from ...evals.dialogue import DialogueLinguisticEvaluator
    from ...evals.dialogue import DonationEvaluator as DialogueDonationEvaluator
except Exception:
    DialogueDonationEvaluator = None
    DialogueLinguisticEvaluator = None

__all__ = [
    "DiversityEvaluator",
    "TTCTEvaluator",
    "CreativityIndexEvaluator",
    "LengthEvaluator",
    "ResponseCountEvaluator",
    "CreativeWritingV3Evaluator",
    "NgramEvaluator",
    "BaseEvaluator",
    "FactualityEvaluator",
    "JokeQualityEvaluator",
    "SyntheticDataQualityEvaluator",
    # "SafetyEvaluator",  # TODO: Missing file
    # "AccuracyEvaluator",  # TODO: Missing file
    # Plotting functionality
    "ComparisonPlotter",
    "plot_evaluation_comparison",
    "plot_comparison_chart",
]


def get_evaluator(metric: str, **kwargs):
    if metric == "diversity":
        return DiversityEvaluator(**kwargs)
    elif metric == "ttct" or metric == "quality":
        return TTCTEvaluator(**kwargs)
    elif metric == "creativity_index":
        return CreativityIndexEvaluator(**kwargs)
    elif metric == "length":
        return LengthEvaluator(**kwargs)
    elif metric == "response_count":
        return ResponseCountEvaluator(**kwargs)
    elif metric == "creative_writing_v3" or metric == "cwv3":
        return CreativeWritingV3Evaluator(**kwargs)
    elif metric == "ngram":
        return NgramEvaluator(**kwargs)
    elif metric == "factuality":
        return FactualityEvaluator(**kwargs)
    elif metric == "joke_quality":
        return JokeQualityEvaluator(**kwargs)
    elif metric == "synthetic_data_quality":
        return SyntheticDataQualityEvaluator(**kwargs)
    elif metric == "donation":
        if DialogueDonationEvaluator is None:
            raise ImportError("Dialogue donation evaluator not available")
        return DialogueDonationEvaluator(**kwargs)
    elif metric in ("dialogue_linguistic", "linguistic_dialogue"):
        if DialogueLinguisticEvaluator is None:
            raise ImportError("Dialogue linguistic evaluator not available")
        return DialogueLinguisticEvaluator(**kwargs)
    elif metric == "safety":
        raise NotImplementedError("SafetyEvaluator not available - missing implementation")
    elif metric == "accuracy":
        raise NotImplementedError("AccuracyEvaluator not available - missing implementation")
    else:
        raise ValueError(
            f"Evaluator {metric} not found. Available evaluators: accuracy, diversity, ttct, creativity_index, length, response_count, creative_writing_v3, ngram, factuality, joke_quality, synthetic_data_quality, safety"
        )
