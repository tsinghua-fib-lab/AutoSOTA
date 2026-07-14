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

"""ProEval evaluator — LLM prediction and evaluation.

Public API::

    from proeval.evaluator import LLMPredictor, OpenRouterClient, DATASET_CONFIGS
    from proeval.evaluator import UnifiedCSVManager, load_dataset_data
"""

from proeval.evaluator.client import MODEL_MAPPING, OpenRouterClient, resolve_model_name
from proeval.evaluator.csv_manager import (
    UnifiedCSVManager,
    convert_numpy_types,
    load_dataset_data,
    load_predictions_from_csv,
    save_predictions_to_csv,
)
from proeval.evaluator.predictor import (
    DATASET_CONFIGS,
    DatasetConfig,
    LLMPredictor,
    create_dices_config,
    create_dices_t2i_config,
    create_gqa_config,
    create_gsm8k_config,
    create_jigsaw_config,
    create_mmlu_config,
    create_strategyqa_config,
    create_svamp_config,
    create_toxicchat_config,
)

__all__ = [
    "OpenRouterClient",
    "MODEL_MAPPING",
    "resolve_model_name",
    "LLMPredictor",
    "DatasetConfig",
    "DATASET_CONFIGS",
    "UnifiedCSVManager",
    "load_dataset_data",
    "save_predictions_to_csv",
    "load_predictions_from_csv",
    "convert_numpy_types",
    "create_strategyqa_config",
    "create_gsm8k_config",
    "create_svamp_config",
    "create_mmlu_config",
    "create_jigsaw_config",
    "create_toxicchat_config",
    "create_gqa_config",
    "create_dices_config",
    "create_dices_t2i_config",
]
