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

"""
Sampling method definitions for verbalized sampling experiments.
"""

from .factory import (
    Method,
    PromptFactory,
    PromptTemplate,
    SamplingPromptTemplate,
    is_method_base_model,
    is_method_combined,
    is_method_multi_turn,
    is_method_structured,
)
from .parser import ResponseParser

__all__ = [
    "PromptFactory",
    "PromptTemplate",
    "SamplingPromptTemplate",
    "Method",
    "is_method_structured",
    "is_method_multi_turn",
    "is_method_combined",
    "is_method_base_model",
    "ResponseParser",
]
