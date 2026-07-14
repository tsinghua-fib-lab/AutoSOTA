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

"""ProEval — Active Evaluation with Bayesian Quadrature.

A library for efficient LLM evaluation through Bayesian Quadrature active
sampling, topic-aware test case generation, and structured LLM prediction.

Quick start::

    from proeval import BQPriorSampler, LLMPredictor, TopicAwareGenerator

    # Active sampling
    sampler = BQPriorSampler(noise_variance=0.3)
    result = sampler.sample(predictions="svamp", target_model=0, budget=20)

    # LLM evaluation
    predictor = LLMPredictor(model="google/gemma-3-4b-it")
    resp, pred, score = predictor.evaluate(
        "Is the sky blue?", True, DATASET_CONFIGS["strategyqa"]
    )

    # Test case generation — pass a Dataset (or DataFrame) plus a GP prior
    gen = TopicAwareGenerator(df=df, dataset="gsm8k", prior_u=u, prior_S=S)
    case = gen.generate(strategy="tss")
"""

__version__ = "0.1.0"

from proeval.evaluator import DATASET_CONFIGS, DatasetConfig, LLMPredictor, OpenRouterClient
from proeval.generator import TopicAwareGenerator
from proeval.sampler import BQPriorSampler, BQSampler, SamplingResult
from proeval.utils import Dataset

__all__ = [
    "BQPriorSampler",
    "BQSampler",  # backward compat alias
    "SamplingResult",
    "TopicAwareGenerator",
    "LLMPredictor",
    "OpenRouterClient",
    "Dataset",
    "DatasetConfig",
    "DATASET_CONFIGS",
]
