"""
DiscoverLLM — a framework for LLM-based dialogue simulation, evaluation, and
training against an intent-aware user simulator.

Top-level packages:

* :mod:`discoverllm.core`     — LLM dispatch + prompt templates + atomic tasks
* :mod:`discoverllm.pipeline` — stateful simulators (user, assistant, reward)
* :mod:`discoverllm.simulate` — experiment orchestration and dataset builders
* :mod:`discoverllm.metrics`  — scoring helpers for evaluation outputs

The training entrypoints live in :mod:`discoverllm.training`.
"""

__version__ = "0.1.0"
