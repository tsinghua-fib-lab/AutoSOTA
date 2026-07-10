"""
Experiment orchestration.

Main entry point: ``python -m discoverllm.simulate.run`` (CLI).
Library entry: :func:`discoverllm.simulate.runner.run_experiment`.

Key exports:

* :data:`MODE_BEST_OF_1`, :data:`MODE_BEST_OF_N` — string constants for
  ``--mode``.
* :class:`ExperimentConfig`, :class:`AssistantConfig`, :class:`UserConfig`,
  :class:`ConversationResult` — config + result dataclasses.
* :func:`load_artifacts`, :func:`load_assistant_configs`,
  :func:`load_user_config` — I/O helpers used by the CLI.
"""

from discoverllm.simulate.config import (
    MODE_BEST_OF_1,
    MODE_BEST_OF_N,
    AssistantConfig,
    ConversationResult,
    ExperimentConfig,
    UserConfig,
    make_assistant_from_config,
)
from discoverllm.simulate.io import (
    load_artifacts,
    load_assistant_configs,
    load_user_config,
)
from discoverllm.simulate.logging_utils import (
    close_error_log,
    init_error_log,
    log_error,
    log_warning,
)

# Runner is imported lazily to avoid circular imports.
# Use: from discoverllm.simulate.runner import run_experiment
