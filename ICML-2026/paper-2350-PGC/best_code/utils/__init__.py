"""Project-wide utilities for the PGC pipeline.

Re-exports the symbols the entry points (:mod:`train` and :mod:`test`)
import directly:

- CLI: :func:`build_train_parser`, :func:`build_test_parser`,
  :func:`parse_devices`, :func:`finalize_opt`, :func:`cli_message_for_opt`
- Logging: :func:`setup_logging`, :func:`log_training_config`,
  :func:`log_test_config`
- Metrics: :func:`compute_accuracy`, :func:`compute_real_accuracy`,
  :func:`compute_fake_accuracy`, :func:`compute_average_precision`,
  :func:`compute_auc`, :func:`compute_all_metrics`,
  :func:`compute_mean_metrics`, :func:`log_metrics`
- Reproducibility: :func:`set_seed`
"""

from .cli import (
    build_test_parser,
    build_train_parser,
    cli_message_for_opt,
    finalize_opt,
    parse_devices,
)
from .logging_utils import (
    log_test_config,
    log_training_config,
    setup_logging,
)
from .metrics import (
    compute_accuracy,
    compute_all_metrics,
    compute_auc,
    compute_average_precision,
    compute_fake_accuracy,
    compute_mean_metrics,
    compute_real_accuracy,
    log_metrics,
)
from .seed import set_seed

__all__ = [
    # CLI
    "build_train_parser",
    "build_test_parser",
    "cli_message_for_opt",
    "finalize_opt",
    "parse_devices",
    # Logging
    "setup_logging",
    "log_training_config",
    "log_test_config",
    # Metrics
    "compute_accuracy",
    "compute_real_accuracy",
    "compute_fake_accuracy",
    "compute_average_precision",
    "compute_auc",
    "compute_all_metrics",
    "compute_mean_metrics",
    "log_metrics",
    # Reproducibility
    "set_seed",
]
