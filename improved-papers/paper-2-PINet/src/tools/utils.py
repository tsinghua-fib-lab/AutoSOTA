"""Utility functions and classes for logging and timing code execution."""

import logging
import signal
from typing import Any, Dict, Optional

import wandb
import yaml

logger = logging.getLogger(__name__)


def load_configuration(file_path: str) -> dict:
    """Load configuration file from yaml.

    Args:
        file_path (str): Path to the configuration file.

    Returns:
        dict: A dictionary containing the configuration parameters.
    """
    with open(file_path, "r") as file:
        hyperparameters = yaml.safe_load(file)
    return hyperparameters


class Logger:
    """Encapsulates logging functionalities."""

    _logged_in = False

    def __init__(self, run_name: str, project_name: str = "hcnn") -> None:
        """Initializes the Logger and creates a new wandb run.

        Args:
            run_name (str): The name of the run to be logged.
            project_name (str): The name of the project.
        """
        if not Logger._logged_in:
            wandb.login()
            Logger._logged_in = True

        self.run_name = run_name
        self.run = wandb.init(
            project=project_name,
            name=self.run_name,
            id=self.run_name,
        )

    def __enter__(self) -> "Logger":
        """Enters the runtime context for Logger.

        Returns:
            Logger: The current instance.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exits the runtime context and finishes the wandb run."""
        wandb.finish()

    def log(self, t: int, data: Dict[str, Any]) -> None:
        """Logs data.

        Args:
            t (int): An indexing parameter (for example, the epoch).
            data (Dict[str, Any]): A dictionary of variable names and values to log.
        """
        wandb.log(data, step=t)


class GracefulShutdown:
    """A context manager for graceful shutdowns."""

    stop = False

    def __init__(self, exit_message: Optional[str] = None):
        """Initializes the GracefulShutdown context manager.

        Args:
            exit_message (str): The message to log upon shutdown.
        """
        self.exit_message = exit_message

    def __enter__(self):
        """Register the signal handler."""

        def handle_signal(signum, frame):
            self.stop = True
            if self.exit_message:
                logger.info(self.exit_message)

        signal.signal(signal.SIGINT, handle_signal)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Unregister the signal handler."""
        pass
