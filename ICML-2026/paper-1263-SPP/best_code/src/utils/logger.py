"""
Logging utility module
Provides unified logging functionality
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(
        name: str,
        log_dir: str = "logs",
        level: int = logging.INFO,
        console: bool = True
) -> logging.Logger:
    """
    Set up a logger

    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        console: Whether to output to the console

    Returns:
        Logger object
    """
    # Create the log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Create the logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers more than once
    if logger.handlers:
        return logger

    # Log format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler
    log_file = log_path / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger, or create a new one if it does not exist

    Args:
        name: Logger name

    Returns:
        Logger object
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger

