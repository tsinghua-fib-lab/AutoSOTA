"""File and logging utilities for MONICA."""
import json
import logging
from pathlib import Path


def setup_logging(output_dir: Path):
    """Set up file logging for an experiment run.
    
    Args:
        output_dir: directory to write log files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / "run.log"
    
    # Remove existing handlers for the root logger
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        if isinstance(handler, (logging.FileHandler, logging.StreamHandler)):
            root_logger.removeHandler(handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    root_logger.setLevel(logging.INFO)


def save_json(path: Path, data: dict):
    """Save data as a JSON file.
    
    Args:
        path: output file path
        data: data to serialize
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_json_line(path: Path, data: dict):
    """Append data as a JSON line to a file.
    
    Args:
        path: output file path
        data: data to serialize
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
