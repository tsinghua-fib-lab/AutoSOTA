from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

import logging


def get_logger(name):
    """Return a logger with a consistent, simple format."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
