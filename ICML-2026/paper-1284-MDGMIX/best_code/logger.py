import logging
import time
from datetime import timedelta
from pathlib import Path

from sympy.codegen.ast import Raise


class DotDict(dict):
    """dict.key access"""
    def __getattr__(*args):  # nested
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def create_logger(filepath, colored=False, debug=False):
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
       raise FileExistsError(f"Failed to create Log file {filepath}: {str(e)}")
    log_formatter = LogFormatter(colored=colored)

    # create file handler and set level
    if filepath is not None:
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.INFO)
        if debug:
            file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    return logger


class LogFormatter:
    def __init__(self, colored=False):
        self.colored = colored
        self.start_time = time.time()

    def format(self, record):
        BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
        RESET_SEQ = "\033[0m"
        COLOR_SEQ = "\033[1;%dm"

        COLORS = {
            'WARNING': GREEN,
            'INFO': WHITE,
            'DEBUG': BLUE,
            'CRITICAL': YELLOW,
            'ERROR': RED
        }
        elapsed_seconds = round(record.created - self.start_time)
        levelname = record.levelname
        if self.colored:
            levelname = COLOR_SEQ % (
                30 + COLORS[record.levelname]) + record.levelname + RESET_SEQ

        prefix = "%s - %s - %s" % (
            levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ''