"""
Defines logger
"""

import sys
import logging


def switch_to_file_logger(filename):
    "Switch logger to file logger"
    logger.removeHandler(ch)
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

# see here for explanation: https://docs.python.org/3/howto/logging.html


# create logger
logger = logging.getLogger('fbpinns')
logger.setLevel(logging.INFO)

# create console handler and set level to debug
ch = logging.StreamHandler(stream=sys.stdout)
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter(fmt='[%(levelname)s] %(asctime)s - %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
if not logger.handlers:
    logger.addHandler(ch)

# make sure the logger doesn't propagate to its parents (standalone logger)
logger.propagate = False


if __name__ == "__main__":

    logger.debug("hello world")
    logger.setLevel("DEBUG")
    logger.debug("hello world")
    logger.info("hello world")