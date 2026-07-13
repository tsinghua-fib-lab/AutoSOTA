import logging
import colorlog
from logging import Logger


class LoggerConfig:
    file_handler = None
    console_handler = None
    level = None

    @staticmethod
    def setup_logger(
        level=logging.DEBUG,
        log_file_path=None,
    ):
        LoggerConfig.level = level

        # Colored formatter for console
        console_formatter = colorlog.ColoredFormatter(
            "%(asctime)s %(log_color)s[%(levelname)s] %(purple)s%(name)s: %(blue)s%(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )

        # Plain formatter for file
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )

        if log_file_path is not None:
            LoggerConfig.file_handler = logging.FileHandler(log_file_path)
            LoggerConfig.file_handler.setLevel(logging.DEBUG)
            LoggerConfig.file_handler.setFormatter(file_formatter)

        LoggerConfig.console_handler = logging.StreamHandler()
        LoggerConfig.console_handler.setLevel(logging.DEBUG)
        LoggerConfig.console_handler.setFormatter(console_formatter)

    @staticmethod
    def get_logger(name: str) -> Logger:
        if LoggerConfig.console_handler is None:
            LoggerConfig.setup_logger()

        logger = logging.getLogger(name)
        logger.setLevel(LoggerConfig.level)
        
        # Remove all handlers
        logger.handlers = []
        
        # Add handlers
        logger.addHandler(LoggerConfig.console_handler)
        if LoggerConfig.file_handler is not None:
            logger.addHandler(LoggerConfig.file_handler)
                
        # Prevent propagation to root logger
        logger.propagate = False
        
        return logger
