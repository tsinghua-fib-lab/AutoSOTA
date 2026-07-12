import logging
import os
import sys
from typing import Optional, Union


class Logger:
    """
    A configurable logger that can output to console, file, or both.
    Implements a singleton pattern to ensure the same logger is used throughout the application.
    """
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self, 
        name: str = "cfda", 
        log_to_console: bool = True,
        log_to_file: bool = False,
        log_file: Optional[str] = None,
        log_level: Union[int, str] = logging.INFO
    ):
        """
        Initialize the logger.
        
        Args:
            name: Logger name
            log_to_console: Whether to output logs to console
            log_to_file: Whether to output logs to file
            log_file: Log file path, if None and log_to_file is True, 
                      a default 'logs/{name}.log' will be used
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        if self._initialized:
            return
            
        self.name = name
        self.logger = logging.getLogger(name)
        
        # Convert string log level to int if needed
        if isinstance(log_level, str):
            log_level = getattr(logging, log_level.upper())
        
        self.logger.setLevel(log_level)
        self.logger.handlers = []  # Clear any existing handlers
        
        # Create formatters
        console_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Add console handler if requested
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(console_format)
            self.logger.addHandler(console_handler)
            
        # Add file handler if requested
        if log_to_file:
            if log_file is None:
                # Create default logs directory if it doesn't exist
                os.makedirs('logs', exist_ok=True)
                log_file = f'logs/{name}.log'
                
            # Create directory for log file if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
                
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)
            
        self._initialized = True
    
    def debug(self, message: str) -> None:
        """Log a debug message."""
        self.logger.debug(message)
        
    def info(self, message: str) -> None:
        """Log an info message."""
        self.logger.info(message)
        
    def warning(self, message: str) -> None:
        """Log a warning message."""
        self.logger.warning(message)
        
    def error(self, message: str) -> None:
        """Log an error message."""
        self.logger.error(message)
        
    def critical(self, message: str) -> None:
        """Log a critical message."""
        self.logger.critical(message)
        
    def set_level(self, level: Union[int, str]) -> None:
        """Set the logging level."""
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        self.logger.setLevel(level)
        
    def add_file_handler(self, log_file: str) -> None:
        """Add a file handler to the logger."""
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        
    def get_logger(self) -> logging.Logger:
        """Get the underlying logger object."""
        return self.logger
