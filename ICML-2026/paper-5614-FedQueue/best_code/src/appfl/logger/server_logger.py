import os
import pathlib
import logging
from .utils import LevelFilter
from typing import Optional
from datetime import datetime
from colorama import Fore, Style


class ServerAgentFileLogger:
    """
    ServerAgentFileLogger is a class that logs FL server-side messages to the console and to a file.
    :param file_dir: The directory to save the log file.
    :param file_name: The name of the log file.
    :param experiment_id: An optional string to identify the experiment.
        If not provided, the current date and time will be used.
    """

    def __init__(
        self, file_dir: str = "", file_name: str = "", experiment_id: str = ""
    ) -> None:
        if file_name != "":
            file_name += f"_Server_{experiment_id if experiment_id != '' else datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        info_fmt = logging.Formatter(
            f"{Fore.BLUE}{Style.BRIGHT}appfl: ✅{Style.RESET_ALL}[%(asctime)s server]: %(message)s"
        )
        debug_fmt = logging.Formatter(
            f"{Fore.BLUE}{Style.BRIGHT}appfl: 💡{Style.RESET_ALL}[%(asctime)s server]: %(message)s"
        )
        error_fmt = logging.Formatter(
            f"{Fore.BLUE}{Style.BRIGHT}appfl: ❌{Style.RESET_ALL}[%(asctime)s server]: %(message)s"
        )
        warning_fmt = logging.Formatter(
            f"{Fore.BLUE}{Style.BRIGHT}appfl: ❗️{Style.RESET_ALL}[%(asctime)s server]: %(message)s"
        )

        num_s_handlers = len(
            [h for h in self.logger.handlers if isinstance(h, logging.StreamHandler)]
        )
        num_f_handlers = len(
            [h for h in self.logger.handlers if isinstance(h, logging.FileHandler)]
        )

        if num_s_handlers == 0:
            s_handler_info = logging.StreamHandler()
            s_handler_info.setFormatter(info_fmt)
            s_handler_info.addFilter(LevelFilter(logging.INFO))
            s_handler_debug = logging.StreamHandler()
            s_handler_debug.setFormatter(debug_fmt)
            s_handler_debug.addFilter(LevelFilter(logging.DEBUG))
            s_handler_error = logging.StreamHandler()
            s_handler_error.setFormatter(error_fmt)
            s_handler_error.addFilter(LevelFilter(logging.ERROR))
            s_handler_warning = logging.StreamHandler()
            s_handler_warning.setFormatter(warning_fmt)
            s_handler_warning.addFilter(LevelFilter(logging.WARNING))
            self.logger.addHandler(s_handler_info)
            self.logger.addHandler(s_handler_debug)
            self.logger.addHandler(s_handler_error)
            self.logger.addHandler(s_handler_warning)
        if file_dir != "" and file_name != "" and num_f_handlers == 0:
            if not os.path.exists(file_dir):
                pathlib.Path(file_dir).mkdir(parents=True, exist_ok=True)
            real_file_name = f"{file_dir}/{file_name}.txt"
            self.log_filepath = real_file_name
            f_handler_info = logging.FileHandler(real_file_name)
            f_handler_info.setFormatter(info_fmt)
            f_handler_info.addFilter(LevelFilter(logging.INFO))
            f_handler_debug = logging.FileHandler(real_file_name)
            f_handler_debug.setFormatter(debug_fmt)
            f_handler_debug.addFilter(LevelFilter(logging.DEBUG))
            f_handler_error = logging.FileHandler(real_file_name)
            f_handler_error.setFormatter(error_fmt)
            f_handler_error.addFilter(LevelFilter(logging.ERROR))
            f_handler_warning = logging.FileHandler(real_file_name)
            f_handler_warning.setFormatter(warning_fmt)
            f_handler_warning.addFilter(LevelFilter(logging.WARNING))
            self.logger.addHandler(f_handler_info)
            self.logger.addHandler(f_handler_debug)
            self.logger.addHandler(f_handler_error)
            self.logger.addHandler(f_handler_warning)
            self.info(f"Logging to {real_file_name}")

    def info(self, info: str) -> None:
        self.logger.info(info)

    def debug(self, debug: str) -> None:
        self.logger.debug(debug)

    def error(self, error: str) -> None:
        self.logger.error(error)

    def warning(self, warning: str) -> None:
        self.logger.warning(warning)

    def get_log_filepath(self) -> Optional[str]:
        if hasattr(self, "log_filepath"):
            return self.log_filepath
