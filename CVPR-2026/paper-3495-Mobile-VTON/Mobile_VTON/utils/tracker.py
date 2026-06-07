import logging
import sys
import os.path as osp
import os
from accelerate.tracking import GeneralTracker, on_main_process
from torch.utils.tensorboard import SummaryWriter
from typing import Union

# Setup logging
logger = logging.getLogger(__name__)

# Setup working directory
WORK_DIR = osp.abspath(osp.join(osp.dirname(__file__), "../../.."))
logger.debug(f"Working directory: {WORK_DIR}")
if WORK_DIR not in sys.path:
    logger.warning(f"Working directory ({WORK_DIR}) is not in sys.path. Adding it.")
    sys.path.append(WORK_DIR)


class TensorBoardTracker(GeneralTracker):
    """
    Custom `Tracker` class that supports `tensorboard`. Should be initialized at the start of your script.

    Args:
        logging_dir (`str`, `os.PathLike`):
            Location for TensorBoard logs to be stored.
        kwargs:
            Additional key word arguments passed along to the `tensorboard.SummaryWriter.__init__` method.
    """
    name = "tensorboard"
    requires_logging_directory = True

    @on_main_process
    def __init__(self, logging_dir: Union[str, os.PathLike], **kwargs):
        super().__init__()
        self.logging_dir = logging_dir
        self.writer = SummaryWriter(self.logging_dir, **kwargs)

    @property
    def tracker(self):
        return self.writer

    @on_main_process
    def add_scalar(self, tag, scalar_value, **kwargs):
        self.writer.add_scalar(tag=tag, scalar_value=scalar_value, **kwargs)

    @on_main_process
    def add_text(self, tag, text_string, **kwargs):
        self.writer.add_text(tag=tag, text_string=text_string, **kwargs)

    @on_main_process
    def add_figure(self, tag, figure, **kwargs):
        self.writer.add_figure(tag=tag, figure=figure, **kwargs)
