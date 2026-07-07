from contextlib import contextmanager
from logging import getLogger

from torch import nn

logger = getLogger(__name__)


@contextmanager
def eval_mode(model: nn.Module):
    """
    Context manager to put a model in eval mode.
    """
    was_training = model.training
    if was_training:
        model.eval()
    try:
        yield model
    finally:
        if was_training:
            model.train()
