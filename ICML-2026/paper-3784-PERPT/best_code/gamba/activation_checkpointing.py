from typing import Sequence, Type

import torch.distributed.algorithms._checkpoint.checkpoint_wrapper as cw
import torch.nn as nn


class _ActivationCheckpointingPolicy:
    def __init__(self, block_types: Sequence[Type[nn.Module]], every: int):
        self.block_types = tuple(set(block_types))
        self.every = every
        self.counter = 0

    def __call__(self, module: nn.Module) -> bool:
        if isinstance(module, self.block_types):
            self.counter += 1
            return self.every == 0 or self.counter % self.every == 0


def apply_activation_checkpointing(model: nn.Module, blocks: Sequence[Type[nn.Module]], every: int = 0) -> None:
    cw.apply_activation_checkpointing(
        model=model,
        checkpoint_wrapper_fn=cw.checkpoint_wrapper,
        check_fn=_ActivationCheckpointingPolicy(blocks, every),
    )
