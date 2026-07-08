import torch
from .graph_hook import OffloadSavedTensorHook


class SavedTensorOffloadContext:
    def __init__(self):
        self.savedTensorOffloadContext = torch.autograd.graph.saved_tensors_hooks(
            pack_hook=OffloadSavedTensorHook.pack,
            unpack_hook=OffloadSavedTensorHook.unpack
        )

    def __enter__(self):
        self.savedTensorOffloadContext.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.savedTensorOffloadContext.__exit__(exc_type, exc_val, exc_tb)
