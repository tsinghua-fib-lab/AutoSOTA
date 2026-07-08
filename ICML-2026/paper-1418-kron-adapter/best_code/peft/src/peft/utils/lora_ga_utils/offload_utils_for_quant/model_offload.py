from .saved_tensor_offload import SavedTensorOffloadContext
from .forward_backward_offload import ForwardBackwardOffloadHookContext


class ModelOffloadHookContext:
    def __init__(
        self,
        model,
        no_split_module_classes=None,
        num_block: int = 2,
        enable=True,
        # =========================
        device="cuda",
        strategy="block",
        with_backward_hook=False,
    ):
        """
        Initializes the ModelOffloadHookContext to manage offloading of model computations and saved tensors.

        Args:
            model (torch.nn.Module): The model to which the hooks will be applied.
            no_split_module_classes (list of type, optional): List of module classes that should not be split during offloading. Defaults to None.
            num_block (int, optional): The number of blocks to use when the strategy is set to "block". Defaults to 2.
            enable (bool, optional): If True, enables the hook. Defaults to True.
            device (str, optional): The device to which activations and gradients will be offloaded. Defaults to "cuda".
            strategy (str, optional): The offloading strategy to use. Options are "module" or "block". Defaults to "block".
            with_backward_hook (bool, optional): If True, enables the backward hook for debugging purposes. Defaults to False.
        """
        self.enable = enable
        if not enable:
            return
        self.forwardBackwardOffloadHookContext = ForwardBackwardOffloadHookContext(
            model=model,
            device=device,
            no_split_module_classes=no_split_module_classes,
            with_backward_hook=with_backward_hook,  # for debug
            enable=True,
            num_block=num_block,
            strategy=strategy,  # enum["module","block"],
        )
        self.savedTensorOffloadContext = SavedTensorOffloadContext()

    def __enter__(self):
        if not self.enable:
            return
        self.forwardBackwardOffloadHookContext.__enter__()
        self.savedTensorOffloadContext.__enter__()
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enable:
            return
        self.forwardBackwardOffloadHookContext.__exit__(exc_type, exc_val, exc_tb)
        self.savedTensorOffloadContext.__exit__(exc_type, exc_val, exc_tb)
        pass
