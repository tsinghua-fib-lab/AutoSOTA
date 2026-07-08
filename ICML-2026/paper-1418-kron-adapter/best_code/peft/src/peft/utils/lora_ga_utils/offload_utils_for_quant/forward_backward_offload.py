import torch
import numpy as np
from .graph_hook import OffloadSavedTensorHook
from .forward_hook import ForwardHookForDevice


class ForwardBackwardOffloadHookContext(ForwardHookForDevice):
    mode = ("release",)  # enum["debug","release"]

    def __init__(
        self,
        model,
        offload_proportion=0.5,
        device="cuda",
        no_split_module_classes=None,
        with_backward_hook=False,  # for debug
        enable=True,
        num_block: int = 2,
        strategy="block",  # enum["module","block"],
    ):
        """Offload model during forward and backward.

        Args:
            model (torch.nn.Module): The model to which the hook will be applied.
            offload_proportion (float, optional): The proportion of activations to offload. Defaults to 0.5.
            device (str, optional): The device to which activations will be offloaded. Defaults to "cuda".
            no_split_module_classes (list of type, optional): List of module classes that should not be split during offloading. Defaults to None.
            with_backward_hook (bool, optional): If True, enables the backward hook for debugging purposes. Defaults to False.
            enable (bool, optional): If True, enables the hook. Defaults to True.
            num_block (int, optional): The number of blocks to use when the strategy is set to "block". Defaults to 2.
            strategy (str, optional): The offloading strategy to use. Options are "module" or "block". Defaults to "block".
        """
        self.enable = enable
        if not enable:
            return
        super().__init__()
        self.strategy = strategy
        self.num_block = num_block
        self.device = device  # computing device for offloaded modules
        self.with_backward_hook = with_backward_hook
        self.model = model
        self.handle_list = list()
        if no_split_module_classes is None:
            no_split_module_classes = ["LlamaDecoderLayer", "GPT2TransformerBlock"]
        self.module_list = ForwardHookForDevice.get_module_list(model, no_split_module_classes=no_split_module_classes)
        if ForwardBackwardOffloadHookContext.mode == "debug":
            print(f"module_list:{self.module_list}")
        if self.strategy == "module":
            self.offload_list = self.module_list[: int(offload_proportion * len(self.module_list))]
            if ForwardBackwardOffloadHookContext.mode == "debug":
                print(f"self.offload_list={self.offload_list}")
        elif self.strategy == "block":
            self.module_info = self.get_partition_block(self.module_list, self.num_block)
            if ForwardBackwardOffloadHookContext.mode == "debug":
                print(f"self.module_info={self.module_info}")

    def __enter__(self):
        """
        Register the hook in the appropriate module
        """
        if not self.enable:
            return
        if ForwardBackwardOffloadHookContext.mode == "debug":
            print("ForwardBackwardOffloadHookContext.__enter__(self):")
        if self.strategy == "module":
            self.register_forward_hook_by_module(self.model)
        else:
            self.register_hook_by_block(self.model)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        remove hook registered by __enter__
        """
        if not self.enable:
            return
        if ForwardBackwardOffloadHookContext.mode == "debug":
            print("ForwardBackwardOffloadHookContext.__exit__(self, exc_type, exc_val, exc_tb)")
        for handle in self.handle_list:
            handle.remove()

    def register_hook_by_block(self, module: torch.nn.Module, parent_name=""):
        if self.with_backward_hook and parent_name in self.module_list:
            handle = module.register_full_backward_pre_hook(hook=self.get_backward_hook(pre=True))
            self.handle_list.append(handle)
            handle = module.register_full_backward_hook(hook=self.get_backward_hook(pre=False))
            self.handle_list.append(handle)
        if parent_name in self.module_list:
            if ForwardBackwardOffloadHookContext.mode == "debug":
                print(f"register_hook_by_block(self, module, parent_name={parent_name}")
            # forward hook==============================================================
            handle = module.register_forward_pre_hook(
                hook=self.get_forward_hook_by_block(info=self.module_info[parent_name], pre=True, with_kwargs=True),
                with_kwargs=True,
            )
            self.handle_list.append(handle)
            handle = module.register_forward_hook(
                hook=self.get_forward_hook_by_block(info=self.module_info[parent_name], pre=False, with_kwargs=True),
                with_kwargs=True,
            )
            self.handle_list.append(handle)
            # backward hook==============================================================
            handle = module.register_full_backward_pre_hook(
                hook=self.get_backward_hook_by_block(info=self.module_info[parent_name], pre=True)
            )
            self.handle_list.append(handle)
            handle = module.register_full_backward_hook(
                hook=self.get_backward_hook_by_block(info=self.module_info[parent_name], pre=False)
            )
            self.handle_list.append(handle)
            return

        for name, sub_module in module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name
            self.register_hook_by_block(sub_module, full_name)

    @staticmethod
    def get_forward_hook_by_block(info: dict, pre=True, device="cuda", with_kwargs=True):
        if device is None:
            device = "cuda"
        offload_device = "cpu"
        first_block_flag = info["first_block_flag"]
        last_block_flag = info["last_block_flag"]
        first_module_flag = info["first_module_flag"]
        last_module_flag = info["last_module_flag"]

        def pre_hook_with_kwargs(module, args, kwargs):
            if ForwardBackwardOffloadHookContext.mode == "debug":
                from .resource_monitor import show_gpu_and_cpu_memory

                show_gpu_and_cpu_memory()
            # model
            module.to(device)
            args = tuple(arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in args)
            kwargs = {n: v.to(device) if isinstance(v, torch.Tensor) else v for n, v in kwargs.items()}
            # saved_tensor,such as activations.
            if not last_block_flag and first_module_flag:
                if ForwardBackwardOffloadHookContext.mode == "debug":
                    print(f"set OffloadSavedTensorHook.offload_device = offload_device:{offload_device}")
                OffloadSavedTensorHook.offload_device = offload_device
            elif last_block_flag and first_module_flag:
                if ForwardBackwardOffloadHookContext.mode == "debug":
                    print(f"set OffloadSavedTensorHook.offload_device = device:{device}")
                OffloadSavedTensorHook.offload_device = device
            return args, kwargs

        def after_hook_with_kwargs(module, args, kwargs, output):
            if not last_block_flag:
                module.to(offload_device)
                # output = output.to(offload_device) if isinstance(output, torch.Tensor) else output
                # if isinstance(output, tuple):
                #     output = tuple(o.to(offload_device) if isinstance(o, torch.Tensor) else o for o in output)
            elif last_block_flag:
                module.to(device)
                # output = output.to(device) if isinstance(output, torch.Tensor) else output
                # if isinstance(output, tuple):
                #     output = tuple(o.to(device) if isinstance(o, torch.Tensor) else o for o in output)
            return output

        if pre:
            return pre_hook_with_kwargs
        else:
            return after_hook_with_kwargs

    @staticmethod
    def get_backward_hook_by_block(info: dict, pre=True, device="cuda"):
        if device is None:
            device = "cuda"
        offload_device = "cpu"
        first_block_flag = info["first_block_flag"]
        last_block_flag = info["last_block_flag"]
        first_module_flag = info["first_module_flag"]
        last_module_flag = info["last_module_flag"]

        def pre_hook(module, grad_output):
            module.to(device)
            return grad_output

        def after_hook(module, grad_input, grad_output):
            if not first_block_flag:
                module.to(offload_device)
            else:
                pass
            return grad_input

        if pre:
            return pre_hook
        else:
            return after_hook

    @staticmethod
    def get_backward_hook(pre=True):
        def pre_hook(module, grad_output):
            if ForwardBackwardOffloadHookContext.mode == "debug":
                from .resource_monitor import show_gpu_and_cpu_memory

                show_gpu_and_cpu_memory()
                pass
            return grad_output

        def after_hook(module, grad_input, grad_output):
            if ForwardBackwardOffloadHookContext.mode == "debug":
                from .resource_monitor import show_gpu_and_cpu_memory

                show_gpu_and_cpu_memory()
                pass
            return grad_input

        if pre:
            return pre_hook
        else:
            return after_hook

    def register_forward_hook_by_module(self, module: torch.nn.Module, parent_name=""):
        if ForwardBackwardOffloadHookContext.mode == "debug":
            print(f"register_forward_hook_by_module(self, module, parent_name={parent_name}")
        if self.with_backward_hook and parent_name in self.module_list:
            handle = module.register_full_backward_pre_hook(hook=self.get_backward_hook())
            self.handle_list.append(handle)
            handle = module.register_full_backward_hook(hook=self.get_backward_hook(pre=False))
            self.handle_list.append(handle)

        if parent_name in self.offload_list:
            handle = module.register_forward_pre_hook(
                self.get_forward_hook(pre=True, device=self.device, with_kwargs=True),
                with_kwargs=True,
            )
            self.handle_list.append(handle)
            handle = module.register_forward_hook(
                self.get_forward_hook(pre=False, device=self.device, with_kwargs=True),
                with_kwargs=True,
            )
            self.handle_list.append(handle)
            return
        elif parent_name in self.module_list:
            handle = module.register_forward_pre_hook(
                self.get_align_device_pre_forward_hook(device="cuda", with_kwargs=True),
                with_kwargs=True,
            )
            self.handle_list.append(handle)
            return
        for name, sub_module in module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name
            self.register_forward_hook_by_module(sub_module, full_name)

    @staticmethod
    def get_partition_block(module_list: list, num_block: int) -> dict:
        block_list = list()
        module_groups = [list(e) for e in np.array_split(module_list, num_block)]
        for i in range(num_block):
            block = dict()
            block["module_list"] = module_groups[i]
            block["first_block_flag"] = True if i == 0 else False
            block["last_block_flag"] = True if i == (num_block - 1) else False
            block_list.append(block)
        if ForwardBackwardOffloadHookContext.mode == "debug":
            print(block_list)
        module_info = dict()
        for block in block_list:
            n_module = len(block["module_list"])
            for i in range(n_module):
                module_name = block["module_list"][i]
                module_info[module_name] = dict()
                module_info[module_name].update(
                    {
                        "first_block_flag": block["first_block_flag"],
                        "last_block_flag": block["last_block_flag"],
                        "first_module_flag": True if i == 0 else False,
                        "last_module_flag": True if i == (n_module - 1) else False,
                    }
                )
        return module_info
