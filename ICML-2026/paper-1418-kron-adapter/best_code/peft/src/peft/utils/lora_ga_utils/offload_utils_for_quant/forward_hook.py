import torch


class ForwardHookForDevice:
    def __init__(self):
        pass

    @staticmethod
    def get_align_device_pre_forward_hook(device="cuda", with_kwargs=False):
        """
        ensure same device for input and module
        """

        def hook(module: torch.nn.Module, args):
            if device is not None:
                align_device = device
            elif len(list(module.parameters())) > 0:
                align_device = next(module.parameters()).device
            else:
                align_device = "cuda"
            module.to(align_device)
            args = tuple(arg.to(align_device) if isinstance(arg, torch.Tensor) else arg for arg in args)
            return args

        def hook_with_kwargs(module: torch.nn.Module, args, kwargs):
            if device is not None:
                align_device = device
            elif len(list(module.parameters())) > 0:
                align_device = next(module.parameters()).device
            else:
                align_device = "cuda"
            module.to(align_device)
            args = tuple(arg.to(align_device) if isinstance(arg, torch.Tensor) else arg for arg in args)
            _kwargs = dict()
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    _kwargs[k] = v.to(align_device)
                else:
                    _kwargs[k] = v
            kwargs = _kwargs
            return args, kwargs

        if with_kwargs:
            return hook_with_kwargs
        else:
            return hook

    @staticmethod
    def get_forward_hook(pre: bool, device=None, with_kwargs=False):
        """
        device is executing device
        origin_device is the device where tensor is saved after forward
        """
        origin_device = "cpu"
        if device is not None:
            device = device
        else:
            device = "cuda"

        def pre_hook(module: torch.nn.Module, args):
            module.to(device)
            args = tuple(arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in args)
            return args

        def after_hook(module: torch.nn.Module, args, output):
            module.to(origin_device)
            output = output.to(origin_device) if isinstance(output, torch.Tensor) else output
            if isinstance(output, tuple):
                output = tuple(o.to(origin_device) if isinstance(o, torch.Tensor) else o for o in output)
            return output

        def pre_hook_with_kwargs(module, args, kwargs):
            module.to(device)
            args = tuple(arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in args)
            kwargs = {n: v.to(device) if isinstance(v, torch.Tensor) else v for n, v in kwargs.items()}
            return args, kwargs

        def after_hook_with_kwargs(module, args, kwargs, output):
            module.to(origin_device)
            output = output.to(origin_device) if isinstance(output, torch.Tensor) else output
            if isinstance(output, tuple):
                output = tuple(o.to(origin_device) if isinstance(o, torch.Tensor) else o for o in output)
            return output

        if pre and with_kwargs:
            return pre_hook_with_kwargs
        elif pre and not with_kwargs:
            return pre_hook
        elif not pre and with_kwargs:
            return after_hook_with_kwargs
        elif not pre and not with_kwargs:
            return after_hook

    @staticmethod
    def get_full_name_list(model):
        """
        Get the module name list of the leaf nodes of the module tree
        """
        full_name_list = list()

        def _get_full_name_list(module, parent_name=''):

            """
            get full name list of all submodule. result is self.
            """
            if len(list(module.named_children())) == 0:
                full_name_list.append(parent_name)
            for name, sub_module in module.named_children():
                full_name = f'{parent_name}.{name}' if parent_name else name
                _get_full_name_list(sub_module, full_name)

        _get_full_name_list(model)

        return full_name_list

    @staticmethod
    def get_module_list(model, no_split_module_classes=None):
        """
        Get the module name list of the leaf nodes of the module tree,
         and stop recursing when the specified node(no_split_module_class) is reached.
        """
        module_list = list()

        def _get_module_list(module: torch.nn.Module, parent_name=""):
            flag = False
            if module.__class__.__name__ in no_split_module_classes:
                flag = True
            if flag:
                module_list.append(parent_name)
                return
            if len(list(module.named_children())) == 0:
                module_list.append(parent_name)
                return

            for name, sub_module in module.named_children():
                extend_name = f"{parent_name}.{name}" if parent_name else name
                _get_module_list(sub_module, extend_name)

        _get_module_list(model)
        return module_list
