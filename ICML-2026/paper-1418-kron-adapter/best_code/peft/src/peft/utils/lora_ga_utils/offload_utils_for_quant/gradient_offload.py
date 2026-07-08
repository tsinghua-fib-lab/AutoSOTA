import torch


class GradientOffloadHookContext:
    def __init__(
            self,
            model: torch.nn.Module,
            record_dict: dict,
            enable: bool = True,
            *args,
            **kwargs,
    ):
        """Offload gradient to cpu

        Args:
            model (torch.nn.Module): The model whose gradients will be offloaded.
            record_dict (dict): A dictionary to record offloaded gradient (named_grad)
            enable (bool, optional): If True, enables the gradient offloading. Defaults to True.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """

        if enable:
            self.gradient_device = "cpu"
        else:
            self.gradient_device = "cuda"
        self.handle_list = list()
        self.model = model
        self.record_dict = record_dict

    def __enter__(self):
        self.register_gradient_hook()

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self.handle_list:
            handle.remove()

    def register_gradient_hook(self):
        for _, param in self.model.named_parameters():
            hook = param.register_hook(
                self.get_record_gradient_hook(self.model, self.record_dict)
            )
            self.handle_list.append(hook)

    def get_record_gradient_hook(self, model, record_dict):
        def record_gradient_hook(grad):
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    if n not in record_dict:
                        record_dict[n] = p.grad.to(self.gradient_device)
                    else:
                        record_dict[n] += p.grad.to(self.gradient_device)
                    p.grad = None
            return grad

        return record_gradient_hook
