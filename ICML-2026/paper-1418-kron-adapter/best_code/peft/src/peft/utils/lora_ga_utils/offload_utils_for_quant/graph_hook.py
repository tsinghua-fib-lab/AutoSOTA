import torch


def offload_condition(x: torch.Tensor) -> bool:
    """
    The offload conditions are:
    the tensor is on cuda,
    and the number of bytes occupied by the tensor is equal to the theoretical number of bytes
    """
    return x.device.type == "cuda" \
        and x.numel() * x.element_size() == x.untyped_storage().size()


class OffloadSavedTensorHook:
    mode = "release"  # ["release", "debug"]
    offload_device = "cpu"  # cpu or cuda

    @staticmethod
    def unpack(packed):
        origin_device, x = packed
        return x.to(origin_device)

    @staticmethod
    def pack(x: torch.Tensor):
        if offload_condition(x):
            return x.device, x.to(OffloadSavedTensorHook.offload_device)
        else:
            return x.device, x
