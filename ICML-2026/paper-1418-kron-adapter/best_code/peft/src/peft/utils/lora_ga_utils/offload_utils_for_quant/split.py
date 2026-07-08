import torch.nn


def get_model_memory(model: torch.nn.Module, forward_factor: float = 1.3):
    """
    Calculates the estimated memory usage of a model in gigabytes.

    Args:
        model (torch.nn.Module): The model whose memory usage is to be calculated.
        forward_factor (float, optional): A factor to account for additional memory usage during the forward pass.
                                          Defaults to 1.3.

    Returns:
        float: The estimated memory usage of the model in gigabytes.
    """
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    return forward_factor * total / 1024 ** 3


def get_split_num(origin_type: str = "bf16", quant_type: str = "int8"):
    """
    Calculates the ratio of original type size to quantized type size.

    Args:
        origin_type (str, optional): The data type of the original tensor. Defaults to "bf16".
                                     Options are "fp32" and "bf16".
        quant_type (str, optional): The data type of the quantized tensor. Defaults to "int8".
                                    Options are "int8" and "nf4".

    Raises:
        ValueError: If the origin_type is not "fp32" or "bf16".
        ValueError: If the quant_type is not "int8" or "nf4".

    Returns:
        int: The ratio of the original type size to the quantized type size.
    """
    n_origin_bytes = 16
    n_quant_bytes = 8
    match origin_type:
        case "fp32":
            n_origin_bytes = 32
        case "bf16":
            n_origin_bytes = 16
        case _:
            raise ValueError("Wrong dtype")
    match quant_type:
        case "int8":
            n_quant_bytes = 8
        case "nf4":
            n_quant_bytes = 4
        case _:
            raise ValueError("Wrong dtype")
    return n_origin_bytes // n_quant_bytes
