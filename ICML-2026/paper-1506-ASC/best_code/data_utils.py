import torch

def replace_tensors_with_item(input_dict):
    """
    Takes a dictionary with items that are lists and replaces any Tensor
    in the lists with its .item() value.

    Args:
        input_dict (dict): A dictionary with lists as values.

    Returns:
        dict: A new dictionary with Tensors replaced by their .item() values.
    """
    output_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, list):
            output_dict[key] = [
                item.item() if isinstance(item, torch.Tensor) and item.numel() == 1 else item for item in value
            ]
        else:
            output_dict[key] = value
    return output_dict