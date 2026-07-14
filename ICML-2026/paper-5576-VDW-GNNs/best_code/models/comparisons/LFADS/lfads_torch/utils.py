import torch

from .tuples import SessionBatch


def flatten(dictionary, level=[]):
    """Flattens a dictionary by placing '.' between levels.
    This function flattens a hierarchical dictionary by placing '.'
    between keys at various levels to create a single key for each
    value. It is used internally for converting the configuration
    dictionary to more convenient formats. Implementation was
    inspired by `this StackOverflow post
    <https://stackoverflow.com/questions/6037503/python-unflatten-dict>`_.
    Parameters
    ----------
    dictionary : dict
        The hierarchical dictionary to be flattened.
    level : str, optional
        The string to append to the beginning of this dictionary,
        enabling recursive calls. By default, an empty string.
    Returns
    -------
    dict
        The flattened dictionary.
    """

    tmp_dict = {}
    for key, val in dictionary.items():
        if type(val) == dict:
            tmp_dict.update(flatten(val, level + [key]))
        else:
            tmp_dict[".".join(level + [key])] = val
    return tmp_dict


def transpose_lists(output: list[list]):
    """Transposes the ordering of a list of lists."""
    return list(map(list, zip(*output)))


def send_batch_to_device(batch, device):
    """Recursively searches the batch for tensors and sends them to the device"""

    def send_to_device(obj):
        obj_type = type(obj)
        if obj_type == torch.Tensor:
            return obj.to(device)
        elif obj_type == dict:
            return {k: send_to_device(v) for k, v in obj.items()}
        elif obj_type == list:
            return [send_to_device(o) for o in obj]
        elif obj_type == SessionBatch:
            return SessionBatch(*[send_to_device(o) for o in obj])
        else:
            raise NotImplementedError(
                f"`send_batch_to_device` has not been implemented for {str(obj_type)}."
            )

    return send_to_device(batch)
