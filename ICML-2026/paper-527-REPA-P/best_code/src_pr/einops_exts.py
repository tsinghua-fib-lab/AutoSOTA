from einops import rearrange


def rearrange_many(tensors, pattern, **axes_lengths):
    return tuple(rearrange(tensor, pattern, **axes_lengths) for tensor in tensors)
