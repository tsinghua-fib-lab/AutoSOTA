import torch
from torch import Tensor
from torchvision.transforms import Normalize

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from typing import Callable, Optional


DAVE_CMAP = LinearSegmentedColormap.from_list(
    "white_violet_black", 
    [
        "black", 
        "darkviolet", 
        "darkorange", 
        "orange", 
        "yellow", 
        "white",
    ],
)


def visualize_attribution_batch(
    attribution_map: Tensor, 
    image_batch: Tensor, 
    input_transform: Optional[Callable] = None,
):
    a = attribution_map
    x = image_batch
    
    N_a, C_a, H_a, W_a = a.shape
    N_x, C_x, H_x, W_x = x.shape
    
    assert C_a == 1, f"Expected single-channel explanation. Got {C} channels."
    assert C_x == 3, f"Expected RGB input image. Got {C} channels."
    assert N_a == N_x, f"Number of explanations ({N_a}) does not match numbet of images ({N_x})"

    if input_transform is not None:
        x = denormalize_input(x, input_transform)
        
    x = x.permute(0, 2, 3, 1).detach().cpu()
    a = a.permute(0, 2, 3, 1).detach().cpu()

    for i in range(N_x):
        plt.figure(figsize=(15, 7), dpi=200)
        
        plt.subplot(121)
        plt.title('Image')
        plt.imshow(x[i])
        plt.axis('off')

        plt.subplot(122)
        plt.title('DAVE')
        plt.imshow(a[i], cmap=DAVE_CMAP, vmin=a[i].min(), vmax=a[i].max())
        plt.axis('off')

        plt.show()
        plt.close()


def find_normalize(transform):
    """
    Looks for input normalization. 
    """
    if isinstance(transform, Normalize):
        return transform

    if hasattr(transform, "transforms"):
        for t in transform.transforms:
            norm = find_normalize(t)
            if norm is not None:
                return norm

    return None


def denormalize_input(x: Tensor, input_transform: Callable) -> Tensor:
    """
    Denormalizes input (if normalized by input_transform).
    """
    norm = find_normalize(input_transform)
    
    if norm is None:
        return x

    mean = torch.as_tensor(norm.mean, dtype=x.dtype, device=x.device)
    std = torch.as_tensor(norm.std, dtype=x.dtype, device=x.device)

    shape = (1,) * (x.ndim - 3) + (len(mean), 1, 1)
    mean = mean.view(shape)
    std = std.view(shape)
    return x * std + mean
