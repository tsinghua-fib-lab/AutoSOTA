import math
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image


def tensor_to_image(
    data: Union[Image.Image, torch.Tensor, np.ndarray],
    batched: bool = False,
    format: str = "HWC",
) -> Union[Image.Image, List[Image.Image]]:
    if isinstance(data, Image.Image):
        return data
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    if data.dtype == np.float32 or data.dtype == np.float16:
        data = (data * 255).astype(np.uint8)
    elif data.dtype == np.bool_:
        data = data.astype(np.uint8) * 255
    assert data.dtype == np.uint8
    if format == "CHW":
        if batched and data.ndim == 4:
            data = data.transpose((0, 2, 3, 1))
        elif not batched and data.ndim == 3:
            data = data.transpose((1, 2, 0))

    if batched:
        return [Image.fromarray(d) for d in data]
    return Image.fromarray(data)


def largest_factor_near_sqrt(n: int) -> int:
    """
    Finds the largest factor of n that is closest to the square root of n.

    Args:
        n (int): The integer for which to find the largest factor near its square root.

    Returns:
        int: The largest factor of n that is closest to the square root of n.
    """
    sqrt_n = int(math.sqrt(n))  # Get the integer part of the square root

    # First, check if the square root itself is a factor
    if sqrt_n * sqrt_n == n:
        return sqrt_n

    # Otherwise, find the largest factor by iterating from sqrt_n downwards
    for i in range(sqrt_n, 0, -1):
        if n % i == 0:
            return i

    # If n is 1, return 1
    return 1


def make_image_grid(
    images: List[Image.Image],
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    resize: Optional[int] = None,
) -> Image.Image:
    """
    Prepares a single grid of images. Useful for visualization purposes.
    """
    if rows is None and cols is not None:
        assert len(images) % cols == 0
        rows = len(images) // cols
    elif cols is None and rows is not None:
        assert len(images) % rows == 0
        cols = len(images) // rows
    elif rows is None and cols is None:
        rows = largest_factor_near_sqrt(len(images))
        cols = len(images) // rows

    assert len(images) == rows * cols

    if resize is not None:
        images = [img.resize((resize, resize)) for img in images]

    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid
