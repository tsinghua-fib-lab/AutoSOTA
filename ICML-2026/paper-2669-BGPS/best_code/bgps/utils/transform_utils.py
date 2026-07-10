from typing import Sequence
import numpy as np
import random
from PIL import ImageFilter, ImageOps


__all__ = [
            "GaussianBlur",
            "Solarization",
            "to_tuple",
            "is_rgb_image",
            "is_grayscale_image"
            ]


class GaussianBlur(object):

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img



def to_tuple(param, low=None, bias=None):
    if low is not None and bias is not None:
        raise ValueError("Arguments low and bias are mutually exclusive")

    if param is None:
        return param

    if isinstance(param, (int, float)):
        if low is None:
            param = -param, +param
        else:
            param = (low, param) if low < param else (param, low)
    elif isinstance(param, Sequence):
        if len(param) != 2:
            raise ValueError("to_tuple expects 1 or 2 values")
        param = tuple(param)
    else:
        raise ValueError("Argument param must be either scalar (int, float) or tuple")

    if bias is not None:
        return tuple(bias + x for x in param)

    return tuple(param)

def is_rgb_image(image: np.ndarray) -> bool:
    return len(image.shape) == 3 and image.shape[-1] == 3


def is_grayscale_image(image: np.ndarray) -> bool:
    return (len(image.shape) == 2) or (len(image.shape) == 3 and image.shape[-1] == 1)