import numpy as np


def abs_grayscale_norm(img):
    """Returns absolute value normalized image 2D."""
    if len(img.shape) == 2:
        img = np.absolute(img)
        img = img / float(img.max()) if img.max() > 0 else img
    else:
        image_2d = np.sum(np.abs(img), axis=2)
        vmax = np.percentile(image_2d, 99) + 1e-10
        vmin = np.min(image_2d)
        img = np.clip((image_2d - vmin) / (vmax - vmin), 0, 1)
    return img