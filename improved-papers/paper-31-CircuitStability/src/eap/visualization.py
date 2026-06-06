import sys
import functools

import numpy as np
import matplotlib
import matplotlib.cm


EDGE_TYPE_COLORS = {
    "q": "#FF00FF",  # Purple
    "k": "#00FF00",  # Green
    "v": "#0000FF",  # Blue
    None: "#000000",  # Black
}


def generate_random_color(colorscheme: str) -> str:
    """
    https://stackoverflow.com/questions/28999287/generate-random-colors-rgb
    """

    def rgb2hex(rgb):
        """
        https://stackoverflow.com/questions/3380726/converting-an-rgb-color-tuple-to-a-hexidecimal-string
        """
        return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

    return rgb2hex(color(colorscheme, np.random.randint(0, 256), rgb_order=True))


# ripped from cmapy since it doesn't play nice with new versions of matplotlib
def cmap(cmap_name, rgb_order=False):
    """
    Extract colormap color information as a LUT compatible with cv2.applyColormap().
    Default channel order is BGR.

    Args:
        cmap_name: string, name of the colormap.
        rgb_order: boolean, if false or not set, the returned array will be in
                   BGR order (standard OpenCV format). If true, the order
                   will be RGB.

    Returns:
        A numpy array of type uint8 containing the colormap.
    """

    c_map = matplotlib.colormaps.get_cmap(cmap_name)
    rgba_data = matplotlib.cm.ScalarMappable(cmap=c_map).to_rgba(
        np.arange(0, 1.0, 1.0 / 256.0), bytes=True
    )
    rgba_data = rgba_data[:, 0:-1].reshape((256, 1, 3))

    # Convert to BGR (or RGB), uint8, for OpenCV.
    cmap = np.zeros((256, 1, 3), np.uint8)

    if not rgb_order:
        cmap[:, :, :] = rgba_data[:, :, ::-1]
    else:
        cmap[:, :, :] = rgba_data[:, :, :]

    return cmap


# If python 3, redefine cmap() to use lru_cache.
if sys.version_info > (3, 0):
    cmap = functools.lru_cache(maxsize=200)(cmap)


def color(cmap_name, index, rgb_order=False):
    """Returns a color of a given colormap as a list of 3 BGR or RGB values.

    Args:
        cmap_name: string, name of the colormap.
        index:     floating point between 0 and 1 or integer between 0 and 255,
                   index of the requested color.
        rgb_order: boolean, if false or not set, the returned list will be in
                   BGR order (standard OpenCV format). If true, the order
                   will be RGB.

    Returns:
        List of RGB or BGR values.
    """

    # Float values: scale from 0-1 to 0-255.
    if isinstance(index, float):
        val = round(min(max(index, 0.0), 1.0) * 255)
    else:
        val = min(max(index, 0), 255)

    # Get colormap and extract color.
    colormap = cmap(cmap_name, rgb_order)
    return colormap[int(val), 0, :].tolist()
