# Copied from https://huggingface.co/spaces/PolyU-ChenLab/UniPixel/blob/main/unipixel/utils/visualizer.py

import colorsys
import io
import math
import random
from enum import Enum, unique

import cv2
import imageio.v3 as iio
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
import numpy as np
import pycocotools.mask as mask_util
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg

_SMALL_OBJECT_AREA_THRESH = 1000
_LARGE_MASK_AREA_THRESH = 120000

_COLORS = np.array([
    0.000, 0.447, 0.741, 0.850, 0.325, 0.098, 0.929, 0.694, 0.125, 0.494, 0.184, 0.556, 0.466, 0.674, 0.188, 0.301,
    0.745, 0.933, 0.635, 0.078, 0.184, 0.300, 0.300, 0.300, 0.600, 0.600, 0.600, 1.000, 0.000, 0.000, 1.000, 0.500,
    0.000, 0.749, 0.749, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 1.000, 0.667, 0.000, 1.000, 0.333, 0.333, 0.000,
    0.333, 0.667, 0.000, 0.333, 1.000, 0.000, 0.667, 0.333, 0.000, 0.667, 0.667, 0.000, 0.667, 1.000, 0.000, 1.000,
    0.333, 0.000, 1.000, 0.667, 0.000, 1.000, 1.000, 0.000, 0.000, 0.333, 0.500, 0.000, 0.667, 0.500, 0.000, 1.000,
    0.500, 0.333, 0.000, 0.500, 0.333, 0.333, 0.500, 0.333, 0.667, 0.500, 0.333, 1.000, 0.500, 0.667, 0.000, 0.500,
    0.667, 0.333, 0.500, 0.667, 0.667, 0.500, 0.667, 1.000, 0.500, 1.000, 0.000, 0.500, 1.000, 0.333, 0.500, 1.000,
    0.667, 0.500, 1.000, 1.000, 0.500, 0.000, 0.333, 1.000, 0.000, 0.667, 1.000, 0.000, 1.000, 1.000, 0.333, 0.000,
    1.000, 0.333, 0.333, 1.000, 0.333, 0.667, 1.000, 0.333, 1.000, 1.000, 0.667, 0.000, 1.000, 0.667, 0.333, 1.000,
    0.667, 0.667, 1.000, 0.667, 1.000, 1.000, 1.000, 0.000, 1.000, 1.000, 0.333, 1.000, 1.000, 0.667, 1.000, 0.333,
    0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.167,
    0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000, 1.000, 0.000,
    0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000,
    0.000, 1.000, 0.000, 0.000, 0.000, 0.143, 0.143, 0.143, 0.857, 0.857, 0.857, 1.000, 1.000, 1.000
]).astype(np.float32).reshape(-1, 3)


def random_color(rgb=False, maximum=1):
    idx = np.random.randint(0, len(_COLORS))
    ret = _COLORS[idx] * maximum
    if not rgb:
        ret = ret[::-1]
    return ret


def sample_color(rgb=False, maximum=1):
    inds = list(range(len(_COLORS)))
    random.shuffle(inds)
    ret = _COLORS[inds] * maximum
    if not rgb:
        ret = ret[::-1]
    return ret


@unique
class ColorMode(Enum):
    """
    Enum of different color modes to use for instance visualizations.
    """

    IMAGE = 0
    """
    Picks a random color for every instance and overlay segmentations with low opacity.
    """
    SEGMENTATION = 1
    """
    Let instances of the same category have similar colors
    (from metadata.thing_colors), and overlay them with
    high opacity. This provides more attention on the quality of segmentation.
    """
    IMAGE_BW = 2
    """
    Same as IMAGE, but convert all areas without masks to gray-scale.
    Only available for drawing per-instance mask predictions.
    """


class GenericMask:
    """
    Attribute:
        polygons (list[ndarray]): list[ndarray]: polygons for this mask.
            Each ndarray has format [x, y, x, y, ...]
        mask (ndarray): a binary mask
    """

    def __init__(self, mask_or_polygons, height, width):
        self._mask = self._polygons = self._has_holes = None
        self.height = height
        self.width = width

        m = mask_or_polygons
        if isinstance(m, dict):
            # RLEs
            assert "counts" in m and "size" in m
            if isinstance(m["counts"], list):  # uncompressed RLEs
                h, w = m["size"]
                assert h == height and w == width
                m = mask_util.frPyObjects(m, h, w)
            self._mask = mask_util.decode(m)[:, :]
            return

        if isinstance(m, list):  # list[ndarray]
            self._polygons = [np.asarray(x).reshape(-1) for x in m]
            return

        if isinstance(m, np.ndarray):  # assumed to be a binary mask
            assert m.shape[1] != 2, m.shape
            assert m.shape == (
                height,
                width,
            ), f"mask shape: {m.shape}, target dims: {height}, {width}"
            self._mask = m.astype("uint8")
            return

        raise ValueError("GenericMask cannot handle object {} of type '{}'".format(m, type(m)))

    @property
    def mask(self):
        if self._mask is None:
            self._mask = self.polygons_to_mask(self._polygons)
        return self._mask

    @property
    def polygons(self):
        if self._polygons is None:
            self._polygons, self._has_holes = self.mask_to_polygons(self._mask)
        return self._polygons

    @property
    def has_holes(self):
        if self._has_holes is None:
            if self._mask is not None:
                self._polygons, self._has_holes = self.mask_to_polygons(self._mask)
            else:
                self._has_holes = False  # if original format is polygon, does not have holes
        return self._has_holes

    def mask_to_polygons(self, mask):
        # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
        # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
        # Internal contours (holes) are placed in hierarchy-2.
        # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
        mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
        res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        hierarchy = res[-1]
        if hierarchy is None:  # empty mask
            return [], False
        has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
        res = res[-2]
        res = [x.flatten() for x in res]
        # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
        # We add 0.5 to turn them into real-value coordinate space. A better solution
        # would be to first +0.5 and then dilate the returned polygon by 0.5.
        res = [x + 0.5 for x in res if len(x) >= 6]
        return res, has_holes

    def polygons_to_mask(self, polygons):
        rle = mask_util.frPyObjects(polygons, self.height, self.width)
        rle = mask_util.merge(rle)
        return mask_util.decode(rle)[:, :]

    def area(self):
        return self.mask.sum()

    def bbox(self):

        p = mask_util.frPyObjects(self.polygons, self.height, self.width)
        p = mask_util.merge(p)
        bbox = mask_util.toBbox(p)
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        return bbox


class VisImage:

    def __init__(self, img, scale=1.0):
        """
        Args:
            img (ndarray): an RGB image of shape (H, W, 3) in range [0, 255].
            scale (float): scale the input image
        """
        self.img = img
        self.scale = scale
        self.width, self.height = img.shape[1], img.shape[0]
        self._setup_figure(img)

    def _setup_figure(self, img):
        """
        Args:
            Same as in :meth:`__init__()`.
        Returns:
            fig (matplotlib.pyplot.figure): top level container for all the image plot elements.
            ax (matplotlib.pyplot.Axes): contains figure elements and sets the coordinate system.
        """
        fig = mplfigure.Figure(frameon=False)
        self.dpi = fig.get_dpi()
        # add a small 1e-2 to avoid precision lost due to matplotlib's truncation
        # (https://github.com/matplotlib/matplotlib/issues/15363)
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        self.canvas = FigureCanvasAgg(fig)
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        self.fig = fig
        self.ax = ax
        self.reset_image(img)

    def reset_image(self, img):
        """
        Args:
            img: same as in __init__
        """
        img = img.astype("uint8")
        self.ax.imshow(img, extent=(0, self.width, self.height, 0), interpolation="nearest")

    def save(self, filepath, fig_format=None):
        """
        Args:
            filepath (str): a string that contains the absolute path, including the file name, where
                the visualized image will be saved.
        """
        if fig_format is not None:
            self.fig.savefig(filepath, format=fig_format)
        else:
            self.fig.savefig(filepath)

    def get_image(self):
        """
        Returns:
            ndarray:
                the visualized image of shape (H, W, 3) (RGB) in uint8 type.
                The shape is scaled w.r.t the input image using the given `scale` argument.
        """
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()
        # buf = io.BytesIO()  # works for cairo backend
        # canvas.print_rgba(buf)
        # width, height = self.width, self.height
        # s = buf.getvalue()

        buffer = np.frombuffer(s, dtype="uint8")

        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype("uint8")


class Visualizer:
    """
    Visualizer that draws data about detection/segmentation on images.
    It contains methods like `draw_{text,box,circle,line,binary_mask,polygon}`
    that draw primitive objects to images, as well as high-level wrappers like
    `draw_{instance_predictions,sem_seg,panoptic_seg_predictions,dataset_dict}`
    that draw composite data in some pre-defined style.
    Note that the exact visualization style for the high-level wrappers are subject to change.
    Style such as color, opacity, label contents, visibility of labels, or even the visibility
    of objects themselves (e.g. when the object is too small) may change according
    to different heuristics, as long as the results still look visually reasonable.
    To obtain a consistent style, you can implement custom drawing functions with the
    abovementioned primitive methods instead. If you need more customized visualization
    styles, you can process the data yourself following their format documented in
    tutorials (:doc:`/tutorials/models`, :doc:`/tutorials/datasets`). This class does not
    intend to satisfy everyone's preference on drawing styles.
    This visualizer focuses on high rendering quality rather than performance. It is not
    designed to be used for real-time applications.
    """

    def __init__(self, img_rgb, scale=1.0, instance_mode=ColorMode.IMAGE):
        """
        Args:
            img_rgb: a numpy array of shape (H, W, C), where H and W correspond to
                the height and width of the image respectively. C is the number of
                color channels. The image is required to be in RGB format since that
                is a requirement of the Matplotlib library. The image is also expected
                to be in the range [0, 255].
            instance_mode (ColorMode): defines one of the pre-defined style for drawing
                instances on an image.
        """
        self.img = np.asarray(img_rgb).clip(0, 255).astype(np.uint8)
        self.output = VisImage(self.img, scale=scale)
        self.cpu_device = torch.device("cpu")

        # too small texts are useless, therefore clamp to 9
        self._default_font_size = max(np.sqrt(self.output.height * self.output.width) // 90, 10 // scale)
        self._default_font_size = 18
        self._instance_mode = instance_mode

        import matplotlib.colors as mcolors
        css4_colors = mcolors.CSS4_COLORS
        self.color_proposals = [list(mcolors.hex2color(color)) for color in css4_colors.values()]

    def draw_text(
        self,
        text,
        position,
        *,
        font_size=None,
        color="g",
        horizontal_alignment="center",
        rotation=0,
    ):
        """
        Args:
            text (str): class label
            position (tuple): a tuple of the x and y coordinates to place text on image.
            font_size (int, optional): font of the text. If not provided, a font size
                proportional to the image width is calculated and used.
            color: color of the text. Refer to `matplotlib.colors` for full list
                of formats that are accepted.
            horizontal_alignment (str): see `matplotlib.text.Text`
            rotation: rotation angle in degrees CCW
        Returns:
            output (VisImage): image object with text drawn.
        """
        if not font_size:
            font_size = self._default_font_size

        # since the text background is dark, we don't want the text to be dark
        color = np.maximum(list(mplc.to_rgb(color)), 0.15)
        color[np.argmax(color)] = max(0.8, np.max(color))

        def contrasting_color(rgb):
            """Returns 'white' or 'black' depending on which color contrasts more with the given RGB value."""

            # Decompose the RGB tuple
            R, G, B = rgb

            # Calculate the Y value
            Y = 0.299 * R + 0.587 * G + 0.114 * B

            # If Y value is greater than 128, it's closer to white so return black. Otherwise, return white.
            return 'black' if Y > 128 else 'white'

        bbox_background = contrasting_color(color * 255)

        x, y = position
        self.output.ax.text(
            x,
            y,
            text,
            size=font_size * self.output.scale,
            family="sans-serif",
            bbox={
                "facecolor": bbox_background,
                "alpha": 0.8,
                "pad": 0.7,
                "edgecolor": "none"
            },
            verticalalignment="top",
            horizontalalignment=horizontal_alignment,
            color=color,
            zorder=10,
            rotation=rotation,
        )
        return self.output

    def draw_box(self, box_coord, alpha=0.5, edge_color="g", line_style="-"):
        """
        Args:
            box_coord (tuple): a tuple containing x0, y0, x1, y1 coordinates, where x0 and y0
                are the coordinates of the image's top left corner. x1 and y1 are the
                coordinates of the image's bottom right corner.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
            edge_color: color of the outline of the box. Refer to `matplotlib.colors`
                for full list of formats that are accepted.
            line_style (string): the string to use to create the outline of the boxes.
        Returns:
            output (VisImage): image object with box drawn.
        """
        x0, y0, x1, y1 = box_coord
        width = x1 - x0
        height = y1 - y0

        linewidth = max(self._default_font_size / 12, 1)

        self.output.ax.add_patch(
            mpl.patches.Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor=edge_color,
                linewidth=linewidth * self.output.scale,
                alpha=alpha,
                linestyle=line_style,
            ))
        return self.output

    def draw_rotated_box_with_label(self, rotated_box, alpha=0.5, edge_color="g", line_style="-", label=None):
        """
        Draw a rotated box with label on its top-left corner.
        Args:
            rotated_box (tuple): a tuple containing (cnt_x, cnt_y, w, h, angle),
                where cnt_x and cnt_y are the center coordinates of the box.
                w and h are the width and height of the box. angle represents how
                many degrees the box is rotated CCW with regard to the 0-degree box.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
            edge_color: color of the outline of the box. Refer to `matplotlib.colors`
                for full list of formats that are accepted.
            line_style (string): the string to use to create the outline of the boxes.
            label (string): label for rotated box. It will not be rendered when set to None.
        Returns:
            output (VisImage): image object with box drawn.
        """
        cnt_x, cnt_y, w, h, angle = rotated_box
        area = w * h
        # use thinner lines when the box is small
        linewidth = self._default_font_size / (6 if area < _SMALL_OBJECT_AREA_THRESH * self.output.scale else 3)

        theta = angle * math.pi / 180.0
        c = math.cos(theta)
        s = math.sin(theta)
        rect = [(-w / 2, h / 2), (-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2)]
        # x: left->right ; y: top->down
        rotated_rect = [(s * yy + c * xx + cnt_x, c * yy - s * xx + cnt_y) for (xx, yy) in rect]
        for k in range(4):
            j = (k + 1) % 4
            self.draw_line(
                [rotated_rect[k][0], rotated_rect[j][0]],
                [rotated_rect[k][1], rotated_rect[j][1]],
                color=edge_color,
                linestyle="--" if k == 1 else line_style,
                linewidth=linewidth,
            )

        if label is not None:
            text_pos = rotated_rect[1]  # topleft corner

            height_ratio = h / np.sqrt(self.output.height * self.output.width)
            label_color = self._change_color_brightness(edge_color, brightness_factor=0.7)
            font_size = (np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2) * 0.5 * self._default_font_size)
            self.draw_text(label, text_pos, color=label_color, font_size=font_size, rotation=angle)

        return self.output

    def draw_circle(self, circle_coord, color, radius=3):
        """
        Args:
            circle_coord (list(int) or tuple(int)): contains the x and y coordinates
                of the center of the circle.
            color: color of the polygon. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            radius (int): radius of the circle.
        Returns:
            output (VisImage): image object with box drawn.
        """
        x, y = circle_coord
        self.output.ax.add_patch(mpl.patches.Circle(circle_coord, radius=radius, fill=True, color=color))
        return self.output

    def draw_line(self, x_data, y_data, color, linestyle="-", linewidth=None):
        """
        Args:
            x_data (list[int]): a list containing x values of all the points being drawn.
                Length of list should match the length of y_data.
            y_data (list[int]): a list containing y values of all the points being drawn.
                Length of list should match the length of x_data.
            color: color of the line. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            linestyle: style of the line. Refer to `matplotlib.lines.Line2D`
                for a full list of formats that are accepted.
            linewidth (float or None): width of the line. When it's None,
                a default value will be computed and used.
        Returns:
            output (VisImage): image object with line drawn.
        """
        if linewidth is None:
            linewidth = self._default_font_size / 3
        linewidth = max(linewidth, 1)
        self.output.ax.add_line(
            mpl.lines.Line2D(
                x_data,
                y_data,
                linewidth=linewidth * self.output.scale,
                color=color,
                linestyle=linestyle,
            ))
        return self.output

    def draw_binary_mask(self, binary_mask, color=None, *, edge_color=None, text=None, alpha=0.7, area_threshold=10):
        """
        Args:
            binary_mask (ndarray): numpy array of shape (H, W), where H is the image height and
                W is the image width. Each value in the array is either a 0 or 1 value of uint8
                type.
            color: color of the mask. Refer to `matplotlib.colors` for a full list of
                formats that are accepted. If None, will pick a random color.
            edge_color: color of the polygon edges. Refer to `matplotlib.colors` for a
                full list of formats that are accepted.
            text (str): if None, will be drawn on the object
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
            area_threshold (float): a connected component smaller than this area will not be shown.
        Returns:
            output (VisImage): image object with mask drawn.
        """
        if color is None:
            color = random_color(rgb=True, maximum=1)
        color = mplc.to_rgb(color)

        has_valid_segment = False
        binary_mask = binary_mask.astype("uint8")  # opencv needs uint8
        mask = GenericMask(binary_mask, self.output.height, self.output.width)
        shape2d = (binary_mask.shape[0], binary_mask.shape[1])

        if not mask.has_holes:
            # draw polygons for regular masks
            for segment in mask.polygons:
                area = mask_util.area(mask_util.frPyObjects([segment], shape2d[0], shape2d[1]))
                if area < (area_threshold or 0):
                    continue
                has_valid_segment = True
                segment = segment.reshape(-1, 2)
                self.draw_polygon(segment, color=color, edge_color=edge_color, alpha=alpha)
        else:
            # Use Path/PathPatch to draw vector graphics:
            # https://stackoverflow.com/questions/8919719/how-to-plot-a-complex-polygon
            # rgba = np.zeros(shape2d + (4,), dtype="float32")
            # rgba[:, :, :3] = color
            # rgba[:, :, 3] = (mask.mask == 1).astype("float32") * alpha
            # has_valid_segment = True
            # self.output.ax.imshow(rgba, extent=(0, self.output.width, self.output.height, 0))
            print('has hole')
            for segment in mask.polygons:
                area = mask_util.area(mask_util.frPyObjects([segment], shape2d[0], shape2d[1]))
                if area < (area_threshold or 0):
                    continue
                has_valid_segment = True
                segment = segment.reshape(-1, 2)
                self.draw_polygon(segment, color=color, edge_color=edge_color, alpha=alpha)

        if text is not None and has_valid_segment:
            lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
            self._draw_text_in_mask(binary_mask, text, lighter_color)
        return self.output

    def _draw_number_in_mask(self, binary_mask, text, color, label_mode='1'):
        """
        Find proper places to draw text given a binary mask.
        """

        def number_to_string(n):
            chars = []
            while n:
                n, remainder = divmod(n - 1, 26)
                chars.append(chr(97 + remainder))
            return ''.join(reversed(chars))

        binary_mask = np.pad(binary_mask, ((1, 1), (1, 1)), 'constant')
        mask_dt = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 0)
        mask_dt = mask_dt[1:-1, 1:-1]
        max_dist = np.max(mask_dt)
        coords_y, coords_x = np.where(mask_dt == max_dist)  # coords is [y, x]

        if label_mode == 'a':
            text = number_to_string(int(text))
        else:
            text = text

        self.draw_text(text, (coords_x[len(coords_x) // 2] + 2, coords_y[len(coords_y) // 2] - 6), color=color)

    def draw_binary_mask_with_number(self,
                                     binary_mask,
                                     color=None,
                                     *,
                                     edge_color=None,
                                     text=None,
                                     label_mode='1',
                                     alpha=0.1,
                                     anno_mode=['Mask'],
                                     area_threshold=10):
        """
        Args:
            binary_mask (ndarray): numpy array of shape (H, W), where H is the image height and
                W is the image width. Each value in the array is either a 0 or 1 value of uint8
                type.
            color: color of the mask. Refer to `matplotlib.colors` for a full list of
                formats that are accepted. If None, will pick a random color.
            edge_color: color of the polygon edges. Refer to `matplotlib.colors` for a
                full list of formats that are accepted.
            text (str): if None, will be drawn on the object
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
            area_threshold (float): a connected component smaller than this area will not be shown.
        Returns:
            output (VisImage): image object with mask drawn.
        """
        if color is None:
            randint = random.randint(0, len(self.color_proposals) - 1)
            color = self.color_proposals[randint]
        color = mplc.to_rgb(color)

        has_valid_segment = True
        binary_mask = binary_mask.astype("uint8")  # opencv needs uint8
        mask = GenericMask(binary_mask, self.output.height, self.output.width)
        shape2d = (binary_mask.shape[0], binary_mask.shape[1])

        if 'Mask' in anno_mode:
            if not mask.has_holes:
                # draw polygons for regular masks
                for segment in mask.polygons:
                    area = mask_util.area(mask_util.frPyObjects([segment], shape2d[0], shape2d[1]))
                    if area < (area_threshold or 0):
                        continue
                    has_valid_segment = True
                    segment = segment.reshape(-1, 2)
                    self.draw_polygon(segment, color=color, edge_color=edge_color, alpha=alpha)
            else:
                # Use Path/PathPatch to draw vector graphics:
                # https://stackoverflow.com/questions/8919719/how-to-plot-a-complex-polygon
                for segment in mask.polygons:
                    area = mask_util.area(mask_util.frPyObjects([segment], shape2d[0], shape2d[1]))
                    if area < (area_threshold or 0):
                        continue
                    has_valid_segment = True
                    segment = segment.reshape(-1, 2)
                    self.draw_polygon(segment, color=color, edge_color=edge_color, alpha=alpha)
            # rgba = np.zeros(shape2d + (4,), dtype="float32")
            # rgba[:, :, :3] = color
            # rgba[:, :, 3] = (mask.mask == 1).astype("float32") * alpha
            # self.output.ax.imshow(rgba, extent=(0, self.output.width, self.output.height, 0))

        if 'Box' in anno_mode:
            bbox = mask.bbox()
            self.draw_box(bbox, edge_color=color, alpha=0.75)

        if 'Mark' in anno_mode:
            has_valid_segment = True
        else:
            has_valid_segment = False

        if text is not None and has_valid_segment:
            # lighter_color = tuple([x*0.2 for x in color])
            lighter_color = [1, 1, 1]  # self._change_color_brightness(color, brightness_factor=0.7)
            self._draw_number_in_mask(binary_mask, text, lighter_color, label_mode)
        return self.output

    def draw_polygon(self, segment, color, edge_color=None, alpha=0.5):
        """
        Args:
            segment: numpy array of shape Nx2, containing all the points in the polygon.
            color: color of the polygon. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            edge_color: color of the polygon edges. Refer to `matplotlib.colors` for a
                full list of formats that are accepted. If not provided, a darker shade
                of the polygon color will be used instead.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
        Returns:
            output (VisImage): image object with polygon drawn.
        """
        if edge_color is None:
            # make edge color darker than the polygon color
            if alpha > 0.8:
                edge_color = self._change_color_brightness(color, brightness_factor=-0.7)
            else:
                edge_color = color
        edge_color = mplc.to_rgb(edge_color) + (1, )

        polygon = mpl.patches.Polygon(
            segment,
            fill=True,
            facecolor=mplc.to_rgb(color) + (alpha, ),
            edgecolor=edge_color,
            linewidth=1,  # max(self._default_font_size // 5 * self.output.scale, 1),
        )
        self.output.ax.add_patch(polygon)
        return self.output

    """
    Internal methods:
    """

    def _jitter(self, color):
        """
        Randomly modifies given color to produce a slightly different color than the color given.
        Args:
            color (tuple[double]): a tuple of 3 elements, containing the RGB values of the color
                picked. The values in the list are in the [0.0, 1.0] range.
        Returns:
            jittered_color (tuple[double]): a tuple of 3 elements, containing the RGB values of the
                color after being jittered. The values in the list are in the [0.0, 1.0] range.
        """
        color = mplc.to_rgb(color)
        # np.random.seed(0)
        vec = np.random.rand(3)
        # better to do it in another color space
        vec = vec / np.linalg.norm(vec) * 0.5
        res = np.clip(vec + color, 0, 1)
        return tuple(res)

    def _create_grayscale_image(self, mask=None):
        """
        Create a grayscale version of the original image.
        The colors in masked area, if given, will be kept.
        """
        img_bw = self.img.astype("f4").mean(axis=2)
        img_bw = np.stack([img_bw] * 3, axis=2)
        if mask is not None:
            img_bw[mask] = self.img[mask]
        return img_bw

    def _change_color_brightness(self, color, brightness_factor):
        """
        Depending on the brightness_factor, gives a lighter or darker color i.e. a color with
        less or more saturation than the original color.
        Args:
            color: color of the polygon. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            brightness_factor (float): a value in [-1.0, 1.0] range. A lightness factor of
                0 will correspond to no change, a factor in [-1.0, 0) range will result in
                a darker color and a factor in (0, 1.0] range will result in a lighter color.
        Returns:
            modified_color (tuple[double]): a tuple containing the RGB values of the
                modified color. Each value in the tuple is in the [0.0, 1.0] range.
        """
        assert brightness_factor >= -1.0 and brightness_factor <= 1.0
        color = mplc.to_rgb(color)
        polygon_color = colorsys.rgb_to_hls(*mplc.to_rgb(color))
        modified_lightness = polygon_color[1] + (brightness_factor * polygon_color[1])
        modified_lightness = 0.0 if modified_lightness < 0.0 else modified_lightness
        modified_lightness = 1.0 if modified_lightness > 1.0 else modified_lightness
        modified_color = colorsys.hls_to_rgb(polygon_color[0], modified_lightness, polygon_color[2])
        return modified_color

    def _draw_text_in_mask(self, binary_mask, text, color):
        """
        Find proper places to draw text given a binary mask.
        """
        # sometimes drawn on wrong objects. the heuristics here can improve.
        _num_cc, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, 8)
        if stats[1:, -1].size == 0:
            return
        largest_component_id = np.argmax(stats[1:, -1]) + 1

        # draw text on the largest component, as well as other very large components.
        for cid in range(1, _num_cc):
            if cid == largest_component_id or stats[cid, -1] > _LARGE_MASK_AREA_THRESH:
                # median is more stable than centroid
                # center = centroids[largest_component_id]
                center = np.median((cc_labels == cid).nonzero(), axis=1)[::-1]
                bottom = np.max((cc_labels == cid).nonzero(), axis=1)[::-1]
                center[1] = bottom[1] + 2
                self.draw_text(text, center, color=color)

    def get_output(self):
        """
        Returns:
            output (VisImage): the image output containing the visualizations added
            to the image.
        """
        return self.output


def draw_mask(frames, masks, colors=None):
    if colors is None:
        colors = [random_color(rgb=True, maximum=1) for _ in range(len(masks))]

    imgs = []
    for i in range(frames.size(0)):
        vis = Visualizer(frames[i].numpy())

        for j in range(len(masks)):
            fig = vis.draw_binary_mask_with_number(masks[j][0, i].bool().numpy(), color=colors[j], alpha=0.3)

        buffer = io.BytesIO()
        fig.save(buffer)
        buffer.seek(0)
        img = iio.imread(buffer)
        imgs.append(img)

    return imgs