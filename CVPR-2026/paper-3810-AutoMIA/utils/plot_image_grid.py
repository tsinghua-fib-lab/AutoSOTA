"""Image grid plotting utility"""

import matplotlib.pyplot as plt
import numpy as np
import torch


def image_grid(
    images,
    rows=None,
    cols=None,
    fill: bool = True,
    show_axes: bool = False,
    rgb: bool = True,
):
    """将图像排列成网格

    Args:
        images: 图像列表或张量
        rows: 行数，默认为None（自动计算）
        cols: 列数，默认为None（自动计算）
        fill: 是否填充网格，默认为True
        show_axes: 是否显示坐标轴，默认为False
        rgb: 是否为RGB图像，默认为True

    Returns:
        网格化的图像
    """
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()

    if rows is None != cols is None:
        raise ValueError("行数和列数必须都指定或都不指定")

    if rows is None:
        rows = int(np.sqrt(len(images)))
        cols = (len(images) + rows - 1) // rows

    if fill:
        n_miss = rows * cols - len(images)
        if n_miss > 0:
            if rgb:
                images = np.append(images, np.zeros((n_miss, *images.shape[1:])), axis=0)
            else:
                images = np.append(images, np.zeros((n_miss, *images.shape[1:])), axis=0)

    fig, axarr = plt.subplots(rows, cols, figsize=(15, 15 * rows / cols))
    bleed = 0
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    for ax, im in zip(axarr.flatten(), images):
        if rgb:
            # RGB图像
            ax.imshow(im)
        else:
            # 灰度图像
            ax.imshow(im, cmap="gray")
        if not show_axes:
            ax.set_axis_off()

    return fig 