import io
from typing import Optional, Union
from IPython.core.display_functions import display

from scipy.interpolate import griddata
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch


def untransform_img(img: torch.Tensor) -> Image.Image:
    if len(img.shape) == 4:
        img = img[0]

    img_np = img.permute((1, 2, 0)).detach().float().cpu().numpy()
    img_np = ((img_np + 1) / 2) * 255  # Un-normalize from [-1, 1] to [0, 255]
    img_pil = Image.fromarray(img_np.astype(np.uint8))
    return img_pil


def untrasform_mask(mask: torch.Tensor) -> Image.Image:
    if len(mask.shape) == 4:
        mask = mask[0, 0]
    mask_np = mask.detach().float().cpu().numpy()
    mask_pil = Image.fromarray(mask_np.astype(np.uint8))
    return mask_pil


def to_HWC(img):
    if isinstance(img, np.ndarray) and img.shape[0] in [1, 3]:
        return img.transpose((1, 2, 0))
    elif isinstance(img, torch.Tensor) and img.shape[0] in [1, 3]:
        return img.permute((1, 2, 0)).detach().cpu().numpy()
    elif isinstance(img, torch.Tensor) and len(img.shape) == 2:
        return img.detach().cpu().numpy()
    elif isinstance(img, Image.Image):
        return np.asarray(img)
    else:
        return img


def imshow(
    img: Union[torch.Tensor, np.ndarray, Image.Image],
    title: Optional[str] = None,
    colorbar=False,
    cmap: Optional[str] = None,
    alpha: float = 1.0,
    show: float = False,
):
    # assert len(img.shape) < 4, "Please provide a max of 3D input"
    img = to_HWC(img)
    plt.imshow(img, cmap=cmap, alpha=alpha)
    plt.axis("off")
    if colorbar:
        plt.colorbar()
    if title is not None:
        plt.title(title)
    if show:
        plt.show()


def imshowp(img):
    if isinstance(img, np.ndarray):
        display(Image.fromarray(img.astype(np.uint8)))
    else:
        display(img)


def plot_correspondences(
    img1: Image.Image,
    img2: Image.Image,
    source_points_s: np.ndarray,
    target_points_s: np.ndarray,
    matching_scores_s: np.ndarray,
    source_points_v: np.ndarray,
    target_points_v: np.ndarray,
    matching_scores_v: np.ndarray,
    score_threshold: float = 0.6,
):
    to_plot = [
        ("input", torch.zeros(1, 2), torch.zeros(1, 2), torch.ones(1)),
        ("semantic", source_points_s, target_points_s, matching_scores_s),
        ("visual", source_points_v, target_points_v, matching_scores_v),
    ]
    for title, points_1, points_2, scores in to_plot:
        img1_p = img1.resize((768, 768))
        img2_p = img2.resize((768, 768))

        title = title

        # Generate unique colors
        num_points = len(points_1)
        colors = plt.cm.get_cmap("hsv", num_points)

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Source image with points
        axes[0].imshow(img1_p)
        for pt_idx in range(num_points):
            if scores[pt_idx] > score_threshold:
                axes[0].scatter(points_1[pt_idx, 0], points_1[pt_idx, 1], color=colors(pt_idx), s=40)
        axes[0].set_title("Source Image")
        axes[0].axis("off")

        # # Target image with points
        axes[1].imshow(img2_p)
        for pt_idx in range(num_points):
            if scores[pt_idx] > score_threshold:
                axes[1].scatter(points_2[pt_idx, 0], points_2[pt_idx, 1], color=colors(pt_idx), s=40)
        axes[1].set_title("Target Image")
        axes[1].axis("off")

        plt.tight_layout()
        # fig.suptitle(title, fontsize=16)
        plt.show()


def plot_consistency_scores(img_rgb: Image.Image, mask_pil: Image.Image, points, scores, pad=2):
    """
    Returns a PIL.Image with the consistency overlay.
    """
    # ------------------------------------------------------------------
    # 1. build sparse score map
    # ------------------------------------------------------------------
    H, W = 768, 768
    img_np = np.zeros((H, W), dtype=np.float32)

    pts = np.asarray(points, dtype=np.int32)
    for score, (x, y) in zip(scores, pts):
        y0 = max(y - pad, 0)
        y1 = min(y + pad, H)
        x0 = max(x - pad, 0)
        x1 = min(x + pad, W)
        img_np[y0:y1, x0:x1] = score

    # ------------------------------------------------------------------
    # 2. interpolate onto a full grid
    # ------------------------------------------------------------------
    y_idx, x_idx = np.nonzero(img_np)
    if len(y_idx) == 0:  # safety
        raise ValueError("No non-zero scores to interpolate.")
    interp = griddata(
        points=np.column_stack([x_idx, y_idx]),
        values=img_np[y_idx, x_idx],
        xi=(np.arange(W)[None, :].repeat(H, axis=0), np.arange(H)[:, None].repeat(W, axis=1)),  # grid_x  # grid_y
        method="linear",
        fill_value=0,
    )

    # ------------------------------------------------------------------
    # 3. prepare mask
    # ------------------------------------------------------------------
    mask = np.asarray(mask_pil.resize((W, H)), dtype=np.float32)
    overlay = interp * mask

    # ------------------------------------------------------------------
    # 4. draw the figure
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6), dpi=128)
    ax.set_position([0, 0, 1, 1])  # fill the whole canvas
    ax.imshow(img_rgb, alpha=0.5)
    ax.imshow(overlay, cmap="turbo", alpha=0.6)
    ax.axis("off")
    # ------------------------------------------------------------------
    # 5. grab as PNG into memory
    # ------------------------------------------------------------------
    buf = io.BytesIO()
    fig.savefig(buf, format="png", pad_inches=0)  # <- no bbox_inches='tight'

    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


# def plot_benchmark_grid(imgs_to_plot, consistency_maps, metrics_to_plot, METHODS):

#     # --- FIGURE SETUP ---
#     fig = plt.figure(figsize=(20, 10))
#     gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.3], wspace=0.1, hspace=0.1)

#     titles = ['Subject'] + METHODS

#     # Top row
#     for col in range(4):
#         ax = fig.add_subplot(gs[0, col])
#         ax.axis('off')
#         ax.imshow(imgs_to_plot[titles[col]])
#         ax.set_title(titles[col], pad=12, fontsize=14)

#     # Bottom row
#     for col in range(4):
#         ax = fig.add_subplot(gs[1, col])
#         ax.axis('off')
#         if col == 0:
#             continue
#         # draw the image in the upper 80% of the cell
#         ax.imshow(consistency_maps[titles[col]], extent=(0, 1, 0.2, 1), aspect='auto')
#         # add the caption text underneath
#         met = "\n".join([f"{k}: {v:.3f}" for k,v in metrics_to_plot[titles[col]].items()])
#         ax.text(
#             0.5, 0.0, met,
#             ha='center', va='bottom', wrap=True, fontsize=12
#         )

#      # ------------------------------------------------------------------
#     # 5. grab as PNG into memory
#     # ------------------------------------------------------------------
#     buf = io.BytesIO()
#     fig.savefig(buf, format="png", pad_inches=0)  # <- no bbox_inches='tight'
#     plt.close(fig)
#     buf.seek(0)
#     return Image.open(buf)


def plot_benchmark_grid(
    imgs_to_plot, consistency_maps, metrics_to_plot, METHODS, format="png", title=None, paper_mode=False
):
    """
    Return a PIL.Image containing a benchmark grid.

    If paper_mode=False: 3-row × 4-column grid with:
        Row 0: subject / method images
        Row 1: consistency heat-maps
        Row 2: metric values (text only)

    If paper_mode=True: A grid with a reference image and three method comparisons.
    """
    titles = ["Subject"] + METHODS
    assert len(titles) >= 4, "Need at least three METHODS."

    if paper_mode:
        # ------------- PAPER MODE: Final Correct Layout using Inset Axes -----------------
        fig = plt.figure(figsize=(20, 4), dpi=150)

        # A 1x4 grid for the main images. We'll add metrics/maps as insets.
        gs_main = fig.add_gridspec(1, 4, wspace=0.0, left=0.0, right=1)

        # --- Column 0: Reference Image ---
        ax_ref = fig.add_subplot(gs_main[0, 0])
        ax_ref.imshow(imgs_to_plot[titles[0]])
        ax_ref.set_axis_off()
        ax_ref.set_aspect("equal", adjustable="box")

        # --- Columns 1-3: Method Images with Inset Metrics/Maps ---
        for i in range(3):
            method_name = titles[i + 1]

            # --- Main Prediction Image ---
            ax_img = fig.add_subplot(gs_main[0, i + 1])
            ax_img.imshow(imgs_to_plot[method_name])
            ax_img.set_axis_off()
            ax_img.set_aspect("equal", adjustable="box")

            # --- Inset for Metrics Text ---
            # Positioned to the left of the main image, in the top half.
            ax_text = ax_img.inset_axes((-0.5, 0.5, 0.45, 0.5))
            metrics_str = "\n".join(
                f"{k.split('_')[0]:<6}:{float(v):.3f}"
                for k, v in metrics_to_plot[method_name].items()
                if k in ["VSM_NEW_0.6", "CLIP", "DINO", "VLM", "Oracle"]
            )
            metrics_str = metrics_str.replace("VSM_NEW_0.6", "VSM")
            ax_text.text(
                0.0,
                0.5,
                metrics_str,
                ha="left",
                va="center",
                fontsize=14,
                transform=ax_text.transAxes,
                family="monospace",
            )
            ax_text.set_axis_off()

            # --- Inset for Consistency Map ---
            # Positioned to the left, in the bottom half, ensuring it's square and bottom-aligned.
            ax_map = ax_img.inset_axes((-0.5, 0, 0.45, 0.5))
            ax_map.imshow(consistency_maps[method_name])
            ax_map.set_axis_off()
            ax_map.set_aspect("equal", adjustable="box", anchor="S")  # Anchor to South (bottom)

    else:
        # ------------- 1 ▸ figure & 3-row grid ---------------------------
        fig = plt.figure(figsize=(16, 9), dpi=100)
        if title is not None:
            fig.suptitle(title)
        gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.45], wspace=0.02, hspace=0.05)  # images · maps · text

        # ------------- 2 ▸ row 0 – original images -----------------------
        for col in range(4):
            ax = fig.add_subplot(gs[0, col])
            ax.imshow(imgs_to_plot[titles[col]], extent=(0, 1, 0, 1), aspect="auto")  # fill whole axes
            ax.set_title(titles[col], fontsize=13, pad=6)
            ax.set_axis_off()

        # ------------- 3 ▸ row 1 – heat-maps -----------------------------
        for col in range(4):
            ax = fig.add_subplot(gs[1, col])
            ax.set_axis_off()

            if col == 0:  # keep bottom-left blank
                continue

            ax.imshow(consistency_maps[titles[col]], extent=(0, 1, 0, 1), aspect="auto")  # fill whole axes

        # ------------- 4 ▸ row 2 – metrics text --------------------------
        for col in range(4):
            ax = fig.add_subplot(gs[2, col])
            ax.set_axis_off()

            if col == 0:
                continue

            metrics_str = "\n".join(f"{k}: {v:.3f}" for k, v in metrics_to_plot[titles[col]].items())
            ax.text(0.7, 0.6, metrics_str, ha="right", va="center", fontsize=16, wrap=True, transform=ax.transAxes)

        # ------------- 5 ▸ tighten outer margins -------------------------
        fig.subplots_adjust(left=0, right=1, bottom=0, top=0.94, wspace=0, hspace=0)

    # ------------- 6 ▸ save to PNG in-memory & return ----------------
    buf = io.BytesIO()
    fig.savefig(buf, format=format, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    buf.seek(0)
    return Image.open(buf)


def plot_comparison_grid(imgs_to_plot, consistency_map, metrics_to_plot, format="png"):
    """
    Used for logging images during training to W&B

    Return a PIL.Image containing a 2-row × 3-column comparison grid.

    Row 1: Reference image, Inpainted image, consistency heat-map
    Row 2: metric values (text only)
    """
    # Define column titles
    titles = ["Reference", "Inpainted", "Mask", "Consistency"]

    # ------------- 1 ▸ figure & 2-row grid ---------------------------
    fig = plt.figure(figsize=(16, 8), dpi=100)
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.45], wspace=0.02, hspace=0.05)  # images · text

    # ------------- 2 ▸ row 0 – images and consistency map ------------
    for col in range(4):
        ax = fig.add_subplot(gs[0, col])
        ax.set_axis_off()

        if col < 3:  # Reference and Inpainted images
            ax.imshow(imgs_to_plot[titles[col]], extent=[0, 1, 0, 1], aspect="auto")  # fill whole axes
        else:  # Consistency map
            ax.imshow(consistency_map, extent=[0, 1, 0, 1], aspect="auto")  # fill whole axes

        ax.set_title(titles[col], fontsize=13, pad=6)

    # ------------- 3 ▸ row 1 – metrics text --------------------------
    for col in range(4):
        ax = fig.add_subplot(gs[1, col])
        ax.set_axis_off()

        # if col < 3 and titles[col] in metrics_to_plot:
        #     metrics_str = "\n".join(f"{k}: {v:.3f}" for k, v in metrics_to_plot[titles[col]].items())
        #     ax.text(0.5, 0.6, metrics_str, ha="center", va="center", fontsize=16, wrap=True, transform=ax.transAxes)
        if col == 3:
            metrics_str = "\n".join(f"{k}: {v:.3f}" for k, v in metrics_to_plot.items())
            ax.text(0.5, 0.6, metrics_str, ha="center", va="center", fontsize=16, wrap=True, transform=ax.transAxes)

    # ------------- 4 ▸ tighten outer margins -------------------------
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.94, wspace=0, hspace=0)

    # ------------- 5 ▸ save to PNG in-memory & return ----------------
    buf = io.BytesIO()
    fig.savefig(buf, format=format, pad_inches=0)
    plt.close(fig)

    buf.seek(0)
    return Image.open(buf)
