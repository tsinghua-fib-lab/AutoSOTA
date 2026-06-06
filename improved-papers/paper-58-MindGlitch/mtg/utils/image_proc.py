from git import Optional
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, binary_erosion
import einops
import numpy as np


def get_pix_diff(img1, img2, mask):
    img1_np = np.array(img1)
    img2_np = np.array(img2)
    return np.abs(img1_np[mask] - img2_np[mask]).mean()


def get_mask_width_height(mask):
    if isinstance(mask, Image.Image):
        mask = np.asarray(mask)
    coords = np.column_stack(np.where(mask > 0))
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    height = y_max - y_min
    width = x_max - x_min
    return width, height


def compuate_relative_aspect_ratio(obj_mask, part_mask):
    obj_w, obj_h = get_mask_width_height(obj_mask)
    part_w, part_h = get_mask_width_height(part_mask)
    w_ratio = part_w / obj_w
    h_ratio = part_h / obj_h
    return w_ratio, h_ratio


def compute_aspect_ratio(mask):
    if isinstance(mask, Image.Image):
        mask = np.array(mask)
    # Find the bounding box of the mask
    coords = np.column_stack(np.where(mask > 0))
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    height = y_max - y_min
    width = x_max - x_min
    return width / height


def fit_bb_to_mask(mask, pad=(50, 50)):
    xy = np.argwhere(mask)
    xx = xy[..., 1]
    yy = xy[..., 0]
    x_max = xx.max()
    x_min = xx.min()
    width = x_max - x_min
    pad_x = pad[0]  # * width
    y_max = yy.max()
    y_min = yy.min()
    height = y_max - y_min
    pad_y = pad[1]  # * height
    bb_x1 = x_min - pad_x
    bb_y1 = y_min - pad_y
    bb_x2 = x_max + pad_x
    bb_y2 = y_max + pad_y
    return bb_x1, bb_y1, bb_x2, bb_y2


def split_image_vertically(img, pad=(8, 0)):
    # Get the dimensions of the image
    width, height = img.size

    # Calculate the middle of the image
    middle = width // 2

    # Split the image into two halves
    left_half = img.crop((pad[0], 0, middle, height))
    right_half = img.crop((middle + pad[0], 0, width, height))

    return left_half, right_half


def split_double_image(double_img, sz=512, pad=0):
    """Split images of Subjects200k dataset to get Image 1 and Image 2"""
    double_img_np = np.array(double_img)
    img1_np = double_img_np[:sz, :sz] # left image
    img2_np = double_img_np[:sz, sz + pad :] # right image 
    return Image.fromarray(img1_np), Image.fromarray(img2_np), img1_np, img2_np


def pad_image_to_1024(image, mask):
    W, H = image.size

    target_size = (1024, 1024)

    img_np = np.asarray(image)
    mask_np = np.asarray(mask)

    # Pad the image
    padded_image = np.zeros((*target_size, 3)).astype(np.uint8)
    padded_mask = np.zeros(target_size).astype(bool)

    padded_image[:H, :W, :] = img_np
    padded_mask[:H, :W] = mask_np

    return Image.fromarray(padded_image), Image.fromarray(padded_mask)


def crop_image_to_bb(image, bb):
    image_np = np.array(image)
    cropped_image = image_np[bb[1] : bb[3], bb[0] : bb[2]]
    return Image.fromarray(cropped_image)


def crop_image_to_original(padded_image, original_size):
    H, W = original_size
    padded_image_np = np.array(padded_image)
    unpadded_image = padded_image_np[:W, :H]
    return Image.fromarray(unpadded_image)


def compute_flatness_metrics(score_map, mask=None):
    """
    Compute entropy, variance, max score, and flatness score from a 2D score map,
    optionally within a masked region.

    Args:
        score_map (Tensor): 2D tensor (H, W) of unnormalized scores.
        mask (Tensor, optional): Binary mask (H, W). 1 = valid region, 0 = ignore.

    Returns:
        dict: Dictionary with entropy, variance, max_score, and flatness_score.
    """
    H, W = score_map.shape
    device = score_map.device

    if mask is not None:
        mask = mask.bool()
        if mask.shape != score_map.shape:
            raise ValueError("Mask and score_map must have the same shape.")
        # Mask the score map and flatten
        masked_scores = score_map[mask]
        flat_scores = masked_scores.view(-1)
    else:
        mask = torch.ones_like(score_map, dtype=torch.bool)
        flat_scores = score_map.view(-1)

    # Normalize scores over the masked area
    prob = F.softmax(flat_scores, dim=0)

    # ------------------------
    # Entropy
    # ------------------------
    entropy = -torch.sum(prob * torch.log(prob + 1e-8))
    max_entropy = torch.log(torch.tensor(len(flat_scores), dtype=prob.dtype, device=device))
    norm_entropy = entropy / max_entropy  # ∈ [0,1]

    # ------------------------
    # Coordinates for variance
    # ------------------------
    y_coords, x_coords = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    coords = torch.stack([x_coords, y_coords], dim=-1)  # (H, W, 2)
    masked_coords = coords[mask].float()  # (N, 2)

    # Mean
    mean = torch.sum(prob.unsqueeze(1) * masked_coords, dim=0)  # (2,)
    # Variance
    diffs = masked_coords - mean.unsqueeze(0)  # (N, 2)
    var = torch.sum(prob.unsqueeze(1) * diffs**2)  # scalar

    # ------------------------
    # Max score (raw, not softmax)
    # ------------------------
    max_score = flat_scores.max()

    # ------------------------
    # Flatness score (tunable)
    # ------------------------
    flatness_score = norm_entropy + 0.0005 * var - max_score

    return {
        "entropy": norm_entropy.item(),
        "variance": var.item(),
        "max_score": max_score.item(),
        "flatness_score": flatness_score.item(),
    }


def compute_masked_skewness(data: torch.Tensor, mask: Optional[torch.Tensor] = None):

    data = data.float()
    data_flat = einops.rearrange(data, "h w -> (h w)")

    if mask is not None:
        mask_flat = einops.rearrange(mask, "1 1 h w -> (h w)")
        data_flat = data_flat[mask_flat]

    n = data_flat.numel()
    mean = torch.mean(data_flat)
    std_dev = torch.std(data_flat, unbiased=True)  # ddof=1 is equivalent to unbiased=True

    # Handle the case where standard deviation is zero
    if std_dev == 0:
        skewness = 0.0
    else:
        skewness = (n / ((n - 1) * (n - 2))) * torch.sum(((data_flat - mean) / std_dev) ** 3)

    return {"skewness": skewness.item()}


# Donwsampe masks
def downsample_mask(mask, size) -> torch.Tensor:
    mask_resized = F.interpolate(mask.float(), size=size, mode="nearest").bool().detach().cpu()  # [1,1,H,W]
    return mask_resized


def clean_mask_morph(mask):
    # Convert to NumPy array
    image_array = np.array(mask)
    # Define a structure element (kernel), e.g., a 3x3 square
    structure_element = np.ones((3, 3), dtype=bool)
    image_array = binary_erosion(image_array, structure=structure_element)
    image_array = binary_dilation(image_array, structure=structure_element, iterations=2)

    # morphed_image = (image_array * 255).astype(np.uint8)
    return image_array


def compute_masked_lpips(mask, original_img: np.ndarray, inpainted_img: np.ndarray, lipis_model, DEVICE) -> torch.Tensor:

    mask_3ch = mask[..., None].repeat(3, -1)
    original_img_np_masked = original_img * mask_3ch
    inpainted_img_np_masked = inpainted_img * mask_3ch

    # Step 1: Find the bounding box of the mask
    mask_indices = np.argwhere(mask)
    y_min, x_min = mask_indices.min(axis=0)
    y_max, x_max = mask_indices.max(axis=0)

    # Step 2: Apply the bounding box to crop the images
    cropped_original_img = original_img_np_masked[y_min : y_max + 1, x_min : x_max + 1, :]
    cropped_inpainted_img = inpainted_img_np_masked[y_min : y_max + 1, x_min : x_max + 1, :]

    original_img_t = torch.from_numpy(np.array(cropped_original_img)).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    inpainted_img_t = torch.from_numpy(np.array(cropped_inpainted_img)).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        d = lipis_model.forward(original_img_t, inpainted_img_t)
    return d
