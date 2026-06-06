from typing import Optional

import einops
import torch
from torchvision.transforms.functional import to_tensor
from PIL.Image import Image
import numpy as np

from mtg.utils.image_proc import downsample_mask
from mtg.utils.visualization import plot_correspondences


def compute_vsm_metric(
    source_points_s,
    target_points_s,
    matching_score_maps_max_s,
    source_points_v,
    target_points_v,
    matching_score_maps_max_v,
    visual_thresholds=[0.5, 0.6, 0.7],
    semantic_threshold=0.7,
    visualize_matches=False,
    img1: Optional[Image] = None,
    img2: Optional[Image] = None,
):

    # Make sure that every point in source matches to on point in target (with highest confidence)
    best = {}  # key: (x, y) target point -> (src_idx, score)
    for i, (score, tpt) in enumerate(zip(matching_score_maps_max_s, target_points_s)):
        if score <= semantic_threshold:
            continue
        key = (int(tpt[0]), int(tpt[1]))
        if key not in best or score > best[key][1]:
            best[key] = (i, score)

    conf_idx = [i for i, _ in best.values()]

    # if isinstance(matching_score_maps_max_v, torch.Tensor):
    #     matching_score_maps_max_v = matching_score_maps_max_v.float().detach().cpu().numpy()

    conf_scores_v = torch.tensor([matching_score_maps_max_v[i] for i in conf_idx])

    vsm_metric = {
        f"vsm_{threshold}": (torch.sum(conf_scores_v > threshold) / conf_scores_v.shape[0]).item()
        for threshold in visual_thresholds
    }

    correspondence_img = None
    if visualize_matches and img1 and img2:
        # Semantic Correspondences
        conf_source_pts_s = np.array([source_points_s[i] for i in conf_idx])
        conf_target_pts_s = np.array([target_points_s[i] for i in conf_idx])
        conf_scores_s = np.array([matching_score_maps_max_s[i] for i in conf_idx])

        # Visual Correspondences
        conf_source_pts_v = np.array([source_points_v[i] for i in conf_idx])
        conf_target_pts_v = np.array([target_points_v[i] for i in conf_idx])
        correspondence_img = plot_correspondences(
            img1,
            img2,
            conf_source_pts_s,
            conf_target_pts_s,
            conf_scores_s,
            conf_source_pts_v,
            conf_target_pts_v,
            conf_scores_v,
            score_threshold=0.7,
        )

    return vsm_metric, correspondence_img


def reshape_feats(feats, normalize=True):
    # Normalize the features
    if normalize:
        feats = feats / torch.norm(feats, dim=1, keepdim=True)
    return einops.rearrange(feats, "1 c h w -> (h w) c").float().detach().cpu().numpy()


def extract_features(
    img_source: Image, img_target: Image, cleandift_model, item_desc, img_size=(768, 768), feat_key="us6"
):
    """Extract features from two image using CleanDIFT model"""

    img_source_tensor = to_tensor(img_source.resize(img_size))[None].to("cuda") * 2 - 1
    img_target_tensor = to_tensor(img_target.resize(img_size))[None].to("cuda") * 2 - 1

    # GET FEATURES FOR BOTH IMAGES
    feat_key = "us6"
    # print(f"Using feature {feat_key}")

    with torch.no_grad():
        source_features = cleandift_model.get_features(
            img_source_tensor.bfloat16(), [item_desc], t=None, feat_key=feat_key
        )  # [1,C,H,W]
        # print("Got features for source image")
        target_features = cleandift_model.get_features(
            img_target_tensor.bfloat16(), [item_desc], t=None, feat_key=feat_key
        )  # [1,C,H,W]
        # print("Got features for target image")

    return source_features, target_features


def create_coordinate_tensor(h, w):
    # Create a meshgrid for x and y coordinates
    y_coords, x_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    # Stack x and y coordinates to create a tensor
    coordinate_tensor = torch.stack((x_coords, y_coords))
    return coordinate_tensor.reshape(-1, 2)


def match_features_masked(
    source_features: torch.Tensor, target_features: torch.Tensor, source_mask: torch.Tensor, target_mask: torch.Tensor
):  # TODO: support batch size > 1
    """Supports a batch size of 1 for now

    Args:
        source_features (torch.Tensor): B x C x H x W
        target_features (torch.Tensor): B x C x H x W
        source_mask (torch.Tensor): B x 1 x H x W
        target_mask (torch.Tensor): B x 1 x H x W

    Returns:
        _type_: _description_
    """
    image_resolution = source_mask.shape[-1]
    source_mask = downsample_mask(source_mask, source_features.shape[-2:])
    target_mask = downsample_mask(target_mask, target_features.shape[-2:])

    # source_points = create_coordinate_tensor(*source_features.shape[-2:])  # Dense Correspondence
    source_points = torch.argwhere(source_mask[0, 0] != 0).flip(-1).to(source_features.device).long()  # Only on object

    # Normalize over the feature dimensions then flatten H,W
    target_mask_flat_t = target_mask[:1].reshape(1, -1).to(target_features.device)
    target_features_upsampled_norm_flat = einops.rearrange(
        target_features / target_features.norm(p=2, dim=1, keepdim=True), "1 c h w -> c (h w)"
    )  # [num_feats, (H*W)]
    target_features_upsampled_norm_flat = target_features_upsampled_norm_flat * target_mask_flat_t

    source_point_feats = source_features[0, :, source_points[:, 1], source_points[:, 0]].T[
        :, None
    ]  # [num_points, 1, num_feats]
    source_point_feats_norm = source_point_feats / source_point_feats.norm(
        p=2, dim=-1, keepdim=True
    )  # [num_points, 1, num_feats]

    matching_score_maps = einops.rearrange(
        source_point_feats_norm @ target_features_upsampled_norm_flat, "b 1 (h w) -> b h w", h=target_features.shape[-2]
    )
    # [num_points, 1, num_feats] @ [num_feats, (H*W)] => [num_point, (H*W)] => [num_points, H, W]
    matching_score_maps_max, matching_score_maps_max_idx = torch.max(
        einops.rearrange(matching_score_maps, "b h w -> b (h w)"), -1
    )
    target_points_matched = torch.stack(
        torch.unravel_index(matching_score_maps_max_idx, matching_score_maps.shape[1:])[::-1]
    ).T

    ### Scale the points to the original image resoution
    scale_factor = image_resolution / target_features.shape[-1]
    source_points_hr = source_points * scale_factor
    target_points_hr = target_points_matched * scale_factor
    return (
        source_points_hr,
        target_points_hr,
        matching_score_maps,
        matching_score_maps_max,
        source_points.cpu().numpy(),
        target_points_matched.cpu().numpy(),
    )
    # Source points in image resolution, Matched points in target image, score maps of the matches, max similarity of the matches
