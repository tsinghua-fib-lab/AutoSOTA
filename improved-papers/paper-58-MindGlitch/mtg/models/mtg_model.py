from math import sqrt
from typing import Optional

import torch
import torch.nn as nn
from omegaconf.dictconfig import DictConfig
import torch.nn.functional as F
from wandb import Image
from torchvision import transforms

from mtg.models.detectron2_resnet import ResNet, BottleneckBlock
from mtg.models.mtg_base import MindTheGlitchBase
from mtg.utils.correspondence import match_features_masked, compute_vsm_metric
from mtg.utils.metrics import init_metrics_dict
from mtg.utils.visualization import plot_consistency_scores, plot_comparison_grid, untransform_img, untrasform_mask


def flatten_feats(feats):
    # (b, c, w, h) -> (b, w*h, c)
    b, c, w, h = feats.shape
    feats = feats.view((b, c, -1))
    feats = feats.permute((0, 2, 1))
    return feats


def normalize_feats(feats):
    # (w*h, c)
    feats = feats / torch.linalg.norm(feats, dim=-1)[:, None]
    return feats


class MindTheGlitchModel(MindTheGlitchBase):
    def __init__(self, cleandift_model, config: DictConfig, DEVICE, DTYPE):
        super().__init__(cleandift_model, config, DEVICE, DTYPE)

        ## Aggregation Networks ##
        # This architecture is inspired by Diffusion HyperNetwork, but it can be optimized further
        self.bottleneck_layers_semantic = nn.ModuleList()
        self.bottleneck_layers_visual = nn.ModuleList()

        resnet_config = self.config.model.resnet
        # Semantic aggregation network:
        for l, feature_dim in enumerate(self.feature_dims):
            bottleneck_layer = nn.Sequential(
                *ResNet.make_stage(
                    BottleneckBlock,
                    num_blocks=resnet_config.num_blocks,  # 1
                    in_channels=feature_dim,
                    bottleneck_channels=resnet_config.projection_dim // 4,
                    out_channels=resnet_config.projection_dim,
                    norm="GN",
                    num_norm_groups=resnet_config.num_norm_groups,
                )
            )
            self.bottleneck_layers_semantic.append(bottleneck_layer)

        # Visual aggregation network
        for l, feature_dim in enumerate(self.feature_dims):
            bottleneck_layer = nn.Sequential(
                *ResNet.make_stage(
                    BottleneckBlock,
                    num_blocks=resnet_config.num_blocks,  # 1
                    in_channels=feature_dim,
                    bottleneck_channels=resnet_config.projection_dim // 4,
                    out_channels=resnet_config.projection_dim,
                    norm="GN",
                    num_norm_groups=resnet_config.num_norm_groups,
                )
            )
            self.bottleneck_layers_visual.append(bottleneck_layer)

        # The learnable mixing weights for each aggregation network
        self.mixing_weights_semantic = nn.Parameter(torch.randn(len(self.bottleneck_layers_semantic)))
        self.mixing_weights_visual = nn.Parameter(torch.randn(len(self.bottleneck_layers_visual)))

        self.to(device=DEVICE, dtype=DTYPE)

        if self.config.model.get("load_ckpt", None):
            self.load_state_dict(torch.load(self.config.model.load_ckpt))

    def aggregate_feats(self, img_feats, bottleneck_layers, mixing_weights, normalize=True, flatten=True):
        img_feats_out = torch.empty(0).to(self.device, dtype=self.dtype)
        for i, feat_type in enumerate(self.feat_type):
            bottleneck_layer = bottleneck_layers[i]

            if img_feats_out.numel() == 0:
                img_feats_out = bottleneck_layer(img_feats[feat_type])
            else:
                img_feats_out = img_feats_out + mixing_weights[i] * bottleneck_layer(img_feats[feat_type])

        if normalize:
            img_feats_out = F.normalize(img_feats_out, dim=1)  # B x C x H x W

        if flatten:
            b, c, w, h = img_feats_out.shape
            img_feats_out = img_feats_out.view((b, c, -1))
            img_feats_out = img_feats_out.permute((0, 2, 1))  # B x (H*W) x C
        return img_feats_out

    @staticmethod
    def get_valid_points(img1_points, img2_points):
        # Removes invalid points that were used for padding
        valid_point_idxs = [idx for idx, pt in enumerate(img1_points) if pt[0] != -1]
        valid_points_img1 = img1_points[valid_point_idxs]
        valid_points_img2 = img2_points[valid_point_idxs]
        return valid_points_img1, valid_points_img2

    @staticmethod
    def points_to_idxs(points, out_size):
        points_y = points[:, 0]
        points_x = points[:, 1]
        idx = out_size[1] * torch.round(points_y) + torch.round(points_x)
        return idx

    def compute_similarity(self, img1_feats, img2_feats, bottleneck_layers, mixing_weights):
        mixing_weights_sm = torch.nn.functional.softmax(mixing_weights, dim=0)
        img1_feats_aggr = self.aggregate_feats(img1_feats, bottleneck_layers, mixing_weights_sm)
        img2_feats_aggr = self.aggregate_feats(img2_feats, bottleneck_layers, mixing_weights_sm)
        similarity12 = torch.matmul(img1_feats_aggr, img2_feats_aggr.permute((0, 2, 1))) * self.logit_scale.exp()
        similarity21 = torch.matmul(img2_feats_aggr, img1_feats_aggr.permute((0, 2, 1))) * self.logit_scale.exp()
        return similarity12, similarity21, img1_feats_aggr, img2_feats_aggr

    def compute_loss(
        self,
        sim_semantic12,
        sim_semantic21,
        sim_visual12,
        sim_visual21,
        img1_part_points,
        img2_part_points,
        img1_outside_part_points,
        img2_outside_part_points,
        visual_in_sign=-1,
    ):
        losses = {loss: torch.tensor(0.0).to(self.device, self.dtype) for loss in self.config.losses}

        feat_res = (
            self.config.model.final_feat_resolution,
            self.config.model.final_feat_resolution,
        )
        num_imgs_per_batch = img1_part_points.shape[0]
        for i in range(num_imgs_per_batch):
            img1_pts_in, img2_pts_in = self.get_valid_points(img1_part_points[i], img2_part_points[i])
            img1_pts_out, img2_pts_out = self.get_valid_points(img1_outside_part_points[i], img2_outside_part_points[i])

            if img1_pts_in.shape[0] == 0 or img1_pts_out.shape[0] == 0:
                num_imgs_per_batch -= 1
                continue

            # (x,y) -> (y,x)
            img1_pts_in = img1_pts_in.flip(1)
            img2_pts_in = img2_pts_in.flip(1)
            img1_pts_out = img1_pts_out.flip(1)
            img2_pts_out = img2_pts_out.flip(1)

            img1_pts_in_idxs = self.points_to_idxs(img1_pts_in, feat_res).to(self.device).long()  # NUM_PTS x 1
            img2_pts_in_idxs = self.points_to_idxs(img2_pts_in, feat_res).to(self.device).long()  # NUM_PTS x 1
            img1_pts_out_idxs = self.points_to_idxs(img1_pts_out, feat_res).to(self.device).long()  # NUM_PTS x 1
            img2_pts_out_idxs = self.points_to_idxs(img2_pts_out, feat_res).to(self.device).long()  # NUM_PTS x 1

            for loss in losses:
                if loss == "semantic_in":
                    losses[loss] += (
                        torch.nn.functional.cross_entropy(sim_semantic12[i, img1_pts_in_idxs], img2_pts_in_idxs)
                        + torch.nn.functional.cross_entropy(sim_semantic21[i, img2_pts_in_idxs], img1_pts_in_idxs)
                    ) / 2
                elif loss == "semantic_out":
                    losses[loss] += (
                        torch.nn.functional.cross_entropy(sim_semantic12[i, img1_pts_out_idxs], img2_pts_out_idxs)
                        + torch.nn.functional.cross_entropy(sim_semantic21[i, img2_pts_out_idxs], img1_pts_out_idxs)
                    ) / 2
                elif loss == "visual_in":
                    losses[loss] += self.config.model.lr_alpha * (
                        torch.nn.functional.cross_entropy(
                            visual_in_sign * sim_visual12[i, img1_pts_in_idxs], img2_pts_in_idxs
                        )
                        + torch.nn.functional.cross_entropy(
                            visual_in_sign * sim_visual21[i, img2_pts_in_idxs], img1_pts_in_idxs
                        )
                    )
                elif loss == "visual_out":
                    losses[loss] += self.config.model.lr_alpha * (
                        torch.nn.functional.cross_entropy(sim_visual12[i, img1_pts_out_idxs], img2_pts_out_idxs)
                        + torch.nn.functional.cross_entropy(sim_visual21[i, img2_pts_out_idxs], img1_pts_out_idxs)
                    )
        # Average losses and check for NaN
        total_loss = 0
        for loss in losses:
            if torch.isnan(losses[loss]):
                print(f"NaN {loss}")
                losses[loss] = torch.tensor(0.0).to(self.device, dtype=self.dtype)
            losses[loss] /= num_imgs_per_batch
            total_loss += losses[loss]

        losses["total"] = total_loss
        return losses

    def compute_metrics(
        self,
        img1_feats_s,
        img2_feats_s,
        img1_feats_v,
        img2_feats_v,
        img1_obj_mask,
        img2_obj_mask,
        img1_part_mask=None,
        img1_oracle=None,
        img2_oracle=None,
    ):

        THRESHOLDS = getattr(self.config, "metric_thresholds", [0.5, 0.6, 0.7])
        METRIC_NAMES = getattr(self.config, "metrics", ["vsm"])  # backward-compatibility

        with torch.no_grad():
            B, HW, C = img1_feats_s.shape
            h = int(sqrt(HW))
            # Reshape features from flattented to 2D for compatibility with `match_features_masked` function
            img1_feats_s = img1_feats_s.permute((0, 2, 1)).view(B, C, h, h)
            img2_feats_s = img2_feats_s.permute((0, 2, 1)).view(B, C, h, h)

            B, HW, C = img1_feats_v.shape
            h = int(sqrt(HW))
            img1_feats_v = img1_feats_v.permute((0, 2, 1)).view(B, C, h, h)
            img2_feats_v = img2_feats_v.permute((0, 2, 1)).view(B, C, h, h)

            # Compute the percentage of consistent part of the object to be used as oracle
            if img1_oracle is None and img1_part_mask is not None:
                img1_oracle = 1 - img1_part_mask.view(B, -1).sum(-1) / img1_obj_mask.view(B, -1).sum(-1)

            metrics = init_metrics_dict(METRIC_NAMES, THRESHOLDS)
            # Compute for each image individuall as `match_features_masked` supports only batch size of 1
            for i in range(img1_feats_s.shape[0]):

                (
                    source_points_s,
                    target_points_s,
                    matching_score_maps_s,
                    matching_score_maps_max_s,
                    _,
                    _,
                ) = match_features_masked(
                    img1_feats_s[i][None],
                    img2_feats_s[i][None],
                    img1_obj_mask[i][None],
                    img2_obj_mask[i][None],
                )
                (
                    source_points_v,
                    target_points_v,
                    matching_score_maps_v,
                    matching_score_maps_max_v,
                    _,
                    _,
                ) = match_features_masked(
                    img1_feats_v[i][None],
                    img2_feats_v[i][None],
                    img1_obj_mask[i][None],
                    img2_obj_mask[i][None],
                )

                if "oracle" in METRIC_NAMES and img1_oracle is not None:
                    metrics["oracle"].append(img1_oracle[i].item())

                if "vsm" in METRIC_NAMES:
                    vsm_metrics, _ = compute_vsm_metric(
                        source_points_s,
                        target_points_s,
                        matching_score_maps_max_s,
                        source_points_v,
                        target_points_v,
                        matching_score_maps_max_v,
                        visual_thresholds=THRESHOLDS,
                        semantic_threshold=0.7,
                        visualize_matches=False,
                    )
                    for thresh in THRESHOLDS:
                        metrics[f"vsm_{thresh}"].append(vsm_metrics[f"vsm_{thresh}"])
        return metrics

    def one_pass(
        self,
        img1,
        img2,
        img1_part_points,
        img2_part_points,
        img1_outside_part_points,
        img2_outside_part_points,
        loss_sign,
    ):
        img1_feats, img2_feats = self.get_dift_feats(img1, img2, resize=True)

        # Compute Semantic similarity
        sim_semantic12, sim_semantic21, img1_feats_s, img2_feats_s = self.compute_similarity(
            img1_feats, img2_feats, self.bottleneck_layers_semantic, self.mixing_weights_semantic
        )

        # Compute Visual similarity
        sim_visual12, sim_visual21, img1_feats_v, img2_feats_v = self.compute_similarity(
            img1_feats, img2_feats, self.bottleneck_layers_visual, self.mixing_weights_visual
        )

        # Compute the loss per image-pair
        losses = self.compute_loss(
            sim_semantic12,
            sim_semantic21,
            sim_visual12,
            sim_visual21,
            img1_part_points,
            img2_part_points,
            img1_outside_part_points,
            img2_outside_part_points,
            visual_in_sign=loss_sign,
        )
        return losses, img1_feats_s, img1_feats_v, img2_feats_s, img2_feats_v

    def forward(self, batch, **kwargs):

        (
            ds_idxs,
            img1_orig,
            img2_orig,
            img1_inpainted,
            img2_inpainted,
            img1_obj_mask,
            img2_obj_mask,
            img1_part_mask,
            img2_part_mask,
            img1_part_points,
            img2_part_points,
            img1_outside_part_points,
            img2_outside_part_points,
            img1_oracle,
            img2_oracle,
        ) = self.unpack_batch(batch, self.dtype, self.device)

        # Compute correspondence between inpainted_image_1 and the original_image_2
        # This uses a negative loss to push features within the inpainted region in inpainted_image_1 apart from features in original_image_2
        losses_neg, img1_feats_s, img1_feats_v, img2_feats_s, img2_feats_v = self.one_pass(
            img1_inpainted,
            img2_orig,
            img1_part_points,
            img2_part_points,
            img1_outside_part_points,
            img2_outside_part_points,
            loss_sign=-1,
        )

        # Compute correspondence between original_image_1 and the original_image_2
        # This uses a positive loss to push features within the same region in original_image_1 apart from features in original_image_2
        losses_pos, _, _, _, _ = self.one_pass(
            img1_orig,
            img2_orig,
            img1_part_points,
            img2_part_points,
            img1_outside_part_points,
            img2_outside_part_points,
            loss_sign=1,
        )

        # Average the losses
        losses = {k: (losses_pos[k] + losses_neg[k]) / 2 for k in losses_pos}

        # Compute metrics including the VSM metric
        metrics = self.compute_metrics(
            img1_feats_s,
            img2_feats_s,
            img1_feats_v,
            img2_feats_v,
            img1_obj_mask,
            img2_obj_mask,
            img1_part_mask,
            img1_oracle,
            img2_oracle,
        )

        # Create log images for debugging in W&B
        imgs_to_log = []
        idxs_to_log = kwargs.get("idxs_to_log", [])
        if idxs_to_log:
            for batch_idx, dataset_idx in enumerate(ds_idxs):
                if dataset_idx in idxs_to_log:
                    imgs_to_log.append(
                        {
                            "idx": dataset_idx,
                            "img": self.make_visualization_img(
                                batch_idx,
                                img1_inpainted,
                                img1_obj_mask,
                                img1_feats_v,
                                img1_part_mask,
                                img2_orig,
                                img2_obj_mask,
                                img2_feats_v,
                                metrics,
                                img1_oracle,
                            ),
                        }
                    )
        return losses, metrics, imgs_to_log

    def make_visualization_img(
        self,
        batch_idx,
        img1_inpainted: torch.Tensor,
        img1_obj_mask: torch.Tensor,
        img1_feats_v: torch.Tensor,
        img1_part_mask: torch.Tensor,
        img2_orig: torch.Tensor,
        img2_obj_mask: torch.Tensor,
        img2_feats_v: torch.Tensor,
        metrics,
        img1_oracle,
    ):
        # Extract the specific batch item while preserving 4D dimensionality
        (
            img1_inpainted_item,
            img1_obj_mask_item,
            img1_part_mask_item,
            img2_orig_item,
            img2_obj_mask_item,
            img1_feats_v_item,
            img2_feats_v_item,
        ) = (
            torch.narrow(tensor, 0, batch_idx, 1)
            for tensor in (
                img1_inpainted,
                img1_obj_mask,
                img1_part_mask,
                img2_orig,
                img2_obj_mask,
                img1_feats_v,
                img2_feats_v,
            )
        )

        source_points_v, target_points_v, matching_score_maps_v, matching_score_maps_max_v = (
            self.compute_correspondences(img1_feats_v_item, img2_feats_v_item, img1_obj_mask_item, img2_obj_mask_item)
        )

        img1_inpainted_item = untransform_img(img1_inpainted_item)
        img1_obj_mask_item = untrasform_mask(img1_obj_mask_item)

        consistency_map = plot_consistency_scores(
            img1_inpainted_item,
            img1_obj_mask_item,
            source_points_v.detach().cpu().float(),
            matching_score_maps_max_v.detach().cpu().float(),
        )

        metrics_to_plot = {
            "VSM_0.5": metrics["vsm_0.5"][batch_idx],
            "VSM_0.6": metrics["vsm_0.6"][batch_idx],
            "oracle": img1_oracle[batch_idx],
        }

        imgs_to_plot = {
            "Reference": untransform_img(img2_orig_item),
            "Mask": untrasform_mask(img1_part_mask_item),
            "Inpainted": img1_inpainted_item,
        }

        viz_img = plot_comparison_grid(imgs_to_plot, consistency_map, metrics_to_plot, format="jpg")
        return viz_img

    def compute_correspondences(self, img1_feats, img2_feats, img1_obj_mask_t, img2_obj_mask_t):

        b, hw, c = img1_feats.shape
        h = int(sqrt(hw))
        img1_feats = img1_feats.permute((0, 2, 1)).view(b, c, h, h)
        img2_feats = img2_feats.permute((0, 2, 1)).view(b, c, h, h)

        (
            source_points,
            target_points,
            matching_score_maps,
            matching_score_maps_max,
            _,
            _,
        ) = match_features_masked(img1_feats, img2_feats, img1_obj_mask_t, img2_obj_mask_t)

        return source_points, target_points, matching_score_maps, matching_score_maps_max

    def inference(
        self,
        img1_p: Image,
        img2_p: Image,
        img1_obj_mask_p: Image,
        img2_obj_mask_p: Image,
        img1_part_mask_p: Optional[Image] = None,
        img2_part_mask_p: Optional[Image] = None,
        transform=transforms.ToTensor(),
        return_correspondences=False,
    ):
        """Inference function that is used in real subject-driven generation benchmark setup"""
        img1 = transform(img1_p).unsqueeze(0).to(self.device, dtype=self.dtype) * 2 - 1
        img2 = transform(img2_p).unsqueeze(0).to(self.device, dtype=self.dtype) * 2 - 1
        img1_part_mask_t = (
            transform(img1_part_mask_p).unsqueeze(0).to(self.device, dtype=self.dtype) if img1_part_mask_p else None
        )
        img1_obj_mask_t = transform(img1_obj_mask_p).unsqueeze(0).to(self.device, dtype=self.dtype)
        img2_obj_mask_t = transform(img2_obj_mask_p).unsqueeze(0).to(self.device, dtype=self.dtype)

        img1_feats, img2_feats = self.get_dift_feats(img1, img2, resize=True)

        # Compute Semantic similarity
        sim_semantic12, sim_semantic21, img1_feats_s, img2_feats_s = self.compute_similarity(
            img1_feats, img2_feats, self.bottleneck_layers_semantic, self.mixing_weights_semantic
        )

        # Compute Visual similarity
        sim_visual12, sim_visual21, img1_feats_v, img2_feats_v = self.compute_similarity(
            img1_feats, img2_feats, self.bottleneck_layers_visual, self.mixing_weights_visual
        )

        # Compute metrics only if the masks are provided

        metrics = self.compute_metrics(
            img1_feats_s,
            img2_feats_s,
            img1_feats_v,
            img2_feats_v,
            img1_obj_mask_t,
            img2_obj_mask_t,
            img1_part_mask_t,
        )

        if return_correspondences:
            source_points_s, target_points_s, matching_score_maps_s, matching_score_maps_max_s = (
                self.compute_correspondences(img1_feats_s, img2_feats_s, img1_obj_mask_t, img2_obj_mask_t)
            )
            source_points_v, target_points_v, matching_score_maps_v, matching_score_maps_max_v = (
                self.compute_correspondences(img1_feats_v, img2_feats_v, img1_obj_mask_t, img2_obj_mask_t)
            )

            return {
                "metrics": metrics,
                "source_points_s": source_points_s.cpu().float().numpy(),
                "target_points_s": target_points_s.cpu().float().numpy(),
                "matching_score_maps_max_s": matching_score_maps_max_s.cpu().detach().float().numpy(),
                "source_points_v": source_points_v.cpu().float().numpy(),
                "target_points_v": target_points_v.cpu().float().numpy(),
                "matching_score_maps_max_v": matching_score_maps_max_v.cpu().detach().float().numpy(),
                "img1_feats_v": img1_feats_v.cpu().detach().float().numpy(),
                "img2_feats_v": img2_feats_v.cpu().detach().float().numpy(),
                "img1_feats_s": img1_feats_s.cpu().detach().float().numpy(),
                "img2_feats_s": img2_feats_s.cpu().detach().float().numpy(),
            }
        else:
            return {"metrics": metrics}
