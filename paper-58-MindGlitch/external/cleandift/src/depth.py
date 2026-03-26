from typing import Literal
from omegaconf import OmegaConf
import hydra
import torch
from torch import nn
from jaxtyping import Float
import torch.nn.functional as F


class SigLoss(nn.Module):
    """SigLoss.

        This follows `AdaBins <https://arxiv.org/abs/2011.14141>`_.

    Args:
        valid_mask (bool): Whether filter invalid gt (gt > 0). Default: True.
        loss_weight (float): Weight of the loss. Default: 1.0.
        max_depth (int): When filtering invalid gt, set a max threshold. Default: None.
        warm_up (bool): A simple warm up stage to help convergence. Default: False.
        warm_iter (int): The number of warm up stage. Default: 100.

    Adapted from: https://github.com/facebookresearch/dinov2/blob/main/dinov2/eval/depth/models/losses/sigloss.py
    """

    def __init__(
        self, valid_mask=True, loss_weight=1.0, max_depth=None, warm_up=False, warm_iter=100, loss_name="sigloss"
    ):
        super(SigLoss, self).__init__()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight
        self.max_depth = max_depth
        self.loss_name = loss_name

        self.eps = 0.001

        self.warm_up = warm_up
        self.warm_iter = warm_iter
        self.warm_up_counter = 0

    def sigloss(self, input, target):
        if self.valid_mask:
            valid_mask = target > 0
            if self.max_depth is not None:
                valid_mask = torch.logical_and(target > 0, target <= self.max_depth)
            input = input[valid_mask]
            target = target[valid_mask]

        if self.warm_up:
            if self.warm_up_counter < self.warm_iter:
                g = torch.log(input + self.eps) - torch.log(target + self.eps)
                g = 0.15 * torch.pow(torch.mean(g), 2)
                self.warm_up_counter += 1
                return torch.sqrt(g)

        g = torch.log(input + self.eps) - torch.log(target + self.eps)
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return torch.sqrt(Dg)

    def forward(self, depth_pred, depth_gt):
        """Forward function."""

        loss_depth = self.loss_weight * self.sigloss(depth_pred, depth_gt)
        return loss_depth


class DepthPred(torch.nn.Module):
    """
    Adapted from:
        - https://github.com/facebookresearch/dinov2/blob/main/dinov2/eval/depth/models/decode_heads/decode_head.py
        - https://github.com/facebookresearch/dinov2/blob/main/dinov2/eval/depth/models/decode_heads/linear_head.py
    """

    def __init__(
        self,
        model_config_path: str,
        channels: int,
        loss: nn.Module,
        diffusion_image_size: int,
        classify=True,
        n_bins=256,
        min_depth=0.1,
        max_depth=10.0,
        bins_strategy="UD",
        norm_strategy="linear",
        scale_up=False,
        use_base_model_features=False,
        base_model_timestep: None | int = None,
        adapter_timestep: int | None = None,
        extraction_layer="us6",
        interpolate_features: Literal["DINO", "FULL", "NONE"] = "NONE",
        n_vis_samples: int = 0,
    ):
        super().__init__()
        self.loss = loss
        self.classify = classify
        self.n_bins = n_bins
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.scale_up = scale_up
        self.use_base_model_features = use_base_model_features
        self.base_model_timestep = base_model_timestep
        self.diffusion_image_size = diffusion_image_size
        self.extraction_layer = extraction_layer
        self.interpolate_features = interpolate_features
        self.adapter_timestep = adapter_timestep
        self.n_vis_samples = n_vis_samples

        if use_base_model_features:
            assert base_model_timestep is not None, "Need to provide base_model_timestep if using base model features"

        cfg_model = OmegaConf.load(model_config_path)
        OmegaConf.resolve(cfg_model)
        self.feature_extractor = hydra.utils.instantiate(cfg_model).model

        self.feature_extractor.requires_grad_(False)
        self.feature_extractor.eval()

        if self.classify:
            assert bins_strategy in ["UD", "SID"], "Support bins_strategy: UD, SID"
            assert norm_strategy in ["linear", "softmax", "sigmoid"], "Support norm_strategy: linear, softmax, sigmoid"

            self.bins_strategy = bins_strategy
            self.norm_strategy = norm_strategy
            self.conv_depth = nn.Conv2d(channels, n_bins, kernel_size=1, padding=0, stride=1)
        else:
            self.conv_depth = nn.Conv2d(channels, 1, kernel_size=1, padding=0, stride=1)

    def depth_pred(self, feat: Float[torch.Tensor, "B C H W"]) -> Float[torch.Tensor, "B 1 H W"]:
        """Prediction each pixel."""
        if self.classify:
            logit = self.conv_depth(feat)

            if self.bins_strategy == "UD":
                bins = torch.linspace(self.min_depth, self.max_depth, self.n_bins, device=feat.device, dtype=feat.dtype)
            elif self.bins_strategy == "SID":
                bins = torch.logspace(self.min_depth, self.max_depth, self.n_bins, device=feat.device, dtype=feat.dtype)

            # following Adabins, default linear
            if self.norm_strategy == "linear":
                logit = torch.relu(logit)
                eps = 0.1
                logit = logit + eps
                logit = logit / logit.sum(dim=1, keepdim=True)
            elif self.norm_strategy == "softmax":
                logit = torch.softmax(logit, dim=1)
            elif self.norm_strategy == "sigmoid":
                logit = torch.sigmoid(logit)
                logit = logit / logit.sum(dim=1, keepdim=True)

            output = torch.einsum("ikmn,k->imn", [logit, bins]).unsqueeze(dim=1)

        else:
            if self.scale_up:
                output = torch.sigmoid(self.conv_depth(feat)) * self.max_depth
            else:
                output = torch.relu(self.conv_depth(feat)) + self.min_depth

        return output

    def forward(self, img, depth_gt, *args, **kwargs):
        depth_target = depth_gt.data[0].cuda().bfloat16()
        image = img.data[0].cuda().bfloat16()

        depth_pred = self.predict(image)

        if self.interpolate_features != "FULL":
            H, W = depth_pred.shape[-2:]
            depth_target = F.interpolate(depth_target, (H, W), mode="bilinear", align_corners=False)

        return self.loss(depth_pred, depth_target)

    def predict(self, image):

        B, C, H, W = image.shape
        image = F.interpolate(
            image, size=(self.diffusion_image_size, self.diffusion_image_size), mode="bilinear", align_corners=False
        )

        with torch.no_grad():
            timestep = None
            if self.use_base_model_features:
                timestep = torch.tensor([self.base_model_timestep] * B, device=image.device)
            elif self.adapter_timestep is not None:
                timestep = torch.tensor([self.adapter_timestep] * B, device=image.device)
            features = self.feature_extractor.get_features(
                image,
                ["A photo of a room"] * B,
                timestep,
                self.extraction_layer,
                use_base_model=self.use_base_model_features,
            )

            if self.interpolate_features == "FULL":
                features = F.interpolate(features, size=(H, W), mode="bilinear", align_corners=False)
            elif self.interpolate_features == "DINO":
                H, W = features.shape[-2:]
                features = F.interpolate(features, size=(H * 4, W * 4), mode="bilinear", align_corners=False)

        depth_pred = self.depth_pred(features)

        return depth_pred
