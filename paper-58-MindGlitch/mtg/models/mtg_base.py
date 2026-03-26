import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from omegaconf.dictconfig import DictConfig


FEAT_TYPE_TO_DIM = {
    "us1": 1280,
    "us2": 1280,
    "us3": 1280,
    "us4": 1280,
    "us5": 1280,
    "us6": 1280,
    "us7": 640,
    "us8": 640,
    "us9": 640,
    "us10": 320,
}

FEAT_TYPE_TO_RES = {
    "us1": 12,
    "us2": 12,
    "us3": 24,
    "us4": 24,
    "us5": 24,
    "us6": 48,
    "us7": 48,
    "us8": 48,
    "us9": 96,
    "us10": 96,
}


class MindTheGlitchBase(nn.Module):
    def __init__(self, cleandift_model, config: DictConfig, DEVICE, DTYPE):
        super().__init__()
        self.device = DEVICE
        self.dtype = DTYPE
        self.cleandift_model = cleandift_model
        self.config = config
        self.feat_type = self.config.model.get("feat_type", ["us6"])
        self.feature_dims = [FEAT_TYPE_TO_DIM[f] for f in self.feat_type]
        self.out_feat_res = self.config.model.final_feat_resolution

        # For CLIP symmetric cross entropy loss during training
        self.logit_scale = torch.ones([]) * np.log(1 / 0.07)

        if self.config.model.get("freeze_cleandift", True):
            print("Freezing CleanDIFT model parameters")
            for param in self.parameters():
                param.requires_grad = False

    def get_dift_feats(self, imgs_source, imgs_target, resize=False):
        B = imgs_source.shape[0]

        imgs_tensor = torch.cat([imgs_source, imgs_target], dim=0)

        item_desc = ""
        with torch.no_grad():
            dift_features = self.cleandift_model.get_features(imgs_tensor, [item_desc], t=None, feat_key=None)

        selected_features_source = {f: dift_features[f][:B] for f in self.feat_type}
        selected_features_target = {f: dift_features[f][B:] for f in self.feat_type}
        if resize:
            selected_features_source = self.resize_feats(selected_features_source)
            selected_features_target = self.resize_feats(selected_features_target)
        return selected_features_source, selected_features_target

    @staticmethod
    def unpack_batch(batch, dtype, device):
        ds_idxs = batch["idx"]
        img1_orig = batch["img1_orig"].to(dtype=dtype, device=device)
        img2_orig = batch["img2_orig"].to(dtype=dtype, device=device)
        img1_inpainted = batch["img1_inpainted"].to(dtype=dtype, device=device)
        img2_inpainted = batch["img2_inpainted"].to(dtype=dtype, device=device)
        img1_obj_mask = batch["img1_obj_mask"].to(dtype=dtype, device=device)
        img2_obj_mask = batch["img2_obj_mask"].to(dtype=dtype, device=device)
        img1_part_mask = batch["img1_part_mask"].to(dtype=dtype, device=device)
        img2_part_mask = batch["img2_part_mask"].to(dtype=dtype, device=device)
        img1_oracle = batch["img1_oracle"].to(dtype=dtype, device=device)
        img2_oracle = batch["img2_oracle"].to(dtype=dtype, device=device)

        # Corresspondence
        img1_part_points = torch.round(batch["img1_part_points"]).to(dtype=torch.int, device=device)
        img2_part_points = torch.round(batch["img2_part_points"]).to(dtype=torch.int, device=device)
        img1_outside_part_points = torch.round(batch["img1_outside_part_points"]).to(dtype=torch.int, device=device)
        img2_outside_part_points = torch.round(batch["img2_outside_part_points"]).to(dtype=torch.int, device=device)
        return (
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
        )

    def resize_feats(self, feats):
        resized_feats = {}
        for feat_type in self.feat_type:
            resized_feats[feat_type] = F.interpolate(
                feats[feat_type], size=(self.out_feat_res, self.out_feat_res), mode="bilinear", align_corners=False
            )
        return resized_feats
