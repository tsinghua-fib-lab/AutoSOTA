import copy
import torch
import torch.nn as nn
from hydra.utils import instantiate
from accelerate import load_checkpoint_and_dispatch
from util.util import freeze_module

class Feat_transfer(nn.Module):
    def __init__(self, dim):
        super(Feat_transfer, self).__init__()
        self.conv4x = nn.Sequential(
            nn.Conv2d(in_channels=dim[3] + dim[0], out_channels=dim[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim[0]), nn.ReLU(True)
            )
        self.conv8x = nn.Sequential(
            nn.Conv2d(in_channels=dim[3] + dim[1], out_channels=dim[1], kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim[1]), nn.ReLU(True)
            )
        self.conv16x = nn.Sequential(
            nn.Conv2d(in_channels=dim[3] + dim[2], out_channels=dim[2], kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim[2]), nn.ReLU(True)
            )
        self.conv32x = nn.Sequential(
            nn.Conv2d(in_channels=dim[3], out_channels=dim[3], kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim[3]), nn.ReLU(True)
            )
        self.conv_up_32x = nn.ConvTranspose2d(dim[3],
                                dim[2],
                                kernel_size=3,
                                padding=1,
                                output_padding=1,
                                stride=2
                                )
        self.conv_up_16x = nn.ConvTranspose2d(dim[2],
                                dim[1],
                                kernel_size=3,
                                padding=1,
                                output_padding=1,
                                stride=2
                                )
        self.conv_up_8x = nn.ConvTranspose2d(dim[1],
                                dim[0],
                                kernel_size=3,
                                padding=1,
                                output_padding=1,
                                stride=2
                                )
        
        self.res_16x = nn.Conv2d(dim[3], dim[2], kernel_size=1, padding=0, stride=1)
        self.res_8x = nn.Conv2d(dim[3], dim[1], kernel_size=1, padding=0, stride=1)
        self.res_4x = nn.Conv2d(dim[3], dim[0], kernel_size=1, padding=0, stride=1)

    def forward(self, features):
        features_mono_list = []
        feat_32x = self.conv32x(features[3])
        feat_32x_up = self.conv_up_32x(feat_32x)
        feat_16x = self.conv16x(torch.cat((features[2], feat_32x_up), 1)) + self.res_16x(features[2])
        feat_16x_up = self.conv_up_16x(feat_16x)
        feat_8x = self.conv8x(torch.cat((features[1], feat_16x_up), 1)) + self.res_8x(features[1])
        feat_8x_up = self.conv_up_8x(feat_8x)
        feat_4x = self.conv4x(torch.cat((features[0], feat_8x_up), 1)) + self.res_4x(features[0])
        features_mono_list.append(feat_4x)
        features_mono_list.append(feat_8x)
        features_mono_list.append(feat_16x)
        features_mono_list.append(feat_32x)
        return features_mono_list

class FeatureExtractor(nn.Module):
    def __init__(self, cfg, pretrained_state):
        super(FeatureExtractor, self).__init__()
        vit = cfg.pretrained_model.instance
        if pretrained_state:
            vit.load_state_dict(pretrained_state, strict=True)

        self.encoder = vit.encoder
        self.intermediate_layer_idx = vit.intermediate_layer_idx
        self.pretrained = vit.pretrained
        self.mono_head = vit.depth_head
        self.stereo_head = copy.deepcopy(vit.depth_head)
        self.feat_transfer = Feat_transfer(cfg.feat_dim)

        freeze_module(self.pretrained)
        freeze_module(self.mono_head)

    def forward(self, x):
        with torch.no_grad():
            patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
            vit_feat = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)
            mono_feat, depth = self.mono_head(vit_feat, patch_h, patch_w, return_mono=True)
        stereo_feat = self.stereo_head(vit_feat, patch_h, patch_w, return_stereo=True)
        stereo_feat = self.feat_transfer(stereo_feat)

        return mono_feat, stereo_feat, depth
