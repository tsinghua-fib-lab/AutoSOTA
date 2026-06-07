import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import load_checkpoint_and_dispatch
from .module import *
from util.util import *

def pool2x(x):
    return F.avg_pool2d(x, 3, stride=2, padding=1)

def interp(x, dest):
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    return F.interpolate(x, dest.shape[-2:], **interp_args)

class DispHead(nn.Module):
    def __init__(self, cfg):
        super(DispHead, self).__init__()
        self.conv1 = nn.Conv2d(cfg.pretrained_model.features, cfg.pretrained_model.features, 3, padding=1)
        self.conv2 = nn.Conv2d(cfg.pretrained_model.features, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class ResidualConvUnit(nn.Module):
    def __init__(self, features, activation, bn):
        super().__init__()
        self.bn = bn
        self.groups = 1
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)

        if self.bn == True:
            out = self.bn1(out)
       
        out = self.activation(out)
        out = self.conv2(out)

        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)

class StructureEncoder(nn.Module):
    def __init__(self, cfg):
        super(StructureEncoder, self).__init__()
        self.convc1 = nn.Conv2d(cfg.pretrained_model.features // 2, cfg.pretrained_model.features // 2, 1, padding=0)
        self.convc2 = nn.Conv2d(cfg.pretrained_model.features // 2, cfg.pretrained_model.features // 2, 3, padding=1)
        self.convd1 = nn.Conv2d(1, cfg.pretrained_model.features // 2, 7, padding=3)
        self.convd2 = nn.Conv2d(cfg.pretrained_model.features // 2, cfg.pretrained_model.features // 2, 3, padding=1)
        self.conv = nn.Conv2d(cfg.pretrained_model.features, cfg.pretrained_model.features - 1, 3, padding=1)

    def forward(self, ctx, norm_depth, norm_disp):
        diff = torch.abs(norm_depth - norm_disp)

        c = F.relu(self.convc1(ctx), True)
        c = F.relu(self.convc2(c), True)
        d = F.relu(self.convd1(diff), True)
        d = F.relu(self.convd2(d), True)

        out = torch.cat((c, d), dim=1)
        out = F.relu(self.conv(out), True)

        return torch.cat((out, diff), dim=1)

class MotionEncoder(nn.Module):
    def __init__(self, cfg):
        super(MotionEncoder, self).__init__()
        cor_plane = (cfg.gwc_group + 1) * (2 * cfg.corr_radius + 1) * cfg.corr_level
        self.convc1 = nn.Conv2d(cor_plane, cfg.pretrained_model.features // 2, 1, padding=0)
        self.convc2 = nn.Conv2d(cfg.pretrained_model.features // 2, cfg.pretrained_model.features // 2, 3, padding=1)
        self.convd1 = nn.Conv2d(1, cfg.pretrained_model.features // 2, 7, padding=3)
        self.convd2 = nn.Conv2d(cfg.pretrained_model.features // 2, cfg.pretrained_model.features // 2, 3, padding=1)
        self.conv = nn.Conv2d(cfg.pretrained_model.features, cfg.pretrained_model.features - 1, 3, padding=1)
    
    def forward(self, corr, disp):
        cor = F.relu(self.convc1(corr), True)
        cor = F.relu(self.convc2(cor), True)
        dis = F.relu(self.convd1(disp), True)
        dis = F.relu(self.convd2(dis), True)

        out = torch.cat([cor, dis], dim=1)
        out = F.relu(self.conv(out), True)

        return torch.cat([out, disp], dim=1)

class PromptStereoRecurrentUnit(nn.Module):
    def __init__(self, cfg, features, activation=nn.ReLU(False), deconv=False, bn=False, expand=False, align_corners=True, motion=False, size=None):
        super(PromptStereoRecurrentUnit, self).__init__()
        self.deconv = deconv
        self.expand = expand
        self.align_corners = align_corners
        self.size = size
        self.groups = 1

        out_features = features
        if self.expand == True:
            out_features = features // 2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)
        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)

        if motion:
            self.resConfUnitStructure = nn.Sequential(
                BasicConv(features, features, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0)
                )
            
            self.resConfUnitMotion = nn.Sequential(
                BasicConv(features, features, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0)
                )

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs, structure=None, motion=None, size=None):
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        if motion is not None:
            
            structure = self.resConfUnitStructure(structure)
            output = self.skip_add.add(output, structure)
            
            motion = self.resConfUnitMotion(motion)
            output = self.skip_add.add(output, motion)

        output = self.out_conv(output)

        return output

class MultiPromptUpdateBlock(nn.Module):
    def __init__(self, cfg, pretrained_state):
        super(MultiPromptUpdateBlock, self).__init__()
        self.stereo_pru = nn.ModuleList([PromptStereoRecurrentUnit(cfg, features=cfg.pretrained_model.features, bn=cfg.pretrained_model.use_bn, motion=(i == 0)) for i in range(len(cfg.pretrained_model.out_channels))])
        self.structure_encoder = StructureEncoder(cfg)
        self.motion_encoder = MotionEncoder(cfg)
        self.disp_head = DispHead(cfg)
        self.mask = nn.Sequential(
            nn.Conv2d(cfg.pretrained_model.features, cfg.pretrained_model.features, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(cfg.pretrained_model.features, (2 ** cfg.n_downsample ** 2) * 9, 1, padding=0)
        )

        self.update = nn.ModuleList([
            nn.Sequential(
                BasicConv(cfg.pretrained_model.features * (2 + (i == 0)), cfg.pretrained_model.features, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(cfg.pretrained_model.features, cfg.pretrained_model.features, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid()                
            ) for i in range(len(cfg.pretrained_model.out_channels))
        ])

        if pretrained_state:
            block_state = self.state_dict()
            new_dict = {}

            for i in range(len(cfg.pretrained_model.out_channels)):
                ref_name = f'scratch.refinenet{i + 1}'
                for module_name in ['mono_pru', 'stereo_pru']:
                    tar_name = f'{module_name}.{i}'

                    for k, v in pretrained_state.items():
                        if k.startswith(ref_name):
                            new_k = k.replace(ref_name, tar_name)
                            if new_k in block_state:
                                new_dict[new_k] = v
                
            block_state.update(new_dict)
            self.load_state_dict(block_state, strict=True)

    def forward(self, net, corr, disp, ctx, norm_depth, itr=0):
        norm_disp, _, _ = normalize_disparity(disp)
        structure = self.structure_encoder(ctx, norm_depth, norm_disp)
        motion = self.motion_encoder(corr, disp)

        for i in reversed(range(len(net))):
            if (i == len(net) - 1):
                z = self.update[i](torch.cat([net[i], pool2x(net[i - 1])], dim=1))
                net[i] = (1 - z) * net[i] + z * (self.stereo_pru[i](net[i]))
            elif (i == 0):
                z = self.update[i](torch.cat([net[i], structure, motion], dim=1))
                net[i] = (1 - z) * net[i] + z * (self.stereo_pru[i](net[i], interp(net[i + 1], net[i]), structure=structure, motion=motion))
            else:
                z = self.update[i](torch.cat([net[i], pool2x(net[i - 1])], dim=1))
                net[i] = (1 - z) * net[i] + z * (self.stereo_pru[i](net[i], interp(net[i + 1], net[i])))

        delta_disp = self.disp_head(net[0])
        mask = .25 * self.mask(net[0])

        # Adaptive dampening: reduce updates where structure and motion disagree
        # norm_depth and norm_disp are already normalized — their difference indicates
        # regions where monocular depth and stereo are inconsistent
        diff = torch.abs(norm_depth - norm_disp)
        # Exponential dampening: large diff → small weight, small diff → weight near 1
                # Progressive: less dampening in later iterations
        dampening = 0.3 + 0.7 * torch.exp(-2.0 * diff / (1.0 + 0.05 * itr))
        delta_disp = delta_disp * dampening

        return net, delta_disp, mask
