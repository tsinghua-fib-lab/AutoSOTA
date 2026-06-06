######################################
# https://github.com/sczhou/ProPainter
######################################
import torch.nn as nn
import torch.nn.functional as F
import torch
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d


class SecondOrderDeformableAlignment(ModulatedDeformConv2d):
    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 5)
        super().__init__(*args, **kwargs)
        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )

    def forward(self, x, extra_feat):
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset = torch.cat([offset_1, offset_2], dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.deform_groups)


class BidirectionalPropagation(nn.Module):
    def __init__(self, channel):
        super().__init__()
        modules = ['backward_', 'forward_']
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        self.channel = channel
        for i, module in enumerate(modules):
            self.deform_align[module] = SecondOrderDeformableAlignment(2 * channel, channel, 3, padding=1, deform_groups=16)
            self.backbone[module] = nn.Sequential(
                nn.Conv2d((2 + i) * channel, channel, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(channel, channel, 3, 1, 1),
            )
        self.fusion = nn.Conv2d(2 * channel, channel, 1, 1, 0)

    def forward(self, x):
        b, t, c, h, w = x.shape
        feats = {}
        feats['spatial'] = [x[:, i, :, :, :] for i in range(0, t)]
        for module_name in ['backward_', 'forward_']:
            feats[module_name] = []
            frame_idx = range(0, t)
            mapping_idx = list(range(0, len(feats['spatial'])))
            mapping_idx += mapping_idx[::-1]
            if 'backward' in module_name:
                frame_idx = frame_idx[::-1]
            feat_prop = x.new_zeros(b, self.channel, h, w)
            for i, idx in enumerate(frame_idx):
                feat_current = feats['spatial'][mapping_idx[idx]]
                if i > 0:
                    cond_n1 = feat_prop
                    feat_n2 = torch.zeros_like(feat_prop)
                    cond_n2 = torch.zeros_like(cond_n1)
                    if i > 1:
                        feat_n2 = feats[module_name][-2]
                        cond_n2 = feat_n2
                    cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                    feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                    feat_prop = self.deform_align[module_name](feat_prop, cond)
                feat = [feat_current] + [feats[k][idx] for k in feats if k not in ['spatial', module_name]] + [feat_prop]
                feat = torch.cat(feat, dim=1)
                feat_prop = feat_prop + self.backbone[module_name](feat)
                feats[module_name].append(feat_prop)
            if 'backward' in module_name:
                feats[module_name] = feats[module_name][::-1]
        outputs = []
        for i in range(0, t):
            align_feats = [feats[k].pop(0) for k in feats if k != 'spatial']
            align_feats = torch.cat(align_feats, dim=1)
            outputs.append(self.fusion(align_feats))
        return torch.stack(outputs, dim=1) + x


class deconv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return self.conv(x)


class P3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, kernel_size, kernel_size), stride=(1, stride, stride), padding=(0, padding, padding)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(nn.Conv3d(out_channels, out_channels, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(2, 0, 0), dilation=(2, 1, 1)))

    def forward(self, feats):
        feat1 = self.conv1(feats)
        feat2 = self.conv2(feat1)
        return feat2


class FCNet(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.downsample = nn.Sequential(nn.Conv3d(3, 32, kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 2, 2), padding_mode='replicate'), nn.LeakyReLU(0.2, inplace=True))
        self.encoder1 = nn.Sequential(P3DBlock(32, 32, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True), P3DBlock(32, 64, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True))
        self.encoder2 = nn.Sequential(P3DBlock(64, 64, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True), P3DBlock(64, 128, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True))
        self.mid_dilation = nn.Sequential(
            nn.Conv3d(128, 128, (1, 3, 3), (1, 1, 1), padding=(0, 3, 3), dilation=(1, 3, 3)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 128, (1, 3, 3), (1, 1, 1), padding=(0, 2, 2), dilation=(1, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 128, (1, 3, 3), (1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.feat_prop_module = BidirectionalPropagation(128)
        self.decoder2 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True), deconv(128, 64, 3, 1), nn.LeakyReLU(0.2, inplace=True))
        self.decoder1 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True), deconv(64, 32, 3, 1), nn.LeakyReLU(0.2, inplace=True))
        self.upsample = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.LeakyReLU(0.2, inplace=True), deconv(32, 2, 3, 1))
        self.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)

    def forward(self, masked_flows, masks):
        b, t, _, h, w = masked_flows.size()
        masked_flows = masked_flows.permute(0, 2, 1, 3, 4)
        masks = masks.permute(0, 2, 1, 3, 4)
        inputs = torch.cat((masked_flows, masks), dim=1)
        x = self.downsample(inputs)
        feat_e1 = self.encoder1(x)
        feat_e2 = self.encoder2(feat_e1)
        feat_mid = self.mid_dilation(feat_e2)
        feat_mid = feat_mid.permute(0, 2, 1, 3, 4)
        feat_prop = self.feat_prop_module(feat_mid)
        feat_prop = feat_prop.view(-1, 128, h // 8, w // 8)
        _, c, _, h_f, w_f = feat_e1.shape
        feat_e1 = feat_e1.permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h_f, w_f)
        feat_d2 = self.decoder2(feat_prop) + feat_e1
        feat_d1 = self.decoder1(feat_d2)
        flow = self.upsample(feat_d1)
        flow = flow.view(b, t, 2, h, w)
        return flow

    def forward_bidirect_flow(self, masked_flows_bi, masks):
        masks_forward = masks[:, :-1, ...].contiguous()
        masks_backward = masks[:, 1:, ...].contiguous()
        masked_flows_forward = masked_flows_bi[0] * (1 - masks_forward)
        masked_flows_backward = masked_flows_bi[1] * (1 - masks_backward)
        pred_flows_forward = self.forward(masked_flows_forward, masks_forward)
        masked_flows_backward = torch.flip(masked_flows_backward, dims=[1])
        masks_backward = torch.flip(masks_backward, dims=[1])
        pred_flows_backward = self.forward(masked_flows_backward, masks_backward)
        pred_flows_backward = torch.flip(pred_flows_backward, dims=[1])
        return pred_flows_forward, pred_flows_backward

    def combine_flow(self, masked_flows_bi, pred_flows_bi, masks):
        masks_forward = masks[:, :-1, ...].contiguous()
        masks_backward = masks[:, 1:, ...].contiguous()
        pred_flows_forward = pred_flows_bi[0] * masks_forward + masked_flows_bi[0] * (1 - masks_forward)
        pred_flows_backward = pred_flows_bi[1] * masks_backward + masked_flows_bi[1] * (1 - masks_backward)
        return pred_flows_forward, pred_flows_backward
