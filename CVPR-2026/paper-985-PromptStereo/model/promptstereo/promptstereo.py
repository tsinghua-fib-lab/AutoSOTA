import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import load_checkpoint_and_dispatch
from .corr import CombinedGeometryEncodingVolume
from .update import MultiPromptUpdateBlock
from .extractor import FeatureExtractor
from .module import *
from util.util import *

class PromptStereo(nn.Module):
    def __init__(self, cfg):
        super(PromptStereo, self).__init__()
        self.cfg = cfg

        vit = cfg.pretrained_model.instance
        vit = load_checkpoint_and_dispatch(vit, cfg.pretrained_model.checkpoint, strict=True)

        self.fnet = FeatureExtractor(cfg, vit.state_dict())
        self.update_block = MultiPromptUpdateBlock(cfg, vit.depth_head.state_dict())

        del vit

        self.hourglass = HourGlass(cfg)
        self.classifier = nn.Conv3d(cfg.gwc_group, 1, 3, 1, 1, bias=False)

        self.stem = nn.ModuleList([
            nn.Sequential(
                BasicConv(in_channel, out_channel, kernel_size=3, stride=2, padding=1),
                BasicConv(out_channel, out_channel, relu='relu', kernel_size=3, stride=1, padding=1)
            ) for in_channel, out_channel in zip([3] + cfg.stem_dim[:-1], cfg.stem_dim)
        ])

        self.spx_4 = nn.Sequential(
            BasicConv(cfg.feat_dim[0], cfg.stem_dim[1], kernel_size=3, stride=1, padding=1),
            BasicConv(cfg.stem_dim[1], cfg.stem_dim[1], relu='relu', kernel_size=3, stride=1, padding=1)
        )
        self.spx_2 = Conv2x(cfg.stem_dim[1], cfg.stem_dim[0], deconv=True)
        self.spx = nn.ConvTranspose2d(2 * cfg.stem_dim[0], 9, kernel_size=4, stride=2, padding=1)

        self.desc = nn.Sequential(
            BasicConv(cfg.feat_dim[0] + cfg.stem_dim[1], cfg.feat_dim[0] + cfg.stem_dim[1], kernel_size=3, stride=1, padding=1),
            nn.Conv2d(cfg.feat_dim[0] + cfg.stem_dim[1], cfg.feat_dim[0], kernel_size=1, stride=1, padding=0)
        )

        self.cnet = nn.ModuleList([
            nn.Sequential(
                BasicConv(feat_dim + stem_dim, feat_dim + stem_dim, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(feat_dim + stem_dim, cfg.pretrained_model.features, 1, 1, 0)
            ) for feat_dim, stem_dim in zip(cfg.feat_dim, cfg.stem_dim[1:])
        ])
        
        self.hnet = nn.ModuleList([
            nn.Sequential(
                BasicConv(cfg.pretrained_model.features * 2, cfg.pretrained_model.features * 2, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(cfg.pretrained_model.features * 2, cfg.pretrained_model.features, 1, 1, 0)
            ) for _ in range(len(cfg.pretrained_model.out_channels))
        ])

        self.conf = nn.Sequential(
            BasicConv(cfg.pretrained_model.features * 2, cfg.pretrained_model.features * 2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(cfg.pretrained_model.features * 2, 1, 1, 1, 0),
            nn.Sigmoid()
        )

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
            if isinstance(m, nn.SyncBatchNorm):
                m.eval()

    def upsample_disp(self, disp, mask):
        B, C, H, W = disp.shape
        factor = 2 ** self.cfg.n_downsample
        mask = mask.view(B, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        disp = F.unfold(factor * disp, [3, 3], padding=1)
        disp = disp.view(B, C, 9, 1, 1, H, W)
        disp = torch.sum(mask * disp, dim=2)
        disp = disp.permute(0, 1, 4, 2, 5, 3).contiguous()
        
        return disp.view(B, C, factor * H, factor * W)

    def forward(self, left, right, iters=16, test_mode=False):
        B, _, H, W = left.shape
        stem_left, left = normalize_image(left)
        stem_right, right = normalize_image(right)

        feat_mono, feat_stereo, depth = self.fnet(torch.cat((left, right), dim=0))
        ctx_mono = feat_mono[:B]
        feat_left = [stereo[:B] for stereo in feat_stereo]
        feat_right = [stereo[B:] for stereo in feat_stereo]

        stem_list = []
        for i, block in enumerate(self.stem):
            if i >= 1:
                stem_list.append(block(stem_list[-1]))
            else:
                stem_list.append(block(torch.cat((stem_left, stem_right), dim=0)))

        match_left = self.desc(torch.cat((feat_left[0], stem_list[1][:B]), dim=1))
        match_right = self.desc(torch.cat((feat_right[0], stem_list[1][B:]), dim=1))

        gwc_volume = build_gwc_volume(match_left, match_right, self.cfg.gwc_max_disp // (2 ** self.cfg.n_downsample), self.cfg.gwc_group)
        geometry_encoding_volume = self.hourglass(gwc_volume, feat_left)
        prob = F.softmax(self.classifier(geometry_encoding_volume).squeeze(1) * 1.5, dim=1)
        init_disp = disparity_regression(prob, self.cfg.gwc_max_disp // (2 ** self.cfg.n_downsample))      

        del gwc_volume, prob

        if not test_mode:
            xspx = self.spx_4(match_left)
            xspx = self.spx_2(xspx, stem_list[0][:B])
            spx_pred = self.spx(xspx)
            spx_pred = F.softmax(spx_pred, 1)

        corr_block = CombinedGeometryEncodingVolume(match_left, match_right, geometry_encoding_volume, self.cfg.corr_level, self.cfg.corr_radius)

        ctx_stereo = [block(torch.cat((x, y), dim=1)) for block, x, y in zip(self.cnet, feat_stereo, stem_list[1:])]
        ctx_left = [stereo[:B] for stereo in ctx_stereo]
        ctx_right = [stereo[B:] for stereo in ctx_stereo]
        warped_ctx_right = fmap_sampler(ctx_right, init_disp)
        net = [block(torch.cat((x, y), dim=1)) for block, x, y in zip(self.hnet, ctx_left, warped_ctx_right)]
        
        conf = self.conf(torch.cat((ctx_left[0], warped_ctx_right[0]), dim=1))
        norm_depth, _, _ = normalize_disparity(depth)
        _, scale, shift = normalize_disparity(init_disp)
        aligned_depth = norm_depth * scale[..., None, None] + shift[..., None, None]
        disp = conf * init_disp + (1 - conf) * aligned_depth
        
        disp_pred = []
        for itr in range(iters):
            disp = disp.detach()
            corr = corr_block(disp)
            net, delta_disp, mask = self.update_block(net, corr, disp, ctx_mono, norm_depth, itr=itr)
            disp = disp + delta_disp

            # LEAP: Adaptive disparity truncation to mono depth range
            # Keeps disparity within a decreasing margin of aligned mono depth
            # Prevents catastrophic drift in ambiguous regions
            if aligned_depth.shape[-2:] != disp.shape[-2:]:
                aligned_scale = torch.nn.functional.interpolate(
                    aligned_depth, disp.shape[-2:], mode='bilinear', align_corners=True)
            else:
                aligned_scale = aligned_depth
            max_disp_scale = self.cfg.gwc_max_disp / (2 ** self.cfg.n_downsample)
            margin_frac = 0.5 - 0.4 * itr / max(iters - 1, 1)
            margin = margin_frac * max_disp_scale
            disp = torch.clamp(disp, aligned_scale - margin, aligned_scale + margin)

            if test_mode and itr < iters - 1:
                continue

            up_disp = self.upsample_disp(disp, mask)
            disp_pred.append(up_disp)

        if test_mode:
            # Edge-aware refinement: use left image gradients to sharpen disparity at edges
            # This addresses soft-argmin over-smoothing at depth discontinuities
            # stem_left is at full resolution (H,W), matches up_disp
            left_gray = stem_left[:, :1, :, :]  # Use R channel as guidance
            # Compute local gradient magnitude via simple 3x3 differences
            grad_x = left_gray[:, :, :, 2:] - left_gray[:, :, :, :-2]
            grad_y = left_gray[:, :, 2:, :] - left_gray[:, :, :-2, :]
            # Pad gradients back to original size
            grad_x = torch.nn.functional.pad(grad_x, (1, 1, 0, 0), mode='replicate')
            grad_y = torch.nn.functional.pad(grad_y, (0, 0, 1, 1), mode='replicate')
            edge_strength = (grad_x.abs() + grad_y.abs()).clamp(0, 1)
            # Apply edge-guided sharpening: at edges, trust current disparity more
            # by reducing local smoothing effect of upsampling
            # Simple unsharp mask: sharpened = disp + alpha * (disp - blurred)
            kernel = torch.ones(1, 1, 3, 3, device=up_disp.device) / 9.0
            disp_blurred = torch.nn.functional.conv2d(
                torch.nn.functional.pad(up_disp, (1,1,1,1), mode='replicate'),
                kernel, groups=1
            )
            alpha = 0.15 * edge_strength.mean(dim=1, keepdim=True)
            up_disp = up_disp + alpha * (up_disp - disp_blurred)
            return up_disp

        factor = 2 ** self.cfg.n_downsample
        init_disp = context_upsample(init_disp * 4, spx_pred, factor)

        return init_disp, disp_pred