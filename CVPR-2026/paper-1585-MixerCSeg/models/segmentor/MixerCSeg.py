import torch
import torch.nn.functional as F
from torch import nn
from models.decoder import SRFModule
from models.encoder import VSSEncoder



class MixerCSeg(nn.Module):
    def __init__(self, backbone, embed_dims, args=None):
        super().__init__()
        self.args = args
        self.backbone = backbone    
        self.decoder = SRFModule(embed_dims, mid_dim=8, size=(args.load_width, args.load_height))
        self.use_tta = getattr(args, 'use_tta', False)
        self.use_morph = getattr(args, 'use_morph', False)
        self.tta_scales = getattr(args, 'tta_scales', [1.0, 1.25])

    def _forward_single(self, x):
        """Single forward pass without TTA."""
        outs = self.backbone(x)
        out = self.decoder(outs)
        if hasattr(self, 'use_morph') and self.use_morph and not self.training:
            prob = torch.sigmoid(out)
            prob = self._apply_morph(prob)
            eps = 1e-7
            prob = torch.clamp(prob, eps, 1 - eps)
            out = torch.log(prob / (1 - prob))
        return out

    def _apply_morph(self, prob_map, kernel_size=3, min_area=20):
        """Apply morphological post-processing to clean up predictions."""
        B, C, H, W = prob_map.shape
        # Binary threshold at 0.5
        binary = (prob_map > 0.5).float()
        # Closing: fill small holes in crack predictions
        kernel = torch.ones(kernel_size, kernel_size, device=prob_map.device)
        # Use max pooling for dilation, min pooling for erosion
        dilated = F.max_pool2d(binary, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        closed = -F.max_pool2d(-dilated, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        # Opening: remove isolated noise
        eroded = -F.max_pool2d(-closed, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        opened = F.max_pool2d(eroded, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        # Blend: keep original probs for confident regions, use morph for boundaries
        alpha = 0.3  # blend factor
        result = prob_map * (1 - alpha) + opened * alpha
        return result

    def _forward_tta(self, x):
        """Test-time augmentation: multi-scale + flip ensemble."""
        B, C, H, W = x.shape
        preds = []

        for scale in self.tta_scales:
            # Resize to target scale (round to multiples of 4 for compatibility with patch splitting)
            if scale != 1.0:
                new_h = max(64, int(H * scale) // 64 * 64)
                new_w = max(64, int(W * scale) // 64 * 64)
                x_scaled = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
            else:
                x_scaled = x

            # Original orientation at all scales
            out = self._forward_single(x_scaled)
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
            preds.append(torch.sigmoid(out))

            # Flip augmentations only at base scale (for speed)
            if scale == 1.0:
                # Horizontal flip
                out_hf = self._forward_single(torch.flip(x_scaled, dims=[3]).contiguous())
                out_hf = torch.flip(out_hf, dims=[3])
                out_hf = F.interpolate(out_hf, size=(H, W), mode='bilinear', align_corners=False)
                preds.append(torch.sigmoid(out_hf))



        # Average probabilities then convert back to logits
        avg_prob = torch.stack(preds).mean(dim=0)
        # Inverse sigmoid to get logits (for compatibility with BCEWithLogitsLoss)
        eps = 1e-7
        avg_prob = torch.clamp(avg_prob, eps, 1 - eps)
        avg_logit = torch.log(avg_prob / (1 - avg_prob))
        return avg_logit

    def forward(self, samples):
        if self.use_tta and not self.training:
            return self._forward_tta(samples)
        else:
            return self._forward_single(samples)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2, -1)):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims

    def forward(self, x, y):
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()

        return 1 - dc

class bce_dice(nn.Module):
    def __init__(self, args):
        super(bce_dice, self).__init__()
        self.bce_fn = nn.BCEWithLogitsLoss()
        self.dice_fn = DiceLoss()
        self.args = args

    def forward(self, y_pred, y_true):
        bce = self.bce_fn(y_pred, y_true)
        dice = self.dice_fn(y_pred.sigmoid(), y_true)
        return self.args.BCELoss_ratio * bce + self.args.DiceLoss_ratio * dice



def build_MixerCSeg(args):
    device = torch.device(args.device)
    args.device = torch.device(args.device)

    embed_dim=[16,32,64,128]

    depths = [1,1,1,1]
    state_dim=[8,8,16,16]

    backbone = VSSEncoder(
        in_dim=3,
        embed_dim=embed_dim,
        depths=depths,
        mlp_ratio=2.,
        state_dim=state_dim,
        )
    model = MixerCSeg(backbone, embed_dim, args).to(device)

    criterion = bce_dice(args)
    criterion.to(device)
    
    return model, criterion

