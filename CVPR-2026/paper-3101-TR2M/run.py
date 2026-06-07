"""TR2M single-image demo.

Loads one RGB image plus a text description, runs DepthAnything (relative depth),
DINOv2 (image features), CLIP (text features) and the TR2M ScaleMap head to predict
per-pixel scale/shift maps, and writes a 1x5 visualisation to disk.

Example:
    python run.py --image figures/sample.jpg --text "a classroom scene"
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from CLIP import clip
from depth_anything.dpt import DepthAnything
from scalemap_depth import ScaleMapModel


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Defaults match the eval-time NYU resolution (must be multiples of 14)
DEFAULT_H = 434
DEFAULT_W = 560

DEPTH_MODEL_CONFIGS = {
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
}


def parse_args():
    p = argparse.ArgumentParser(description="TR2M single-image demo")
    p.add_argument('--image', required=True, help='Path to input RGB image.')
    p.add_argument('--text', required=True, help='Text description of the scene.')
    p.add_argument('--weight', default='weights/da_s_vitl_vitl.pth',
                   help='Path to the trained TR2M ScaleMap weight.')
    p.add_argument('--output', default='output.png',
                   help='Where to save the 1x5 visualisation.')
    p.add_argument('--depth_model', default='da_s', choices=['da_s', 'da_b', 'da_l'],
                   help='DepthAnything backbone size.')
    p.add_argument('--text_encoder', default='ViT-L/14',
                   help='CLIP variant for text encoding.')
    p.add_argument('--img_encoder', default='vitl', choices=['vits', 'vitb', 'vitl'],
                   help='DINOv2 size for image features.')
    p.add_argument('--input_height', type=int, default=DEFAULT_H,
                   help='Resize height fed to the model (must be a multiple of 14).')
    p.add_argument('--input_width', type=int, default=DEFAULT_W,
                   help='Resize width fed to the model (must be a multiple of 14).')
    p.add_argument('--vis_low_percentile', type=float, default=1.0,
                   help='Lower percentile for relative-depth colour range.')
    p.add_argument('--vis_high_percentile', type=float, default=95.0,
                   help='Upper percentile for relative-depth colour range.')
    p.add_argument('--max_depth', type=float, default=10.0,
                   help='Upper bound (metres) for metric-depth colour mapping.')
    p.add_argument('--min_depth', type=float, default=1e-3,
                   help='Lower bound (metres) for metric-depth colour mapping.')
    p.add_argument('--device', default='cuda',
                   help='torch device.')
    # ScaleMap hyperparams (must match training)
    p.add_argument('--image_channels', type=int, default=1024)
    p.add_argument('--text_channels',  type=int, default=768)
    p.add_argument('--out_channels',   type=int, default=192)
    p.add_argument('--features',       type=int, default=128)
    p.add_argument('--cross_attn_nhead', type=int, default=8)
    p.add_argument('--coef_scale', type=float, default=0.01)
    p.add_argument('--coef_shift', type=float, default=0.01)
    return p.parse_args()


def load_image(path, height, width, device):
    """Load an RGB image, return (display_uint8 HxWx3, normalised tensor 1x3xHxW)."""
    img_pil = Image.open(path).convert('RGB')
    orig_w, orig_h = img_pil.size  # PIL gives (W, H)

    display = np.array(img_pil)  # uint8, HxWx3, for the leftmost panel

    tfm = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),  # -> [0, 1] float, CxHxW
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tensor = tfm(img_pil).unsqueeze(0).to(device)
    return display, tensor, (orig_h, orig_w)


def build_models(args, device):
    print(f"== Building DepthAnything backbone: {args.depth_model}")
    if not args.depth_model.startswith('da'):
        raise ValueError(f"Unsupported depth_model: {args.depth_model}")
    encoder = 'vit' + args.depth_model[-1]
    depth_model = DepthAnything(DEPTH_MODEL_CONFIGS[encoder])
    da_weight = f'./depth_anything/depth_anything_{encoder}14.pth'
    if not os.path.exists(da_weight):
        raise FileNotFoundError(
            f"DepthAnything weight not found at {da_weight}. "
            f"Download it from https://github.com/LiheYoung/Depth-Anything"
        )
    depth_model.load_state_dict(torch.load(da_weight, map_location='cpu'))
    depth_model.to(device).eval()

    print(f"== Loading CLIP text encoder: {args.text_encoder}")
    clip_model, _ = clip.load(args.text_encoder, device=device)
    clip_model.eval()

    print(f"== Loading DINOv2 image encoder: {args.img_encoder}")
    image_f_model = torch.hub.load('facebookresearch/dinov2',
                                   f'dinov2_{args.img_encoder}14').to(device)
    image_f_model.eval()

    print("== Building TR2M ScaleMap head")
    scalemap = ScaleMapModel(
        args.image_channels, args.out_channels, args.text_channels,
        args.features, args.cross_attn_nhead, args.coef_scale, args.coef_shift,
    ).to(device).eval()

    print(f"== Loading TR2M weight: {args.weight}")
    ckpt = torch.load(args.weight, map_location='cpu')
    state_dict = (ckpt["state_dict"]
                  if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt)
    scalemap.load_state_dict(state_dict)

    return depth_model, clip_model, image_f_model, scalemap


@torch.no_grad()
def run_pipeline(args, device, image_tensor, text):
    depth_model, clip_model, image_f_model, scalemap = build_models(args, device)

    H, W = args.input_height, args.input_width
    if H % 14 != 0 or W % 14 != 0:
        raise ValueError(f"input_height ({H}) and input_width ({W}) must be multiples of 14.")

    # Text features
    text_tokens = clip.tokenize([text], truncate=True).to(device)
    text_features = clip_model.encode_text(text_tokens).unsqueeze(1)  # [1, 1, D_text]

    # Image features (DINOv2 patch tokens)
    image_features = image_f_model.get_intermediate_layers(
        image_tensor, n=1, return_class_token=False
    )[0]  # [1, P, D_img]

    # Scale + shift maps from TR2M
    scale_pred, shift_pred, _, _ = scalemap(
        image_features.float(),
        text_features.float(),
        patch_h=H // 14,
        patch_w=W // 14,
    )  # each [1, 1, H, W]

    # Relative depth
    relative_depth = depth_model(image_tensor).unsqueeze(1)  # [1, 1, H_da, W_da]
    if relative_depth.shape[-2:] != scale_pred.shape[-2:]:
        relative_depth = F.interpolate(
            relative_depth, size=scale_pred.shape[-2:], mode='bilinear', align_corners=True,
        )

    # Metric depth via the TR2M formula
    metric_depth = 1.0 / (scale_pred * relative_depth + shift_pred)

    return scale_pred, shift_pred, relative_depth, metric_depth


def upsample_to(tensor, size):
    """Upsample a [1,1,h,w] tensor to (H, W) and return a HxW float numpy array."""
    out = F.interpolate(tensor, size=size, mode='bilinear', align_corners=True)
    return out.squeeze().detach().cpu().numpy()


def percentile_range(arr, low_pct, high_pct):
    """Return (vmin, vmax) at the given percentiles, with a degenerate-range guard."""
    vmin = float(np.percentile(arr, low_pct))
    vmax = float(np.percentile(arr, high_pct))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def save_panel(rgb, relative, scale_map, shift_map, metric, low_pct, high_pct, max_depth, min_depth, out_path, text=''):
    fig, axes = plt.subplots(1, 6, figsize=(26, 4.5))

    # Text panel — wrap at ~30 chars per line for readability
    import textwrap
    wrapped = '\n'.join(textwrap.wrap(text, width=30))
    axes[0].set_facecolor('#f5f5f5')
    axes[0].text(0.5, 0.5, wrapped, ha='center', va='center',
                 fontsize=9, transform=axes[0].transAxes,
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#cccccc'))
    axes[0].set_title('Text description')
    axes[0].axis('off')

    axes[1].imshow(rgb)
    axes[1].set_title('Input image')
    axes[1].axis('off')

    rel_vmin, rel_vmax = percentile_range(relative, low_pct, high_pct)
    im1 = axes[2].imshow(relative, cmap='magma', vmin=rel_vmin, vmax=rel_vmax)
    axes[2].set_title('Relative depth')
    axes[2].axis('off')
    fig.colorbar(im1, ax=axes[2], fraction=0.046, pad=0.04)

    im2 = axes[3].imshow(scale_map, cmap='hot')
    axes[3].set_title(f'Scale map\n{scale_map.mean():.4f} ± {scale_map.std():.4f}')
    axes[3].axis('off')
    fig.colorbar(im2, ax=axes[3], fraction=0.046, pad=0.04)

    im3 = axes[4].imshow(shift_map, cmap='hot')
    axes[4].set_title(f'Shift map\n{shift_map.mean():.4f} ± {shift_map.std():.4f}')
    axes[4].axis('off')
    fig.colorbar(im3, ax=axes[4], fraction=0.046, pad=0.04)

    metric_clipped = np.clip(metric, min_depth, max_depth)
    im4 = axes[5].imshow(metric_clipped, cmap='magma', vmin=min_depth, vmax=max_depth)
    axes[5].set_title(f'Metric depth (m)')
    axes[5].axis('off')
    fig.colorbar(im4, ax=axes[5], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    if device.type != args.device:
        print(f"[warn] CUDA unavailable, falling back to CPU.")

    if device.type == 'cuda':
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)

    rgb_uint8, image_tensor, (orig_h, orig_w) = load_image(
        args.image, args.input_height, args.input_width, device
    )

    scale_t, shift_t, rel_t, metric_t = run_pipeline(args, device, image_tensor, args.text)

    # Resample everything to the original image resolution so the panels line up
    scale_np    = upsample_to(scale_t,  (orig_h, orig_w))
    shift_np    = upsample_to(shift_t,  (orig_h, orig_w))
    relative_np = upsample_to(rel_t,    (orig_h, orig_w))
    metric_np   = upsample_to(metric_t, (orig_h, orig_w))

    print(f"\nText: {args.text!r}")
    print(f"Predicted metric depth: min={metric_np.min():.3f} m, "
          f"max={metric_np.max():.3f} m, mean={metric_np.mean():.3f} m")

    save_panel(rgb_uint8, relative_np, scale_np, shift_np, metric_np,
               args.vis_low_percentile, args.vis_high_percentile,
               args.max_depth, args.min_depth, args.output, text=args.text)
    print(f"Saved visualisation to {args.output}")


if __name__ == '__main__':
    main()
