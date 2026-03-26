from i3d import *
from argparse import ArgumentParser
import os
import lpips
import numpy as np
from PIL import Image
import skimage
import torch


parser = ArgumentParser()
parser.add_argument('--root', default=None, type=str, help='root directory of videos')
parser.add_argument('--res', default='240p', choices=['240p', '480p', '2K'], help='input resolution')
args = parser.parse_args()


def calculate_scores(img1, img2, lpips_model):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    s0 = 20 * np.log10(255 / (mse ** 0.5))
    s1 = skimage.metrics.structural_similarity(img1, img2, data_range=255, channel_axis=2)
    img1_tensor = 2 * (torch.Tensor(img1) / 255).permute(2, 0, 1) - 1
    img2_tensor = 2 * (torch.Tensor(img2) / 255).permute(2, 0, 1) - 1
    s2 = lpips_model(img1_tensor, img2_tensor).item()
    return s0, s1, s2


# load pre-trained models
lpips_model = lpips.LPIPS(net='alex')
i3d_model_weight = 'weights/i3d_rgb_imagenet.pth'
i3d_model = InceptionI3d(400, in_channels=3, final_endpoint='Logits')
i3d_model.load_state_dict(torch.load(i3d_model_weight))

# memorize results
total_psnr = []
total_ssim = []
total_lpips = []
img1_i3d_lst = []
img2_i3d_lst = []
seqs = sorted(os.listdir(os.path.join(args.root, 'GTImages', args.res)))
for seq in seqs:
    psnr_lst, ssim_lst, lpips_lst, img1_lst, img2_lst = [], [], [], [], []
    init_frame = sorted(os.listdir(os.path.join(args.root, 'GTImages', args.res, seq)))[0]
    frames = sorted(os.listdir(os.path.join('outputs', args.root.split('/')[-1], seq)))

    # load data
    img1_pil_lst = []
    img2_pil_lst = []
    for i, frame in enumerate(frames):
        new_frame = '{:05d}.jpg'.format(int(init_frame[:-4]) + i)
        img1 = np.array(Image.open(os.path.join(args.root, 'GTImages', args.res, seq, new_frame)).convert('RGB'))
        img2 = np.array(Image.open(os.path.join('outputs', args.root.split('/')[-1], seq, frame)).convert('RGB'))

        # calculate psnr, ssim, and lpips scores
        s0, s1, s2 = calculate_scores(img1, img2, lpips_model)
        psnr_lst.append(s0)
        ssim_lst.append(s1)
        lpips_lst.append(s2)

        # store images for vfid calculation
        img1_pil_lst.append(Image.fromarray(img1))
        img2_pil_lst.append(Image.fromarray(img2))

    # video-level evaluation
    total_psnr.append(sum(psnr_lst) / len(frames))
    total_ssim.append(sum(ssim_lst) / len(frames))
    total_lpips.append(sum(lpips_lst) / len(frames))

    # saving i3d activations
    img1_i3d, img2_i3d = calculate_i3d_activations(img1_pil_lst, img2_pil_lst, i3d_model)
    img1_i3d_lst.append(img1_i3d)
    img2_i3d_lst.append(img2_i3d)
    print('seq: {}, psnr: {:.2f}, ssim: {:.4f}, lpips: {:.4f}'.format(seq, sum(psnr_lst) / len(frames), sum(ssim_lst) / len(frames), sum(lpips_lst) / len(frames)))

# total evaluation
vfid = calculate_vfid(img1_i3d_lst, img2_i3d_lst)
print('psnr: {:.2f}, ssim: {:.4f}, lpips: {:.4f}, vfid: {:.4f}\n'.format(sum(total_psnr) / len(seqs), sum(total_ssim) / len(seqs), sum(total_lpips) / len(seqs), vfid))
