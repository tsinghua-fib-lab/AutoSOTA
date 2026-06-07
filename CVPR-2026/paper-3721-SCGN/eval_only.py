"""
Evaluation-only script for SCGN.
Loads pre-trained weights and computes PSNR, SSIM, IoU on test set.
"""
import sys
import os
import torch
import torch.nn as nn
import torchvision
from math import log10

from convLast_std import Net

model_name = 'convLast_std_tem_data4'

def calc_psnr(sr, hr):
    sr, hr = sr.double(), hr.double()
    diff = (sr - hr) / 255.00
    mse = diff.pow(2).mean()
    if mse == 0:
        return float('inf')
    psnr = -10 * log10(mse)
    return float(psnr)

def calc_ssim(sr, hr):
    from pytorch_msssim import ssim
    ssim_val = ssim(sr, hr, size_average=True)
    return float(ssim_val)

def calc_iou(sr, hr, threshold=127.5):
    sr = sr.float()
    hr = hr.float()
    sr_binary = (sr > threshold).float()
    hr_binary = (hr > threshold).float()
    intersection = (sr_binary * hr_binary).sum()
    union = sr_binary.sum() + hr_binary.sum() - intersection
    if union == 0:
        return 0.0
    return float(intersection / union)

def main():
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True

    # Load model
    model = nn.DataParallel(Net()).to(device)
    weight_path = model_name + "_100.pth"
    if not os.path.exists(weight_path):
        print(f"ERROR: weight file {weight_path} not found!")
        sys.exit(1)

    weights = torch.load(weight_path)
    model.load_state_dict(weights['model_state_dict'])
    model.eval()
    torch.set_grad_enabled(False)
    print(f"Loaded weights from {weight_path}")

    # Save denoised images
    result_dir = model_name + '_result'
    os.makedirs(result_dir, exist_ok=True)

    test_data = 'tem_test_data4'
    for i in range(100):
        name = str(i) + '.png'
        in_path = os.path.join(test_data, 'noisy', name)
        out_path = os.path.join(result_dir, name)
        in_img = torchvision.io.read_image(in_path).cuda()
        if in_img.shape[0] > 1:
            in_img = in_img[:1]
        in_img = torch.unsqueeze(in_img, 0).float()
        out_img = model(in_img)
        out_img = torch.clip_(out_img, 0, 255)
        out_img = torch.squeeze(out_img, 0).byte()
        torchvision.io.write_png(out_img.cpu(), out_path)

    print(f"Saved {100} denoised images to {result_dir}")

    # Compute metrics
    gt_dir = os.path.join(test_data, 'gt')
    path = os.listdir(gt_dir)
    psnr_sum, ssim_sum, iou_sum = 0.0, 0.0, 0.0
    for name in path:
        in_path = os.path.join(gt_dir, name)
        out_path = os.path.join(result_dir, name)
        in_img = torchvision.io.read_image(in_path).float()
        out_img = torchvision.io.read_image(out_path).float()
        if in_img.shape[0] > 1:
            in_img = in_img[:1]
        if out_img.shape[0] > 1:
            out_img = out_img[:1]
        in_img = torch.unsqueeze(in_img, 0)
        out_img = torch.unsqueeze(out_img, 0)
        psnr_sum += calc_psnr(in_img, out_img)
        ssim_sum += calc_ssim(in_img, out_img)
        iou_sum += calc_iou(in_img, out_img)

    n = len(path)
    avg_psnr = psnr_sum / n
    avg_ssim = ssim_sum / n
    avg_iou = iou_sum / n
    print(f"Results for {n} test images:")
    print(f"PSNR: {avg_psnr:.4f}")
    print(f"SSIM: {avg_ssim:.4f}")
    print(f"IoU: {avg_iou:.4f}")

if __name__ == "__main__":
    main()
