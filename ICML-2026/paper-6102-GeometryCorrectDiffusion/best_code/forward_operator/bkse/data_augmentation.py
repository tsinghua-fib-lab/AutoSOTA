import argparse
import logging
import os
import os.path as osp
import random

import cv2
import data.util as data_util
import lmdb
import numpy as np
import torch
import utils.util as util
import yaml
from models.kernel_encoding.kernel_wizard import KernelWizard


def read_image(env, key, x, y, h, w):
    img = data_util.read_img(env, key, (3, 720, 1280))
    img = np.transpose(img[x : x + h, y : y + w, [2, 1, 0]], (2, 0, 1))
    return img


def main():
    device = torch.device("cuda")

    parser = argparse.ArgumentParser(description="Kernel extractor testing")

    parser.add_argument("--source_H", action="store", help="source image height", type=int, required=True)
    parser.add_argument("--source_W", action="store", help="source image width", type=int, required=True)
    parser.add_argument("--target_H", action="store", help="target image height", type=int, required=True)
    parser.add_argument("--target_W", action="store", help="target image width", type=int, required=True)
    parser.add_argument(
        "--augmented_H", action="store", help="desired height of the augmented images", type=int, required=True
    )
    parser.add_argument(
        "--augmented_W", action="store", help="desired width of the augmented images", type=int, required=True
    )

    parser.add_argument(
        "--source_LQ_root", action="store", help="source low-quality dataroot", type=str, required=True
    )
    parser.add_argument(
        "--source_HQ_root", action="store", help="source high-quality dataroot", type=str, required=True
    )
    parser.add_argument(
        "--target_HQ_root", action="store", help="target high-quality dataroot", type=str, required=True
    )
    parser.add_argument("--save_path", action="store", help="save path", type=str, required=True)
    parser.add_argument("--yml_path", action="store", help="yml path", type=str, required=True)
    parser.add_argument(
        "--num_images", action="store", help="number of desire augmented images", type=int, required=True
    )

    args = parser.parse_args()

    source_LQ_root = args.source_LQ_root
    source_HQ_root = args.source_HQ_root
    target_HQ_root = args.target_HQ_root

    save_path = args.save_path
    source_H, source_W = args.source_H, args.source_W
    target_H, target_W = args.target_H, args.target_W
    augmented_H, augmented_W = args.augmented_H, args.augmented_W
    yml_path = args.yml_path
    num_images = args.num_images

    # Initializing logger
    logger = logging.getLogger("base")
    os.makedirs(save_path, exist_ok=True)
    util.setup_logger("base", save_path, "test", level=logging.INFO, screen=True, tofile=True)
    logger.info("source LQ root: {}".format(source_LQ_root))
    logger.info("source HQ root: {}".format(source_HQ_root))
    logger.info("target HQ root: {}".format(target_HQ_root))
    logger.info("augmented height: {}".format(augmented_H))
    logger.info("augmented width: {}".format(augmented_W))
    logger.info("Number of augmented images: {}".format(num_images))

    # Initializing mode
    logger.info("Loading model...")
    with open(yml_path, "r") as f:
        print(yml_path)
        opt = yaml.load(f)["KernelWizard"]
    model_path = opt["pretrained"]
    model = KernelWizard(opt)
    model.eval()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    logger.info("Done")

    # processing data
    source_HQ_env = lmdb.open(source_HQ_root, readonly=True, lock=False, readahead=False, meminit=False)
    source_LQ_env = lmdb.open(source_LQ_root, readonly=True, lock=False, readahead=False, meminit=False)
    target_HQ_env = lmdb.open(target_HQ_root, readonly=True, lock=False, readahead=False, meminit=False)
    paths_source_HQ, _ = data_util.get_image_paths("lmdb", source_HQ_root)
    paths_target_HQ, _ = data_util.get_image_paths("lmdb", target_HQ_root)

    psnr_avg = 0

    for i in range(num_images):
        source_key = np.random.choice(paths_source_HQ)
        target_key = np.random.choice(paths_target_HQ)

        source_rnd_h = random.randint(0, max(0, source_H - augmented_H))
        source_rnd_w = random.randint(0, max(0, source_W - augmented_W))
        target_rnd_h = random.randint(0, max(0, target_H - augmented_H))
        target_rnd_w = random.randint(0, max(0, target_W - augmented_W))

        source_LQ = read_image(source_LQ_env, source_key, source_rnd_h, source_rnd_w, augmented_H, augmented_W)
        source_HQ = read_image(source_HQ_env, source_key, source_rnd_h, source_rnd_w, augmented_H, augmented_W)
        target_HQ = read_image(target_HQ_env, target_key, target_rnd_h, target_rnd_w, augmented_H, augmented_W)

        source_LQ = torch.Tensor(source_LQ).unsqueeze(0).to(device)
        source_HQ = torch.Tensor(source_HQ).unsqueeze(0).to(device)
        target_HQ = torch.Tensor(target_HQ).unsqueeze(0).to(device)

        with torch.no_grad():
            kernel_mean, kernel_sigma = model(source_HQ, source_LQ)
            kernel = kernel_mean + kernel_sigma * torch.randn_like(kernel_mean)
            fake_source_LQ = model.adaptKernel(source_HQ, kernel)
            target_LQ = model.adaptKernel(target_HQ, kernel)

        LQ_img = util.tensor2img(source_LQ)
        fake_LQ_img = util.tensor2img(fake_source_LQ)
        target_LQ_img = util.tensor2img(target_LQ)
        target_HQ_img = util.tensor2img(target_HQ)

        target_HQ_dst = osp.join(save_path, "sharp/{:03d}/{:08d}.png".format(i // 100, i % 100))
        target_LQ_dst = osp.join(save_path, "blur/{:03d}/{:08d}.png".format(i // 100, i % 100))

        os.makedirs(osp.dirname(target_HQ_dst), exist_ok=True)
        os.makedirs(osp.dirname(target_LQ_dst), exist_ok=True)

        cv2.imwrite(target_HQ_dst, target_HQ_img)
        cv2.imwrite(target_LQ_dst, target_LQ_img)
        # torch.save(kernel, osp.join(osp.dirname(target_LQ_dst), f'kernel{i:03d}.pth'))

        psnr = util.calculate_psnr(LQ_img, fake_LQ_img)

        logger.info("Reconstruction PSNR of image #{:03d}/{:03d}: {:.2f}db".format(i, num_images, psnr))
        psnr_avg += psnr

    logger.info("Average reconstruction PSNR: {:.2f}db".format(psnr_avg / num_images))


main()
