import os
import cv2
import numpy as np
import argparse

def linear2srgb(img):
    img = img.astype(np.float32) / 255.0
    srgb = np.where(img <= 0.0031308, img * 12.92, 1.055 * (img ** (1/2.4)) - 0.055)
    return np.clip(srgb * 255.0, 0, 255).astype(np.uint8)

def srgb2linear(img):
    img = img.astype(np.float32) / 255.0
    linear = np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)
    return np.clip(linear * 255.0, 0, 255).astype(np.uint8)

def batch_convert(input_folder, output_folder, mode):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for fname in os.listdir(input_folder):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        img_path = os.path.join(input_folder, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
        if mode == 'l2s':
            out_img = linear2srgb(img)
        elif mode == 's2l':
            out_img = srgb2linear(img)
        else:
            out_img = img.copy()
        out_path = os.path.join(output_folder, fname)
        cv2.imwrite(out_path, out_img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='批量色彩空间转换：linear <-> sRGB')
    parser.add_argument('--input', required=True, help='输入文件夹')
    parser.add_argument('--output', required=True, help='输出文件夹')
    parser.add_argument('--mode', choices=['l2s', 's2l'], required=True, help='转换模式')
    args = parser.parse_args()
    batch_convert(args.input, args.output, args.mode)