import os
import shutil
import argparse
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob
import imageio
from imageio import get_writer
import os.path as osp
from glob import glob
import cv2

def save_video_from_images(rgb_images, video_dir, fps=30):
    assert len(rgb_images) > 0 , "image list cannot be empty"
    
    height, width, _ = rgb_images[0].shape
    os.makedirs(video_dir, exist_ok=True)
    rgb_video_path = os.path.join(video_dir, "original_rgb2.mp4")

    rgb_writer = get_writer(rgb_video_path, fps=fps)

    for rgb_img in rgb_images:
        rgb_writer.append_data(rgb_img)

    rgb_writer.close()

def read_rgbs(rgb_dir):
    rgb_paths = sorted(glob(osp.join(rgb_dir, "*.png"))) + sorted(
        glob(osp.join(rgb_dir, "*.jpg")) + sorted(glob(osp.join(rgb_dir, "*.jpeg")))
    )

    rgbs = []
    # rgb_paths = rgb_paths[::3]
    for rgb_path in rgb_paths:
        rgb_image = cv2.imread(rgb_path) # [540,960,3]
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        rgbs.append(rgb_image)
    
    return rgbs

if __name__=='__main__':
    img_dir = "current-data-dir/dynamic_stereo/dynamic_replica_data/train/1be1d3-7_obj_source_right/images"
    video_dir = "testdata/ori"
    os.makedirs(video_dir,exist_ok=True)
    
    rgb_list = read_rgbs(img_dir)

    save_video_from_images(rgb_list, video_dir)