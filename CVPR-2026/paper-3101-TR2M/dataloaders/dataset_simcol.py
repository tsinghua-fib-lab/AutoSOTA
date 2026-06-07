import logging
import glob
import os
import random
import numpy as np
from PIL import Image  # using pillow-simd for increased speed
from PIL import ImageFile
import cv2

import torch
import torch.utils.data as data
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from .data_utils import DistributedSamplerNoEvenlyDivisible

ImageFile.LOAD_TRUNCATED_IMAGES=True

def change_to_simcol(args):
    args.dataset = "simcol"

    args.input_height = 480
    args.input_width = 640
    args.max_depth_eval = 100
    args.min_depth_eval = 0.1

    if getattr(args, "simcol_root", None) is None:
        print("[skip] --simcol_root not set, skipping simcol.")

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class SimCol(Dataset):
    """Superclass for monocular dataloaders

    Args:
        args, mode
    """
    def __init__(self, args):
        super(SimCol, self).__init__()

        self.args = args
        self.data_dir_root = args.simcol_root
        self.height = args.input_height
        self.width = args.input_width
        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.rescale_factor = 200 / (255 * 256)

        self.resize = transforms.Resize((self.height, self.width))
        self.file_dir = os.path.join(self.data_dir_root, 'misc', 'test_file.txt')
        self.filename = readlines(self.file_dir)

        self.scans = []
        for i, video_file in enumerate(self.filename):
            sequence, frame_base = os.path.split(video_file[:-1])
            image_paths = os.path.join(self.data_dir_root, sequence, frame_base, "FrameBuffer*.png")
            seq_image_paths = glob.glob(image_paths)
            seq_image_paths.sort()
            # seq_image_paths = seq_image_paths[::20]

            for j, seq_image_path in enumerate(seq_image_paths):
                filename = os.path.basename(seq_image_path)
                frame_index = int(filename[12:16])
                seq_depth_path = os.path.join(self.data_dir_root, sequence, frame_base, "Depth_{:04d}.png".format(frame_index))

                if os.path.exists(seq_image_path) and os.path.exists(seq_depth_path):
                    self.scans.append(
                        {
                            "seq_index": i,
                            "video_file": video_file[:-1], 
                            "sequence": sequence,
                            "keyframe": frame_base,
                            "frame_index": frame_index,
                        })

        a = 1

    def __len__(self):
        return len(self.scans)
    
    def get_color(self, video_base, frame_index, do_flip):
        color_path = os.path.join(self.data_dir_root, video_base, "FrameBuffer_{:04d}.png".format(frame_index))
        color = self.loader(color_path)
        
        if do_flip:
            color = color.transpose(Image.FLIP_LEFT_RIGHT)

        return color

    def get_depth(self, video_base, frame_index, do_flip):
        depth_path = os.path.join(self.data_dir_root, video_base, "Depth_{:04d}.png".format(frame_index))
        depth_gt = cv2.imread(depth_path, 3)
        depth_gt = np.array(depth_gt[:, :, 0])
        depth_gt = depth_gt.astype(np.float32, order='C') * self.rescale_factor
        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
            
    def __getitem__(self, index):

        scan = self.scans[index]
        video_file = scan["video_file"]
        frame_index = scan["frame_index"]

        image = self.get_color(video_file, frame_index, False)

        image = self.resize(image)
        image = self.to_tensor(image) 
        image = self.normalize(image)

        depth_gt = self.get_depth(video_file, frame_index, False)
        depth_gt = np.expand_dims(depth_gt, 0)
        depth_gt = torch.from_numpy(depth_gt.astype(np.float32))

        sample = {'image': image, 
                  'depth': depth_gt,
                  'text': 'The image shows a medical scene of a cecum with texture type one.',
                  'image_path': os.path.join(video_file, "FrameBuffer_{:04d}.png".format(frame_index)),
                  'dataset': self.args.dataset}

        return sample