from __future__ import absolute_import, division, print_function
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

def change_to_c3vd(args):
    args.dataset = "c3vd"
    if getattr(args, "c3vd_root", None) is None:
        print("[skip] --c3vd_root not set, skipping c3vd.")
        return
    args.data_path = args.c3vd_root
    args.input_height = 480
    args.input_width = 640
    args.max_depth_eval = 100
    args.min_depth_eval = 0.1

    args.lambdad = 1.0

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class C3VDDataLoader(object):
    def __init__(self, args, mode):
        if mode == 'train':
            self.training_samples = C3VDDataset(args, mode)
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples)
            else:
                self.train_sampler = None

            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=args.num_threads,
                                   pin_memory=False,
                                   persistent_workers=True,
                                   sampler=self.train_sampler)
            logging.info('prepared dataset {} for {} with {} sample.'.format(args.dataset, mode, len(self.training_samples)))

        elif mode == 'online_eval' or mode == 'online_eval_all':
            self.testing_samples = C3VDDataset(args, mode)
            if args.distributed:
                self.eval_sampler = DistributedSamplerNoEvenlyDivisible(self.testing_samples, shuffle=False)
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=False,
                                   persistent_workers=True,
                                   sampler=self.eval_sampler)
            logging.info('prepared dataset {} for {} with {} sample.'.format(args.dataset, mode, len(self.testing_samples)))
            # print('prepared dataset {} for {} with {} sample.'.format(args.dataset, mode, len(self.testing_samples)))
        elif mode == 'test':
            self.testing_samples = C3VDDataset(args, mode)
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)
            logging.info('prepared dataset {} for {} with {} sample.'.format(args.dataset, mode, len(self.testing_samples)))
        else:
            logging.info('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))

class C3VDDataset(Dataset):
    """Superclass for monocular dataloaders

    Args:
        args, mode
    """
    def __init__(self, args, mode='train'):
        super(C3VDDataset, self).__init__()

        self.args = args
        self.data_path = args.data_path
        self.height = args.input_height
        self.width = args.input_width
        self.is_train = True if mode == 'train' else False
        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.rescale_factor = 100 / 65535

        self.colon_types = {
            "cecum": "cecum",
            "desc": "descending colon",
            "sigmoid": "sigmoid colon",
            "trans": "transcending colon",
        }
        self.texture_types_cap = {
            "1" : " texture type one",
            "2" : " texture type two",
            "3" : " texture type three",
            "4" : " texture type four",
        }

        if mode == 'train':
            self.valid_files = np.array([1, 3, 4, 6, 7, 8, 10, 11, 13, 14, 15, 17, 18, 20, 22])
        else:
            self.valid_files = np.array([0, 2, 5, 9, 12, 16, 19, 21])

        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.transforms.ColorJitter(self.brightness,self.contrast,self.saturation,self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = transforms.Resize((self.height, self.width))
        self.scans = []
        self.video_files_all = [os.path.join(self.data_path, file) for file in os.listdir(self.data_path)]
        self.video_files_all.sort()
        self.video_files = [self.video_files_all[i] for i in self.valid_files]

            
        self.sequence_len = np.zeros([len(self.video_files)])
        for i, video_file in enumerate(self.video_files):
            image_paths = os.path.join(video_file, "*_color.png")
            video_base = os.path.basename(video_file)
            colon_type = self.colon_types[video_base.split('_')[0]]
            texture_type = self.texture_types_cap[video_base.split('_')[1][1]]
            description = f"The image shows a medical scene of {colon_type} with {texture_type}."
            seq_image_paths = glob.glob(image_paths)
            seq_image_paths.sort()

            if mode == 'online_eval':
                seq_image_paths = seq_image_paths[::20]
            for j, seq_image_path in enumerate(seq_image_paths):
                filename = os.path.basename(seq_image_path)
                frame_index = int(filename[:4])
                seq_depth_path = os.path.join(self.data_path, video_base, "{:04d}_depth.tiff".format(frame_index))
                if os.path.exists(seq_image_path) and os.path.exists(seq_depth_path):
                    self.scans.append(
                        {
                            "seq_index": i,
                            "sequence": video_base,
                            "frame_index": frame_index,
                            "description": description,
                            "length": len(seq_image_paths),
                        })
        # print("Prepared C3VD dataset with %d sets of RGB, depth, normal and occlusion images." % (len(self.scans)))

    def __len__(self):
        return len(self.scans)
    
    def get_color(self, video_base, frame_index, do_flip):
        color_path = os.path.join(self.data_path, video_base, "{:04d}_color.png".format(frame_index))
        color = self.loader(color_path)
        
        if do_flip:
            color = color.transpose(Image.FLIP_LEFT_RIGHT)

        return color

    def get_depth(self, video_base, frame_index, do_flip):
        depth_path = os.path.join(self.data_path, video_base, "{:04d}_depth.tiff".format(frame_index))
        depth_gt = cv2.imread(depth_path, 3)
        depth_gt = cv2.resize(depth_gt,(self.width,self.height), interpolation=cv2.INTER_NEAREST)
        depth_gt = np.array(depth_gt[:, :, 0])
        depth_gt = depth_gt.astype(np.float32, order='C') * self.rescale_factor
        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
            
    def __getitem__(self, index):

        scan = self.scans[index]
        video_base = scan["sequence"]
        frame_index = scan["frame_index"]
        
        inputs = {}
        
        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        image = self.get_color(video_base, frame_index, do_flip)


        if do_color_aug:
            color_aug = transforms.ColorJitter(self.brightness,self.contrast,self.saturation,self.hue)
        else:
            color_aug = (lambda x: x)

        image = self.resize(image)
        image = color_aug(image)
        image = self.to_tensor(image) 
        image = self.normalize(image)

        depth_gt = self.get_depth(video_base, frame_index, do_flip)
        depth_gt = np.expand_dims(depth_gt, 0)
        depth_gt = torch.from_numpy(depth_gt.astype(np.float32))

        sample_path = os.path.join(video_base, "{:04d}_color.png".format(frame_index))
        sample = {'image': image, 
                  'depth': depth_gt,
                  'text': scan["description"],
                  'dataset': self.args.dataset,
                  'sample_path': sample_path}

        return sample
    
    
if __name__ == "__main__":
    ds = C3VDDataset(data_path='/mnt/data-hdd2/Beilei/Dataset/C3VD', height=256, width=320, frame_idxs=[0, -1, 1], num_scales=4, is_train=True)
    
    test = ds[0]