'''
Author: Alex Wong <alexw@cs.ucla.edu>

If you use this code, please cite the following paper:

A. Wong, and S. Soatto. Unsupervised Depth Completion with Calibrated Backprojection Layers.
https://arxiv.org/pdf/2108.10531.pdf

@inproceedings{wong2021unsupervised,
  title={Unsupervised Depth Completion with Calibrated Backprojection Layers},
  author={Wong, Alex and Soatto, Stefano},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={12747--12756},
  year={2021}
}
'''
import logging
import os
import cv2
import numpy as np
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random

from .data_utils import read_paths, load_depth, DistributedSamplerNoEvenlyDivisible

def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

def preprocessing_transforms_void(mode):
    return transforms.Compose([
        ToTensor_void(mode=mode)
    ])

class ToTensor_void(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        image = sample['image']
        image = self.to_tensor(image)
        image = self.normalize(image)
        sample_path=sample['sample_path']
        depth_path=sample['depth_path']

        if self.mode == 'test':
            return {**sample,
                    'image': image}
        else:
            depth = sample['depth']
            depth = self.to_tensor(depth)
            # depth_pseudo = sample['depth_pseudo']
            # depth_pseudo = self.to_tensor(depth_pseudo)
            return {**sample,
                    'image': image, 
                    'depth': depth, 
                    # 'depth_pseudo': depth_pseudo,
                    'sample_path':sample_path,
                    'depth_path':depth_path}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img

def load_depth(depth_path):
    '''
    Load depth

    Arg(s):
        depth_path : str
            path to depth map
    Return:
        numpy[float32] : depth map (1 x H x W)
    '''

    return load_depth(depth_path, data_format='CHW')

def load_validity_map(validity_map_path):
    '''
    Load validity map

    Arg(s):
        validity_map_path : str
            path to validity map
    Returns:
        numpy[float32] : validity map (1 x H x W)
    '''

    return load_validity_map(validity_map_path, data_format='CHW')

def random_crop(inputs, shape, intrinsics=None, crop_type=['none']):
    '''
    Apply crop to inputs e.g. images, depth and if available adjust camera intrinsics

    Arg(s):
        inputs : list[numpy[float32]]
            list of numpy arrays e.g. images, depth, and validity maps
        shape : list[int]
            shape (height, width) to crop inputs
        intrinsics : numpy[float32]
            3 x 3 camera intrinsics matrix
        crop_type : str
            none, horizontal, vertical, anchored, bottom
    Return:
        list[numpy[float32]] : list of cropped inputs
        numpy[float32] : if given, 3 x 3 adjusted camera intrinsics matrix
    '''

    n_height, n_width = shape
    _, o_height, o_width = inputs[0].shape

    # Get delta of crop and original height and width
    d_height = o_height - n_height
    d_width = o_width - n_width

    # By default, perform center crop
    y_start = d_height // 2
    x_start = d_width // 2

    if 'horizontal' in crop_type:

        # Select from one of the pre-defined anchored locations
        if 'anchored' in crop_type:
            # Create anchor positions
            crop_anchors = [
                0.0, 0.50, 1.0
            ]

            widths = [
                anchor * d_width for anchor in crop_anchors
            ]
            x_start = int(widths[np.random.randint(low=0, high=len(widths))])

        # Randomly select a crop location
        else:
            x_start = np.random.randint(low=0, high=d_width)

    # If bottom alignment, then set starting height to bottom position
    if 'bottom' in crop_type:
        y_start = d_height

    elif 'vertical' in crop_type and np.random.rand() <= 0.30:

        # Select from one of the pre-defined anchored locations
        if 'anchored' in crop_type:
            # Create anchor positions
            crop_anchors = [
                0.50, 1.0
            ]

            heights = [
                anchor * d_height for anchor in crop_anchors
            ]
            y_start = int(heights[np.random.randint(low=0, high=len(heights))])

        # Randomly select a crop location
        else:
            y_start = np.random.randint(low=0, high=d_height)

    # Crop each input into (n_height, n_width)
    y_end = y_start + n_height
    x_end = x_start + n_width
    outputs = [
        T[:, y_start:y_end, x_start:x_end] for T in inputs
    ]

    if intrinsics is not None:
        # Adjust intrinsics
        intrinsics = intrinsics + [[0.0, 0.0, -x_start],
                                   [0.0, 0.0, -y_start],
                                   [0.0, 0.0, 0.0     ]]

        return outputs, intrinsics
    else:
        return outputs

class VoidDataLoader(object):
    def __init__(self, args, mode):
        if mode == 'train':
            self.training_samples = KBNetTrainingDataset(args)
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

        elif mode == 'online_eval':
            self.testing_samples = KBNetInferenceDataset(args)
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

        elif mode == 'test':
            logging.info('void mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))

class KBNetTrainingDataset(Dataset):
    '''
    Dataset for fetching:
        (1) image at time t-1, t, and t+1
        (2) sparse depth
        (3) camera intrinsics matrix

    Arg(s):
        image_paths : list[str]
            paths to image triplets
        sparse_depth_paths : list[str]
            paths to sparse depth maps
        intrinsics_paths : list[str]
            paths to 3 x 3 camera intrinsics matrix
        shape : tuple[int]
            shape (height, width) to crop inputs
        random_crop_type : list[str]
            none, horizontal, vertical, anchored, bottom
    '''

    def __init__(self,args):

        self.args = args
        self.image_paths = read_paths(args.data_path, args.train_image_path_void)
        self.depth_gt_paths = read_paths(args.data_path, args.train_ground_truth_path_void)
        self.pseudo_depth_path = getattr(args, 'pseudo_depth_path', None)
        self.pseudo_dataset_path = getattr(args, 'pseudo_dataset_path', None)
        self.pseudo_depth_model = args.depth_model

        self.transform=preprocessing_transforms_void('train')
        self.data_path_len=len(args.data_path)

    def __getitem__(self, index):
        # Load image
        image = Image.open(self.image_paths[index])
        depth_gt = Image.open(self.depth_gt_paths[index])

        if self.args.do_random_rotate and (self.args.aug):
            random_angle = (random.random() - 0.5) * 2 * self.args.degree
            image = self.rotate_image(image, random_angle)
            depth_gt = self.rotate_image(
                depth_gt, random_angle, flag=Image.NEAREST)
                        
        image = np.asarray(image, dtype=np.float32) / 255.0
        depth_gt = np.asarray(depth_gt, dtype=np.float32) / 256.0
        depth_gt = np.expand_dims(depth_gt, axis=2)
        
        if self.args.aug and (self.args.random_crop):
            image, depth_gt = self.random_crop(
                image, depth_gt, self.args.input_height, self.args.input_width)
        
        if self.args.aug and self.args.random_translate:
            # print("Random Translation!")
            image, depth_gt = self.random_translate(image, depth_gt, self.args.max_translation)

        image, depth_gt = self.train_preprocess(image, depth_gt)
        
        # mask = np.logical_and(depth_gt > 0, depth_gt <= self.args.max_depth_eval).squeeze()[None, ...]
        # pseudo_path = os.path.join(self.pseudo_depth_path,
        #                            self.pseudo_depth_model,
        #                            self.pseudo_dataset_path,
        #                            self.depth_gt_paths[index][43:])
        
        # depth_pseudo = np.load(pseudo_path[:-4] + '.npy')
        # depth_pseudo = np.expand_dims(depth_pseudo, axis=2)

        sample = {'image': image, 
                  'depth': depth_gt,
                #   'mask': mask,
                #   'depth_pseudo': depth_pseudo,
                  'sample_path':self.image_paths[index][self.data_path_len:],
                  'depth_path':self.depth_gt_paths[index][self.data_path_len:]}

        sample['dataset'] = self.args.dataset
        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth
    
    def random_translate(self, img, depth, max_t=20):
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        p = self.config.translate_prob
        do_translate = random.random()
        if do_translate > p:
            return img, depth
        x = random.randint(-max_t, max_t)
        y = random.randint(-max_t, max_t)
        M = np.float32([[1, 0, x], [0, 1, y]])
        # print(img.shape, depth.shape)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        depth = cv2.warpAffine(depth, M, (depth.shape[1], depth.shape[0]))
        depth = depth.squeeze()[..., None]  # add channel dim back. Affine warp removes it
        # print("after", img.shape, depth.shape)
        return img, depth
    
    def train_preprocess(self, image, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()

        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        return image, depth_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.args.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug
    
    def __len__(self):
        return len(self.image_paths)


class KBNetInferenceDataset(Dataset):
    '''
    Dataset for fetching:
        (1) image
        (2) sparse depth
        (3) camera intrinsics matrix

    Arg(s):
        image_paths : list[str]
            paths to image triplets
        sparse_depth_paths : list[str]
            paths to sparse depth maps
        intrinsics_paths : list[str]
            paths to 3 x 3 camera intrinsics matrix
    '''

    def __init__(self,args):

        self.args = args
        self.image_paths = read_paths(args.data_path, args.val_image_path_void)
        self.depth_gt_paths = read_paths(args.data_path, args.val_ground_truth_path_void)
        self.pseudo_depth_path = getattr(args, 'pseudo_depth_path', None)
        self.pseudo_dataset_path = getattr(args, 'pseudo_dataset_path', None)
        self.pseudo_depth_model = args.depth_model
        
        self.use_offline_pseudo_depth = self.pseudo_depth_path is not None

        self.transform=preprocessing_transforms_void('eval')
        self.data_path_len=len(args.data_path)

    def __getitem__(self, index):
        # Load image
        image = Image.open(self.image_paths[index])
        image = np.asarray(image, dtype=np.float32) / 255.0

        # image1 = data_utils.load_image(
        #         self.image_paths[index],
        #         normalize=False,
        #         data_format='CHW')
        # image1 = image1.astype(np.float32)

        depth_gt = Image.open(self.depth_gt_paths[index])
        depth_gt = np.asarray(depth_gt, dtype=np.float32) / 256.0
        depth_gt = np.expand_dims(depth_gt, axis=2)

        # mask = np.logical_and(depth_gt > 0, depth_gt <= self.args.max_depth_eval).squeeze()[None, ...]
        # pseudo_path = os.path.join(self.pseudo_depth_path,
        #                            self.pseudo_depth_model,
        #                            self.pseudo_dataset_path,
        #                            self.depth_gt_paths[index][self.data_path_len:])
        
        # depth_pseudo = np.load(pseudo_path[:-4] + '.npy')
        # depth_pseudo = np.expand_dims(depth_pseudo, axis=2)

        sample = {'image': image, 
                  'depth': depth_gt,
                #   'mask': mask,
                #   'depth_pseudo': depth_pseudo,
                  'sample_path':self.image_paths[index][self.data_path_len:],
                  'depth_path':self.depth_gt_paths[index][self.data_path_len:]}

        sample['dataset'] = self.args.dataset
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.image_paths)
