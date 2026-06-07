import copy
import logging
import random

import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
from torchvision import transforms

import numpy as np
from PIL import Image
import os
import itertools
import cv2

from .data_utils import DistributedSamplerNoEvenlyDivisible
from .datasets_void import VoidDataLoader
from .dataset_c3vd import C3VDDataLoader, change_to_c3vd
from types import SimpleNamespace

def _require(args, attr, dataset):
    val = getattr(args, attr, None)
    if val is None:
        print(f"[skip] --{attr} not set, skipping {dataset}.")
    return val


def change_to_kitti(args):
    args.dataset = "kitti"
    kitti_root = _require(args, "kitti_root", "kitti").rstrip("/") + "/"
    kitti_gt_root = _require(args, "kitti_gt_root", "kitti").rstrip("/") + "/"
    args.data_path = kitti_root
    args.txt_path = "./text/text_all/kitti/train"
    args.gt_path = kitti_gt_root
    args.filenames_file = "data_splits/kitti/eigen_train_files_with_gt.txt"
    args.input_height = 352
    args.input_width = 1216
    args.do_kb_crop = True
    args.data_path_eval = kitti_root
    args.txt_path_eval = "./text/text_all/kitti/test"
    args.gt_path_eval = kitti_gt_root
    args.filenames_file_eval = "data_splits/kitti/eigen_test_files_with_gt.txt"
    args.max_depth_eval = 80
    args.min_depth_eval = 0.001
    args.garg_crop = True

    args.lambdad = 1.0

def change_to_nyu(args):
    args.dataset = "nyu"
    nyu_root = _require(args, "nyu_root", "nyu").rstrip("/")
    args.data_path = os.path.join(nyu_root, "sync")
    args.txt_path = "./text/text_all/nyu/train"
    args.gt_path = os.path.join(nyu_root, "sync")
    args.filenames_file = "data_splits/nyudepthv2/nyudepthv2_train_files_with_gt.txt"
    args.input_height = 480
    args.input_width = 640
    args.do_kb_crop = False
    args.data_path_eval = os.path.join(nyu_root, "official_splits", "test")
    args.gt_path_eval = os.path.join(nyu_root, "official_splits", "test")
    args.txt_path_eval = "./text/text_all/nyu/test"
    args.filenames_file_eval = "/container_pip/test_files_with_gt.txt"
    args.max_depth_eval = 10
    args.min_depth_eval = 0.001
    args.garg_crop = False

    args.lambdad = 0.6

def change_to_void(args):
    args.dataset = "void"

    args.do_kb_crop = False
    args.max_depth_eval = 5
    args.min_depth_eval = 0.2
    args.garg_crop = False

    void_root = _require(args, "void_root", "void").rstrip("/") + "/"
    args.data_path = void_root
    args.gt_path = void_root

    args.txt_path = "./text/text_all/void/train"
    args.train_image_path_void = "./data_splits/void/train_image.txt"
    args.train_sparse_depth_path_void = "./data_splits/void/train_sparse_depth.txt"
    args.train_intrinsics_path_void = "./data_splits/void/train_intrinsics.txt"
    args.train_ground_truth_path_void = "./data_splits/void/train_ground_truth.txt"

    args.val_image_path_void = "./data_splits/void/test_image.txt"
    args.val_sparse_depth_path_void = "./data_splits/void/test_sparse_depth.txt"
    args.val_intrinsics_path_void = "./data_splits/void/test_intrinsics.txt"
    args.val_ground_truth_path_void = "./data_splits/void/test_ground_truth.txt"
    args.txt_path_eval = "./text/text_all/void/test"

    args.lambdad = 0.6
    
def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)

def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])

def repetitive_roundrobin(*iterables):
    """
    cycles through iterables but sample wise
    first yield first sample from first iterable then first sample from second iterable and so on
    then second sample from first iterable then second sample from second iterable and so on

    If one iterable is shorter than the others, it is repeated until all iterables are exhausted
    repetitive_roundrobin('ABC', 'D', 'EF') --> A D E B D F C D E
    """
    # Repetitive roundrobin
    iterables_ = [iter(it) for it in iterables]
    exhausted = [False] * len(iterables)
    while not all(exhausted):
        for i, it in enumerate(iterables_):
            try:
                yield next(it)
            except StopIteration:
                exhausted[i] = True
                iterables_[i] = cycle(iterables[i])
                # First elements may get repeated if one iterable is shorter than the others
                yield next(iterables_[i])

class RepetitiveRoundRobinDataLoader(object):
    def __init__(self, *dataloaders):
        self.dataloaders = dataloaders

    def __iter__(self):
        return repetitive_roundrobin(*self.dataloaders)

    def __len__(self):
        # First samples get repeated, thats why the plus one
        return len(self.dataloaders) * (max(len(dl) for dl in self.dataloaders) + 1)
    
class MixedNYUKITTI(object):
    def __init__(self, config, mode, **kwargs):
        if mode == 'train':
            self.config = config
            change_to_nyu(config)
            nyu_conf = copy.deepcopy(vars(config))
            nyu_conf = SimpleNamespace(**nyu_conf)
            change_to_kitti(config)
            kitti_conf = copy.deepcopy(vars(config))
            kitti_conf = SimpleNamespace(**kitti_conf)
            
            nyu_loader = NewDataLoader(nyu_conf, mode).data
            kitti_loader = NewDataLoader(kitti_conf, mode).data
            # It has been changed to repetitive roundrobin
            self.data = RepetitiveRoundRobinDataLoader(
                nyu_loader, kitti_loader)
            
            self.num_samples = len(nyu_loader.dataset) + \
            len(kitti_loader.dataset)
            
        elif mode == 'online_eval':
            self.config = config
            self.data = NewDataLoader(config, mode).data

class MixedNYUKITTIVOID(object):
    def __init__(self, config, mode, **kwargs):
        if mode == 'train':
            self.config = config
            change_to_nyu(config)
            nyu_conf = copy.deepcopy(vars(config))
            nyu_conf = SimpleNamespace(**nyu_conf)
            change_to_kitti(config)
            kitti_conf = copy.deepcopy(vars(config))
            kitti_conf = SimpleNamespace(**kitti_conf)
            change_to_void(config)
            void_conf = copy.deepcopy(vars(config))
            void_conf = SimpleNamespace(**void_conf)

            nyu_loader = NewDataLoader(nyu_conf, mode).data
            kitti_loader = NewDataLoader(kitti_conf, mode).data
            void_loader = VoidDataLoader(void_conf, mode).data
            # It has been changed to repetitive roundrobin
            self.data = RepetitiveRoundRobinDataLoader(
                nyu_loader, kitti_loader, void_loader)
            
            self.num_samples = len(nyu_loader.dataset) + \
            len(kitti_loader.dataset) + \
            len(void_loader.dataset)
            
        elif mode == 'online_eval':
            self.config = config
            if config.dataset == 'nyu' or config.dataset == 'kitti':
                self.data = NewDataLoader(config, mode).data
            elif config.dataset == 'void':
                self.data = VoidDataLoader(config, mode).data

class MixedNYUKITTIVOIDC3VD(object):
    def __init__(self, config, mode, **kwargs):
        if mode == 'train':
            self.config = config
            change_to_nyu(config)
            nyu_conf = copy.deepcopy(vars(config))
            nyu_conf = SimpleNamespace(**nyu_conf)
            change_to_kitti(config)
            kitti_conf = copy.deepcopy(vars(config))
            kitti_conf = SimpleNamespace(**kitti_conf)
            # change_to_void(config)
            # void_conf = copy.deepcopy(vars(config))
            # void_conf = SimpleNamespace(**void_conf)
            change_to_c3vd(config)
            c3vd_conf = copy.deepcopy(vars(config))
            c3vd_conf = SimpleNamespace(**c3vd_conf)

            self.nyu_loader = NewDataLoader(nyu_conf, mode)
            self.kitti_loader = NewDataLoader(kitti_conf, mode)
            # self.void_loader = VoidDataLoader(void_conf, mode)
            self.c3vd_loader = C3VDDataLoader(c3vd_conf, mode)
            # It has been changed to repetitive roundrobin
            self.data = RepetitiveRoundRobinDataLoader(
                self.nyu_loader.data, self.kitti_loader.data, self.c3vd_loader.data)
            
            self.num_samples = len(self.nyu_loader.data.dataset) + \
            len(self.kitti_loader.data.dataset) + len(self.c3vd_loader.data.dataset)# + len(self.void_loader.data.dataset)
            print('nyu samples:', len(self.nyu_loader.data.dataset))
            print('kiti samples:', len(self.kitti_loader.data.dataset))
            # print('void samples:', len(self.void_loader.data.dataset))
            print('c3vd samples:', len(self.c3vd_loader.data.dataset))
            print('total samples:', self.num_samples)
            
        elif mode == 'online_eval' or mode == 'online_eval_all':
            self.config = config
            if config.dataset == 'nyu' or config.dataset == 'kitti':
                self.data = NewDataLoader(config, mode).data
            elif config.dataset == 'void':
                self.data = VoidDataLoader(config, mode).data
            elif config.dataset == 'c3vd':
                self.data = C3VDDataLoader(config, mode).data

class NewDataLoader(object):
    def __init__(self, args, mode):
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
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
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
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
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)
            logging.info('prepared dataset {} for {} with {} sample.'.format(args.dataset, mode, len(self.testing_samples)))
        else:
            logging.info('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))

        

class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, transform=None, is_for_online_eval=False):
        self.args = args
        if mode == 'online_eval':
            with open(args.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(args.filenames_file, 'r') as f:
                self.filenames = f.readlines()

        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor
        self.is_for_online_eval = is_for_online_eval

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        # focal = float(sample_path.split()[2])
        focal = 518.8579

        if self.mode == 'train':
            if self.args.dataset == 'kitti':
                image_path = os.path.join(self.args.data_path, sample_path.split()[0])
                depth_path = os.path.join(self.args.gt_path, sample_path.split()[1])
                # pseudo_path = os.path.join(self.args.pseudo_depth_path,
                #                            self.args.depth_model,
                #                            self.args.pseudo_dataset_path,
                #                            sample_path.split()[1])

            else:
                image_path = os.path.join(self.args.data_path, "./" + sample_path.split()[0])
                depth_path = os.path.join(self.args.gt_path, "./" + sample_path.split()[1])
                # pseudo_path = os.path.join(self.args.pseudo_depth_path,
                #                            self.args.depth_model,
                #                            self.args.pseudo_dataset_path,
                #                            sample_path.split()[1][1:])
                
            image = Image.open(image_path)
            depth_gt = Image.open(depth_path)
            w, h = image.size
            # depth_pseudo = np.load(pseudo_path[:-4] + '.npy')
            # depth_pseudo = np.expand_dims(depth_pseudo, axis=2)

            if self.args.do_kb_crop is True:
                height = image.height
                width = image.width
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                # depth_pseudo = depth_pseudo[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]

            # To avoid blank boundaries due to pixel registration
            if self.args.dataset == 'nyu':
                if self.args.input_height == 480:
                    depth_gt = np.array(depth_gt)
                    valid_mask = np.zeros_like(depth_gt)
                    valid_mask[45:472, 43:608] = 1
                    depth_gt[valid_mask==0] = 0
                    depth_gt = Image.fromarray(depth_gt)
                else:
                    depth_gt = depth_gt.crop((43, 45, 608, 472))
                    image = image.crop((43, 45, 608, 472))
                    # depth_pseudo = depth_pseudo[45:472, 43:608, :]
            
            if self.args.do_random_rotate and (self.args.aug):
                random_angle = (random.random() - 0.5) * 2 * self.args.degree
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(
                    depth_gt, random_angle, flag=Image.NEAREST)
                
            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)

            if self.args.dataset == 'nyu':
                depth_gt = depth_gt / 1000.0
            else:
                depth_gt = depth_gt / 256.0
            if self.args.aug and (self.args.random_crop):
                image, depth_gt = self.random_crop(
                    image, depth_gt, self.args.input_height, self.args.input_width)
            
            if self.args.aug and self.args.random_translate:
                # print("Random Translation!")
                image, depth_gt = self.random_translate(image, depth_gt, self.args.max_translation)

            image, depth_gt = self.train_preprocess(image, depth_gt)
                            
            # mask = np.logical_and(depth_gt > 0, depth_gt <= self.args.max_depth_eval).squeeze()[None, ...]

            sample = {'image': image, 
                      'depth': depth_gt, 
                    #   'mask': mask,
                    #   'depth_pseudo': depth_pseudo,
                      'focal': focal, 
                      'sample_path':sample_path}


        else:
            if self.mode == 'online_eval':
                data_path = self.args.data_path_eval
            else:
                data_path = self.args.data_path

            image_path = os.path.join(data_path, "./" + sample_path.split()[0])
            image = Image.open(image_path)
            image = np.asarray(image, dtype=np.float32) / 255.0

            if self.mode == 'online_eval':
                gt_path = self.args.gt_path_eval
                depth_path = os.path.join(gt_path, "./" + sample_path.split()[1])
                # pseudo_path = os.path.join(self.args.pseudo_depth_path,
                #                            self.args.depth_model,
                #                            self.args.pseudo_dataset_path_eval,
                #                            sample_path.split()[1])
                has_valid_depth = False
                try:
                    depth_gt = Image.open(depth_path)
                    # depth_pseudo = np.load(pseudo_path[:-4] + '.npy')
                    # depth_pseudo = np.expand_dims(depth_pseudo, axis=2)
                    has_valid_depth = True
                except IOError:
                    depth_gt = False
                    # print('Missing gt for {}'.format(image_path))

                if has_valid_depth:
                    depth_gt = np.asarray(depth_gt, dtype=np.float32)
                    depth_gt = np.expand_dims(depth_gt, axis=2)
                    if self.args.dataset == 'nyu':
                        depth_gt = depth_gt / 1000.0
                    else:
                        depth_gt = depth_gt / 256.0
                    # mask = np.logical_and(depth_gt > 0, depth_gt <= self.args.max_depth_eval).squeeze()[None, ...]

            if self.args.do_kb_crop is True:
                height = image.shape[0]
                width = image.shape[1]
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
                if self.mode == 'online_eval' and has_valid_depth:
                    depth_gt = depth_gt[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
                    # depth_pseudo = depth_pseudo[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
            
            if self.mode == 'online_eval':
                sample = {'image': image, 
                          'depth': depth_gt,
                        #   'mask': mask, 
                        #   'depth_pseudo': depth_pseudo,
                          'focal': focal, 
                          'has_valid_depth': has_valid_depth, 
                          'sample_path':sample_path}
            else:
                sample = {'image': image, 
                          'focal': focal, 
                          'sample_path':sample_path}

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
        return len(self.filenames)


class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        image, focal = sample['image'], sample['focal']
        image = self.to_tensor(image)
        image = self.normalize(image)
        sample_path=sample['sample_path']

        if self.mode == 'test':
            return {'image': image, 'focal': focal}

        depth = sample['depth']
        # depth_pseudo = sample['depth_pseudo']
        if self.mode == 'train':
            depth = self.to_tensor(depth)
            # depth_pseudo = self.to_tensor(depth_pseudo)
            return {**sample,
                    'image': image, 
                    'depth': depth, 
                    # 'depth_pseudo': depth_pseudo,
                    'focal': focal, 
                    'sample_path':sample_path}
        else:
            has_valid_depth = sample['has_valid_depth']
            depth = self.to_tensor(depth)
            # depth_pseudo = self.to_tensor(depth_pseudo)
            return {**sample,
                    'image': image, 
                    'depth': depth, 
                    # 'depth_pseudo': depth_pseudo,
                    'focal': focal, 
                    'has_valid_depth': has_valid_depth, 
                    'sample_path':sample_path}

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
