import os
import copy
import random
import math
import json # 新增导入
from PIL import Image # 新增导入

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data
from .ImagenetV2 import ImageNetV2Dataset


# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
tr_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
									transforms.RandomHorizontalFlip(),
                                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
									transforms.ToTensor(),
									normalize])
te_transforms = transforms.Compose([transforms.Resize(256),
									transforms.CenterCrop(224),
									transforms.ToTensor(),
									normalize])
te_transforms_imageC = transforms.Compose([transforms.CenterCrop(224),
									transforms.ToTensor(),
									normalize])

rotation_tr_transforms = tr_transforms
rotation_te_transforms = te_transforms

common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
	                    'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
	                    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']


class ImagePathFolder(datasets.ImageFolder):
	def __init__(self, traindir, train_transform):
		super(ImagePathFolder, self).__init__(traindir, train_transform)	

	def __getitem__(self, index):
		path, _ = self.imgs[index]
		img = self.loader(path)
		if self.transform is not None:
			img = self.transform(img)
		path, pa = os.path.split(path)
		path, pb = os.path.split(path)
		return img, 'val/%s/%s' %(pb, pa)


# =========================Rotate ImageFolder Preparations Start======================
# Assumes that tensor is (nchannels, height, width)
def tensor_rot_90(x):
	return x.flip(2).transpose(1, 2)

def tensor_rot_180(x):
	return x.flip(2).flip(1)

def tensor_rot_270(x):
	return x.transpose(1, 2).flip(2)

def rotate_single_with_label(img, label):
	if label == 1:
		img = tensor_rot_90(img)
	elif label == 2:
		img = tensor_rot_180(img)
	elif label == 3:
		img = tensor_rot_270(img)
	return img

def rotate_batch_with_labels(batch, labels):
	images = []
	for img, label in zip(batch, labels):
		img = rotate_single_with_label(img, label)
		images.append(img.unsqueeze(0))
	return torch.cat(images)

def rotate_batch(batch, label='rand'):
	if label == 'rand':
		labels = torch.randint(4, (len(batch),), dtype=torch.long)
	else:
		assert isinstance(label, int)
		labels = torch.zeros((len(batch),), dtype=torch.long) + label
	return rotate_batch_with_labels(batch, labels), labels


# =========================Rotate ImageFolder Preparations End======================


# The following ImageFolder supports sample a subset from the entire dataset by index/classes/sample number, at any time after the dataloader created. 
class SelectedRotateImageFolder(datasets.ImageFolder):
    def __init__(self, root, train_transform, original=True, rotation=True, rotation_transform=None):
        super(SelectedRotateImageFolder, self).__init__(root, train_transform)
        self.original = original
        self.rotation = rotation
        self.rotation_transform = rotation_transform

        self.original_samples = self.samples
        random.shuffle(self.samples)

    def __getitem__(self, index):
        # path, target = self.imgs[index]
        path, target = self.samples[index]
        img_input = self.loader(path)

        if self.transform is not None:
            img = self.transform(img_input)
        else:
            img = img_input

        results = []
        if self.original:
            results.append(img)
            results.append(target)
        if self.rotation:
            if self.rotation_transform is not None:
                img = self.rotation_transform(img_input)
            target_ssh = np.random.randint(0, 4, 1)[0]
            img_ssh = rotate_single_with_label(img, target_ssh)
            results.append(img_ssh)
            results.append(target_ssh)
        return results

    def switch_mode(self, original, rotation):
        self.original = original
        self.rotation = rotation

    def set_target_class_dataset(self, target_class_index, logger=None):
        self.target_class_index = target_class_index
        self.samples = [(path, idx) for (path, idx) in self.original_samples if idx in self.target_class_index]
        self.targets = [s[1] for s in self.samples]

    def set_dataset_size(self, subset_size):
        num_train = len(self.targets)
        indices = list(range(num_train))
        random.shuffle(indices)
        self.samples = [self.samples[i] for i in indices[:subset_size]]
        self.targets = [self.targets[i] for i in indices[:subset_size]]
        return len(self.targets)

    def set_specific_subset(self, indices):
        self.samples = [self.original_samples[i] for i in indices]
        self.targets = [s[1] for s in self.samples]


def reset_data_sampler(sampler, dset_length, dset):
    sampler.dataset = dset
    if dset_length % sampler.num_replicas != 0 and False:
        sampler.num_samples = math.ceil((dset_length - sampler.num_replicas) / sampler.num_replicas)
    else:
        sampler.num_samples = math.ceil(dset_length / sampler.num_replicas)
    sampler.total_size = sampler.num_samples * sampler.num_replicas


def prepare_train_dataset(args):
    print('Preparing data...')
    traindir = os.path.join(args.data, 'train')
    trset = SelectedRotateImageFolder(traindir, tr_transforms, original=True, rotation=args.rotation,
                                                        rotation_transform=rotation_tr_transforms)
    return trset


def prepare_train_dataloader(args, trset=None, sampler=None):
    if sampler is None:
        trloader = torch.utils.data.DataLoader(trset, batch_size=args.batch_size, shuffle=True,
                                                        num_workers=args.workers, pin_memory=True)
        train_sampler = None
    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trset)
        trloader = torch.utils.data.DataLoader(
            trset, batch_size=args.batch_size,
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True) #sampler=None shuffle=True,
    return trloader, train_sampler

class ImageNetSubsetFromList(torch.utils.data.Dataset):
    def __init__(self, item_list, transform=None):
        self.item_list = item_list # item_list 已经是 (full_path, label) 元组的列表
        self.transform = transform

    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, index):
        path, target = self.item_list[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target

def prepare_test_data(args, use_transforms=True, semi=False):
    if args.corruption == 'original' or args.corruption == 'rendition' or args.corruption == 'v2' or args.corruption == 'sketch':
        te_transforms_local = te_transforms if use_transforms else None
    elif args.corruption in common_corruptions:
        te_transforms_local = te_transforms_imageC if use_transforms else None
    else:
        assert False, NotImplementedError
    
    # --- 根据 args.continual 的值选择数据集加载方式 ---
    if args.continual:
        # 当 args.continual 为 True 时，我们总是使用 5k 子集。
        # args.corruption 参数将指定要应用于此子集的损坏类型。
        
        # 加载类别映射和 5k 图像 ID 列表
        class_map_path = args.imagenet_class_map_file
        indices_file_path = args.imagenet_5k_indices_file

        if not os.path.exists(class_map_path):
            raise FileNotFoundError(f"ImageNet class map file not found at: {class_map_path}")
        with open(class_map_path, 'r') as f:
            class_to_idx = json.load(f)

        if not os.path.exists(indices_file_path):
            raise FileNotFoundError(f"ImageNet 5k indices file not found at: {indices_file_path}")
        with open(indices_file_path, 'r') as f:
            fnames_relative = f.readlines() # 这些是相对路径，例如 "nXXXX/image.JPEG"

        item_list_for_current_corruption = []
        
        # 确定当前损坏类型的数据根目录
        base_dir_for_current_corruption = None
        if args.corruption == 'original':
            # 对于原始图像，使用 ImageNet 的 val 目录
            base_dir_for_current_corruption = os.path.join(args.data, 'val')
        elif args.corruption in common_corruptions:
            # 对于 ImageNet-C 损坏，使用 ImageNet-C 的结构
            base_dir_for_current_corruption = os.path.join(args.data_corruption, args.corruption, str(args.level))
        elif args.corruption == 'rendition':
            base_dir_for_current_corruption = args.data_rendition
            print(f"Warning: args.continual is True, but corruption '{args.corruption}' is not 'original' or a common ImageNet-C corruption. "
                  f"Loading full dataset for '{args.corruption}' as no specific 5k indices are provided for it.")
            teset = datasets.ImageFolder(base_dir_for_current_corruption, te_transforms_local)
            shuffle_dataloader = args.if_shuffle
            teloader = torch.utils.data.DataLoader(teset, batch_size=args.batch_size, shuffle=shuffle_dataloader, 
                                                    num_workers=args.workers, pin_memory=True)
            return teset, teloader # 直接返回，不走 5k 逻辑
        elif args.corruption == 'sketch':
            base_dir_for_current_corruption = args.data_sketch
            print(f"Warning: args.continual is True, but corruption '{args.corruption}' is not 'original' or a common ImageNet-C corruption. "
                  f"Loading full dataset for '{args.corruption}' as no specific 5k indices are provided for it.")
            teset = datasets.ImageFolder(base_dir_for_current_corruption, te_transforms_local)
            shuffle_dataloader = args.if_shuffle
            teloader = torch.utils.data.DataLoader(teset, batch_size=args.batch_size, shuffle=shuffle_dataloader, 
                                                    num_workers=args.workers, pin_memory=True)
            return teset, teloader # 直接返回，不走 5k 逻辑
        elif args.corruption == 'v2':
            print(f"Warning: args.continual is True, but corruption '{args.corruption}' is not 'original' or a common ImageNet-C corruption. "
                  f"Loading full dataset for '{args.corruption}' as no specific 5k indices are provided for it.")
            teset = ImageNetV2Dataset(transform=te_transforms_local, location=args.data_v2)
            shuffle_dataloader = args.if_shuffle
            teloader = torch.utils.data.DataLoader(teset, batch_size=args.batch_size, shuffle=shuffle_dataloader, 
                                                    num_workers=args.workers, pin_memory=True)
            return teset, teloader # 直接返回，不走 5k 逻辑
        else:
            raise Exception(f"Unknown corruption type '{args.corruption}' with --continual flag.")

        # 遍历 5k 索引文件，构建 (完整路径, 标签) 列表
        for fn_rel in fnames_relative:
            fn_rel = fn_rel.strip() # 移除换行符
            if not fn_rel: continue # 跳过空行

            # 从相对路径中提取 synset ID (例如 "n09468604" from "n09468604/ILSVRC2012_val_00015253.JPEG")
            synset_id = fn_rel.split(os.sep)[0] 

            if synset_id not in class_to_idx:
                print(f"Warning: Synset ID '{synset_id}' not found in class map for path '{fn_rel}'. Skipping.")
                continue
            
            label = class_to_idx[synset_id]
            full_image_path = os.path.join(base_dir_for_current_corruption, fn_rel)

            if not os.path.exists(full_image_path):
                print(f"Warning: Image file not found at: {full_image_path}. Skipping sample.")
                continue
            item_list_for_current_corruption.append((full_image_path, label))

        teset = ImageNetSubsetFromList(item_list_for_current_corruption, transform=te_transforms_local)
        print(f"Loaded {len(teset)} samples for continual evaluation of '{args.corruption}' (level {args.level}) using 5k ImageNet subset.")
        shuffle_dataloader = True # 在 continual 模式下，通常会打乱数据以确保批次多样性

    # --- 非 continual 模式下的原有逻辑 ---
    elif args.corruption == 'original':
        validdir = os.path.join(args.data, 'val')
        teset = SelectedRotateImageFolder(validdir, te_transforms_local, original=False, rotation=False,
                                                    rotation_transform=rotation_te_transforms)
        teset.switch_mode(True, False)
        shuffle_dataloader = args.if_shuffle
    elif args.corruption in common_corruptions:
        if not semi:
            print('Test on %s level %d' %(args.corruption, args.level))
        validdir = os.path.join(args.data_corruption, args.corruption, str(args.level))
        teset = SelectedRotateImageFolder(validdir, te_transforms_local, original=False, rotation=False,
                                                    rotation_transform=rotation_te_transforms)
        teset.switch_mode(True, False)
        shuffle_dataloader = args.if_shuffle
    elif args.corruption == 'rendition':
        validdir = args.data_rendition
        teset = datasets.ImageFolder(validdir, te_transforms_local)
        shuffle_dataloader = args.if_shuffle
    elif args.corruption == 'sketch':
        validdir = args.data_sketch
        teset = datasets.ImageFolder(validdir, te_transforms_local)
        shuffle_dataloader = args.if_shuffle
    elif args.corruption == 'v2':
         teset = ImageNetV2Dataset(transform=te_transforms_local, location=args.data_v2)
         shuffle_dataloader = args.if_shuffle
    else:
        raise Exception('Corruption not found!')
    
    if not hasattr(args, 'workers'):
        args.workers = 1

    teloader = torch.utils.data.DataLoader(teset, batch_size=args.batch_size, shuffle=shuffle_dataloader, 
                                                    num_workers=args.workers, pin_memory=True)
    return teset, teloader