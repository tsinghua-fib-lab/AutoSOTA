# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torchvision.transforms.v2 import Compose, RandomHorizontalFlip, ToDtype, ToImage, Resize, Grayscale
import torchvision

def get_transform_cifar(is_for_fid):
    if is_for_fid:  # For FID evaluation, there should be no augmentation
        transform_list = [
            ToImage(),
            ToDtype(torch.float32, scale=True),
        ]
    else:
        transform_list = [
            ToImage(),
            RandomHorizontalFlip(),
            ToDtype(torch.float32, scale=True),
            # torchvision.transforms.Resize((224, 224)),
            # torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            
        ]
    return Compose(transform_list)


def get_transform_mnist():
    transform_list = [
        Resize((32, 32)),  # Resize images to 32x32 pixels for debugging
        Grayscale(num_output_channels=3),  # Convert to three channels
        ToImage(),
        ToDtype(torch.float32, scale=True),
    ]
    return Compose(transform_list)
