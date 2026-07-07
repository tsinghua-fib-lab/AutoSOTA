from functools import partial
from typing import List, Union

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

BICUBIC = InterpolationMode.BICUBIC


def get_simclr_color_jitter(s=1):
    # SimCLR type jitter
    color_jitter = transforms.ColorJitter(
        brightness=0.8 * s, contrast=0.8 * s, saturation=0.8 * s, hue=0.2 * s
    )
    return transforms.RandomApply([color_jitter], p=0.8)


def get_color_jitter():
    color_jitter = transforms.ColorJitter(
        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
    )
    return transforms.RandomApply([color_jitter], p=0.5)


def five_crop(size):
    return transforms.Compose(
        [transforms.FiveCrop(size), transforms.Lambda(lambda crops: torch.stack(crops))]
    )


AUGMENTATIONS = {
    "random_flip": transforms.RandomHorizontalFlip(),
    "random_resized_crop": transforms.RandomResizedCrop,
    "normalize": transforms.Normalize,
    "random_crop": transforms.RandomCrop,
    "resize": transforms.Resize,
    "random_translation": transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    "center_crop": transforms.CenterCrop,
    "color_jitter": get_color_jitter(),
    "simclr_color_jitter": get_simclr_color_jitter(),
    "random_grayscale": transforms.RandomGrayscale(p=0.2),
    "gaussian_blur": transforms.GaussianBlur,
    "five_crop": five_crop,
}


def create_augmentations(
    augmentations: List,
    mean: List,
    std: List,
    crop_size: Union[List, int] = None,
    resize: Union[List, int] = None,
    video_inputs: bool = False,
):
    """Create a list of augmentations to be applied sequentially."""
    mean = torch.tensor(mean) if isinstance(mean, list) else mean
    std = torch.tensor(std) if isinstance(std, list) else std

    if video_inputs:
        mean = mean.view(-1, 1, 1, 1)
        std = std.view(-1, 1, 1, 1)

    selected_transforms = []

    for aug in augmentations:
        if aug in ["random_resized_crop", "random_crop", "center_crop", "five_crop"]:
            selected_transforms.append(AUGMENTATIONS[aug](size=crop_size))

        elif aug == "normalize":
            selected_transforms.append(AUGMENTATIONS[aug](mean=mean, std=std))

        elif aug == "resize":

            def resize_biubic(tensor, resize_aug):
                # TODO: remove if this is fixed in newer versions
                # Bicubic overshoots the range of pixel values sometimes
                return resize_aug(tensor).clamp(0.0, 1.0)

            resize_biubic_func = partial(
                resize_biubic,
                resize_aug=AUGMENTATIONS[aug](size=resize, interpolation=BICUBIC),
            )
            selected_transforms.append(resize_biubic_func)

        elif aug == "gaussian_blur":
            size = crop_size if crop_size is not None else resize
            selected_transforms.append(AUGMENTATIONS[aug](kernel_size=int(0.1 * size)))

        else:
            selected_transforms.append(AUGMENTATIONS[aug])

    return transforms.Compose(selected_transforms)
