import random
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageOps


class MTGAugmentations:
    """
    Custom augmentations for the Mind The Glitch dataset.
    Handles both image transformations and corresponding point transformations.
    Applies augmentations to only one of the images (either img1 or img2) and its related items.
    """

    def __init__(
        self,
        img_size=(256, 256),
        horizontal_flip_prob=0.5,
        color_jitter_prob=0.5,
        brightness_range=(0.8, 1.2),
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        training=True,
    ):
        """
        Args:
            img_size: Target image size (height, width)
            horizontal_flip_prob: Probability of horizontal flip
            brightness_range: Range for brightness adjustment
            contrast_range: Range for contrast adjustment
            saturation_range: Range for saturation adjustment
            hue_range: Range for hue adjustment
            training: Whether in training mode (apply augmentations)
        """
        self.img_size = img_size
        self.horizontal_flip_prob = horizontal_flip_prob
        self.color_jitter_prob = color_jitter_prob
        self.training = training

        # Color jitter transform for visual augmentations
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness_range, contrast=contrast_range, saturation=saturation_range
        )

    def augment_pil_image(self, img):
        """
        Apply visual augmentations to a PIL image
        """
        if not self.training:
            return img

        # Apply color jitter with 50% probability
        if random.random() < self.color_jitter_prob:
            img = self.color_jitter(img)

        # Apply horizontal flip with specified probability
        if random.random() < self.horizontal_flip_prob:
            img = ImageOps.mirror(img)

        return img

    def apply_single_image_augmentations(self, images_dict, points_lr, image_id=1):
        """
        Apply augmentations to only one image (either img1 or img2) and its related points

        Args:
            images_dict: Dictionary containing PIL images
            points_lr: Dictionary containing correspondence points in low resolution
            image_id: Which image to augment (1 or 2)

        Returns:
            Tuple of (augmented_images_dict, augmented_points_lr, applied_vflip)
        """
        # Randomly choose which image to augment if not specified
        if image_id not in [1, 2]:
            image_id = random.choice([1, 2])

        # Determine which image prefix to use
        img_prefix = f"img{image_id}"

        # Decide once if we should apply horizontal flip
        applied_vflip = random.random() < self.horizontal_flip_prob

        # Apply augmentations to the selected images
        for key in list(images_dict.keys()):
            if key.startswith(img_prefix):

                # Apply visual augmentations (contrast, brightness, etc.)
                if random.random() < self.color_jitter_prob and ("orig" in key or "inpainted" in key):
                    images_dict[key] = self.color_jitter(images_dict[key])

                # Apply horizontal flip if needed
                if applied_vflip:
                    images_dict[key] = ImageOps.mirror(images_dict[key])

        # If horizontal flip was applied, update the correspondence points
        if applied_vflip and points_lr is not None:
            width = self.img_size[1] // 16  # Feature resolution

            # Flip the x-coordinates of the points for the selected image
            if image_id == 1 and "img1_points_lr" in points_lr:
                points_lr["img1_points_lr"][:, 0] = width - 1 - points_lr["img1_points_lr"][:, 0]
            elif image_id == 2 and "img2_points_lr" in points_lr:
                points_lr["img2_points_lr"][:, 0] = width - 1 - points_lr["img2_points_lr"][:, 0]

        return images_dict, points_lr, applied_vflip
