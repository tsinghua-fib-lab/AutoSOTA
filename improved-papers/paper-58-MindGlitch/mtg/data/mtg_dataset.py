import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional, Any

import cv2
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torchvision.utils import make_grid
from omegaconf.dictconfig import DictConfig

from generate_dataset import Status
from configs.config import DatasetConfig
from mtg.data.augmentations import MTGAugmentations


def pad_points_list(img1_points, img2_points, max_points, pad_value=-1):
    padded_img1_points = np.ones((max_points, 2)) * pad_value  # -1 indicates invalid points
    padded_img2_points = np.ones((max_points, 2)) * pad_value

    num_valid_points = min(len(img1_points), max_points)
    if len(img1_points) > max_points:  # We have more points than max_points
        # Randomly select max_points from img1_pojnts
        valid_idxs = np.random.choice(np.arange(len(img1_points)), max_points, replace=False)
        padded_img1_points[:num_valid_points] = img1_points[valid_idxs]
        padded_img2_points[:num_valid_points] = img2_points[valid_idxs]
    else:  # We have less points than max_points
        padded_img1_points[:num_valid_points] = img1_points
        padded_img2_points[:num_valid_points] = img2_points
    return padded_img1_points, padded_img2_points


class MTGDataset(Dataset):
    def __init__(
        self,
        split_json_path: Union[str, Path],
        dataset_config: DatasetConfig,
        transform=None,
        training: bool = True,
    ):
        self.dataset_path = dataset_config.dataset_path
        self.split_json_path = split_json_path
        self.valid_idxs = load_split(self.split_json_path)
        self.json_path = self.dataset_path / "json"
        self.imgs_path = self.dataset_path / "imgs"
        self.correspondence_path = self.dataset_path / "correspondences"
        self.dataset_config = dataset_config
        self.training = training

        # Set up transforms
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

        # Get augmentation parameters from config
        if self.training and self.dataset_config.augmentations.enabled:
            self.augmentations = MTGAugmentations(
                img_size=self.dataset_config.img_size,
                horizontal_flip_prob=self.dataset_config.augmentations.horizontal_flip_prob,
                color_jitter_prob=self.dataset_config.augmentations.color_jitter_prob,
                brightness_range=tuple(self.dataset_config.augmentations.brightness_range),
                contrast_range=tuple(self.dataset_config.augmentations.contrast_range),
                saturation_range=tuple(self.dataset_config.augmentations.saturation_range),
                training=training,
            )
        else:
            self.augmentations = None

        if self.training:
            self.apply_data_filtering()

        self.max_correspond_points = int((self.dataset_config.img_size[0] // 16) ** 2)

    def apply_data_filtering(self):
        # Filter samples with low skewness
        skewness_thresh = self.dataset_config.filtering.skewness_thresh
        if skewness_thresh != -1:
            self.valid_idxs = self.filter_with_metric("matching_skewness", skewness_thresh)

        # Filter samples with low LIPIPS
        lpips_threshold = self.dataset_config.filtering.lpips_thresh
        if lpips_threshold != -1:
            self.valid_idxs = self.filter_with_metric("source_lpips", lpips_threshold, operator=">")

        # Filter samples with low LIPIPS
        rel_size_threshold = self.dataset_config.filtering.source_target_rel_sz_diff
        if rel_size_threshold != -1:
            self.valid_idxs = self.filter_with_metric("source_target_rel_sz_diff", rel_size_threshold, operator="<")

    def filter_with_metric(self, metric_name, metric_thresh, operator=">"):
        print(f"Filtering samples with {metric_name} {operator} {metric_thresh}...")

        new_valid_idxs = []
        for idx in self.valid_idxs:
            file_idx = idx["file_idx"]
            pt_idx = idx["pt_idx"]
            with open(self.json_path / f"{file_idx}.json", "r") as f:
                meta_data = json.load(f)
                pt_seg_id = f"pt_seg_{pt_idx}"
                if pt_seg_id in meta_data:
                    metric_value = meta_data[pt_seg_id][metric_name]
                    if operator == ">" and metric_value > metric_thresh:
                        new_valid_idxs.append(idx)
                    elif operator == "<" and metric_value < metric_thresh:
                        new_valid_idxs.append(idx)
        print(
            f"Applying filter {metric_name} {operator} {metric_thresh}. Remaining samples: {len(new_valid_idxs)} out of {len(self.valid_idxs)}"
        )
        return new_valid_idxs

    def __len__(self):
        return len(self.valid_idxs)

    def __getitem__(self, idx):
        file_idx = self.valid_idxs[idx]["file_idx"]
        pt_idx = self.valid_idxs[idx]["pt_idx"]

        # Load all images as PIL images
        images_dict = {
            "img1_orig": Image.open(self.imgs_path / f"{file_idx}_img_1.png"),
            "img2_orig": Image.open(self.imgs_path / f"{file_idx}_img_2.png"),
            "img1_inpainted": Image.open(self.imgs_path / f"{file_idx}_img_1i_pt{pt_idx}.png"),
            "img2_inpainted": Image.open(self.imgs_path / f"{file_idx}_img_2i_pt{pt_idx}.png"),
            "img1_obj_mask": Image.open(self.imgs_path / f"{file_idx}_obj_mask_1.png"),
            "img2_obj_mask": Image.open(self.imgs_path / f"{file_idx}_obj_mask_2.png"),
            "img1_part_mask": Image.open(self.imgs_path / f"{file_idx}_part_mask_img_1_pt{pt_idx}.png"),
            "img2_part_mask": Image.open(self.imgs_path / f"{file_idx}_part_mask_img_2_pt{pt_idx}.png"),
        }

        # Load correspondence data
        corr_data = torch.load(self.correspondence_path / f"{file_idx}.pt", weights_only=False)
        img1_points_lr = corr_data["source_points_lr"]
        img2_points_lr = corr_data["target_points_lr"]
        matching_score = corr_data["matching_scores_max"]

        # Apply augmentations to one image and its related points before tensor conversion
        if self.training and self.augmentations is not None:
            # Create points dictionary for augmentation
            points_lr = {"img1_points_lr": img1_points_lr, "img2_points_lr": img2_points_lr}

            # Apply augmentations to only image 1 and its related items
            images_dict, points_lr, _ = self.augmentations.apply_single_image_augmentations(
                images_dict, points_lr, image_id=1
            )
            # Update the points after augmentation
            img1_points_lr = points_lr["img1_points_lr"]
            img2_points_lr = points_lr["img2_points_lr"]

        # Convert all images to tensors
        img1 = self.transform(images_dict["img1_orig"]) * 2 - 1
        img2 = self.transform(images_dict["img2_orig"]) * 2 - 1
        img1_inpainted = self.transform(images_dict["img1_inpainted"]) * 2 - 1
        img2_inpainted = self.transform(images_dict["img2_inpainted"]) * 2 - 1
        img1_obj_mask = self.transform(images_dict["img1_obj_mask"])
        img2_obj_mask = self.transform(images_dict["img2_obj_mask"])
        img1_part_mask = self.transform(images_dict["img1_part_mask"])
        img2_part_mask = self.transform(images_dict["img2_part_mask"])

        # Downscale part masks to feature map resolution
        feat_size = (self.dataset_config.img_size[0] // 16, self.dataset_config.img_size[1] // 16)
        img1_part_mask_lr_np = np.array(images_dict["img1_part_mask"].resize(feat_size, Image.Resampling.NEAREST))
        img2_part_mask_lr_np = np.array(images_dict["img2_part_mask"].resize(feat_size, Image.Resampling.NEAREST))

        # Split correspondence points to on-part and outside-part points
        on_part_points = []
        outside_part_points = []
        for idx, pt in enumerate(img1_points_lr):

            # Skip inconfident matches
            if matching_score[idx] < self.dataset_config.filtering.matching_score_thresh:
                continue

            # Ensure points are within bounds
            y1, x1 = min(pt[1], img1_part_mask_lr_np.shape[0] - 1), min(pt[0], img1_part_mask_lr_np.shape[1] - 1)
            y2, x2 = min(img2_points_lr[idx][1], img2_part_mask_lr_np.shape[0] - 1), min(
                img2_points_lr[idx][0], img2_part_mask_lr_np.shape[1] - 1
            )

            # Check if the point is on or off the part mask
            if img1_part_mask_lr_np[y1, x1] > 0 and img2_part_mask_lr_np[y2, x2] > 0:
                on_part_points.append(idx)
            elif img1_part_mask_lr_np[y1, x1] == 0 and img2_part_mask_lr_np[y2, x2] == 0:
                outside_part_points.append(idx)
            else:
                continue

        img1_part_points = img1_points_lr[on_part_points]
        img2_part_points = img2_points_lr[on_part_points]
        img1_outside_part_points = img1_points_lr[outside_part_points]
        img2_outside_part_points = img2_points_lr[outside_part_points]

        # Unify the number of points so that samples can be batched together
        img1_part_points, img2_part_points = pad_points_list(
            img1_part_points, img2_part_points, self.max_correspond_points
        )
        img1_outside_part_points, img2_outside_part_points = pad_points_list(
            img1_outside_part_points, img2_outside_part_points, self.max_correspond_points
        )

        # No need for additional augmentations here as they were applied before tensor conversion

        # Create the batch dictionary
        batch = {
            "idx": f"{file_idx}_pt{pt_idx}",
            "img1_orig": img1,  # with background
            "img2_orig": img2,  # with white background
            "img1_inpainted": img1_inpainted,
            "img2_inpainted": img2_inpainted,
            "img1_obj_mask": img1_obj_mask,
            "img2_obj_mask": img2_obj_mask,
            "img1_part_mask": img1_part_mask,
            "img2_part_mask": img2_part_mask,
            "img1_part_points": torch.tensor(img1_part_points),
            "img2_part_points": torch.tensor(img2_part_points),
            "img1_outside_part_points": torch.tensor(img1_outside_part_points),
            "img2_outside_part_points": torch.tensor(img2_outside_part_points),
            "img1_oracle": 1 - img1_part_mask.sum() / img1_obj_mask.sum(),
            "img2_oracle": 1 - img2_part_mask.sum() / img2_obj_mask.sum(),
        }

        return batch

    def visualize_sample(self, idx):
        sample = self.__getitem__(idx)
        images = [
            sample["img1_orig"],
            sample["img1_inpainted"],
            sample["img1_obj_mask"].repeat(3, 1, 1),
            sample["img1_pt_mask"].repeat(3, 1, 1),
            sample["img2_orig"],
            sample["img2_inpainted"],
            sample["img2_obj_mask"].repeat(3, 1, 1),
            sample["img2_pt_mask"].repeat(3, 1, 1),
        ]

        sample_grid = make_grid(images, nrow=4)
        sample_grid = transforms.ToPILImage()(sample_grid)
        return sample_grid


# load json file
def load_split(json_path):
    with open(json_path) as f:
        split = json.load(f)
    return split


def get_valid_idxs(dataset_path: Path) -> List[Dict[str, Any]]:
    # Get the indices of the files that have been successfully segmented
    jsons_path = dataset_path / "json"
    imgs_path = dataset_path / "imgs"
    print("Getting valid indices in the dataset...")
    valid_idxs: List[Dict[str, Any]] = []
    for path in jsons_path.glob("*.json"):
        with open(path, "r") as f:
            meta_data = json.load(f)
        for k in meta_data:
            if "pt_seg_" in k:
                pt_idx = int(k.split("_")[-1])
                if (
                    meta_data[k]["status"] == Status.SUCCEEDED.value
                    and (imgs_path / f"{meta_data['idx']}_img_1i_pt{pt_idx}.png").exists()
                ):
                    valid_idxs.append({"file_idx": meta_data["idx"], "pt_idx": pt_idx})
    return valid_idxs


def create_train_val_split(dataset_path, train_val_split=0.9):
    valid_idxs = get_valid_idxs(dataset_path)

    # Check if we have valid indices
    if len(valid_idxs) == 0:
        raise ValueError(f"No valid indices found for dataset {dataset_path}. Please check the dataset.")

    indices = np.random.permutation(len(valid_idxs))

    # Calculate the split point
    val_size = int(len(valid_idxs) * (1 - train_val_split))

    # Get the validation indices
    val_indices = indices[:val_size]

    # Get the training indices
    train_indices = indices[val_size:]

    # Create the train and val lists
    val_idxs = [valid_idxs[i] for i in val_indices]
    train_idxs = [valid_idxs[i] for i in train_indices]

    return train_idxs, val_idxs


def load_dataset(
    train_dataset_cfg: DatasetConfig,
    val_dataset_cfg: DatasetConfig,
    train_transform=None,
    val_transform=None,
):
    train_dataset_path = Path(train_dataset_cfg.dataset_path)
    val_dataset_path = Path(val_dataset_cfg.dataset_path)

    train_json_path = train_dataset_path / f"train_split_{train_dataset_cfg.train_val_split}.json"
    val_json_path = val_dataset_path / f"val_split_{val_dataset_cfg.train_val_split}.json"

    # If validation dataset is the same as training dataset => take split from training dataset
    if train_dataset_cfg.dataset_name == val_dataset_cfg.dataset_name:
        assert (
            train_dataset_cfg.train_val_split == val_dataset_cfg.train_val_split
        ), "When using the same dataset for training and validation, the train_val_split must be the same"

        print(f"Loading training and validation splits from {train_dataset_path}")
        # If the splits don't exist, create them
        if not train_json_path.exists() or not val_json_path.exists():
            # Save the training indices to json file
            np.random.seed(train_dataset_cfg.seed)
            train_idxs, val_idxs = create_train_val_split(train_dataset_path, train_dataset_cfg.train_val_split)

            with open(train_json_path, "w") as f:
                json.dump(train_idxs, f)
            print(f"Training split saved to {train_json_path}")

            with open(val_json_path, "w") as f:
                json.dump(val_idxs, f)
            print(f"Validation split saved to {val_json_path}")

    else:
        print(f"Loading training split from {train_dataset_path}")
        # Get training set
        if not train_json_path.exists():
            np.random.seed(train_dataset_cfg.seed)
            train_idxs, _ = create_train_val_split(train_dataset_path, train_dataset_cfg.train_val_split)
            with open(train_json_path, "w") as f:
                json.dump(train_idxs, f)
            print(f"Training split saved to {train_json_path}")

        # Get validation set
        print(f"Loading validation split from {val_dataset_path}")
        if not val_json_path.exists():
            # Save the validation indices to json file
            np.random.seed(val_dataset_cfg.seed)
            _, val_idxs = create_train_val_split(val_dataset_path, val_dataset_cfg.train_val_split)

            with open(val_json_path, "w") as f:
                json.dump(val_idxs, f)
            print(f"Validation split saved to {val_json_path}")

    # Load the training and validation splits
    train_dataset = MTGDataset(
        train_json_path,
        train_dataset_cfg,
        train_transform,
        training=True,
    )
    val_dataset = MTGDataset(
        val_json_path,
        val_dataset_cfg,
        val_transform,
        training=False,
    )
    print(f"Available Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    ### Randomly select samples if num_samples is set
    def limit_dataset_samples(dataset, num_samples, seed):
        """Randomly select a subset of samples from the dataset if num_samples is set."""
        np.random.seed(seed)
        if num_samples != -1 and num_samples < len(dataset.valid_idxs):
            dataset.valid_idxs = np.random.choice(dataset.valid_idxs, num_samples).tolist()
        return dataset

    train_dataset = limit_dataset_samples(train_dataset, train_dataset_cfg.num_samples, train_dataset_cfg.seed)
    val_dataset = limit_dataset_samples(val_dataset, val_dataset_cfg.num_samples, val_dataset_cfg.seed)

    print(f"Using Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    ### Choose some samples for visualization
    def load_or_create_visualization_samples(dataset, dataset_path, dataset_cfg, name):
        log_path = dataset_path / f"{name}_idxs_to_log.json"
        if not log_path.exists():
            print(f"Saving {name} samples to visualize...")
            samples_to_log = np.random.choice(
                dataset.valid_idxs, dataset_cfg.num_visualization_samples, replace=False
            ).tolist()
            with open(log_path, "w") as f:
                json.dump(samples_to_log, f)
            print(f"{name.capitalize()} samples to visualize saved to {log_path}")
        else:
            with open(log_path, "r") as f:
                samples_to_log = json.load(f)
            samples_to_log = [f"{f['file_idx']}_pt{f['pt_idx']}" for f in samples_to_log]
        return samples_to_log

    train_to_log = load_or_create_visualization_samples(
        train_dataset, train_dataset_path, train_dataset_cfg, "training"
    )
    val_to_log = load_or_create_visualization_samples(val_dataset, val_dataset_path, val_dataset_cfg, "val")

    return train_dataset, val_dataset, train_to_log, val_to_log
