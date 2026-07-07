
import json
import os
from pathlib import Path
from typing import List

from loguru import logger
from torchvision.transforms import Compose
from tqdm.auto import tqdm

from dataset.base_video import VideoDataset

from .utils import load_data_from_dir, parse_video_paths

AVAIL_DATASETS = ["K400", "K700", "HMDB51", "UCF101"]


class VideoDatasetFromDirectory(VideoDataset):
    # Here, the dataset is in the form of folders
    # each folder is a single class with all the videos
    # not used since it is slow to load at startup, takes 1 min to scan
    # the folders due so os.scan or walk

    mode: str = None
    dataset_name: str = None
    label_names_path: str = None
    train_split_path: str = None
    test_split_path: str = None
    actions2names: str = None
    extension: str = None

    def __init__(
        self,
        split: str,
        data_path: str,
        num_frames: int,
        temporal_strides: List,
        sampler_type: str,
        num_clips: int = 1,
        target_fps: int = 30,
        transforms: Compose = None,
        **_,
    ):

        assert split in ["train", "val"]
        logger.info(
            f"Creating dataset {self.dataset_name} | Split: {split}.")

        file_paths, labels, label_names, label2name = self.parse_data(
            split, Path(data_path))

        super().__init__(
            file_paths=file_paths,
            labels=labels,
            label_names=label_names,
            label2name=label2name,
            num_frames=num_frames,
            temporal_strides=temporal_strides,
            sampler_type=sampler_type,
            num_clips=num_clips,
            target_fps=target_fps,
            transforms=transforms
        )

    def parse_data(self, split: str, data_path: Path):
        logger.info("Parsing data paths ...")

        with open(self.label_names_path, "r", encoding="utf-8") as f:
            label_names = f.read().splitlines()
            label_names = [item.strip()
                           for item in label_names if len(item) > 0]

        with open(self.actions2names, "r", encoding="utf-8") as f:
            actions2names = json.load(f)

        name2label = {name: i for i, name in enumerate(label_names)}
        label2name = {i: actions2names[name]
                      for i, name in enumerate(label_names)}

        split_path = data_path / Path(split)

        file_paths, labels, = load_data_from_dir(
            split_path, class_names=label_names, extensions=[self.extension])

        labels = [name2label[item] for item in labels]

        return (
            file_paths,
            labels,
            label_names,
            label2name
        )


class VideoDatasetFromSplits(VideoDataset):
    mode: str = None
    dataset_name: str = None
    label_names_path: str = None
    train_split_path: str = None
    test_split_path: str = None
    actions2names: str = None
    extension: str = None

    def __init__(
        self,
        split: str,
        data_path: str,
        num_frames: int,
        temporal_strides: List,
        sampler_type: str,
        num_clips: int = 1,
        target_fps: int = 30,
        transforms: Compose = None,
        **_,
    ):

        logger.info(f"Creating dataset {self.dataset_name} ....")

        assert split in ["train", "val"]

        file_paths, labels, label_names, label2name = self.parse_data(
            split, Path(data_path))

        super().__init__(
            file_paths=file_paths,
            labels=labels,
            label_names=label_names,
            label2name=label2name,
            num_frames=num_frames,
            temporal_strides=temporal_strides,
            sampler_type=sampler_type,
            num_clips=num_clips,
            target_fps=target_fps,
            transforms=transforms
        )

    def parse_data(self, split: str, data_path: str):
        logger.info("Parsing data paths ....")

        split_path = self.train_split_path if split == "train" else self.test_split_path

        with open(split_path, "r", encoding="utf-8") as f:
            video_paths = f.read().splitlines()
            video_paths = [item.strip()
                           for item in video_paths if len(item) > 0]

        return parse_video_paths(
            data_path=data_path,
            video_paths=video_paths,
            label_names_path=self.label_names_path,
            actions2names=self.actions2names,
            extension=self.extension
        )


class UCF101(VideoDatasetFromSplits):
    mode = "video"
    dataset_name = "UCF101"
    label_names_path = "dataset/video_data/ucf101/actions.list"
    train_split_path = "dataset/video_data/ucf101/trainlist01.txt"
    test_split_path = "dataset/video_data/ucf101/testlist01.txt"
    actions2names = "dataset/video_data/ucf101/actions2names.json"
    extension = "avi"


class HMDB51(VideoDatasetFromSplits):
    mode = "video"
    dataset_name = "HMDB51"
    label_names_path = "dataset/video_data/hmdb51/actions.list"
    train_split_path = "dataset/video_data/hmdb51/train_split01.txt"
    test_split_path = "dataset/video_data/hmdb51/test_split01.txt"
    actions2names = "dataset/video_data/hmdb51/actions2names.json"
    extension = "avi"


class K400(VideoDatasetFromSplits):
    mode = "video"
    dataset_name = "K400"
    label_names_path = "dataset/video_data/k400/actions.list"
    train_split_path = "dataset/video_data/k400/train_split.txt"
    test_split_path = "dataset/video_data/k400/val_split.txt"
    actions2names = "dataset/video_data/k400/actions2names.json"
    extension = "h5"


class K700(VideoDatasetFromSplits):
    mode = "video"
    dataset_name = "K700"
    label_names_path = "dataset/video_data/k700/actions.list"
    train_split_path = "dataset/video_data/k700/train_split.txt"
    test_split_path = "dataset/video_data/k700/val_split.txt"
    actions2names = "dataset/video_data/k700/actions2names.json"
    extension = "mp4"


class SSv2(VideoDataset):
    mode = "hdf5"
    dataset_name = "SSv2"
    actions2labels = "dataset/video_data/SSv2/actions2labels.json"
    train_split_path = "dataset/video_data/SSv2/train_split.json"
    test_split_path = "dataset/video_data/SSv2/val_split.json"
    extension = "h5"

    def __init__(
        self,
        split: str,
        data_path: str,
        num_frames: int,
        temporal_strides: int,
        sampler_type: str,
        num_clips: int = 1,
        target_fps: int = 30,
        transforms: Compose = None,
        **_
    ):

        assert split in ["train", "val"]
        logger.info(
            f"Creating dataset {self.dataset_name} | Split: {split}.")

        file_paths, labels, label_names, label2name = self.parse_data(
            split, Path(data_path))

        super().__init__(
            file_paths=file_paths,
            labels=labels,
            label_names=label_names,
            label2name=label2name,
            num_frames=num_frames,
            temporal_strides=temporal_strides,
            sampler_type=sampler_type,
            num_clips=num_clips,
            target_fps=target_fps,
            transforms=transforms
        )

    def parse_data(self, split: str, data_path: Path):
        logger.info("Parsing data paths ...")

        # load json
        with open(self.actions2labels, "r", encoding="utf-8") as f:
            actions2labels = json.load(f)

        actions2labels = {k: int(v) for k, v in actions2labels.items()}
        label2name = {v: k for k, v in actions2labels.items()}
        label_names = list(actions2labels.keys())

        split_path = self.train_split_path if split == "train" else \
            self.test_split_path

        with open(split_path, "r", encoding="utf-8") as f:
            paths_and_labels = json.load(f)

        file_paths, labels = [], []

        tbar = tqdm(paths_and_labels)
        for idx, item in enumerate(tbar):
            video_name = item["id"]

            video_name = f"{video_name}.{self.extension}" \
                if self.extension not in video_name else video_name

            video_path = data_path / Path(video_name)

            if idx % 1000 == 0:
                # check every N iter, otherwise too slow
                assert os.path.isfile(
                    video_path), f"Video {video_path} is not a valid file"

            # detailed_description = item["label"]
            # Plugging [something] into [something] -> Plugging something into something
            label_name = item["template"].replace("[something]", "something")
            # Some trailing [ and ]
            label_name = label_name.replace("[", "").replace("]", "")

            label = actions2labels[label_name]

            file_paths.append(video_path)
            labels.append(label)

        return (
            file_paths,
            labels,
            label_names,
            label2name
        )
