from abc import abstractmethod
from typing import List, Union

import torch
import torchvision
from loguru import logger

from .base import AbstractDataset, Example
from .video_decoder import load_h5_frames, load_image_frames, torchvision_decode_video
from .video_utils import clip_sampler, get_video_info, get_video_type


def get_clip_loading_func(file_type):
    if file_type == "video":
        return torchvision_decode_video
        # return pyav_decode_video
    elif file_type == "frames":
        return load_image_frames
    return load_h5_frames


class VideoExample(Example):
    def __init__(
        self,
        file_path: str,
        label: int,
        label_name: str,
        num_frames: int,
        temporal_strides: Union[List, int],
        sampler_type: float,
        target_fps: int,
        num_clips: int = 1,
    ):
        super().__init__(file_path, label, label_name)

        self.target_fps = target_fps
        self.num_frames = num_frames
        self.sampler_type = sampler_type
        self.num_clips = num_clips

        self.file_type = get_video_type(file_path)
        self.clip_loading_func = get_clip_loading_func(self.file_type)
        self.temporal_strides = (
            [temporal_strides]
            if isinstance(temporal_strides, int)
            else temporal_strides
        )

        if num_clips > 1:
            assert (
                sampler_type == "sliding" and (len(self.temporal_strides) == 1)
            ) or (
                sampler_type == "center" and num_clips == len(self.temporal_strides)
            ), "multi-view either with sliding window or center and diff strides"

    def set_data_info(self):
        fps, width, height, frame_count = get_video_info(
            file_path=self.file_path,
            target_fps=self.target_fps,
            file_type=self.file_type,
        )

        self.data_info["fps"] = fps
        self.data_info["num_frames"] = self.num_frames
        self.data_info["duration"] = float(frame_count) / fps
        self.data_info["width"] = width
        self.data_info["height"] = height
        self.data_info["video_num_frames"] = frame_count

    @abstractmethod
    def load_clips(self) -> list:
        if len(self.data_info) == 0:
            self.set_data_info()

        clips_starts_ends, temporal_strides = clip_sampler(
            num_frames=self.num_frames,
            sampler_type=self.sampler_type,
            temporal_strides=self.temporal_strides,
            video_duration=self.data_info["duration"],
            fps=self.data_info["fps"],
            target_fps=self.target_fps,
            num_clips=self.num_clips,
        )

        sampeled_clips = self.clip_loading_func(
            file_path=self.file_path,
            clips_starts_ends=clips_starts_ends,
            num_frames=self.num_frames,
            fps=self.target_fps,
            temporal_strides=temporal_strides,
            duration=self.data_info["duration"],
        )

        return sampeled_clips

    def load_example(self) -> Union[torch.Tensor, List]:
        sampeled_clips = self.load_clips()

        if len(sampeled_clips) == 1:
            return sampeled_clips[0]
        return sampeled_clips


class VideoDataset(AbstractDataset):
    def __init__(
        self,
        file_paths: List,
        labels: List,
        label_names: List,
        label2name: dict,
        num_frames: int,
        temporal_strides: Union[List, int],
        sampler_type: str,
        target_fps: int,
        num_clips: int = 1,
        transforms: torchvision.transforms.Compose = None,
    ):
        self.transforms = transforms

        examples = self.create_examples(
            file_paths=file_paths,
            labels=labels,
            label_names=[label2name[i] for i in labels],
            num_frames=num_frames,
            temporal_strides=temporal_strides,
            sampler_type=sampler_type,
            target_fps=target_fps,
            num_clips=num_clips,
        )

        super().__init__(
            label_names=label_names,
            label2name=label2name,
            examples=examples,
        )

        logger.info(f"For dataset: {self.dataset_name}:")
        logger.info(f"      - number of classes: {self.num_classes}")
        logger.info(f"      - number of examples: {len(examples)}")

    @abstractmethod
    def parse_data(self) -> tuple:
        raise NotImplementedError

    def preprocess(self, data: torch.Tensor) -> torch.Tensor:
        if self.transforms is not None:
            data = (
                [self.transforms(item) for item in data]
                if isinstance(data, list)
                else self.transforms(data)
            )

        return data

    def create_examples(
        self,
        file_paths: List,
        labels: List,
        label_names: List,
        num_frames: int,
        temporal_strides: int,
        sampler_type: str,
        target_fps: int,
        num_clips: int,
    ) -> list:
        logger.info("Creating the examples ....")

        return [
            VideoExample(
                file_path=file_path,
                label=label,
                label_name=label_name,
                num_frames=num_frames,
                temporal_strides=temporal_strides,
                sampler_type=sampler_type,
                target_fps=target_fps,
                num_clips=num_clips,
            )
            for file_path, label, label_name in zip(file_paths, labels, label_names)
        ]
