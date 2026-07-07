
import math
import random
from pathlib import Path
from typing import List, Union


import h5py
import numpy as np
import torch
import torchvision
import cv2


def get_video_info_with_opencv(video_file):

    cap = cv2.VideoCapture(str(video_file))

    assert cap.isOpened(), "Error opening video file"

    # Get the video's properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    return fps, width, height, frame_count


def get_video_info_with_torchvision(video_file):
    video_reader = torchvision.io.VideoReader(
        str(video_file), stream="video", num_threads=2
    )

    fps = video_reader.get_metadata()["video"]["fps"][0]
    duration = video_reader.get_metadata()["video"]["duration"][0]
    frame_count = int(math.ceil(duration * fps))

    video_reader.seek(0.0)
    frame = next(video_reader)["data"]
    _, height, width = frame.shape

    return fps, width, height, frame_count


def secs_to_pts(time_in_seconds: float, time_base: float,
                start_pts: int, round_mode: str = "floor") -> int:
    """
    Converts seconds to a pts for video decoding
    from pytorchvideo
    """
    if time_in_seconds == math.inf:
        return math.inf

    assert round_mode in [
        "floor", "ceil"], f"round_mode={round_mode} is not supported!"

    if round_mode == "floor":
        return math.floor(time_in_seconds / time_base) + start_pts
    else:
        return math.ceil(time_in_seconds / time_base) + start_pts


def read_image_frame(frame_path):
    with open(frame_path, "rb") as f:
        frame_str = np.frombuffer(f.read(), dtype=np.uint8)
        # of shape HWC
        frame = cv2.imdecode(frame_str, flags=cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame


def read_h5_frames(file_path, frame_indices: List) -> List:
    frames = []
    video_name = Path(file_path).name.split(".")[0]

    with h5py.File(file_path, 'r') as video_h5:
        for frame_idx in frame_indices:
            frame_str = video_h5[video_name][frame_idx]
            frame = cv2.imdecode(frame_str, flags=cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frames.append(frame)
    return frames


def sample_frames_with_stride(
    frames: torch.Tensor,
    num_frames: int,
    temporal_stride: int
) -> torch.Tensor:
    """
    Samples frames with a given stride
    duplicates the last one if len(frames) < num_frames
    """
    index = torch.linspace(0, (num_frames - 1) *
                           temporal_stride, num_frames).long()
    index = torch.clamp(index, 0, frames.shape[0] - 1)
    frames = torch.index_select(frames, 0, index)
    return frames


def center_sampler(delta, clip_duration):
    # return the center of the video
    start = delta / 2.0
    return (start, start+clip_duration)


def random_sampler(delta, clip_duration):
    # randomly sampled video starts
    start = random.uniform(0, delta)
    return (start, start+clip_duration)


def uniform_sampler(clip_duration, clip_index, clip_starts):
    # return one of the clip starts
    start = clip_starts[clip_index]
    return (start, start+clip_duration)


def sample(sampler_type, delta, clip_duration, video_duration):

    if sampler_type == "random":
        start, end = random_sampler(delta, clip_duration)
    elif sampler_type == "center":
        start, end = center_sampler(delta, clip_duration)
    else:
        raise ValueError(
            f"Sampler type {sampler_type} is not supported")

    start, end = np.round(start, 2), np.round(end, 2)
    end = min(video_duration, end)
    return (start, end)


def get_clip_duration(
    num_frames: int,
    temporal_strides: List,
    fps: float,
    target_fps: float,
    video_duration: float,
):

    #temporal_stride = random.choice(temporal_stride)
    clip_durations, adjusted_strides = [], []
    for temporal_stride in temporal_strides:
        # adjust the temporal stride for the target FPS
        # 8 @ 30fps -> 6.6 @ 25fps to cover same duration
        temporal_stride = temporal_stride * (fps / target_fps)
        num_frames_per_clip = int(
            math.ceil(max(num_frames - 1, 1) * temporal_stride))

        clip_duration = float(num_frames_per_clip) / fps
        clip_duration = min(clip_duration, video_duration)
        assert video_duration >= clip_duration, \
            "Video duration must be greater than clip duration"

        clip_durations.append(clip_duration)
        adjusted_strides.append(temporal_stride)

    return clip_durations, adjusted_strides


def clip_sampler(
    num_frames: int,
    sampler_type: str,
    temporal_strides: List,
    video_duration: float,
    fps: float,
    target_fps: float,
    num_clips: int = 1
):

    assert sampler_type in ["random", "sliding", "center"]

    clip_durations, temporal_strides = get_clip_duration(
        num_frames=num_frames,
        temporal_strides=temporal_strides,
        fps=fps,
        target_fps=target_fps,
        video_duration=video_duration
    )

    if num_clips == 1:
        # training (can be a randomly chosen stride)
        # or single view testing (single fixed stride)
        assert sampler_type in ["random", "center"]
        assert len(clip_durations) == len(temporal_strides)

        random_selection = random.randint(0, len(clip_durations)-1)
        temporal_strides = [temporal_strides[random_selection]]
        clip_duration = clip_durations[random_selection]

        delta = max(video_duration - clip_duration, 0)
        clips_starts_ends = [sample(sampler_type, delta,
                                    clip_duration, video_duration)]

    elif len(clip_durations) > 1:
        # multi-view testing with different strides
        # must sampling type of "center"
        assert sampler_type == "center" and num_clips == len(temporal_strides)

        clips_starts_ends = [
            sample(sampler_type, max(video_duration - clip_duration, 0),
                   clip_duration, video_duration) for clip_duration in clip_durations
        ]

    else:
        # multi-view testing with sliding temporal window
        assert sampler_type == "sliding"
        assert len(clip_durations) == 1 and len(temporal_strides) == 1

        delta = max(video_duration - clip_durations[0], 0)
        clip_starts = np.linspace(0.0, delta, num_clips)
        clips_starts_ends = [(start, start+clip_durations[0])
                             for start in clip_starts]
        temporal_strides = temporal_strides * len(clips_starts_ends)

    return clips_starts_ends, temporal_strides


def get_video_info(file_path: str, target_fps: int, file_type: str):
    assert file_type in ["video", "hdf5", "frames"]

    if file_type == "video":
        fps, width, height, frame_count = get_video_info_with_torchvision(
            file_path)

        return fps, width, height, frame_count

    if file_type == "frames":
        dummy_frame = read_image_frame(file_path[0])
        frame_count = len(file_path)

    elif file_type == "hdf5":
        video_name = Path(file_path).name.split(".")[0]

        with h5py.File(file_path, 'r') as video_h5:
            frame_count = len(video_h5[video_name])
            dummy_frame = cv2.imdecode(
                video_h5[video_name][0], flags=cv2.IMREAD_COLOR)
            dummy_frame = cv2.cvtColor(dummy_frame, cv2.COLOR_BGR2RGB)

    height = dummy_frame.shape[0]
    width = dummy_frame.shape[1]
    return target_fps, width, height, frame_count


def get_video_type(file_path: Union[List, str]):
    path = file_path[0] if isinstance(file_path, list) else file_path

    if Path(path).suffix == ".h5":
        return "hdf5"

    if Path(path).suffix in [".jpg", ".png", ".jpeg"]:
        return "frames"

    if Path(path).suffix in [".mp4", ".avi"]:
        return "video"

    raise ValueError(f"File type {Path(path).suffix} is not supported")
