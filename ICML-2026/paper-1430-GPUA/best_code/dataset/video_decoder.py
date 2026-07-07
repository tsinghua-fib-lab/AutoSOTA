import io
import math
from typing import List

import torch
import torchvision
from loguru import logger
from .transforms import BICUBIC


from .video_utils import (
    read_h5_frames,
    read_image_frame,
    sample_frames_with_stride,
    secs_to_pts,
)

try:
    import av
except ImportError:
    logger.warning("PyAV not found.")
    av = None


def torchvision_decode_video(
    file_path: str,
    clips_starts_ends: List,
    num_frames: int,
    temporal_strides: List,
    **_,
) -> list:

    video_reader = torchvision.io.VideoReader(
        str(file_path), stream="video", num_threads=2
    )

    fps = video_reader.get_metadata()["video"]["fps"][0]
    time_step = 1.0 / fps

    vid_start = min([start for start, _ in clips_starts_ends])
    vid_end = max([end for _, end in clips_starts_ends])

    all_frames = []
    curr_frame_time = vid_start
    for frame in video_reader.seek(vid_start):
        all_frames.append(frame["data"])

        if curr_frame_time > vid_end:
            break
        curr_frame_time += time_step

    # each frame is already CHW frame
    all_frames = [
        frame.div(255) if frame.max() > 1.0 else frame for frame in all_frames
    ]

    all_frames = torch.stack(all_frames).to(torch.float32)

    sampeled_clips = []

    for idx, (start_secs, end_secs) in enumerate(clips_starts_ends):
        start_idx = int(math.floor((start_secs - vid_start) * fps))
        end_idx = int(math.ceil((end_secs - vid_start) * fps)) + 1

        frames = all_frames[start_idx:end_idx]

        # Sample num_frames with a gap == stride
        temporal_stride = temporal_strides[idx]
        frames = sample_frames_with_stride(frames, num_frames, temporal_stride)
        assert frames.shape[0] == num_frames

        # TCHW -> CTHW
        frames = frames.permute(1, 0, 2, 3)
        sampeled_clips.append(frames)

    return sampeled_clips


def pyav_decode_video(
    file_path: str,
    clips_starts_ends: List,
    num_frames: int,
    temporal_strides: List,
    **_,
) -> list:

    with open(file_path, "rb") as f:
        video_file = io.BytesIO(f.read())

    container = av.open(video_file)

    assert (
        container is not None and len(container.streams.video) > 0
    ), f"Video stream not found for file {file_path}"

    video_stream = container.streams.video[0]
    video_time_base = float(video_stream.time_base)
    video_start_pts = (
        0.0 if video_stream.start_time is None else video_stream.start_time
    )
    fps = float(container.streams.video[0].average_rate)

    vid_start = min([start for start, _ in clips_starts_ends])
    end_start = max([end for _, end in clips_starts_ends])

    start_pts = secs_to_pts(
        vid_start, video_time_base, video_start_pts, round_mode="floor"
    )
    end_pts = secs_to_pts(
        end_start, video_time_base, video_start_pts, round_mode="ceil"
    )

    seek_offset = max(start_pts - 1024, 0)

    container.seek(
        int(seek_offset), any_frame=False, backward=True, stream=video_stream
    )

    all_frames = {}
    for frame in container.decode(video=0):
        if frame.pts >= start_pts and frame.pts <= end_pts:
            all_frames[frame.pts] = frame
        elif frame.pts >= end_pts:
            break

    container.close()

    all_frames = [all_frames[i] for i in sorted(all_frames.keys())]
    all_frames = [frame.to_rgb().to_ndarray()
                  for frame in all_frames]  # each frame is HWC
    all_frames = [
        torchvision.transforms.functional.to_tensor(frame) for frame in all_frames
    ]  # now each frame is CHW

    # THWC
    all_frames = torch.stack(all_frames).to(torch.float32)

    sampeled_clips = []

    for idx, (start_secs, end_secs) in enumerate(clips_starts_ends):
        start_idx = int(math.floor((start_secs - vid_start) * fps))
        end_idx = int(math.ceil((end_secs - vid_start) * fps)) + 1
        frames = all_frames[start_idx:end_idx]

        # Sample num_frames with a gap == stride
        temporal_stride = temporal_strides[idx]
        frames = sample_frames_with_stride(frames, num_frames, temporal_stride)
        assert frames.shape[0] == num_frames

        # TCHW -> CTHW
        frames = frames.permute(1, 0, 2, 3)
        sampeled_clips.append(frames)

    return sampeled_clips


def load_image_frames(
    file_path: str,
    clips_starts_ends: List,
    num_frames: int,
    fps: int,
    temporal_strides: List,
    duration: float,
) -> list:
    # this is > 1 only in multi crop inference
    sampeled_clips = []

    for idx, (start_secs, end_secs) in enumerate(clips_starts_ends):
        assert (
            start_secs >= 0 and start_secs < duration
        ), "start_secs must be between 0 and duration"

        end_secs = min(end_secs, duration)

        start_frame_idx = int(math.floor(fps * start_secs))
        end_frame_idx = int(math.ceil(fps * end_secs))
        frame_indices = list(range(start_frame_idx, end_frame_idx))

        frame_paths = [file_path[i] for i in frame_indices]
        frames = [
            read_image_frame(frame_path) for frame_path in frame_paths
        ]  # each frame is HWC
        frames = [
            torchvision.transforms.functional.to_tensor(frame) for frame in frames
        ]  # now each frame is CHW

        frames = torch.stack(frames).to(torch.float32)

        # Sample num_frames with a gap == stride
        temporal_stride = temporal_strides[idx]
        frames = sample_frames_with_stride(frames, num_frames, temporal_stride)
        assert frames.shape[0] == num_frames

        # TCHW to CTHW
        frames = frames.permute(1, 0, 2, 3)

        sampeled_clips.append(frames)

    return sampeled_clips


def load_h5_frames(
    file_path: str,
    clips_starts_ends: List,
    num_frames: int,
    fps: int,
    temporal_strides: List,
    duration: float,
) -> list:
    # this is > 1 only in multi crop inference
    sampeled_clips = []

    for idx, (start_secs, end_secs) in enumerate(clips_starts_ends):
        assert (
            start_secs >= 0 and start_secs < duration
        ), "start_secs must be between 0 and duration"

        end_secs = min(end_secs, duration)

        start_frame_idx = int(math.floor(fps * start_secs))
        end_frame_idx = int(math.ceil(fps * end_secs))
        frame_indices = list(range(start_frame_idx, end_frame_idx))

        frames = read_h5_frames(file_path, frame_indices)  # each frame is HWC
        frames = [
            torchvision.transforms.functional.to_tensor(frame) for frame in frames
        ]  # now each frame is CHW

        frames = torch.stack(frames).to(torch.float32)

        # Sample num_frames with a gap == stride
        temporal_stride = temporal_strides[idx]
        frames = sample_frames_with_stride(frames, num_frames, temporal_stride)
        assert frames.shape[0] == num_frames

        # TCHW to CTHW
        frames = frames.permute(1, 0, 2, 3)

        sampeled_clips.append(frames)

    return sampeled_clips
