from __future__ import annotations

import argparse
import gc
import os
from typing import Any

import numpy as np
import torch

from macd.io import ensure_dir, read_jsonl
from macd.masks import aggregate_masks, build_object_masks, build_time_grid

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional progress dependency
    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return iterable


def read_video(video_path: str, fps: float | None = None) -> tuple[torch.Tensor, float]:
    try:
        import decord

        reader = decord.VideoReader(video_path)
        orig_fps = float(reader.get_avg_fps()) if hasattr(reader, "get_avg_fps") else 0.0
        use_fps = fps if fps and fps > 0 else (orig_fps if orig_fps > 0 else 1.0)
        step = max(1, int(round(orig_fps / use_fps))) if orig_fps > 0 and use_fps > 0 else 1
        indices = np.arange(0, len(reader), step)
        if len(indices) == 0:
            indices = np.arange(min(len(reader), 1))
        frames = reader.get_batch(indices).asnumpy()
        tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
        return tensor, use_fps
    except Exception:
        pass

    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        orig_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        use_fps = fps if fps and fps > 0 else (orig_fps if orig_fps > 0 else 1.0)
        step = max(1, int(round(orig_fps / use_fps))) if orig_fps > 0 and use_fps > 0 else 1
        frames = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            idx += 1
        cap.release()
        if not frames:
            raise RuntimeError(f"No frames decoded from: {video_path}")
        array = np.stack(frames, axis=0)
        tensor = torch.from_numpy(array).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
        return tensor, use_fps
    except Exception:
        pass

    from torchvision.io import read_video as tv_read_video

    video, _, info = tv_read_video(video_path, pts_unit="sec")
    if video.numel() == 0:
        raise RuntimeError(f"No frames decoded from: {video_path}")
    orig_fps = float(info.get("video_fps", 0.0)) if isinstance(info, dict) else 0.0
    use_fps = fps if fps and fps > 0 else (orig_fps if orig_fps > 0 else 1.0)
    tensor = video.permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
    return tensor, use_fps


def write_video(path: str, video: torch.Tensor, fps: float) -> None:
    from fractions import Fraction
    fps = Fraction(fps).limit_denominator()

    ensure_dir(os.path.dirname(path))
    if video.ndim != 4:
        raise ValueError(f"Expected video as (C,T,H,W), got {tuple(video.shape)}")
    frames = video.clamp(0.0, 1.0).permute(1, 2, 3, 0)
    frames = (frames * 255.0).to(torch.uint8).cpu()
    T, H, W, C = frames.shape
    if H % 2 or W % 2:
        padded = torch.zeros((T, H + (H % 2), W + (W % 2), C), dtype=frames.dtype)
        padded[:, :H, :W, :] = frames
        frames = padded
    # Use av directly since torchvision 0.20+av 13+ has fps type issues
    import av
    container = av.open(path, mode="w")
    stream = container.add_stream("mpeg4", rate=fps)
    stream.width = frames.shape[2]
    stream.height = frames.shape[1]
    stream.pix_fmt = "yuv420p"
    for frame_np in frames.numpy():
        av_frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
        av_frame = av_frame.reformat(format="yuv420p")
        for packet in stream.encode(av_frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def sample_indices(total: int, keep: float | None) -> np.ndarray:
    if keep is None or keep <= 0:
        return np.arange(total, dtype=np.int64)
    target = int(round(total * keep)) if keep <= 1.0 else int(round(keep))
    target = max(1, min(total, target))
    if target >= total:
        return np.arange(total, dtype=np.int64)
    idx = np.floor(np.arange(target) * (total / float(target))).astype(np.int64)
    idx[-1] = total - 1
    return idx


def load_r_values(path: str) -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], float] = {}
    for item in read_jsonl(path):
        values[(str(item.get("video_id", "")), str(item.get("object_id", "")))] = float(item.get("r", 0.0))
    return values


def synthesize_counterfactuals(args: argparse.Namespace) -> None:
    manifests = read_jsonl(args.manifest_file)
    r_values = load_r_values(args.r_file)
    ensure_dir(args.output_dir)

    for entry in tqdm(manifests, desc="videos"):
        video_id = str(entry.get("video_id", ""))
        video_path = entry.get("orig_video_path")
        objects = entry.get("objects", [])
        if not video_path or not os.path.isfile(video_path) or not objects:
            continue

        object_ids = {object_id for (vid, object_id), value in r_values.items() if vid == video_id and value != 0.0}
        if not object_ids:
            print(f"[WARN] No nonzero r values for {video_id}")
            continue

        V, fps = read_video(video_path, fps=args.video_fps)
        _, _, T, H, W = V.shape
        keep_idx = sample_indices(T, args.frame_ratio)
        V = V[:, :, keep_idx, :, :]
        t_sample = build_time_grid(T, fps)[keep_idx]
        _, _, T, H, W = V.shape

        masks = build_object_masks(objects, t_sample, H, W, object_ids=object_ids)
        r_by_object = {object_id: r_values[(video_id, object_id)] for object_id in masks.keys() if (video_id, object_id) in r_values}
        if not r_by_object:
            continue
        mask = aggregate_masks(masks, r_by_object, mode=args.mode).to(V.dtype)
        V_counterfactual = V * (1.0 - mask)

        out_path = os.path.join(args.output_dir, f"{video_id}_merged_{args.mode}.mp4")
        write_video(out_path, V_counterfactual[0], fps=fps)

        del V, masks, mask, V_counterfactual
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize MACD counterfactual videos from manifest and r-values.")
    parser.add_argument("--manifest-file", required=True)
    parser.add_argument("--r-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--video-fps", type=float, default=1.0)
    parser.add_argument("--mode", choices=["max", "sum"], default="max")
    parser.add_argument("--frame-ratio", type=float, default=None, help="Optional ratio/count of frames to keep.")
    synthesize_counterfactuals(parser.parse_args())


if __name__ == "__main__":
    main()
