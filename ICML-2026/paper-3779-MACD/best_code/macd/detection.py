from __future__ import annotations

import argparse
import math
import os
from collections import defaultdict
from typing import Any

import numpy as np

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional progress dependency
    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return iterable

from macd.io import ensure_dir, read_jsonl


def xyxy_to_cxcywh_norm(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[float, float, float, float]:
    x1 = max(0.0, min(float(x1), width - 1.0))
    y1 = max(0.0, min(float(y1), height - 1.0))
    x2 = max(0.0, min(float(x2), width - 1.0))
    y2 = max(0.0, min(float(y2), height - 1.0))
    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    cx = x1 + box_w / 2.0
    cy = y1 + box_h / 2.0
    return (
        0.0 if width <= 0 else cx / float(width),
        0.0 if height <= 0 else cy / float(height),
        0.0 if width <= 0 else box_w / float(width),
        0.0 if height <= 0 else box_h / float(height),
    )


def build_sample_grid(duration_sec: float, metadata_fps: float) -> np.ndarray:
    if metadata_fps <= 0:
        raise ValueError("metadata_fps must be positive")
    step = 1.0 / metadata_fps
    count = max(1, int(math.floor(duration_sec / step)) + 1)
    return np.clip(np.arange(count, dtype=np.float64) * step, 0.0, max(0.0, duration_sec))


def group_consecutive(indices: list[int]) -> list[list[int]]:
    if not indices:
        return []
    groups = [[indices[0]]]
    for idx in indices[1:]:
        if idx == groups[-1][-1] + 1:
            groups[-1].append(idx)
        else:
            groups.append([idx])
    return groups


def category_name(names_obj: Any, cls_id: int) -> str:
    if isinstance(names_obj, dict):
        return str(names_obj.get(cls_id, cls_id))
    if isinstance(names_obj, (list, tuple)) and 0 <= cls_id < len(names_obj):
        return str(names_obj[cls_id])
    return str(cls_id)


def process_video_with_yolo(
    video_path: str,
    model: Any,
    device: str,
    conf: float,
    iou: float,
    metadata_fps: float,
    min_frames_per_object: int,
    min_score: float,
    max_objects_per_video: int,
    vid_stride: int,
    imgsz: int,
    half: bool,
    classes: list[int] | None = None,
) -> dict[str, Any]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = float(frame_count) / fps if fps > 0 else 0.0
    cap.release()

    track_data: dict[int, list[tuple[int, float, float, float, float, float, int]]] = defaultdict(list)
    frame_index = -1
    for result in model.track(
        source=video_path,
        conf=conf,
        iou=iou,
        device=device,
        classes=classes,
        verbose=False,
        stream=True,
        persist=True,
        vid_stride=max(1, int(vid_stride)),
        imgsz=int(imgsz),
        half=bool(half),
    ):
        frame_index += 1
        boxes = getattr(result, "boxes", None)
        if boxes is None or getattr(boxes, "id", None) is None:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy() if getattr(boxes, "conf", None) is not None else np.ones(len(ids))
        clss = boxes.cls.cpu().numpy().astype(int) if getattr(boxes, "cls", None) is not None else np.full(len(ids), -1)
        for i, track_id in enumerate(ids):
            score = float(confs[i])
            if score < min_score:
                continue
            cx, cy, box_w, box_h = xyxy_to_cxcywh_norm(*map(float, xyxy[i].tolist()), width=width, height=height)
            track_data[int(track_id)].append((frame_index, cx, cy, box_w, box_h, score, int(clss[i])))

    t_grid = build_sample_grid(duration_sec, metadata_fps)
    names_obj = getattr(model, "names", None) or getattr(getattr(model, "model", None), "names", None)
    objects: list[dict[str, Any]] = []
    sorted_tracks = sorted(track_data.items(), key=lambda item: len(item[1]), reverse=True)
    if max_objects_per_video > 0:
        sorted_tracks = sorted_tracks[:max_objects_per_video]

    for track_id, detections in sorted_tracks:
        if len(detections) < max(1, min_frames_per_object):
            continue
        grid_map: dict[int, tuple[float, float, float, float, float, int]] = {}
        for frame_idx, cx, cy, box_w, box_h, score, cls_id in sorted(detections):
            t_sec = 0.0 if fps <= 0 else float(frame_idx) / fps
            grid_idx = max(0, min(int(round(t_sec * metadata_fps)), len(t_grid) - 1))
            if grid_idx not in grid_map or score > grid_map[grid_idx][4]:
                grid_map[grid_idx] = (cx, cy, box_w, box_h, score, cls_id)
        segments = []
        for group in group_consecutive(sorted(grid_map.keys())):
            if len(group) < min_frames_per_object:
                continue
            values = [grid_map[idx] for idx in group]
            segments.append(
                {
                    "t_start": float(t_grid[group[0]]),
                    "t_end": float(t_grid[group[-1]]),
                    "t": [float(t_grid[idx]) for idx in group],
                    "cx": [float(v[0]) for v in values],
                    "cy": [float(v[1]) for v in values],
                    "w": [float(v[2]) for v in values],
                    "h": [float(v[3]) for v in values],
                    "score": [float(v[4]) for v in values],
                }
            )
        if not segments:
            continue
        cls_ids = [value[5] for value in grid_map.values()]
        cls_id = max(set(cls_ids), key=cls_ids.count) if cls_ids else -1
        objects.append(
            {
                "object_id": f"track_{track_id}",
                "tracker_id": int(track_id),
                "category": int(cls_id),
                "category_name": category_name(names_obj, int(cls_id)),
                "avg_score": float(np.mean([value[4] for value in grid_map.values()])),
                "segments": segments,
            }
        )

    objects.sort(key=lambda obj: obj.get("avg_score", 0.0), reverse=True)
    return {
        "orig_video_path": video_path,
        "original_fps": fps,
        "width": width,
        "height": height,
        "duration_sec": duration_sec,
        "metadata_fps": metadata_fps,
        "objects": objects,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a YOLO object-track manifest for MACD.")
    parser.add_argument("--question-file", required=True, help="Input question JSONL with image/video_id/text fields.")
    parser.add_argument("--orig-video-dir", required=True, help="Directory containing original videos.")
    parser.add_argument("--output-manifest", required=True, help="Output manifest JSONL.")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics YOLO weights or model name.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--classes", type=int, nargs="*", default=None)
    parser.add_argument("--metadata-fps", type=float, default=1.0)
    parser.add_argument("--min-frames", type=int, default=1)
    parser.add_argument("--min-score", type=float, default=0.25)
    parser.add_argument("--max-objects-per-video", type=int, default=200)
    parser.add_argument("--vid-stride", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true")
    args = parser.parse_args()

    from ultralytics import YOLO

    ensure_dir(os.path.dirname(args.output_manifest))
    questions = read_jsonl(args.question_file)
    yolo = YOLO(args.model)
    seen: set[str] = set()
    with open(args.output_manifest, "w", encoding="utf-8") as output:
        for question in tqdm(questions, desc="videos"):
            video_id = str(question.get("video_id", ""))
            if not video_id or video_id in seen:
                continue
            seen.add(video_id)
            rel_path = question.get("image")
            if not rel_path:
                continue
            video_path = os.path.join(args.orig_video_dir, rel_path)
            if not os.path.isfile(video_path):
                print(f"[WARN] Missing video for {video_id}: {video_path}")
                continue
            manifest = process_video_with_yolo(
                video_path=video_path,
                model=yolo,
                device=args.device,
                conf=args.conf,
                iou=args.iou,
                metadata_fps=args.metadata_fps,
                min_frames_per_object=args.min_frames,
                min_score=args.min_score,
                max_objects_per_video=args.max_objects_per_video,
                vid_stride=args.vid_stride,
                imgsz=args.imgsz,
                half=args.half,
                classes=args.classes,
            )
            manifest.update({"video_id": video_id, "question_id": question.get("question_id"), "prompt": question.get("text", "")})
            for idx, obj in enumerate(manifest["objects"], start=1):
                obj["object_id"] = f"{video_id}_obj_{idx}"
            output.write(__import__("json").dumps(manifest, ensure_ascii=False) + "\n")
            output.flush()


if __name__ == "__main__":
    main()
