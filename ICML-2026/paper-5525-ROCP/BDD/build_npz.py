import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

# Default dataset layout (can be overridden by CLI flags)
BDD_DATA_ROOT = Path("/path/to/your/bdd100k")
DEFAULT_IMAGES_DIR = BDD_DATA_ROOT / "images" / "100k" / "val"
DEFAULT_LABELS_JSON = BDD_DATA_ROOT / "labels" / "bdd100k_labels_images_val.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# ROI thresholds (normalized)
LEFT_MAX_U = 1.0 / 3.0
RIGHT_MIN_U = 2.0 / 3.0
AHEAD_MIN_VBOT = 0.6
AHEAD_RHO_MIN = 0.15
SIDE_MIN_VBOT = 0.5
SIDE_RHO_MIN = 0.10

# Category filters to avoid treating irrelevant objects as obstacles.
# BDD labels (truth)
PED_CATEGORIES = {"person", "rider"}
VEH_CATEGORIES = {"car", "truck", "bus", "train", "motor", "bike"}
AHEAD_CATEGORIES = PED_CATEGORIES | VEH_CATEGORIES

# YOLO class names (prediction). IDs are built from model.names at runtime.
PRED_PED_NAMES = {"person", "rider"}
PRED_VEH_NAMES = {
    "car",
    "truck",
    "bus",
    "train",
    "motor",
    "motorcycle",
    "motorbike",
    "bike",
    "bicycle",
}


def parse_args():
    parser = argparse.ArgumentParser(description="One-step build of (scores, true_bits) NPZ.")
    parser.add_argument("--labels-json", type=Path, default=DEFAULT_LABELS_JSON)
    parser.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--out-npz", type=Path, default=DEFAULT_OUTPUT_DIR / "prob_true.npz")
    parser.add_argument("--model", type=str, default="yolo11n.pt")
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.7, help="YOLO internal NMS IoU threshold.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def load_bdd_labels(labels_json_path: Path):
    with open(labels_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_bdd_samples(labels_json_path: Path):
    data = load_bdd_labels(labels_json_path)
    for item in data:
        name = item.get("name")
        frames = item.get("frames") or []
        objects = []
        if frames:
            objects = frames[0].get("objects") or []
        yield name, objects


def get_image_size(image_path: Path):
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height


def resolve_image_path(images_root: Path, name: str):
    name_path = Path(name)
    if name_path.is_absolute() and name_path.exists():
        return name_path
    cand1 = images_root / name_path
    if cand1.exists():
        return cand1
    cand2 = images_root / name_path.name
    if cand2.exists():
        return cand2
    if name_path.suffix == "":
        for ext in (".jpg", ".png"):
            cand3 = cand1.with_suffix(ext)
            if cand3.exists():
                return cand3
            cand4 = cand2.with_suffix(ext)
            if cand4.exists():
                return cand4
    return None


def compute_y_true(objects, width, height):
    y1 = 0  # ahead-close (includes vehicles)
    y2 = 0  # left
    y3 = 0  # right

    for obj in objects:
        category = obj.get("category")
        box2d = obj.get("box2d")
        if category is None or box2d is None:
            continue
        x1 = box2d.get("x1")
        y1b = box2d.get("y1")
        x2 = box2d.get("x2")
        y2b = box2d.get("y2")
        if None in (x1, y1b, x2, y2b):
            continue

        cx = (x1 + x2) / 2.0
        u = cx / float(width)
        v_bot = y2b / float(height)
        rho = (y2b - y1b) / float(height)

        left = u <= LEFT_MAX_U
        right = u >= RIGHT_MIN_U
        ahead_close = (
            (u >= LEFT_MAX_U)
            and (u <= RIGHT_MIN_U)
            and (v_bot >= AHEAD_MIN_VBOT)
            and (rho >= AHEAD_RHO_MIN)
        )
        side_close = (v_bot >= SIDE_MIN_VBOT) and (rho >= SIDE_RHO_MIN)
        left_close = left and side_close
        right_close = right and side_close

        if category in AHEAD_CATEGORIES and ahead_close:
            y1 = 1
        if category in VEH_CATEGORIES and left_close:
            y2 = 1
        if category in VEH_CATEGORIES and right_close:
            y3 = 1

        if y1 and y2 and y3:
            break

    return y1, y2, y3


def _class_mask(class_ids: torch.Tensor, allowed_ids):
    mask = torch.zeros_like(class_ids, dtype=torch.bool)
    for cid in allowed_ids:
        mask |= class_ids == cid
    return mask


def _masked_max_conf(scores: torch.Tensor, mask: torch.Tensor):
    if mask.sum() == 0:
        return 0.0
    return float(scores[mask].max().item())


def compute_s_scores(
    boxes_xyxy: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    width: int,
    height: int,
    ahead_ids,
    veh_ids,
):
    if boxes_xyxy.numel() == 0:
        return 0.0, 0.0, 0.0

    centers = (boxes_xyxy[:, 0:2] + boxes_xyxy[:, 2:4]) / 2.0
    u = centers[:, 0] / float(width)
    v_bot = boxes_xyxy[:, 3] / float(height)
    rho = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) / float(height)

    left_mask = u <= LEFT_MAX_U
    right_mask = u >= RIGHT_MIN_U
    ahead_close_mask = (v_bot >= AHEAD_MIN_VBOT) & (rho >= AHEAD_RHO_MIN)
    side_close_mask = (v_bot >= SIDE_MIN_VBOT) & (rho >= SIDE_RHO_MIN)
    ahead_mask = (
        (u >= LEFT_MAX_U)
        & (u <= RIGHT_MIN_U)
        & ahead_close_mask
    )

    ahead_class_mask = _class_mask(class_ids, ahead_ids)
    veh_mask = _class_mask(class_ids, veh_ids)

    s1 = _masked_max_conf(scores, ahead_class_mask & ahead_mask)
    s2 = _masked_max_conf(scores, veh_mask & left_mask & side_close_mask)
    s3 = _masked_max_conf(scores, veh_mask & right_mask & side_close_mask)
    return s1, s2, s3


def build_fx(p1, p2, p3):
    p1c = np.clip(p1, 0.0, 1.0)
    p2c = np.clip(p2, 0.0, 1.0)
    p3c = np.clip(p3, 0.0, 1.0)
    f000 = (1 - p1c) * (1 - p2c) * (1 - p3c)
    f001 = (1 - p1c) * (1 - p2c) * p3c
    f010 = (1 - p1c) * p2c * (1 - p3c)
    f011 = (1 - p1c) * p2c * p3c
    f100 = p1c * (1 - p2c) * (1 - p3c)
    f101 = p1c * (1 - p2c) * p3c
    f110 = p1c * p2c * (1 - p3c)
    f111 = p1c * p2c * p3c
    return np.stack([f000, f001, f010, f011, f100, f101, f110, f111], axis=1)


def _model_name_map(model):
    names = getattr(model, "names", None)
    if names is None:
        raise ValueError("model.names not found; cannot build class id mapping.")
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names)}
    raise ValueError("Unsupported model.names type.")


def _ids_from_names(model, target_names):
    name_map = _model_name_map(model)
    target_lower = {n.lower() for n in target_names}
    ids = {i for i, n in name_map.items() if n.lower() in target_lower}
    return ids


def top1_accuracy(probability: np.ndarray, true_label: np.ndarray) -> float:
    pred_idx = np.argmax(probability, axis=1)
    true_idx = (true_label[:, 0] * 4 + true_label[:, 1] * 2 + true_label[:, 2]).astype(np.int64)
    return float(np.mean(pred_idx == true_idx))


def main():
    args = parse_args()
    args.out_npz.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    ped_ids = _ids_from_names(model, PRED_PED_NAMES)
    veh_ids = _ids_from_names(model, PRED_VEH_NAMES)
    ahead_ids = ped_ids | veh_ids
    print("ped_ids:", sorted(ped_ids), "veh_ids:", sorted(veh_ids), "ahead_ids:", sorted(ahead_ids))
    if not veh_ids:
        raise ValueError("veh_ids is empty: check model.names vs PRED_VEH_NAMES.")
    if not ahead_ids:
        raise ValueError("ahead_ids is empty: check model.names vs PRED_*_NAMES.")

    y_list = []
    s_list = []
    missing_images = 0
    for idx, (name, objects) in enumerate(iter_bdd_samples(args.labels_json)):
        if args.max_samples is not None and idx >= args.max_samples:
            break
        if not name:
            continue

        image_path = resolve_image_path(args.images_root, name)
        if image_path is None:
            missing_images += 1
            continue

        width, height = get_image_size(image_path)
        y1, y2, y3 = compute_y_true(objects, width, height)

        results = model.predict(
            source=str(image_path),
            conf=args.conf_thres,
            iou=args.iou_thres,
            device=args.device,
            imgsz=args.imgsz,
            verbose=False,
        )
        r = results[0]
        height_det, width_det = r.orig_shape

        if r.boxes is None or r.boxes.xyxy is None or r.boxes.xyxy.numel() == 0:
            s1 = s2 = s3 = 0.0
        else:
            boxes_xyxy = r.boxes.xyxy
            scores = r.boxes.conf
            class_ids = r.boxes.cls.to(dtype=torch.int64)
            s1, s2, s3 = compute_s_scores(
                boxes_xyxy=boxes_xyxy,
                scores=scores,
                class_ids=class_ids,
                width=width_det,
                height=height_det,
                ahead_ids=ahead_ids,
                veh_ids=veh_ids,
            )

        y_list.append((y1, y2, y3))
        s_list.append((s1, s2, s3))

    if missing_images:
        print(f"Warning: {missing_images} images were missing and skipped.")

    if not y_list:
        raise ValueError("No samples processed. Check dataset paths.")

    y_arr = np.asarray(y_list, dtype=np.int64)
    s_arr = np.asarray(s_list, dtype=np.float32)
    print("mean(y_true) =", y_arr.mean(axis=0))
    print("mean(s)      =", s_arr.mean(axis=0))
    freq = np.bincount(y_arr[:, 0] * 4 + y_arr[:, 1] * 2 + y_arr[:, 2], minlength=8) / len(y_arr)
    print("freq(y)      =", freq)
    pred_bits = (s_arr > 0.5).astype(np.int64)
    bit_acc = (pred_bits == y_arr).mean(axis=0)
    print("bit-acc@0.5  =", bit_acc)
    # NOTE: We save only raw bit scores and true bits. Downstream evaluation can
    # fit calibrators (e.g., isotonic regression) and then rebuild an 8-class
    # distribution.
    np.savez(args.out_npz, scores=s_arr, true_bits=y_arr)

    # Optional: sanity check (treat raw scores as probabilities).
    probability_raw = build_fx(s_arr[:, 0], s_arr[:, 1], s_arr[:, 2]).astype(np.float32)
    acc_raw = top1_accuracy(probability_raw, y_arr)
    print(f"Saved {len(y_list)} rows to {args.out_npz} (scores, true_bits)")
    print(f"Top-1 accuracy (raw, uncalibrated): {acc_raw:.4f}")


if __name__ == "__main__":
    main()
