from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml
from PIL import Image, ImageOps
from torch.utils.data import Dataset


DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "images"
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file into a python dictionary."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def first_param_point(params_grid: Dict[str, Sequence[Any]]) -> Dict[str, Any]:
    """Select the first value from each parameter list for a quick default run."""

    def _pick(value: Sequence[Any] | Any) -> Any:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if not value:
                raise ValueError("Param grid contains an empty list; cannot determine default.")
            return value[0]
        return value

    return {key: _pick(values) for key, values in params_grid.items()}


@dataclass(frozen=True)
class EditRecord:
    image_path: Path
    src_prompt: str
    tgt_prompt: str
    edit_prompt: str
    edit_id: Optional[str] = None


class LocalEditDataset(Dataset):
    """Simple dataset mirroring src/utils/mydataset.py for local demos."""

    def __init__(self, records: List[EditRecord], image_size: int = 512, use_center_crop: bool = False) -> None:
        if not records:
            raise ValueError("No records found in the dataset root.")
        self._records = records
        self.image_size = int(image_size)
        self._use_center_crop = bool(use_center_crop)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # type: ignore[override]
        record = self._records[idx]
        image = Image.open(record.image_path).convert("RGB")
        if self._use_center_crop:
            image = _center_square_crop(image)
        image = _resize_image(image, (self.image_size, self.image_size))

        blank = Image.new("RGB", image.size, color=(255, 255, 255))
        return {
            "id": record.edit_id or Path(record.image_path).stem,
            "original_image": image,
            "edited_image": blank,
            "original_prompt": record.src_prompt,
            "edited_prompt": record.tgt_prompt,
            "edit_prompt": record.edit_prompt,
            "image_path": str(record.image_path),
        }


def load_local_dataset(
    path: str | Path | None = None,
    image_size: int = 512,
    center_crop: bool = True,
) -> LocalEditDataset:
    root = _resolve_dataset_root(path)
    records = _parse_edit_records(root)
    return LocalEditDataset(records=records, image_size=image_size, use_center_crop=center_crop)


def _resolve_dataset_root(path: str | Path | None) -> Path:
    if path is not None:
        root = Path(path).expanduser().resolve()
    else:
        root = DEFAULT_DATA_ROOT
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    return root


def _parse_edit_records(root: Path) -> List[EditRecord]:
    records: List[EditRecord] = []
    for subdir in sorted(p for p in root.iterdir() if p.is_dir()):
        meta_file = subdir / "meta.jsonl"
        if not meta_file.exists():
            continue
        try:
            image_path = _select_image_file(subdir)
        except FileNotFoundError:
            continue

        with meta_file.open("r", encoding="utf-8") as handle:
            for line_num, raw_line in enumerate(handle, start=1):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {meta_file} at line {line_num}: {exc}") from exc

                records.append(
                    EditRecord(
                        image_path=image_path,
                        src_prompt=record.get("original_prompt", ""),
                        tgt_prompt=record.get("edited_prompt", ""),
                        edit_prompt=record.get("edit_prompt", record.get("edited_prompt", "")),
                        edit_id=record.get("edit_id"),
                    )
                )

    if not records:
        raise FileNotFoundError(
            f"No edit samples found under {root}. Expected subdirectories with 'meta.jsonl' files."
        )
    return records


def _select_image_file(folder: Path) -> Path:
    candidates = [
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    ]
    if not candidates:
        raise FileNotFoundError(f"No RGB image found inside {folder}")

    preferred = sorted(
        (p for p in candidates if p.stem.lower() in {"i", "image", "original"}),
        key=lambda p: p.name,
    )
    if preferred:
        return preferred[0]
    return sorted(candidates, key=lambda p: p.name)[0]


def _center_square_crop(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width == height:
        return image

    target_size = min(width, height)
    try:
        resample = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
    except AttributeError:  # pragma: no cover
        resample = Image.LANCZOS

    return ImageOps.fit(
        image,
        (target_size, target_size),
        method=resample,
        centering=(0.5, 0.5),
    )


def _resize_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    try:
        resample = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
    except AttributeError:  # pragma: no cover
        resample = Image.LANCZOS
    return image.resize(size, resample=resample)
