import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VERSION_PATTERN = re.compile(r"(.*)_v(\d+)$")


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def split_basename_and_version(stem: str) -> Tuple[str, str]:
    match = VERSION_PATTERN.fullmatch(stem)
    if match:
        return match.group(1), f"v{match.group(2)}"
    return stem, ""


class PairedDegradedImageDataset(Dataset):
    """Dataset for paired clean and degraded images under root/source/{ref,dis}."""

    def __init__(self, root_dir: str, image_processor):
        self.root_dir = Path(root_dir).absolute()
        self.image_processor = image_processor
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {self.root_dir}")

        self.samples: List[Dict[str, Any]] = []
        self._scan_sources()
        if not self.samples:
            raise RuntimeError(f"No paired ref/dis images found under {self.root_dir}")

        source_names = sorted({sample["source"] for sample in self.samples})
        print(
            f"[PairedDegradedImageDataset] Loaded {len(self.samples)} pairs "
            f"from {len(source_names)} sources: {source_names}"
        )

    def _scan_sources(self) -> None:
        for source_dir in sorted(self.root_dir.iterdir()):
            if not source_dir.is_dir():
                continue

            ref_dir = source_dir / "ref"
            dis_dir = source_dir / "dis"
            if not (ref_dir.is_dir() and dis_dir.is_dir()):
                continue

            ref_index: Dict[str, Path] = {
                path.stem: path for path in ref_dir.iterdir() if is_image_file(path)
            }
            if not ref_index:
                warnings.warn(f"Source {source_dir.name} has no reference images; skipping.")
                continue

            match_count = 0
            for degraded_path in dis_dir.iterdir():
                if not is_image_file(degraded_path):
                    continue
                base_name, version = split_basename_and_version(degraded_path.stem)
                ref_path = ref_index.get(base_name)
                if ref_path is None:
                    continue
                self.samples.append(
                    {
                        "source": source_dir.name,
                        "ref_path": str(ref_path.absolute()),
                        "dis_path": str(degraded_path.absolute()),
                        "version": version,
                    }
                )
                match_count += 1

            if match_count == 0:
                warnings.warn(f"Source {source_dir.name} has no matching degraded images.")

    def __len__(self) -> int:
        return len(self.samples)

    def _open_rgb(self, path: str) -> Optional[Image.Image]:
        try:
            return Image.open(path).convert("RGB")
        except Exception as exc:
            warnings.warn(f"Failed to open image as RGB: {path} | {exc}")
            return None

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        record = self.samples[idx]
        clean_image = self._open_rgb(record["ref_path"])
        degraded_image = self._open_rgb(record["dis_path"])
        if clean_image is None or degraded_image is None:
            return None

        try:
            clean_batch = self.image_processor(images=[clean_image], return_tensors="pt")
            degraded_batch = self.image_processor(images=[degraded_image], return_tensors="pt")
            clean_tensor = clean_batch["pixel_values"][0]
            degraded_tensor = degraded_batch["pixel_values"][0]
        except Exception as exc:
            warnings.warn(
                "Image processor failed for "
                f"clean={record['ref_path']}, degraded={record['dis_path']} | {exc}"
            )
            return None

        if clean_tensor.dim() != 3 or clean_tensor.size(0) != 3:
            warnings.warn(f"Clean tensor is not 3-channel: {tuple(clean_tensor.shape)}")
            return None
        if degraded_tensor.dim() != 3 or degraded_tensor.size(0) != 3:
            warnings.warn(f"Degraded tensor is not 3-channel: {tuple(degraded_tensor.shape)}")
            return None

        return {
            "degraded_pixel_values": degraded_tensor,
            "clean_pixel_values": clean_tensor,
            "source": record["source"],
            "ref_path": record["ref_path"],
            "dis_path": record["dis_path"],
            "version": record["version"],
        }


def paired_image_collate_fn(batch: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    valid = [item for item in batch if item is not None]
    if not valid:
        return None

    import torch

    return {
        "degraded_pixel_values": torch.stack(
            [item["degraded_pixel_values"] for item in valid], dim=0
        ),
        "clean_pixel_values": torch.stack(
            [item["clean_pixel_values"] for item in valid], dim=0
        ),
        "source": [item["source"] for item in valid],
        "ref_path": [item["ref_path"] for item in valid],
        "dis_path": [item["dis_path"] for item in valid],
        "version": [item["version"] for item in valid],
    }
