"""Stratified 1000-image ImageNet-val subsample for 4x SR benchmarking.

Data sources (tried in order for each image):
    1. <val_root>/raw/ILSVRC2012_val_0000XXXX.JPEG  (if already extracted)
    2. <val_tar_path> opened with tarfile + byte-offset index (NFS-friendly,
       no per-file sync overhead)

Extraction of 50k small files to NFS takes hours because each write pays a
sync round-trip; reading from a sequentially-indexed tar over NFS is ~100x
faster for a ~1000-file stratified subsample, so that path is the default.

The PyTorch-convention class id (0..999) is the alphabetical index of a WNID
within ``sorted(os.listdir(<train_root>))`` -- matching torchvision's
ImageFolder / all pretrained models. The train dir (set via the
``IMAGENET_TRAIN_ROOT`` env var) is only needed to (re)build the manifest;
normal runs read the prebuilt manifest CSV and never touch it.

The stratified subsample picks exactly one validation image per class, with a
deterministic choice driven by ``seed``.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import tarfile
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T


DEFAULT_VAL_ROOT = os.environ.get("IMAGENET_VAL_ROOT", "data/imagenet_val_256")
DEFAULT_VAL_TAR_PATH = os.environ.get("IMAGENET_VAL_TAR") or None
DEFAULT_TRAIN_ROOT_FOR_WNID_ORDER = os.environ.get("IMAGENET_TRAIN_ROOT", "")
_VAL_FILENAME_RE = re.compile(r"^ILSVRC2012_val_(\d{8})\.JPEG$", re.IGNORECASE)


@dataclass(frozen=True)
class ValItem:
    val_index_1based: int          # matches filename number
    filename: str                  # e.g. "ILSVRC2012_val_00000001.JPEG"
    wnid: str                      # e.g. "n01751748"
    pytorch_class_id: int          # 0..999


def _load_wnid_to_pytorch_id(train_root: str) -> dict:
    wnids = sorted(d for d in os.listdir(train_root) if d.startswith("n"))
    if len(wnids) != 1000:
        raise RuntimeError(
            f"Expected 1000 WNID subdirs under {train_root!r}, found {len(wnids)}"
        )
    return {wnid: idx for idx, wnid in enumerate(wnids)}


def _load_val_synsets(val_synsets_txt: str) -> List[str]:
    with open(val_synsets_txt, "r") as f:
        wnids = [line.strip() for line in f if line.strip()]
    if len(wnids) != 50_000:
        raise RuntimeError(
            f"Expected 50000 lines in {val_synsets_txt!r}, found {len(wnids)}"
        )
    return wnids


def build_val_manifest(
    val_root: str = DEFAULT_VAL_ROOT,
    train_root: str = DEFAULT_TRAIN_ROOT_FOR_WNID_ORDER,
) -> List[ValItem]:
    synsets_path = os.path.join(val_root, "val_synsets.txt")
    val_wnids = _load_val_synsets(synsets_path)
    wnid_to_id = _load_wnid_to_pytorch_id(train_root)
    items: List[ValItem] = []
    for i, wnid in enumerate(val_wnids, start=1):
        filename = f"ILSVRC2012_val_{i:08d}.JPEG"
        items.append(
            ValItem(
                val_index_1based=i,
                filename=filename,
                wnid=wnid,
                pytorch_class_id=wnid_to_id[wnid],
            )
        )
    return items


class TarImageStore:
    # Indexed random-access byte store for a seekable ImageNet-val tar.
    #
    # Builds a {filename -> (offset, size)} index once, cached as JSON next
    # to the tar (or to ``index_cache_path`` if provided) so subsequent runs
    # skip the O(N) member scan. Thread-safe for concurrent reads via a
    # coarse-grained lock around the underlying file handle's seek/read.
    def __init__(
        self,
        tar_path: str = DEFAULT_VAL_TAR_PATH,
        index_cache_path: Optional[str] = None,
    ):
        self._tar_path = tar_path
        self._index_cache_path = index_cache_path or (
            os.path.join(os.path.dirname(tar_path), "ILSVRC2012_img_val.tar.index.json")
            if os.access(os.path.dirname(tar_path), os.W_OK)
            else tar_path + ".index.json"
        )
        self._index: Dict[str, List[int]] = self._load_or_build_index()
        self._fh_lock = threading.Lock()
        self._fh: Optional[io.BufferedReader] = None

    def _load_or_build_index(self) -> Dict[str, List[int]]:
        if os.path.exists(self._index_cache_path):
            with open(self._index_cache_path, "r") as f:
                return json.load(f)
        index: Dict[str, List[int]] = {}
        with tarfile.open(self._tar_path, "r|") as tf:
            for member in tf:
                if member.isfile():
                    index[member.name] = [int(member.offset_data), int(member.size)]
        tmp = self._index_cache_path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(index, f)
            os.replace(tmp, self._index_cache_path)
        except OSError:
            if os.path.exists(tmp):
                os.remove(tmp)
        return index

    def _ensure_fh(self) -> io.BufferedReader:
        if self._fh is None or self._fh.closed:
            self._fh = open(self._tar_path, "rb")
        return self._fh

    def read_bytes(self, filename: str) -> bytes:
        if filename not in self._index:
            raise KeyError(filename)
        offset, size = self._index[filename]
        with self._fh_lock:
            fh = self._ensure_fh()
            fh.seek(offset)
            data = fh.read(size)
        if len(data) != size:
            raise IOError(f"short read for {filename}: {len(data)} != {size}")
        return data

    def __contains__(self, filename: str) -> bool:
        return filename in self._index

    def close(self):
        with self._fh_lock:
            if self._fh is not None and not self._fh.closed:
                self._fh.close()


class LocalThenTarImageStore:
    # Prefer a file under ``raw_dir`` if present; otherwise fall back to the
    # tar store. Lets a partial extraction coexist with direct-from-tar reads
    # so we don't waste already-extracted files.
    def __init__(self, raw_dir: str, tar_store: TarImageStore):
        self._raw_dir = raw_dir
        self._tar_store = tar_store

    def read_bytes(self, filename: str) -> bytes:
        raw_path = os.path.join(self._raw_dir, filename)
        if os.path.exists(raw_path):
            with open(raw_path, "rb") as f:
                return f.read()
        return self._tar_store.read_bytes(filename)


def write_manifest_csv(items: Sequence[ValItem], dst_csv: str) -> None:
    with open(dst_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["val_index_1based", "filename", "wnid", "pytorch_class_id"])
        for it in items:
            writer.writerow([it.val_index_1based, it.filename, it.wnid, it.pytorch_class_id])


def read_manifest_csv(src_csv: str) -> List[ValItem]:
    items: List[ValItem] = []
    with open(src_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append(
                ValItem(
                    val_index_1based=int(row["val_index_1based"]),
                    filename=row["filename"],
                    wnid=row["wnid"],
                    pytorch_class_id=int(row["pytorch_class_id"]),
                )
            )
    return items


def stratified_one_per_class(
    items: Sequence[ValItem], seed: int = 0
) -> List[ValItem]:
    # Pick one val image per pytorch_class_id (0..999). The choice is the
    # seed-deterministic first element after a per-class permutation, so
    # re-running with the same seed yields an identical 1000-image subset.
    rng = np.random.default_rng(seed)
    by_class: dict = {}
    for it in items:
        by_class.setdefault(it.pytorch_class_id, []).append(it)
    chosen: List[ValItem] = []
    for class_id in range(1000):
        bucket = by_class.get(class_id, [])
        if not bucket:
            raise RuntimeError(f"class_id {class_id} has no val images in manifest")
        order = rng.permutation(len(bucket))
        chosen.append(bucket[int(order[0])])
    return chosen


def random_categories_subsample(
    items: Sequence[ValItem],
    num_classes: int,
    images_per_class: Optional[int] = None,
    seed: int = 0,
) -> List[ValItem]:
    # Pick `num_classes` pytorch_class_ids without replacement (seed-deterministic),
    # then include `images_per_class` val images per chosen class (None = all
    # available, typically 50). Output is seed-shuffled so classes interleave
    # instead of appearing in 50-image blocks.
    rng = np.random.default_rng(seed)
    by_class: dict = {}
    for it in items:
        by_class.setdefault(it.pytorch_class_id, []).append(it)
    all_class_ids = sorted(by_class.keys())
    if num_classes > len(all_class_ids):
        raise ValueError(
            f"num_classes={num_classes} exceeds available classes={len(all_class_ids)}"
        )
    chosen_class_ids = sorted(
        int(c) for c in rng.choice(all_class_ids, size=num_classes, replace=False)
    )
    chosen: List[ValItem] = []
    for class_id in chosen_class_ids:
        bucket = sorted(by_class[class_id], key=lambda x: x.val_index_1based)
        if images_per_class is not None and int(images_per_class) < len(bucket):
            order = rng.permutation(len(bucket))
            picks = [bucket[int(i)] for i in order[: int(images_per_class)]]
            bucket = sorted(picks, key=lambda x: x.val_index_1based)
        chosen.extend(bucket)
    shuffle_order = rng.permutation(len(chosen))
    return [chosen[int(i)] for i in shuffle_order]


def build_center_crop_256_transform():
    return T.Compose(
        [
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(256),
            T.ToTensor(),  # (C, 256, 256) in [0, 1]
        ]
    )


def decode_jpeg_bytes_to_m1to1(jpeg_bytes: bytes, transform=None) -> torch.Tensor:
    transform = transform or build_center_crop_256_transform()
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    tensor_0_1 = transform(img)
    return tensor_0_1 * 2.0 - 1.0


def load_image_m1to1(jpeg_path: str, transform=None) -> torch.Tensor:
    with open(jpeg_path, "rb") as f:
        return decode_jpeg_bytes_to_m1to1(f.read(), transform=transform)


_SR3_TOP1K_LINE_RE = re.compile(r"^(n\d+)/(ILSVRC2012_val_\d{8}\.JPEG)\s+(\d+)\s*$")


def read_sr3_top1k_manifest(file_path: str) -> List[ValItem]:
    # The community-standard 1000-image ImageNet-256 SR test set introduced by
    # SR3 (Saharia et al., 2021) and adopted by DDRM / DPS / NDTM / oc-guidance.
    # Each line: "<wnid>/<filename> <pytorch_class_id>". The class id uses the
    # same alphabetical-WNID convention as our other code paths, so no remap.
    items: List[ValItem] = []
    with open(file_path, "r") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            match = _SR3_TOP1K_LINE_RE.match(stripped)
            if match is None:
                raise RuntimeError(
                    f"unparseable sr3_top1k line {line_number} in {file_path!r}: {stripped!r}"
                )
            wnid, filename, class_id_str = match.groups()
            val_match = _VAL_FILENAME_RE.match(filename)
            assert val_match is not None  # regex above already enforces format
            items.append(
                ValItem(
                    val_index_1based=int(val_match.group(1)),
                    filename=filename,
                    wnid=wnid,
                    pytorch_class_id=int(class_id_str),
                )
            )
    return items


class StratifiedValSubsample:
    # Lazy iterable: produces (gt_m1to1, class_id_tensor, item) tuples, one at a time,
    # from a 1000-image stratified ImageNet-val subset. Consumers choose their own
    # device; tensors are returned on CPU so the caller is explicit about placement.
    #
    # Image bytes come from whichever store resolves the filename first:
    # local ``<val_root>/raw/`` if extracted, otherwise random-access reads
    # from ``val_tar_path`` via ``TarImageStore``. Both paths are equivalent
    # semantically so downstream code is agnostic to extraction state.
    #
    # ``sr3_top1k_path`` short-circuits the stratified picking and uses the
    # community-standard SR3 list verbatim — for paper-comparable evaluation.
    def __init__(
        self,
        val_root: str = DEFAULT_VAL_ROOT,
        val_tar_path: Optional[str] = DEFAULT_VAL_TAR_PATH,
        manifest_csv: Optional[str] = None,
        seed: int = 0,
        subsample_size: Optional[int] = 1000,
        num_random_categories: Optional[int] = None,
        images_per_category: Optional[int] = None,
        sr3_top1k_path: Optional[str] = None,
    ):
        self._val_root = val_root
        if sr3_top1k_path is not None:
            self._items = read_sr3_top1k_manifest(sr3_top1k_path)
        else:
            if manifest_csv is not None and os.path.exists(manifest_csv):
                all_items = read_manifest_csv(manifest_csv)
            else:
                all_items = build_val_manifest(val_root=val_root)
                if manifest_csv is not None:
                    write_manifest_csv(all_items, manifest_csv)
            if num_random_categories is not None:
                self._items = random_categories_subsample(
                    all_items,
                    num_classes=int(num_random_categories),
                    images_per_class=images_per_category,
                    seed=seed,
                )
            elif subsample_size == 1000:
                self._items = stratified_one_per_class(all_items, seed=seed)
            elif subsample_size is None:
                self._items = list(all_items)
            else:
                # Take the first N from the stratified 1000 ordering (deterministic).
                strat = stratified_one_per_class(all_items, seed=seed)
                self._items = strat[: int(subsample_size)]
        self._transform = build_center_crop_256_transform()
        self._raw_dir = os.path.join(val_root, "raw")
        self._store = self._build_store(val_tar_path)

    def _build_store(self, val_tar_path: Optional[str]):
        if val_tar_path is not None and os.path.exists(val_tar_path):
            tar_store = TarImageStore(tar_path=val_tar_path)
            if os.path.isdir(self._raw_dir):
                return LocalThenTarImageStore(self._raw_dir, tar_store)
            return tar_store
        # Tar missing: fall back to local only. Will raise if a file is absent.
        if not os.path.isdir(self._raw_dir):
            raise FileNotFoundError(
                f"neither {val_tar_path!r} nor {self._raw_dir!r} is usable as an image source"
            )

        class _LocalOnly:
            def __init__(self, raw_dir: str) -> None:
                self._raw_dir = raw_dir

            def read_bytes(self, filename: str) -> bytes:
                with open(os.path.join(self._raw_dir, filename), "rb") as f:
                    return f.read()

        return _LocalOnly(self._raw_dir)

    def __len__(self) -> int:
        return len(self._items)

    def items(self) -> List[ValItem]:
        return list(self._items)

    def __getitem__(self, idx: int):
        it = self._items[idx]
        jpeg_bytes = self._store.read_bytes(it.filename)
        gt = decode_jpeg_bytes_to_m1to1(jpeg_bytes, transform=self._transform)
        label = torch.tensor(it.pytorch_class_id, dtype=torch.long)
        return gt, label, it
