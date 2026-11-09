"""Pack official waveforms into float32 arrays, preserving CSV order and folds.

Scale in float64 BEFORE casting: raw gravitational-wave amplitudes are tiny.
The master archive is retained; no extracted waveform files are required.
"""
import argparse
import io
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=ROOT / "data/raw/g2net-gravitational-wave-detection.zip")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output", type=Path, default=ROOT / "data/packed")
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument('--streaming', action='store_true')
    args = parser.parse_args()
    destination = args.output
    if args.fixture and destination.resolve() == (ROOT / "data/packed").resolve():
        raise ValueError("Fixture outputs must not overwrite the competition data directory")
    destination.mkdir(parents=True, exist_ok=True)
    if args.streaming:
        if args.fixture:
            raise ValueError('Streaming is exclusively for the official range download')
        from streaming_archive import CheckedRangeReader
        source = CheckedRangeReader(args.archive)
    else:
        source = args.archive
    with zipfile.ZipFile(source) as archive:
        members = set(archive.namelist())
        splits = [("train", "training_labels.csv"), ("test", "sample_submission.csv")]
        if args.streaming:
            splits.reverse()  # Official archive stores test waveforms first.
        for split, filename in splits:
            raw_csv = ROOT / 'data/raw' / filename
            csv_bytes = raw_csv.read_bytes() if args.streaming else archive.read(filename)
            if len(csv_bytes) != archive.getinfo(filename).file_size:
                raise ValueError('Official CSV size differs from the archive directory')
            table = pd.read_csv(io.BytesIO(csv_bytes), dtype={"id": str})
            if table.id.duplicated().any():
                raise ValueError("Duplicate sample IDs")
            if split == "train":
                for seed in [2021, 48]:
                    folds = np.full(len(table), -1, dtype=np.int8)
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                    for fold, (_, valid) in enumerate(cv.split(table, table.target)):
                        folds[valid] = fold
                    table[f"fold_{seed}"] = folds
            table.to_csv(destination / f"{split}.csv", index=False)
            if not args.fixture:
                raw_path = ROOT / "data/raw" / filename
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_bytes(csv_bytes)
            target = destination / f"{split}.npy"
            partial = target.with_suffix(".partial.npy")
            if target.exists():
                existing = np.load(target, mmap_mode="r")
                if existing.shape != (len(table), 3, 4096) or existing.dtype != np.float32:
                    raise ValueError(f"Invalid existing array: {target}")
                print(f"Reusing {target}", flush=True)
                continue
            arr = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=(len(table), 3, 4096))

            def read_sample(pair):
                index, sample_id = pair
                member = f"{split}/{sample_id[0]}/{sample_id[1]}/{sample_id[2]}/{sample_id}.npy"
                if member not in members:
                    raise ValueError(f"Missing waveform {member}")
                wave = np.load(io.BytesIO(archive.read(member)), allow_pickle=False)
                if wave.shape != (3, 4096) or not np.isfinite(wave).all():
                    raise ValueError(f"Invalid waveform {member}")
                return index, (wave.astype(np.float64) * 1e20).astype(np.float32)

            # Bound queued work to avoid holding hundreds of thousands of futures.
            ordered = list(enumerate(table.id))
            if args.streaming:
                ordered.sort(key=lambda pair: archive.getinfo(
                    f'{split}/{pair[1][0]}/{pair[1][1]}/{pair[1][2]}/{pair[1]}.npy').header_offset)
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                bar = tqdm(total=len(table), desc=split)
                for start in range(0, len(table), 2048):
                    pairs = ordered[start:start+2048]
                    for index, wave in executor.map(read_sample, pairs):
                        arr[index] = wave
                        bar.update(1)
                bar.close()
            arr.flush()
            del arr
            partial.replace(target)
        if args.streaming:
            # The final CSV occurs after train waveforms; checking it now avoids
            # waiting for the end of the download before packing begins.
            for filename in ['training_labels.csv', 'sample_submission.csv']:
                if archive.read(filename) != (ROOT / 'data/raw' / filename).read_bytes():
                    raise ValueError('Official CSV and archive contents differ')
    (destination / "metadata.json").write_text(json.dumps({
        "amplitude_scale": 1e20, "sample_rate": 2048, "shape_per_sample": [3, 4096],
        "dtype": "float32", "fold_seeds": [2021, 48],
        "source_archive": str(ROOT / 'data/raw/g2net-gravitational-wave-detection.zip') if args.streaming else str(args.archive),
        "fixture": args.fixture,
    }, indent=2))


if __name__ == "__main__":
    main()
