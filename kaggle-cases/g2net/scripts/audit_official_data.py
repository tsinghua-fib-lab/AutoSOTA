"""Check packed-data identity, stratification and exact float32 round trips."""
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    metadata = json.loads((ROOT / 'data/packed/metadata.json').read_text())
    if metadata.get('fixture'):
        raise ValueError('Official audit refuses generated fixture data')
    result = dict(amplitude_scale=metadata['amplitude_scale'], splits={},
                  archive_bytes=(ROOT / 'data/raw/g2net-gravitational-wave-detection.zip').stat().st_size)
    rng = np.random.default_rng(114)
    with zipfile.ZipFile(ROOT / 'data/raw/g2net-gravitational-wave-detection.zip') as archive:
        result['archive_members'] = len(archive.infolist())
        for split, expected_count in [('train', 560000), ('test', 226000)]:
            table = pd.read_csv(ROOT / f'data/packed/{split}.csv', dtype={'id': str})
            waves = np.load(ROOT / f'data/packed/{split}.npy', mmap_mode='r')
            if len(table) != expected_count or waves.shape != (len(table), 3, 4096) or waves.dtype != np.float32:
                raise ValueError('Unexpected official data shape or type')
            if table.id.duplicated().any():
                raise ValueError('Duplicate official IDs')
            selected = rng.choice(len(table), size=512, replace=False)
            samples = np.asarray(waves[selected])
            if not np.isfinite(samples).all() or not np.all(np.any(samples != 0, axis=(1, 2))):
                raise ValueError('Invalid packed waveforms')
            for index in selected[:32]:
                sid = table.id.iloc[index]
                member = f'{split}/{sid[0]}/{sid[1]}/{sid[2]}/{sid}.npy'
                raw = np.load(io.BytesIO(archive.read(member)), allow_pickle=False)
                expected = (raw.astype(np.float64)*1e20).astype(np.float32)
                if not np.array_equal(expected, waves[index]):
                    raise ValueError('Packed amplitude or sample ID differs from the official archive')
            info = dict(samples=len(table), shape=list(waves.shape), dtype=str(waves.dtype),
                        checked_finite_samples=512, exact_archive_round_trips=32,
                        raw_wave_dtype=str(raw.dtype),
                        id_sha256=hashlib.sha256('\n'.join(table.id).encode()).hexdigest(),
                        sampled_scaled_rms_by_detector=np.sqrt(np.mean(samples.astype(np.float64)**2, axis=(0, 2))).tolist())
            if split == 'train':
                info['positive_rate'] = float(table.target.mean())
                info['folds'] = {str(seed): table.groupby(f'fold_{seed}').target.agg(['size', 'mean']).to_dict('index')
                                 for seed in [2021, 48]}
                train_ids = set(table.id)
            else:
                if train_ids.intersection(table.id):
                    raise ValueError('Train/test ID overlap')
            result['splits'][split] = info
    result['passed'] = True
    (ROOT / 'reports/official_data_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
