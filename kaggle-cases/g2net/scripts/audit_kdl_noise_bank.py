"""Compare independent generated noise with training-fold-only real negatives."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]


def spectrum(waves):
    window = np.hanning(4096)
    values = np.abs(np.fft.rfft(waves*window, axis=-1))**2/(2048*np.sum(window**2))
    values[:, :, 1:-1] *= 2
    return values.mean(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bank', type=Path, default=ROOT / 'data/synthetic_kdl_noise_250k')
    parser.add_argument('--report', type=Path, default=ROOT / 'reports/kdl_noise_bank_audit.json')
    args = parser.parse_args()
    bank = args.bank
    generated = np.load(bank / 'noise.npy', mmap_mode='r')
    metadata = json.loads((bank / 'metadata.json').read_text())
    if generated.shape != (metadata['count'], 3, 4096) or metadata['amplitude_scale'] != 1e20:
        raise ValueError('Noise bank shape or scale is incorrect')
    table = pd.read_csv(ROOT / 'data/packed/train.csv')
    official = np.load(ROOT / 'data/packed/train.npy', mmap_mode='r')
    rng = np.random.default_rng(491)
    noise_ids = rng.choice(len(generated), 1024, replace=False)
    real_ids = rng.choice(np.flatnonzero((table.target == 0) & (table.fold_48 != 0)), 1024, replace=False)
    a, b = np.asarray(generated[noise_ids]), np.asarray(official[real_ids])
    if not np.isfinite(a).all() or not np.any(a):
        raise ValueError('Invalid generated noise')
    pa, pb = spectrum(a), spectrum(b)
    freq = np.fft.rfftfreq(4096, 1/2048)
    bands = {}
    for low, high in [(20, 50), (50, 100), (100, 250), (250, 350), (350, 600), (600, 1000)]:
        mask = (freq >= low) & (freq < high)
        bands[f'{low}-{high}Hz'] = np.median(pa[:, mask]/pb[:, mask], axis=1).tolist()
    filt = butter(5, 20.43, btype='highpass', fs=2048, output='sos')
    rms_a = np.sqrt(np.mean(sosfiltfilt(filt, a, axis=-1)**2, axis=-1)).mean(0)
    rms_b = np.sqrt(np.mean(sosfiltfilt(filt, b, axis=-1)**2, axis=-1)).mean(0)
    ratios = np.array(list(bands.values()))
    if np.any((ratios < .8) | (ratios > 1.25)) or np.any((rms_a/rms_b < .8) | (rms_a/rms_b > 1.25)):
        raise ValueError('Generated noise has an unexpected PSD/RMS distribution')
    report = dict(passed=True, count=metadata['count'], shape=list(generated.shape),
                  finite_checked_samples=len(a), real_comparison_scope='Fold 0 seed 48 training negatives only',
                  generated_to_real_median_psd_ratio_by_detector=bands,
                  generated_to_real_highpass_rms_ratio_by_detector=(rms_a/rms_b).tolist(),
                  historical_source_psd_scope='Unknown; published external artifact',
                  metric_scope='Distribution audit, not competition performance')
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
