"""PSD-weighted network SNR for physical templates, checked against PyCBC.

Uses the published adjusted PSD as an external artifact. No validation waveform
or validation label is used. Packed strains are scaled by 1e20.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from pycbc.filter import sigma
from pycbc.types import TimeSeries, load_frequencyseries

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bank', type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads((args.bank / 'metadata.json').read_text())
    if metadata.get('injection') != 'physical':
        raise ValueError('Physical templates are required')
    waves = np.load(args.bank / 'signals.npy', mmap_mode='r')
    freqs = np.fft.rfftfreq(4096, d=1/2048)
    psds = [load_frequencyseries(str(ROOT / f'sources/kdl_generation/ch{i}_psd_adjusted.npy')) for i in range(3)]
    densities = np.stack([np.interp(freqs, p.sample_frequencies.numpy(), p.numpy()) for p in psds])
    selection = (freqs >= 20) & (freqs < 1024)
    # dt FFT has physical units. delta_f=0.5, one-sided inner product.
    weight = 4*0.5/(densities[:, selection]*1e40*2048**2)
    values = np.zeros(len(waves), dtype=np.float32)
    for start in range(0, len(waves), 256):
        transformed = np.fft.rfft(np.asarray(waves[start:start+256], dtype=np.float64), axis=-1)
        values[start:start+256] = np.sqrt(np.sum(np.abs(transformed[:, :, selection])**2*weight, axis=(1, 2)))
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('Invalid physical SNR')
    references = []
    for index in np.linspace(0, len(waves)-1, 24, dtype=int):
        per_detector = [sigma(TimeSeries(waves[index, c].astype(np.float64)/1e20, delta_t=1/2048),
                              psd=psds[c], low_frequency_cutoff=20) for c in range(3)]
        references.append([int(index), float(values[index]), float(np.linalg.norm(per_detector))])
    error = max(abs(a/b-1) for _, a, b in references)
    if error > 1e-5:
        raise ValueError(f'FFT SNR disagrees with PyCBC: {error}')
    np.save(args.bank / 'network_snr.npy', values)
    report = dict(count=len(values), method='4 integral |h(f)|^2 / PSD df; quadrature sum of three detectors',
                  low_frequency_cutoff=20, max_relative_error_vs_pycbc=error, passed=True,
                  quantiles=np.quantile(values, [.01, .1, .5, .9, .99]).tolist(),
                  psd_scope='External historical published PSD; original estimator training scope unknown',
                  reference_samples=references)
    (args.bank / 'snr_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
