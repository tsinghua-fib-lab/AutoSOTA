"""Generate a physical-amplitude bank with the winner's published generator.

Unused segmentation-image saves are disabled and np.save is captured in memory.
All waveform generation, projections, downsampling and physical amplitudes remain
upstream code. Public mass estimates are an external pretrained artifact; the
training scope of that historical predictor is unknown and recorded explicitly.
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'sources/kdl_generation'
sys.path.insert(0, str(SOURCE))
import signals

CDF = None
CAPTURE = None


class CaptureNumpy:
    def __init__(self):
        self.saved = None

    def __getattr__(self, name):
        return getattr(np, name)

    def save(self, path, waveform, *args, **kwargs):
        self.saved = (str(path), np.asarray(waveform))


def initialize(cdf_values, mass_values):
    global CDF, CAPTURE
    CDF = interp1d(cdf_values, mass_values, kind='linear', fill_value=(9, 33.25),
                  bounds_error=False, assume_sorted=True)
    CAPTURE = CaptureNumpy()
    signals.np = CAPTURE
    signals.save_mask = lambda *args, **kwargs: None


def generate(index):
    for attempt in range(32):
        seed = 2000000 + index*32 + attempt
        CAPTURE.saved = None
        success = signals.generate_signal_spin_predicted_mchirp(seed, CDF,
                       catalog=str(ROOT / 'data/kdl_bank_capture'))
        if success and CAPTURE.saved is not None:
            name, waveform = CAPTURE.saved
            value = (waveform.astype(np.float64)*1e20).astype(np.float32)
            if value.shape != (3, 4096) or not np.isfinite(value).all() or not np.any(value):
                continue
            _, _, m1, m2, distance, arrival = Path(name).name.split('_')
            m1, m2, distance, arrival = map(float, [m1, m2, distance, arrival])
            chirp = (m1*m2)**(3/5)/(m1+m2)**(1/5)
            return index, value, [m1, m2, chirp, m1/m2, distance, arrival], attempt
    raise RuntimeError(f'Physical signal {index} could not be generated')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=50000)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/synthetic_kdl_50k_fold0')
    parser.add_argument('--prefix-bank', type=Path)
    args = parser.parse_args()
    table = pd.read_csv(ROOT / 'data/packed/train.csv', dtype={'id': str})
    estimates = pd.read_csv(SOURCE / 'pred_mchirp.csv', dtype={'name': str})
    training_ids = set(table.loc[table.fold_48 != args.fold, 'id'])
    estimates = estimates[(estimates.target == 1) & estimates.name.isin(training_ids)]
    histogram, edges = np.histogram(estimates.mchirp, bins=97, density=True, range=(9, 33.25))
    probabilities = histogram/histogram.sum()
    cumulative = np.r_[0, np.cumsum(probabilities)[:-1]]
    mass_values = edges[:-1]+np.diff(edges)/2
    initialize(cumulative, mass_values)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'signals.npy').exists():
        raise FileExistsError('Completed bank already exists')
    waves = np.lib.format.open_memmap(args.output / 'signals.partial.npy', mode='w+', dtype=np.float32,
                                     shape=(args.count, 3, 4096))
    parameters = np.zeros((args.count, 6), dtype=np.float32)
    offset = 0
    if args.prefix_bank:
        prefix_info = json.loads((args.prefix_bank / 'metadata.json').read_text())
        if prefix_info['fold_seed'] != 48 or prefix_info['excluded_prior_fold'] != args.fold or prefix_info['injection'] != 'physical':
            raise ValueError('Prefix bank must use the exact same excluded fold and physical generator')
        prior = np.load(args.prefix_bank / 'signals.npy', mmap_mode='r')
        prior_parameters = np.load(args.prefix_bank / 'parameters.npy', mmap_mode='r')
        offset = min(len(prior), args.count)
        for start in range(0, offset, 1024):
            waves[start:min(start+1024, offset)] = prior[start:min(start+1024, offset)]
        parameters[:offset] = prior_parameters[:offset]
        print(f'Reused {offset} existing deterministic templates', flush=True)
    started = time.monotonic()
    retries = 0
    with mp.Pool(args.workers, initializer=initialize, initargs=(cumulative, mass_values)) as pool:
        for done, (index, value, params, attempt) in enumerate(pool.imap_unordered(generate, range(offset, args.count), chunksize=4), offset+1):
            waves[index] = value
            parameters[index] = params
            retries += attempt
            if done % 256 == 0 or done == args.count:
                print(f'generated={done}/{args.count} seconds={time.monotonic()-started:.1f} retries={retries}', flush=True)
    waves.flush()
    del waves
    (args.output / 'signals.partial.npy').replace(args.output / 'signals.npy')
    np.save(args.output / 'parameters.npy', parameters)
    metadata = dict(count=args.count, amplitude_scale=1e20, dtype='float32',
                    source_generator='KDL generate_signal_spin_predicted_mchirp',
                    normalization='Physical detector amplitudes at original sampled distance; no per-channel normalization',
                    injection='physical', fold_seed=48, excluded_prior_fold=args.fold,
                    prior_records=len(estimates), prior_scope='Only IDs in the current training fold contribute to the histogram',
                    historical_prior_predictor_training_scope='unknown; external published artifact, not a newly leak-free trained estimator',
                    adaptations=['Packed saves', 'Unused segmentation images omitted'],
                    parameter_columns=['mass1', 'mass2', 'chirp_mass', 'mass1_over_mass2', 'distance', 'arrival_seconds'],
                    parameter_precision='Derived from upstream two-decimal filename metadata',
                    seconds=time.monotonic()-started, retries=retries,
                    complete_champion_solution_reproduction=False)
    metadata['prefix_reused'] = offset
    metadata['prefix_bank'] = str(args.prefix_bank) if args.prefix_bank else None
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
