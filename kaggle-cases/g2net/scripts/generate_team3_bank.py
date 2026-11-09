"""Run third-place notebook's waveform functions in a reproducible worker pool.

The publicly supplied KDL PSD curves replace the notebook's missing PSD object.
Signals retain the original per-detector SNR=1 normalization and float16 storage.
This dependency adaptation is recorded; it is not an exact 300k-bank reproduction.
"""
import argparse
import ast
import json
import math
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
from pycbc.detector import Detector
from pycbc.filter import sigma
from pycbc.psd import interpolate
from pycbc.types import TimeSeries, load_frequencyseries
from pycbc.waveform import get_td_waveform

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'sources/team_3rd/1D_Model/notebooks/SyntheticSignal.ipynb'
FUNCTIONS = {}
PSDS = None


def initialize():
    global FUNCTIONS, PSDS
    namespace = dict(np=np, math=math, Detector=Detector, TimeSeries=TimeSeries,
                     get_td_waveform=get_td_waveform, sigma=sigma)
    notebook = json.loads(NOTEBOOK.read_text())
    selected = []
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        if 'def pick_4096(' in source:
            parsed = ast.parse(source)
            wanted = {'pick_4096', 'generate_signal', 'get_random_param', 'transform_signal'}
            for node in parsed.body:
                if isinstance(node, ast.FunctionDef) and node.name in wanted:
                    exec(compile(ast.Module(body=[node], type_ignores=[]), str(NOTEBOOK), 'exec'), namespace)
                    selected.append(node.name)
    if len(selected) != 4:
        raise RuntimeError(f'Unexpected upstream notebook structure: {selected}')
    namespace.update(DET_H1=Detector('H1'), DET_L1=Detector('L1'), DET_V1=Detector('V1'))
    FUNCTIONS = namespace
    PSDS = [interpolate(load_frequencyseries(str(ROOT / f'sources/kdl_generation/ch{i}_psd_adjusted.npy')), 0.5)
            for i in range(3)]


def generate(index):
    np.random.seed(1000000 + index)
    for attempt in range(10):
        params = FUNCTIONS['get_random_param']()
        try:
            detectors = FUNCTIONS['generate_signal'](params)
            result = np.stack([FUNCTIONS['transform_signal'](wave, PSD, 1, 0)
                               for wave, PSD in zip(detectors, PSDS)])
            result = (result * 1e20).astype(np.float16)
            if result.shape == (3, 4096) and np.isfinite(result).all() and np.any(result):
                return index, result, attempt
        except RuntimeError:
            pass
    raise RuntimeError(f'Waveform {index} failed repeatedly')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=30000)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/synthetic_team3_30k')
    args = parser.parse_args()
    initialize()  # Fail in the parent rather than continually restarting workers.
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'signals.npy').exists():
        raise FileExistsError('A completed bank already exists')
    waves = np.lib.format.open_memmap(args.output / 'signals.partial.npy', mode='w+',
                                     dtype=np.float16, shape=(args.count, 3, 4096))
    started = time.monotonic()
    retries = 0
    with mp.Pool(args.workers, initializer=initialize) as pool:
        for done, (index, value, attempt) in enumerate(pool.imap_unordered(generate, range(args.count), chunksize=8), 1):
            waves[index] = value
            retries += attempt
            if done % 256 == 0 or done == args.count:
                print(f'generated={done}/{args.count} seconds={time.monotonic()-started:.1f} retries={retries}', flush=True)
    waves.flush()
    del waves
    (args.output / 'signals.partial.npy').replace(args.output / 'signals.npy')
    metadata = dict(source_notebook=str(NOTEBOOK), count=args.count, dtype='float16',
                    shape_per_signal=[3, 4096], amplitude_scale=1e20,
                    normalization='SNR=1 per detector using PyCBC sigma',
                    PSD_source='KDL public adjusted PSD curves, interpolated to 0.5 Hz',
                    adaptations=['Missing upstream PSD object replaced by public winner curves',
                                 'Packed array replaces 300000 individual files and pickle'],
                    seed_offset=1000000, waveform_approximant='SEOBNRv4_opt with original IMRPhenomPv2 fallback',
                    seconds=time.monotonic()-started, retries=retries,
                    full_winning_solution_reproduction=False)
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
