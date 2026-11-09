"""Pack output from the champion's original colored-noise generator."""
import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'sources/kdl_generation'
sys.path.insert(0, str(SOURCE))
import noise

BANK = None
COUNT = None
CHUNKS = None
BLOCK = None


class CaptureNumpy:
    def __getattr__(self, name):
        return getattr(np, name)

    def save(self, path, value, *args, **kwargs):
        offset = int(Path(path).name.rsplit('_', 1)[1])
        index = BLOCK*CHUNKS+offset
        if index >= COUNT:
            return
        value = np.asarray(value)
        if value.shape != (3, 4096) or not np.isfinite(value).all():
            raise ValueError('Invalid original generator output')
        BANK[index] = (value.astype(np.float64)*1e20).astype(np.float32)


def initialize(path, count, chunks):
    global BANK, COUNT, CHUNKS
    os.chdir(SOURCE)
    BANK = np.load(path, mmap_mode='r+')
    COUNT, CHUNKS = count, chunks
    noise.np = CaptureNumpy()


def generate(block):
    global BLOCK
    BLOCK = block
    # Different seed range from the generator smoke check and historical bank.
    noise.make_noise_ls(500000+block, catalog=str(ROOT / 'data/kdl_noise_capture'), chunks=CHUNKS)
    BANK.flush()
    return block


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=250000)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--chunks', type=int, default=512)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/synthetic_kdl_noise_250k')
    parser.add_argument('--prefix-bank', type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'noise.npy').exists():
        raise FileExistsError('Completed noise bank already exists')
    partial = args.output / 'noise.partial.npy'
    bank = np.lib.format.open_memmap(partial, mode='w+', dtype=np.float32, shape=(args.count, 3, 4096))
    prefix_blocks = 0
    if args.prefix_bank:
        prefix_info = json.loads((args.prefix_bank / 'metadata.json').read_text())
        if prefix_info['chunks_per_block'] != args.chunks or prefix_info['amplitude_scale'] != 1e20:
            raise ValueError('Prefix noise has incompatible generation settings')
        previous = np.load(args.prefix_bank / 'noise.npy', mmap_mode='r')
        prefix_blocks = min(len(previous), args.count)//args.chunks
        for start in range(0, prefix_blocks*args.chunks, 1024):
            end = min(start+1024, prefix_blocks*args.chunks)
            bank[start:end] = previous[start:end]
        print(f'Reused {prefix_blocks*args.chunks} existing noise samples', flush=True)
    bank.flush()
    del bank
    started = time.monotonic()
    blocks = math.ceil(args.count/args.chunks)
    with mp.Pool(args.workers, initializer=initialize,
                 initargs=(str(partial), args.count, args.chunks)) as pool:
        for done, block in enumerate(pool.imap_unordered(generate, range(prefix_blocks, blocks)), prefix_blocks+1):
            if done % 16 == 0 or done == blocks:
                print(f'blocks={done}/{blocks} seconds={time.monotonic()-started:.1f}', flush=True)
    partial.replace(args.output / 'noise.npy')
    metadata = dict(count=args.count, amplitude_scale=1e20, dtype='float32',
                    source_generator='Unchanged KDL noise.make_noise_ls using PyCBC colored_noise',
                    source_psd='Published ch0/1/2_psd_adjusted.npy; historical estimation scope unknown',
                    waveform_length=4096, channels=3, sample_rate=2048,
                    seed_rule='2**31 + 1 + 500000 + block', chunks_per_block=args.chunks,
                    real_training_waveforms_used=False, complete_champion_solution_reproduction=False,
                    seconds=time.monotonic()-started)
    metadata['prefix_reused'] = prefix_blocks*args.chunks
    metadata['prefix_bank'] = str(args.prefix_bank) if args.prefix_bank else None
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
