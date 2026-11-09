import argparse
from pathlib import Path
import pandas as pd
import pickle
import gc

from kuma_utils.torch import TorchLogger

from configs import HW_CFG
from datasets import load_signal_cache


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default='input/')
    parser.add_argument("--export_dir", type=str, default='input/')
    parser.add_argument("--hardware", type=str, default='A100',
                        help="Your hardware spec config name (this determines the cache size; see configs.py)")
    parser.add_argument("--cache", action='store_true', 
                        help="whether to generate waveform cache file")
   
    opt = parser.parse_args()
    LOGGER = TorchLogger('tmp.log', file=False)

    N_CPU, N_RAM, N_GPU, N_GRAM = HW_CFG[opt.hardware]
    if opt.cache:
        LOGGER(f'Maximum cache size is set to {N_RAM//2} GB')

    root_dir = Path(opt.root_dir).expanduser()
    train = pd.read_csv(root_dir/'training_labels.csv')
    test = pd.read_csv(root_dir/'sample_submission.csv')
    export_dir = Path(opt.export_dir).expanduser()
    export_dir.mkdir(parents=True, exist_ok=True)

    LOGGER('===== TRAIN =====')
    train['path'] = train['id'].apply(lambda x: root_dir/f'train/{x[0]}/{x[1]}/{x[2]}/{x}.npy')
    train.to_csv(export_dir/'train.csv', index=False)
    if opt.cache:
        train_cache = load_signal_cache(
            train['path'].values, N_RAM//2, n_jobs=N_CPU)
        with open(export_dir/'train_cache.pickle', 'wb') as f:
            pickle.dump(train_cache, f)
        del train_cache; gc.collect

    LOGGER('===== TEST =====')
    test['path'] = test['id'].apply(lambda x: root_dir/f'test/{x[0]}/{x[1]}/{x[2]}/{x}.npy')
    test.to_csv(export_dir/'test.csv', index=False)
    if opt.cache:
        test_cache = load_signal_cache(
            test['path'].values, N_RAM//2, n_jobs=N_CPU)
        with open(export_dir/'test_cache.pickle', 'wb') as f:
            pickle.dump(test_cache, f)
        del test_cache; gc.collect
    