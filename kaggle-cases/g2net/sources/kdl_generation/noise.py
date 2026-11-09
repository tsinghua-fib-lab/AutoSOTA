import os
import numpy as np
from pycbc.noise.reproduceable import colored_noise
from pycbc.types import load_frequencyseries
from joblib import Parallel, delayed

def make_noise_ls(i, catalog = "noise_synth", chunks = 1024):
    s = 2 ** 31 + 1 + i
    psd0 = load_frequencyseries("ch0_psd_adjusted.npy")
    psd1 = load_frequencyseries("ch1_psd_adjusted.npy")
    psd2 = load_frequencyseries("ch2_psd_adjusted.npy")
    psd = [psd0, psd1, psd2]
    np.random.seed(s)
    name = str(s) + "_noise"
    path = os.path.join(catalog, name)
    seeds = np.random.randint(0, np.iinfo(np.uint32).max, 3)
    ch0 = colored_noise(psd[0], 0, 2 * chunks, seed = seeds[0], sample_rate = 2048).numpy()
    ch1 = colored_noise(psd[1], 0, 2 * chunks, seed = seeds[1], sample_rate = 2048).numpy()
    ch2 = colored_noise(psd[2], 0, 2 * chunks, seed = seeds[2], sample_rate = 2048).numpy()
    signal = np.vstack([ch0, ch1, ch2])
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    for i in range(chunks):
        np.save(path + "_" + str(i), signal[:, i * 4096 : (i + 1) * 4096])

if __name__ == '__main__':
    _ = Parallel(n_jobs=32, verbose=5)([delayed(make_noise_ls)(i) for i in range(500)])