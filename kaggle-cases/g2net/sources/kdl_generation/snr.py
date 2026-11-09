import argparse
import glob
import math
import os

import numpy as np
import pandas as pd
import pylab
from joblib import Parallel, delayed
from pycbc import conversions, distributions
from pycbc.detector import Detector
from pycbc.filter import matched_filter
from pycbc.filter.resample import resample_to_delta_t
from pycbc.types.frequencyseries import load_frequencyseries
from pycbc.waveform.waveform import get_td_waveform
from scipy.interpolate.interpolate import interp1d


def logpdf(x):
    return x

def snr_by_name(name, psd, cdfinv = None):
    pattern = os.path.split(name)[-1][:-4]
    fid, _ = os.path.splitext(os.path.basename(name))
    if pattern.find("noise") >= 0:
        return fid, 12 ** 0.5
    seed, _, m1, m2, d, t = pattern.split("_")
    np.random.seed(int(seed))
    apx = 'SEOBNRv4_opt'
    num_samples = 1
 
    params_distribution = distributions.Uniform(
                                                inclination = (0, np.pi), 
                                                coa_phase = (0, 2 * np.pi), 
                                                distance = (500, 3000),
                                                polarization = (0, 2 * np.pi),
                                                end_time = (1.5, 1.9),
                                                m1 = (5, 50),
                                                m2 = (5, 50),
                                                s1 = (-1, 1),
                                                s2 = (-1, 1)
    )
    params = params_distribution.rvs(num_samples)
    if abs(float(m1) - params["m1"][0]) < 0.01 and \
       abs(float(m2) - params["m2"][0]) < 0.01 and \
       abs(float(d) - params["distance"][0]) < 0.01 and \
       abs(float(t) - params["end_time"][0]) < 0.01:
       m1 = params["m1"][0]
       m2 = params["m2"][0]
    else:
        np.random.seed(int(seed))
        minq = 1/4
        maxq = 1/minq
        mc_distribution = distributions.External(["x"], logpdf, cdfinv=cdfinv)
        q_distribution = distributions.QfromUniformMass1Mass2(q=(minq,maxq))
        mc_samples = mc_distribution.rvs(size=num_samples)
        q_samples = q_distribution.rvs(size=num_samples)

        m1_t = conversions.mass1_from_mchirp_q(mc_samples, q_samples['q'])
        m2_t = conversions.mass2_from_mchirp_q(mc_samples, q_samples['q'])

        params_distribution = distributions.Uniform(
                                                inclination = (0, np.pi), 
                                                coa_phase = (0, 2 * np.pi), 
                                                distance = (200, 3000),
                                                polarization = (0, 2 * np.pi),
                                                end_time = (1.5, 1.9),
                                                s1 = (-1, 1),
                                                s2 = (-1, 1))
        params = params_distribution.rvs(num_samples)
        if abs(float(m1) - m1_t) < 0.01 and \
           abs(float(m2) - m2_t) < 0.01 and \
           abs(float(d) - params["distance"][0]) < 0.01 and \
           abs(float(t) - params["end_time"][0]) < 0.01:
            m1 = m1_t
            m2 = m2_t
            #print("GtG with hard")
        else:
            return fid, math.sqrt(12)
       
   
    uniform_solid_angle_distribution = distributions.UniformSolidAngle()
    angles = uniform_solid_angle_distribution.rvs(num_samples)

    det_h1 = Detector('H1')
    det_l1 = Detector('L1')
    det_v1 = Detector('V1')
    det = [det_h1, det_l1, det_v1]

    hp, hc = get_td_waveform(approximant=apx,
                             mass1=m1,
                             mass2=m2,
                             spin1z=params["s1"][0],
                             spin2z=params["s2"][0],
                             inclination=params["inclination"][0],
                             coa_phase=params["coa_phase"][0],
                             delta_t=1.0/4096,
                             f_lower=20, distance = params["distance"][0])

    end_time = params["end_time"][0]
    declination = angles["theta"][0] - np.pi / 2
    right_ascension = angles["phi"][0]
    polarization = params["polarization"][0]
    hp, hc = hp.trim_zeros(), hc.trim_zeros()
    hp.prepend_zeros(3 * 4096)
    hc.prepend_zeros(3 * 4096)
    hp.start_time += end_time - hp.end_time
    hc.start_time += end_time - hc.end_time
    hp = hp.crop(-hp.start_time, 0)
    hc = hc.crop(-hc.start_time, 0)
    hc.start_time = 0
    hp.start_time = 0
    hp.resize(4096 * 2)
    hc.resize(4096 * 2)
    snr_sum = 0
    for ch in range(3):
        s = det[ch].project_wave(hp, hc,  right_ascension, declination, polarization)
        s.prepend_zeros(3 * 4096)
        s = s.crop(-s.start_time, 0)
        s.start_time = 0
        s.resize(4096 * 2)
        a = resample_to_delta_t(hp, 1/2048)
        b = resample_to_delta_t(s, 1/2048)
        #n = colored_noise(psd[0], b.start_time, b.end_time, seed =31337, sample_rate=2048)
        snr = matched_filter(a, b, psd=psd[ch], low_frequency_cutoff=20)
        #max_snr = max(abs(snr)) ** 0.5
        #print(max_snr)
        snr_sum += max(abs(snr))
    res_snr = snr_sum ** 0.5
    return fid, res_snr

def parse_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--out', type=str, default="snr.csv")
    arg('--signal_dir', type=str, default="/mnt/sota/datasets/g2net/signal_spin_mc/")
    return parser.parse_args()
if __name__ == '__main__':
    args = parse_args()
    names = glob.glob(os.path.join(args.signal_dir, "*.npy"))
    #names = glob.glob("signal_spin_mc/*.npy")[:1000]
    #names = glob.glob("noise_synth/*.npy")[:1000]
    psd0 = load_frequencyseries("ch0_psd_adjusted.npy")
    psd1 = load_frequencyseries("ch1_psd_adjusted.npy")
    psd2 = load_frequencyseries("ch2_psd_adjusted.npy")
    psd = [psd0, psd1, psd2]
    preds = pd.read_csv("pred_mchirp.csv")
    preds = preds[preds.target == 1]
    hist = pylab.hist(preds["mchirp"], bins = 49 * 2 - 1, density = True, range = (9., 33.25))
    y = hist[0] / hist[0].sum()
    x = hist[1][:-1] + (hist[1][1] - hist[1][0]) / 2
    cy = np.zeros_like(y)
    s = 0
    for i in range(len(y)):
        cy[i] = s
        s = s + y[i]
    cdfinv = interp1d(cy, x, kind="linear", fill_value=(9, 33.25), bounds_error = False, assume_sorted = True)
    data = Parallel(n_jobs=64, verbose=5)([delayed(snr_by_name)(name, psd, cdfinv) for name in names[:]])
    pd.DataFrame(data, columns=["name", "snr"]).to_csv(args.out, index=False)
