import os
import numpy as np
import pandas as pd
import pylab
from joblib import Parallel, delayed
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector
from pycbc import distributions, waveform
from pycbc.filter.resample import resample_to_delta_t
import os
from pycbc import conversions
from scipy.interpolate import interp1d
import cv2
import matplotlib.pyplot as plt

def save_mask(hp, hc, path, dilatation_size = 1):
    hp, hc = hp.trim_zeros(), hc.trim_zeros()
    amp = waveform.utils.amplitude_from_polarizations(hp, hc)
    amp = amp.crop(-amp.start_time, 0)
    amp.start_time = 0
    amp.resize(4096 * 2)
    amp_array = np.abs(amp.numpy()).reshape(-1, 4 * 4 * 2).max(1)

    f = waveform.utils.frequency_from_polarizations(hp, hc)
    f = f.crop(-f.start_time, 0)
    f.start_time = 0
    f.resize(4096 * 2)
    f_array = np.abs(f.numpy()).reshape(-1, 4 * 4 * 2).max(1)


    f_array[np.where(amp_array == max(amp_array))[0][0] + 1: ] = 0
    pts = (f_array / 4).round().astype(int)
    mask = np.zeros((256, 256))
    for i in range(0,256):
        if pts[i] == 0:
            continue
        cv2.line(mask, (i - 1, pts[i - 1]), (i, pts[i]), 1)
    mask = mask[::-1, :]
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                           (dilatation_size, dilatation_size))
    mask = cv2.dilate(mask, element)
    mask = mask.astype(bool)
    plt.imsave(path + ".png", mask)

def generate_signal_spin(seed, catalog = "signal_spin"):
    s = 2 ** 30 + seed
    name = str(s) + "_signal"

    np.random.seed(s)
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

    name += "_%.2f_%.2f_%.2f_%.2f" % (params["m1"][0], params["m2"][0], params["distance"][0], params["end_time"][0])
    path = os.path.join(catalog, name)
    if os.path.exists(path + ".npy"):
        return True
    
    uniform_solid_angle_distribution = distributions.UniformSolidAngle()
    angles = uniform_solid_angle_distribution.rvs(num_samples)

    det_h1 = Detector('H1')
    det_l1 = Detector('L1')
    det_v1 = Detector('V1')
    det = [det_h1, det_l1, det_v1]

    i = 0
    try:
        hp, hc = get_td_waveform(approximant=apx,
                             mass1=params["m1"][i],
                             mass2=params["m2"][i],
                             spin1z=params["s1"][i],
                             spin2z=params["s2"][i],
                             inclination=params["inclination"][i],
                             coa_phase=params["coa_phase"][i],
                             delta_t=1.0/4096,
                             f_lower=10, distance = params["distance"][i])
    except: 
        print(apx + " failed")
        return False
    end_time = params["end_time"][i]
    declination = angles["theta"][i] - np.pi / 2
    right_ascension = angles["phi"][i]
    polarization = params["polarization"][i]
    hp.start_time += end_time
    hc.start_time += end_time

    signal = []
    for ch in [0, 1, 2]:
        s = det[ch].project_wave(hp, hc,  right_ascension, declination, polarization)
        if s.start_time > 0:
            return False
        s = s.crop(-s.start_time, 0)
        s.start_time = 0
        s.resize(4096 * 2)
        s = resample_to_delta_t(s, 1/2048)
        signal += [s]
    signal = np.array(signal)
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    np.save(path, signal)
    save_mask(hp, hc, path)
    return True

def generate_signal_spin_predicted_mchirp(seed, cdfinv, catalog = "signal_spin_mc"):
    def logpdf(x):
        return x
    s = 2 ** 30 + seed
    name = str(s) + "_signal"

    np.random.seed(s)
    apx = 'SEOBNRv4_opt'
    num_samples = 1
    minq = 1/4
    maxq = 1/minq


    mc_distribution = distributions.External(["x"], logpdf, cdfinv=cdfinv)
    q_distribution = distributions.QfromUniformMass1Mass2(q=(minq,maxq))
    mc_samples = mc_distribution.rvs(size=num_samples)
    q_samples = q_distribution.rvs(size=num_samples)

    m1 = conversions.mass1_from_mchirp_q(mc_samples, q_samples['q'])
    m2 = conversions.mass2_from_mchirp_q(mc_samples, q_samples['q'])

    params_distribution = distributions.Uniform(
                                                inclination = (0, np.pi), 
                                                coa_phase = (0, 2 * np.pi), 
                                                distance = (200, 3000),
                                                polarization = (0, 2 * np.pi),
                                                end_time = (1.5, 1.9),
                                                s1 = (-1, 1),
                                                s2 = (-1, 1)
    )
    params = params_distribution.rvs(num_samples)

    name += "_%.2f_%.2f_%.2f_%.2f" % (m1[0], m2[0], params["distance"][0], params["end_time"][0])
    path = os.path.join(catalog, name)
    if os.path.exists(path + ".npy"):
        return True
    
    uniform_solid_angle_distribution = distributions.UniformSolidAngle()
    angles = uniform_solid_angle_distribution.rvs(num_samples)

    det_h1 = Detector('H1')
    det_l1 = Detector('L1')
    det_v1 = Detector('V1')
    det = [det_h1, det_l1, det_v1]

    i = 0
    try:
        hp, hc = get_td_waveform(approximant=apx,
                             mass1=m1[i],
                             mass2=m2[i],
                             spin1z=params["s1"][i],
                             spin2z=params["s2"][i],
                             inclination=params["inclination"][i],
                             coa_phase=params["coa_phase"][i],
                             delta_t=1.0/4096,
                             f_lower=10, distance = params["distance"][i])
    except: 
        print(apx + " failed")
        return False
    end_time = params["end_time"][i]
    declination = angles["theta"][i] - np.pi / 2
    right_ascension = angles["phi"][i]
    polarization = params["polarization"][i]
    hp.start_time += end_time
    hc.start_time += end_time

    signal = []
    for ch in [0, 1, 2]:
        s = det[ch].project_wave(hp, hc,  right_ascension, declination, polarization)
        if s.start_time > 0:
            return False
        s = s.crop(-s.start_time, 0)
        s.start_time = 0
        s.resize(4096 * 2)
        s = resample_to_delta_t(s, 1/2048)
        signal += [s]
    signal = np.array(signal)
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    np.save(path, signal)
    save_mask(hp, hc, path)
    return True

if __name__ == '__main__':
    #_ = Parallel(n_jobs=32, verbose=5)([delayed(generate_signal_spin)(i) for i in range(100007)])
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
    _ = Parallel(n_jobs=32, verbose=5)([delayed(generate_signal_spin_predicted_mchirp)(i, cdfinv) for i in range(300000)])