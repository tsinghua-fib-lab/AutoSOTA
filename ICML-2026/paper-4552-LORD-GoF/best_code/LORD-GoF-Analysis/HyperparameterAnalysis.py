import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import linregress, kstest
from tqdm import tqdm
from collections import defaultdict
import torch


CONFIG = {
    'M_STREAM': 2000,
    'TOKEN_LEN': 400,
    'ALPHA': 0.05,
    'N_CALIB': 10000,
    'DATA_DIR': 'raw_data',

    'W0_LIST': [0.0005, 0.001, 0.005, 0.01, 0.025],
    'GAMMA_LIST': [1.05, 1.2, 1.4, 1.6, 1.8, 2.0]
}


EXPERIMENTS = [
    {'model': 'OPT-1.3B', 'wm': 'Gumbel', 'prefix': 'opt1.3b', 'is_inv': False, 'temp': 0.5},
    {'model': 'OPT-1.3B', 'wm': 'Inverse', 'prefix': 'opt1.3b', 'is_inv': True, 'temp': 0.7},
    {'model': 'Qwen-2.5-3B', 'wm': 'Gumbel', 'prefix': 'qwen2p5_3b', 'is_inv': False, 'temp': 0.5},
    {'model': 'Qwen-2.5-3B', 'wm': 'Inverse', 'prefix': 'qwen2p5_3b', 'is_inv': True, 'temp': 0.7},
    {'model': 'Sheared-LLaMA', 'wm': 'Gumbel', 'prefix': 'sheared_llama_2p7b', 'is_inv': False, 'temp': 0.5},
    {'model': 'Sheared-LLaMA', 'wm': 'Inverse', 'prefix': 'sheared_llama_2p7b', 'is_inv': True, 'temp': 0.7},
]


TEST_NAMES = ['Kolmogorov', 'Kuiper', 'Anderson', 'Cramer', 'Watson', 'Chi_squared', 'Rao', 'Greenwood']



class OnlineLORD:
    def __init__(self, alpha_target=0.05, w0=0.01, gamma_exp=1.2, max_steps=10000):
        self.alpha = alpha_target
        self.w0 = w0

        raw_gamma = (np.arange(1, max_steps + 2) ** -gamma_exp)
        self.gamma = 0.07 * raw_gamma
        self.gamma /= self.gamma.sum()

        self.last_discovery_time = 0
        self.wealth = w0

    def test(self, t, p_value):
        delay = (t + 1) - self.last_discovery_time
        if delay >= len(self.gamma):
            alpha_t = 1e-20
        else:
            alpha_t = self.wealth * self.gamma[delay]

        is_reject = p_value <= alpha_t
        if is_reject:
            self.wealth += (self.alpha - self.w0)
            self.last_discovery_time = t + 1
        else:
            self.wealth -= alpha_t
            if self.wealth < 0: self.wealth = 0
        return is_reject


class GoFStatistics:
    @staticmethod
    def kolmogorov(Y):
        n = len(Y);
        Y_sorted = np.sort(Y);
        r = np.arange(1, n + 1)
        return np.max(np.maximum(r / n - Y_sorted, Y_sorted - (r - 1) / n))

    @staticmethod
    def kuiper(Y):
        n = len(Y);
        Y_sorted = np.sort(Y);
        r = np.arange(1, n + 1)
        return np.max(r / n - Y_sorted) + np.max(Y_sorted - (r - 1) / n)

    @staticmethod
    def anderson(Y):
        n = len(Y);
        Y_sorted = np.sort(Y)
        Y_sorted = np.clip(Y_sorted, 1e-10, 1 - 1e-10)
        S = np.sum((2 * np.arange(1, n + 1) - 1) * (np.log(Y_sorted) + np.log(1 - Y_sorted[::-1]))) / n
        return -n - S

    @staticmethod
    def cramer(Y):
        n = len(Y);
        Y_sorted = np.sort(Y)
        return 1 / (12 * n) + np.sum((Y_sorted - (2 * np.arange(1, n + 1) - 1) / (2 * n)) ** 2)

    @staticmethod
    def watson(Y):
        return GoFStatistics.cramer(Y) - len(Y) * (np.mean(Y) - 0.5) ** 2

    @staticmethod
    def chi_squared(Y, c=10):
        obs, _ = np.histogram(Y, bins=np.linspace(0, 1, c + 1))
        return np.sum((obs - len(Y) / c) ** 2 / (len(Y) / c))

    @staticmethod
    def rao(Y):
        n = len(Y);
        Y_sorted = np.sort(Y)
        sp = np.diff(Y_sorted, prepend=0);
        sp = np.append(sp, 1 - Y_sorted[-1])
        return 0.5 * n * np.sum(np.abs(sp - 1 / (n + 1)))

    @staticmethod
    def greenwood(Y):
        n = len(Y);
        Y_sorted = np.sort(Y)
        sp = np.diff(Y_sorted, prepend=0);
        sp = np.append(sp, 1 - Y_sorted[-1])
        return np.sum(sp ** 2)


def calculate_all_scores(Y):
    return {
        'Kolmogorov': GoFStatistics.kolmogorov(Y), 'Kuiper': GoFStatistics.kuiper(Y),
        'Anderson': GoFStatistics.anderson(Y), 'Cramer': GoFStatistics.cramer(Y),
        'Watson': GoFStatistics.watson(Y), 'Chi_squared': GoFStatistics.chi_squared(Y),
        'Rao': GoFStatistics.rao(Y), 'Greenwood': GoFStatistics.greenwood(Y)
    }


def transform_inv_to_uniform(Ys_neg):
    r = -np.array(Ys_neg);
    r = np.clip(r, 0, 1 - 1e-9)
    return 1 - (1 - r) ** 2


class SemiParametricCalibrator:
    def __init__(self, h0_scores):
        self.h0_scores = np.sort(h0_scores);
        self.n = len(h0_scores)
        tail_frac = 0.1;
        self.tail_start = int(self.n * (1 - tail_frac))
        self.tail_scores = self.h0_scores[self.tail_start:]
        self.thresh = self.tail_scores[0]
        emp_p = (self.n - np.arange(self.tail_start + 1, self.n + 1) + 1) / (self.n + 1)
        res = linregress(self.tail_scores, np.log(emp_p))
        self.slope = res.slope;
        self.intercept = res.intercept

    def get_pval(self, score):
        if score <= self.thresh:
            idx = np.searchsorted(self.h0_scores, score, side='left')
            return (self.n - idx + 1) / (self.n + 1)
        return np.exp(self.slope * score + self.intercept)


def get_calibrators(is_inv):

    np.random.seed(42)
    h0_storage = defaultdict(list)
    for _ in range(CONFIG['N_CALIB']):
        if is_inv:
            u = np.random.rand(CONFIG['TOKEN_LEN']);
            eta = np.random.rand(CONFIG['TOKEN_LEN'])
            y = transform_inv_to_uniform(-np.abs(u - eta))
        else:
            y = np.random.rand(CONFIG['TOKEN_LEN'])

        scores = calculate_all_scores(y)
        for k, v in scores.items(): h0_storage[k].append(v)
    return {k: SemiParametricCalibrator(v) for k, v in h0_storage.items()}


def get_watermark_pool(exp_cfg):
    fname_mid = "_inv" if exp_cfg['is_inv'] else ""
    cnt = 500 if exp_cfg['is_inv'] else 1000
    fname = f"{CONFIG['DATA_DIR']}/{exp_cfg['prefix']}{fname_mid}_temp_{exp_cfg['temp']}_len_{CONFIG['TOKEN_LEN']}_cnt_{cnt}.pkl"

    if not os.path.exists(fname):
        sim_data = []
        for _ in range(100):
            if exp_cfg['is_inv']:
                u = np.random.rand(CONFIG['TOKEN_LEN']);
                eta = np.random.rand(CONFIG['TOKEN_LEN'])
                sim_data.append(-np.abs(u - eta) * 0.3)
            else:
                from scipy.stats import beta
                sim_data.append(beta.rvs(0.1, 1, size=CONFIG['TOKEN_LEN']))
        return np.array(sim_data)
    data = pickle.load(open(fname, "rb"))
    Ys = data['watermark']['Ys']
    if torch.is_tensor(Ys): Ys = Ys.cpu().numpy()
    return Ys



def run_experiment(exp_cfg):
    print(f"\n" + "=" * 60)
    print(f"EXP: {exp_cfg['model']} + {exp_cfg['wm']} (Temp={exp_cfg['temp']})")
    print("=" * 60)

    calibs = get_calibrators(exp_cfg['is_inv'])
    pool = get_watermark_pool(exp_cfg)


    pi, rho = 0.05, 0.7
    np.random.seed(999)
    labels = np.random.choice([0, 1], size=CONFIG['M_STREAM'], p=[1 - pi, pi])
    n_real = np.sum(labels)


    stream_pvals = {tn: [] for tn in TEST_NAMES}

    for t in range(CONFIG['M_STREAM']):
        if labels[t] == 1:
            idx = np.random.randint(len(pool))
            Y_pure = pool[idx]
            mask = np.random.rand(CONFIG['TOKEN_LEN']) < rho
            if exp_cfg['is_inv']:
                u = np.random.rand(CONFIG['TOKEN_LEN']);
                eta = np.random.rand(CONFIG['TOKEN_LEN'])
                Y_doc = -np.abs(u - eta);
                Y_doc[mask] = Y_pure[mask]
                Y_final = transform_inv_to_uniform(Y_doc)
            else:
                Y_doc = np.random.rand(CONFIG['TOKEN_LEN']);
                Y_doc[mask] = Y_pure[mask]
                Y_final = Y_doc
        else:
            if exp_cfg['is_inv']:
                u = np.random.rand(CONFIG['TOKEN_LEN']);
                eta = np.random.rand(CONFIG['TOKEN_LEN'])
                Y_final = transform_inv_to_uniform(-np.abs(u - eta))
            else:
                Y_final = np.random.rand(CONFIG['TOKEN_LEN'])

        scores = calculate_all_scores(Y_final)
        for tn in TEST_NAMES:
            stream_pvals[tn].append(calibs[tn].get_pval(scores[tn]))


    print("\n[A] Impact of Initial Wealth (W0) [Fixed Gamma=1.2]")
    print(f"{'W0':<10} | {'Avg FDR':<10} | {'Avg Power':<10} | {'Min Pow':<8} | {'Max Pow':<8}")
    print("-" * 65)

    rows_w0 = []
    for w0 in CONFIG['W0_LIST']:
        fdr_list, pow_list = [], []

        for tn in TEST_NAMES:
            ctrl = OnlineLORD(CONFIG['ALPHA'], w0=w0, gamma_exp=1.2, max_steps=CONFIG['M_STREAM'] + 100)
            tp, fp = 0, 0
            for t, pval in enumerate(stream_pvals[tn]):
                if ctrl.test(t, pval):
                    if labels[t] == 1:
                        tp += 1
                    else:
                        fp += 1
            fdr = fp / (tp + fp) if (tp + fp) > 0 else 0
            power = tp / n_real if n_real > 0 else 0
            fdr_list.append(fdr);
            pow_list.append(power)

        avg_fdr = np.mean(fdr_list)
        avg_pow = np.mean(pow_list)
        rows_w0.append((w0, avg_fdr, avg_pow))
        print(f"{w0:<10.4f} | {avg_fdr:<10.3f} | {avg_pow:<10.3f} | {min(pow_list):<8.3f} | {max(pow_list):<8.3f}")


    fixed_w0 = 0.005
    print("\n[B] Impact of Decay Exponent (Gamma) [Fixed W0=0.005]")
    print(f"{'Gamma':<10} | {'Avg FDR':<10} | {'Avg Power':<10} | {'Min Pow':<8} | {'Max Pow':<8}")
    print("-" * 65)

    rows_gamma = []
    for g in CONFIG['GAMMA_LIST']:
        fdr_list, pow_list = [], []
        for tn in TEST_NAMES:
            ctrl = OnlineLORD(CONFIG['ALPHA'], w0=fixed_w0, gamma_exp=g, max_steps=CONFIG['M_STREAM'] + 100)
            tp, fp = 0, 0
            for t, pval in enumerate(stream_pvals[tn]):
                if ctrl.test(t, pval):
                    if labels[t] == 1:
                        tp += 1
                    else:
                        fp += 1
            fdr = fp / (tp + fp) if (tp + fp) > 0 else 0
            power = tp / n_real if n_real > 0 else 0
            fdr_list.append(fdr);
            pow_list.append(power)

        avg_fdr = np.mean(fdr_list)
        avg_pow = np.mean(pow_list)
        rows_gamma.append((g, avg_fdr, avg_pow))
        print(f"{g:<10.2f} | {avg_fdr:<10.3f} | {avg_pow:<10.3f} | {min(pow_list):<8.3f} | {max(pow_list):<8.3f}")


if __name__ == "__main__":
    for exp in EXPERIMENTS:
        run_experiment(exp)