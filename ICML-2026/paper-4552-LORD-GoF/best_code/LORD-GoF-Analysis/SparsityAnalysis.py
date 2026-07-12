import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from tqdm import tqdm
from collections import defaultdict
import torch


COMMON_CONFIG = {
    'TOKEN_LEN': 400,
    'M_STREAM': 1000,
    'ALPHA': 0.05,
    'W0': 0.01,
    'GAMMA_EXP': 1.2,
    'N_CALIB': 10000,
    'DATA_DIR': 'raw_data',
    'SAVE_DIR': 'my_plot',
    'FONT_FAMILY': 'DejaVu Sans',
    'LINE_WIDTH': 1.5
}

EXPERIMENTS = [
    {'model': 'Qwen-2.5-3B', 'wm': 'Gumbel', 'file_prefix': 'qwen2p5_3b', 'is_inv': False, 'temp': 0.5, 'rho': 0.7},
    {'model': 'Sheared-LLaMA-2.7B', 'wm': 'Gumbel', 'file_prefix': 'sheared_llama_2p7b', 'is_inv': False, 'temp': 0.5,
     'rho': 0.7},
    {'model': 'OPT-1.3B', 'wm': 'Inverse', 'file_prefix': 'opt1.3b', 'is_inv': True, 'temp': 0.7, 'rho': 0.7},
    {'model': 'Qwen-2.5-3B', 'wm': 'Inverse', 'file_prefix': 'qwen2p5_3b', 'is_inv': True, 'temp': 0.7, 'rho': 0.7},
    {'model': 'Sheared-LLaMA-2.7B', 'wm': 'Inverse', 'file_prefix': 'sheared_llama_2p7b', 'is_inv': True, 'temp': 0.7,
     'rho': 0.7},
]

PI_LIST = [0.01, 0.02, 0.05, 0.10, 0.30, 0.50, 0.70, 0.90, 0.99]



class OnlineLORD:
    def __init__(self, alpha_target=0.05, w0=0.01, gamma_exp=1.2):
        self.alpha = alpha_target
        self.w0 = w0
        self.max_steps = COMMON_CONFIG['M_STREAM'] + 5000
        self.gamma = 0.07 * (np.arange(1, self.max_steps + 2) ** -gamma_exp)
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


class NaiveFixed:
    def __init__(self, alpha_target=0.05): self.alpha = alpha_target

    def test(self, t, p_value): return p_value <= self.alpha



class GoFStatistics:
    @staticmethod
    def kolmogorov(Y):
        n = len(Y);
        Y_sorted = np.sort(Y);
        r = np.arange(1, n + 1)
        d_plus = r / n - Y_sorted;
        d_minus = Y_sorted - (r - 1) / n
        return np.max(np.maximum(d_plus, d_minus))

    @staticmethod
    def kuiper(Y):
        n = len(Y);
        Y_sorted = np.sort(Y);
        r = np.arange(1, n + 1)
        d_plus = r / n - Y_sorted;
        d_minus = Y_sorted - (r - 1) / n
        return np.max(d_plus) + np.max(d_minus)

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
        n = len(Y);
        w2 = GoFStatistics.cramer(Y)
        return w2 - n * (np.mean(Y) - 0.5) ** 2

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


TEST_NAMES = ['Kolmogorov', 'Kuiper', 'Anderson', 'Cramer', 'Watson', 'Chi_squared', 'Rao', 'Greenwood']

# Display codes used in the paper's figures/tables (Rao->Ney, Greenwood->Phi).
DISPLAY = {'Kolmogorov': 'Kol', 'Kuiper': 'Kui', 'Anderson': 'And', 'Cramer': 'Cra',
           'Watson': 'Wat', 'Chi_squared': 'Chi', 'Rao': 'Ney', 'Greenwood': 'Phi'}


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
    for _ in range(COMMON_CONFIG['N_CALIB']):
        if is_inv:
            u = np.random.rand(COMMON_CONFIG['TOKEN_LEN']);
            eta = np.random.rand(COMMON_CONFIG['TOKEN_LEN'])
            y = transform_inv_to_uniform(-np.abs(u - eta))
        else:
            y = np.random.rand(COMMON_CONFIG['TOKEN_LEN'])

        scores = calculate_all_scores(y)
        for k, v in scores.items(): h0_storage[k].append(v)
    return {k: SemiParametricCalibrator(v) for k, v in h0_storage.items()}


def get_watermark_pool(exp_cfg):
    temp = exp_cfg['temp']
    fname_mid = "_inv" if exp_cfg['is_inv'] else ""
    cnt = 500 if exp_cfg['is_inv'] else 1000
    fname = f"{COMMON_CONFIG['DATA_DIR']}/{exp_cfg['file_prefix']}{fname_mid}_temp_{temp}_len_{COMMON_CONFIG['TOKEN_LEN']}_cnt_{cnt}.pkl"

    if not os.path.exists(fname):
        if exp_cfg['is_inv']:
            sim_data = []
            for _ in range(100):
                u = np.random.rand(COMMON_CONFIG['TOKEN_LEN']);
                eta = np.random.rand(COMMON_CONFIG['TOKEN_LEN'])
                sim_data.append(-np.abs(u - eta) * 0.3)
            return np.array(sim_data)
        else:
            from scipy.stats import beta
            return beta.rvs(0.1, 1, size=(100, COMMON_CONFIG['TOKEN_LEN']))

    data = pickle.load(open(fname, "rb"))
    Ys = data['watermark']['Ys']
    if torch.is_tensor(Ys): Ys = Ys.cpu().numpy()
    return Ys



def run_experiment(exp_cfg, idx):
    print(f"Running Exp {idx + 1} (Sparsity): {exp_cfg['model']} + {exp_cfg['wm']}...")
    calibs = get_calibrators(exp_cfg['is_inv'])
    pool = get_watermark_pool(exp_cfg)

    # Methods: 8 Naive, 8 LORD
    methods = []
    for tn in TEST_NAMES:
        methods.append(f"Naive_{tn}")
        methods.append(f"LORD_{tn}")

    results = {m: {'pi': [], 'fdr': [], 'pow': []} for m in methods}

    # Loop over Pi
    for pi in tqdm(PI_LIST, desc="Pi Loop"):
        # Run Simulation
        np.random.seed(int(pi * 1000) + idx * 100)  # Reproducibility
        labels = np.random.choice([0, 1], size=COMMON_CONFIG['M_STREAM'], p=[1 - pi, pi])

        controllers = {}
        for m in methods:
            if m.startswith("Naive"):
                controllers[m] = NaiveFixed(COMMON_CONFIG['ALPHA'])
            else:
                controllers[m] = OnlineLORD(COMMON_CONFIG['ALPHA'])

        stats = {m: {'tp': 0, 'fp': 0} for m in methods}

        for t in range(COMMON_CONFIG['M_STREAM']):
            if labels[t] == 1:
                idx_pool = np.random.randint(len(pool))
                Y_pure = pool[idx_pool]
                mask = np.random.rand(COMMON_CONFIG['TOKEN_LEN']) < exp_cfg['rho']

                if exp_cfg['is_inv']:
                    u = np.random.rand(COMMON_CONFIG['TOKEN_LEN']);
                    eta = np.random.rand(COMMON_CONFIG['TOKEN_LEN'])
                    Y_noise = -np.abs(u - eta)
                    Y_doc = Y_noise.copy();
                    Y_doc[mask] = Y_pure[mask]
                    Y_final = transform_inv_to_uniform(Y_doc)
                else:
                    Y_doc = np.random.rand(COMMON_CONFIG['TOKEN_LEN']);
                    Y_doc[mask] = Y_pure[mask]
                    Y_final = Y_doc
            else:
                if exp_cfg['is_inv']:
                    u = np.random.rand(COMMON_CONFIG['TOKEN_LEN']);
                    eta = np.random.rand(COMMON_CONFIG['TOKEN_LEN'])
                    Y_final = transform_inv_to_uniform(-np.abs(u - eta))
                else:
                    Y_final = np.random.rand(COMMON_CONFIG['TOKEN_LEN'])

            scores = calculate_all_scores(Y_final)

            for tn in TEST_NAMES:
                pval = calibs[tn].get_pval(scores[tn])
                for mode in ["Naive", "LORD"]:
                    key = f"{mode}_{tn}"
                    if controllers[key].test(t, pval):
                        if labels[t] == 1:
                            stats[key]['tp'] += 1
                        else:
                            stats[key]['fp'] += 1

        # Aggregate
        n_real = np.sum(labels == 1)
        for m in methods:
            tp = stats[m]['tp'];
            fp = stats[m]['fp']
            fdr = fp / (tp + fp) if (tp + fp) > 0 else 0.0
            power = tp / n_real if n_real > 0 else 0.0
            results[m]['pi'].append(pi)
            results[m]['fdr'].append(fdr)
            results[m]['pow'].append(power)

    # Plotting
    cmap = plt.get_cmap('tab10')
    colors = {name: cmap(i) for i, name in enumerate(TEST_NAMES)}
    file_tag = f"{exp_cfg['file_prefix']}_{exp_cfg['wm'][:3].lower()}_pi"


    plt.figure(figsize=(8, 6))
    for m in methods:
        prefix, tname = m.split('_', 1)
        ls = '-' if prefix == 'LORD' else ':'
        col = colors[tname]
        alpha = 1.0 if prefix == 'LORD' else 0.6
        plt.plot(results[m]['pi'], results[m]['fdr'], label=f"{prefix}-{DISPLAY[tname]}", linestyle=ls, color=col, alpha=alpha, marker='o',
                 markersize=3)

    plt.axhline(0.05, color='k', ls='--', lw=1.5)
    plt.xscale('log')
    plt.xlabel(r'Global Sparsity $\pi$')
    plt.ylabel('FDR')
    plt.ylim(-0.05, 1.05)
    plt.title(f"{exp_cfg['model']} {exp_cfg['wm']} - FDR vs Sparsity")

    plt.tight_layout()
    plt.savefig(f"{COMMON_CONFIG['SAVE_DIR']}/{file_tag}_fdr.pdf")
    plt.close()


    plt.figure(figsize=(8, 6))
    for m in methods:
        prefix, tname = m.split('_', 1)
        ls = '-' if prefix == 'LORD' else ':'
        col = colors[tname]
        alpha = 1.0 if prefix == 'LORD' else 0.6
        plt.plot(results[m]['pi'], results[m]['pow'], label=f"{prefix}-{DISPLAY[tname]}", linestyle=ls, color=col, alpha=alpha, marker='o',
                 markersize=3)

    plt.xscale('log')
    plt.xlabel(r'Global Sparsity $\pi$')
    plt.ylabel('Power')
    plt.ylim(-0.05, 1.05)
    plt.title(f"{exp_cfg['model']} {exp_cfg['wm']} - Power vs Sparsity")

    plt.tight_layout()
    plt.savefig(f"{COMMON_CONFIG['SAVE_DIR']}/{file_tag}_pow.pdf")
    plt.close()


if __name__ == "__main__":
    if not os.path.exists(COMMON_CONFIG['SAVE_DIR']):
        os.makedirs(COMMON_CONFIG['SAVE_DIR'])

    for i, exp in enumerate(EXPERIMENTS):
        run_experiment(exp, i)
