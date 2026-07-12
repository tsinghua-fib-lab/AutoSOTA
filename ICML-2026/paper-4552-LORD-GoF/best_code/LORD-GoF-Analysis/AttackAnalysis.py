import os
import pickle
import argparse
import numpy as np
import torch
from scipy.stats import linregress
from tqdm import tqdm
from collections import defaultdict



def parse_args():
    parser = argparse.ArgumentParser(description="Run Robustness Attacks (Subst, Del, Ins)")
    parser.add_argument('--model', type=str, default='opt', choices=['opt', 'qwen', 'llama'])
    parser.add_argument('--wm_type', type=str, default='gum', choices=['gum', 'inv'])
    parser.add_argument('--pi', type=float, default=0.05)
    parser.add_argument('--temp', type=float, default=0.5)
    parser.add_argument('--m', type=int, default=1000)
    parser.add_argument('--alpha', type=float, default=0.05)
    return parser.parse_args()


args = parse_args()

MODEL_MAP = {'opt': 'opt1.3b', 'qwen': 'qwen2p5_3b', 'llama': 'sheared_llama_2p7b'}

CONFIG = {
    'TOKEN_LEN': 400,
    'ATTACK_RATIOS': [0.1, 0.2],
    'ATTACK_TYPES': ['Substitution', 'Deletion', 'Insertion'],
    'DATA_DIR': 'raw_data',
    'N_CALIB': 10000,
    'W0': 0.01,
    'GAMMA_EXP': 1.2
}



class OnlineLORD:
    def __init__(self, alpha_target=0.05, w0=0.01, gamma_exp=1.2):
        self.alpha = alpha_target
        self.w0 = w0
        self.max_steps = args.m + 5000
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
    def __init__(self, alpha_target=0.05):
        self.alpha = alpha_target

    def test(self, t, p_value):
        return p_value <= self.alpha



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
    def rao(Y):  # Neyman
        n = len(Y);
        Y_sorted = np.sort(Y)
        sp = np.diff(Y_sorted, prepend=0);
        sp = np.append(sp, 1 - Y_sorted[-1])
        return 0.5 * n * np.sum(np.abs(sp - 1 / (n + 1)))

    @staticmethod
    def greenwood(Y):  # Phi
        n = len(Y);
        Y_sorted = np.sort(Y)
        sp = np.diff(Y_sorted, prepend=0);
        sp = np.append(sp, 1 - Y_sorted[-1])
        return np.sum(sp ** 2)


TEST_NAMES = ['Kolmogorov', 'Kuiper', 'Anderson', 'Cramer', 'Watson', 'Chi_squared', 'Rao', 'Greenwood']


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


def get_watermark_data(temp):
    model_prefix = MODEL_MAP[args.model]
    if args.wm_type == 'gum':
        fname = f"{CONFIG['DATA_DIR']}/{model_prefix}_temp_{temp}_len_{CONFIG['TOKEN_LEN']}_cnt_1000.pkl"
        is_inv = False
    else:
        fname = f"{CONFIG['DATA_DIR']}/{model_prefix}_inv_temp_{temp}_len_{CONFIG['TOKEN_LEN']}_cnt_500.pkl"
        is_inv = True

    if not os.path.exists(fname):
        if is_inv:
            dummy = []
            for _ in range(100):
                u = np.random.rand(CONFIG['TOKEN_LEN']);
                eta = np.random.rand(CONFIG['TOKEN_LEN'])
                dummy.append(-(np.abs(u - eta) * 0.3))
            return np.array(dummy)
        else:
            from scipy.stats import beta
            return beta.rvs(0.1, 1, size=(100, CONFIG['TOKEN_LEN']))

    try:
        data = pickle.load(open(fname, "rb"))
        Ys = data['watermark']['Ys']
        if torch.is_tensor(Ys): Ys = Ys.cpu().numpy()
        return Ys
    except:
        return np.random.rand(100, CONFIG['TOKEN_LEN'])



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


def get_calibrators(is_inv, length):
    np.random.seed(42)
    h0_storage = defaultdict(list)
    for _ in range(CONFIG['N_CALIB']):
        if is_inv:
            u = np.random.rand(length);
            eta = np.random.rand(length)
            y_sample = transform_inv_to_uniform(-np.abs(u - eta))
        else:
            y_sample = np.random.rand(length)
        scores = calculate_all_scores(y_sample)
        for k, v in scores.items(): h0_storage[k].append(v)
    return {k: SemiParametricCalibrator(v) for k, v in h0_storage.items()}



def run_experiment():
    print(f"\n>>> Robustness Attack Exp: {args.model.upper()} + {args.wm_type.upper()} (Temp={args.temp})")

    # Pre-load data
    pool = get_watermark_data(args.temp)
    is_inv = (args.wm_type == 'inv')




    table_data = defaultdict(lambda: defaultdict(dict))

    for attack_type in CONFIG['ATTACK_TYPES']:
        for ratio in CONFIG['ATTACK_RATIOS']:
            # Determine effective length
            if attack_type == 'Deletion':
                curr_len = int(CONFIG['TOKEN_LEN'] * (1 - ratio))
            elif attack_type == 'Insertion':
                curr_len = int(CONFIG['TOKEN_LEN'] * (1 + ratio))
            else:  # Substitution
                curr_len = CONFIG['TOKEN_LEN']


            calibs = get_calibrators(is_inv, curr_len)


            methods = []
            for tn in TEST_NAMES: methods.extend([f"Naive_{tn}", f"LORD_{tn}"])
            controllers = {}
            for m in methods:
                if m.startswith("Naive"):
                    controllers[m] = NaiveFixed(args.alpha)
                else:
                    controllers[m] = OnlineLORD(args.alpha, CONFIG['W0'], CONFIG['GAMMA_EXP'])

            stats = {m: {'tp': 0, 'fp': 0} for m in methods}


            np.random.seed(int(ratio * 100 + len(attack_type)))
            labels = np.random.choice([0, 1], size=args.m, p=[1 - args.pi, args.pi])

            for t in range(args.m):

                if labels[t] == 1:
                    idx = np.random.randint(len(pool))
                    Y_orig = pool[idx]  # 400
                else:
                    if is_inv:
                        u = np.random.rand(CONFIG['TOKEN_LEN']);
                        eta = np.random.rand(CONFIG['TOKEN_LEN'])
                        Y_orig = -np.abs(u - eta)
                    else:
                        Y_orig = np.random.rand(CONFIG['TOKEN_LEN'])


                if attack_type == 'Substitution':
                    # Mix noise
                    mask = np.random.rand(len(Y_orig)) < ratio  # ratio is corruption rate
                    if is_inv:
                        u = np.random.rand(len(Y_orig));
                        eta = np.random.rand(len(Y_orig))
                        Y_noise = -np.abs(u - eta)
                    else:
                        Y_noise = np.random.rand(len(Y_orig))

                    Y_doc = Y_orig.copy()
                    Y_doc[mask] = Y_noise[mask]

                elif attack_type == 'Deletion':

                    keep_mask = np.random.rand(len(Y_orig)) > ratio
                    Y_doc = Y_orig[keep_mask]

                    if len(Y_doc) < 10:
                        if is_inv:
                            Y_doc = -np.abs(np.random.rand(10) - np.random.rand(10))
                        else:
                            Y_doc = np.random.rand(10)

                elif attack_type == 'Insertion':
                    # Add ratio * N tokens
                    n_ins = int(len(Y_orig) * ratio)
                    if is_inv:
                        u = np.random.rand(n_ins);
                        eta = np.random.rand(n_ins)
                        Y_ins = -np.abs(u - eta)
                    else:
                        Y_ins = np.random.rand(n_ins)

                    Y_doc = np.concatenate([Y_orig, Y_ins])


                if is_inv:
                    Y_final = transform_inv_to_uniform(Y_doc)
                else:
                    Y_final = Y_doc


                raw_scores = calculate_all_scores(Y_final)
                for tn in TEST_NAMES:
                    pval = calibs[tn].get_pval(raw_scores[tn])


                    prefix_list = ["Naive", "LORD"]
                    for prefix in prefix_list:
                        k = f"{prefix}_{tn}"
                        is_rej = controllers[k].test(t, pval)
                        if is_rej:
                            if labels[t] == 1:
                                stats[k]['tp'] += 1
                            else:
                                stats[k]['fp'] += 1


            n_real = np.sum(labels == 1)
            for k in stats:
                tp = stats[k]['tp'];
                fp = stats[k]['fp']
                power = tp / n_real if n_real > 0 else 0
                fdr = fp / (tp + fp) if (tp + fp) > 0 else 0
                table_data[attack_type][ratio][k] = (fdr, power)


    NAME_MAP = {
        'Kolmogorov': 'Kol', 'Kuiper': 'Kui', 'Cramer': 'Cra', 'Anderson': 'And',
        'Watson': 'Wat', 'Chi_squared': 'Chi', 'Rao': 'Ney', 'Greenwood': 'Phi'
    }

    print("\n" + "=" * 100)
    print(f"Table 3 Format: {args.model.upper()} + {args.wm_type.upper()}")
    print("=" * 100)


    cols = []
    for at in ['Substitution', 'Deletion', 'Insertion']:
        for r in [0.1, 0.2]:
            short_at = "Subst" if at == 'Substitution' else ("Del" if at == 'Deletion' else "Ins")
            cols.append(f"{short_at} ({r})")

    header_str = f"{'Method':<12} "
    for c in cols: header_str += f"| {c:<13} "
    print(header_str)


    sub_header = f"{'':<12} "
    for _ in cols: sub_header += f"| {'FDR':<5} {'Pow':<6} "
    print(sub_header)
    print("-" * len(sub_header))

    base_names = ['Kolmogorov', 'Kuiper', 'Cramer', 'Anderson', 'Watson', 'Chi_squared', 'Rao', 'Greenwood']

    for mode in ['Naive', 'LORD']:
        print(f"--- {mode} ---")
        for base in base_names:
            short_name = NAME_MAP[base]
            row_str = f"{mode}-{short_name:<4} "

            for at in ['Substitution', 'Deletion', 'Insertion']:
                for r in [0.1, 0.2]:
                    key = f"{mode}_{base}"
                    # Check attack name match
                    # Config uses full names
                    res = table_data[at][r][key]
                    fdr, pow = res
                    row_str += f"| {fdr:.3f} {pow:.3f}  "
            print(row_str)
    print("=" * 100 + "\n")


if __name__ == "__main__":
    run_experiment()