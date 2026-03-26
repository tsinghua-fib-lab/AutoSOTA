import numpy as np
import argparse
import pandas as pd
import jax
from models import nn_model
from utils import l2_complexity
import jax.numpy as jnp
import optax
import numpy as onp
import os 

import sys
sys.path.insert(0,'loci')

from loci.causa.loci import loci

from utils import nullable_int 

def standardize_data(x, return_statistics = False, trim_outliers = 0):
    """
    Standardize the data and do a box-trim of outliers (pure numpy version)).
    """
    if x.shape[1] > 2:
        x = x[:, 0:2]

    if trim_outliers > 0:
        x_ = x[:, 0]
        y_ = x[:, 1]
        x_low, x_high = np.quantile(x_, np.array([trim_outliers/2, 1 - trim_outliers/2]))
        y_low, y_high = np.quantile(y_, np.array([trim_outliers/2, 1 - trim_outliers/2]))
        x = x[(x_ > x_low) & (x_ < x_high) & (y_ > y_low) & (y_ < y_high)]


    mean = np.mean(x, axis = 0, keepdims = True)
    std = np.std(x, axis = 0, keepdims = True)
    if return_statistics:
        return (x - mean) / std, mean.squeeze(), std.squeeze()
    return x - mean / std

def parse_arguments():
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument("--dataset", metavar = "DATASET", type = str, default = "tuebingen", help = "Dataset to use for benchmarking. See data/ for options.")
    parser.add_argument("--outlier_trim", metavar = "OUTLIER_TRIM", type = float, default = 0, help = "What percent of outliers to remove from data.")
    parser.add_argument("--n_steps", metavar = "N_STEPS", type = int, default = 1000, help = "Number of steps for optimization.")
    parser.add_argument("--model", metavar = "MODEL", type = str, default = "hsic", help = "LOCI variant (['hsic', 'hsic_spline', 'lik', 'lik_spline']).")
    parser.add_argument("--fix_samples", metavar = "MAX_SAMPLES", type = nullable_int, default = 1000, help = "Fix a dataset size for benchmarking: if none, use all samples, otherwise, resample if needed to get to this size.")
    parser.add_argument("--outfile", metavar = "OUTFILE", type = str, default = "results.csv", help = "Output file for results.")
    parser.add_argument("--verbose", action = "store_true", help = "Print verbose output.")
    parser.add_argument("--remote", action='store_true', default = False, help = "Use remote data path.")
    return parser.parse_args()

args = parse_arguments()

FIX_SAMPLES = args.fix_samples
VERBOSE = args.verbose

if args.remote:
    BASE_PATH = "/scratch/st-benbr-1/xijohnny/causal-velocity/data/"
else:
    BASE_PATH = "data/"
    from cdt.causality.pairwise import ANM, CDS, IGCI, RECI
    from cdt.independence.stats import NormalizedHSIC
    
OUTFILE  = args.outfile

if args.dataset == "tuebingen":
    DATA_PATH = BASE_PATH + "Tuebingen"
if args.dataset == "tuebingen_new":
    DATA_PATH = BASE_PATH + "Tuebingen"
elif args.dataset == "sim":
    DATA_PATH = BASE_PATH + "Benchmark_simulated/SIM"
elif args.dataset == "simc":
    DATA_PATH = BASE_PATH + "Benchmark_simulated/SIM-c"
elif args.dataset == "simg":
    DATA_PATH = BASE_PATH + "Benchmark_simulated/SIM-G"
elif args.dataset == "simln":
    DATA_PATH = BASE_PATH + "Benchmark_simulated/SIM-ln"
elif args.dataset == "an":
    DATA_PATH = BASE_PATH + "ANLSMN_pairs/AN"
elif args.dataset == "ans":
    DATA_PATH = BASE_PATH + "ANLSMN_pairs/AN-s"
elif args.dataset == "ls":
    DATA_PATH = BASE_PATH + "ANLSMN_pairs/LS"
elif args.dataset == "lss":
    DATA_PATH = BASE_PATH + "ANLSMN_pairs/LS-s"
elif args.dataset == "mnu":
    DATA_PATH = BASE_PATH + "ANLSMN_pairs/MN-U"
elif args.dataset == "velocity":
    DATA_PATH = BASE_PATH + "synthetic/velocity"
elif args.dataset == "sigmoid":
    DATA_PATH = BASE_PATH + "synthetic/sigmoid"
elif args.dataset == "anm":
    DATA_PATH = BASE_PATH + "synthetic/anm"
elif args.dataset == "lsnm":
    DATA_PATH = BASE_PATH + "synthetic/lsnm"
else:
    DATA_PATH = BASE_PATH + args.dataset

#### CONDITIONAL DIVERGENCE (https://github.com/baosws/CDCI/blob/main/CDCI.py)

EPSILON = 1e-8
SEED = 0

from collections import Counter

def cond_dist(x, y, max_dev=3):
    vmax =  2 * max_dev
    vmin = -2 * max_dev

    x = (x - x.mean()) / (x.std() + EPSILON)
    t = x[np.abs(x) < max_dev]
    x = (x - t.mean()) / (t.std() + EPSILON)
    xd = np.round(x * 2)
    xd[xd > vmax] = vmax
    xd[xd < vmin] = vmin

    x_count = Counter(xd)
    vrange = range(vmin, vmax + 1)

    pyx = []
    for x in x_count:
        if x_count[x] > 12:
            yx = y[xd == x]
            yx = (yx - np.mean(yx)) / (np.std(yx) + EPSILON)
            yx = np.round(yx * 2)
            yx[yx > vmax] = vmax
            yx[yx < vmin] = vmin
            count_yx = Counter(yx)
            pyx_x = np.array([count_yx[i] for i in vrange], dtype=np.float64)
            pyx_x = pyx_x / pyx_x.sum()
            pyx.append(pyx_x)
    return pyx

def CKL(A, B, **kargs):
    '''Causal score via Kullback-Leibler divergence'''
    pyx = cond_dist(A, B, **kargs)
    if len(pyx) == 0:
        return 0
    pyx = np.array(pyx) # axis 0: x; axis 1: y
    mean_y = pyx.mean(axis=0)
    return (pyx * np.log((pyx + EPSILON) / (mean_y + EPSILON))).sum(axis=1).mean()

def CKM(A, B, **kargs):
    '''Causal score via Kolmogorov metric'''
    pyx = cond_dist(A, B, **kargs)
    if len(pyx) == 0:
        return 0
    pyx = np.array(pyx) # axis 0: x; axis 1: y
    mean_y = pyx.mean(axis=0).cumsum()
    pyx = pyx.cumsum(axis=1)

    return np.abs(pyx - mean_y).max(axis=1).mean()

def CHD(A, B, **kargs):
    '''Causal score via Hellinger Distance'''
    pyx = cond_dist(A, B, **kargs)
    if len(pyx) == 0:
        return 0
    pyx = np.array(pyx) # axis 0: x; axis 1: y
    mean_y = pyx.mean(axis=0)
    return (((pyx ** 0.5 - mean_y ** 0.5) ** 2).sum(axis=1) ** 0.5).mean()

def CCS(A, B, **kargs):
    '''Causal score via Chi-Squared distance'''
    pyx = cond_dist(A, B, **kargs)
    if len(pyx) == 0:
        return 0
    pyx = np.array(pyx) # axis 0: x; axis 1: y
    mean_y = pyx.mean(axis=0)
    return ((pyx - mean_y) ** 2 / (mean_y + EPSILON)).sum(axis=1).mean()

def CTV(A, B, **kargs):
    '''Causal score via Total Variation distance'''
    pyx = cond_dist(A, B, **kargs)
    if len(pyx) == 0:
        return 0
    pyx = np.array(pyx) # axis 0: x; axis 1: y
    mean_y = pyx.mean(axis=0)
    return 0.5 * np.abs(pyx - mean_y).sum(axis=1).mean()

def causal_score(variant, A, B, **kargs):
    variant = eval(variant)
    return variant(B, A, **kargs) - variant(A, B, **kargs)

class anm(nn_model):
        """
        anm model. 
        """
        def forward(self, x, params):
            if self.layers>1 :
                h = x.reshape(-1, 1)
                for i in (range(self.layers-1)):
                    weight = params[f"w{i+1}_l2"]
                    h = jnp.dot(h, weight) + params[f"b{i+1}"]
                    h = jax.nn.relu(h)
                out = jnp.dot(h, params[f"w{self.layers}_l2"]) + params[f"b{self.layers}"]
            else:
                out = jnp.dot(x, params["w1_l2"]) + params["b1"]
            return out
        
        def __call__(self, x, params):
            out = self.forward(x, params)
            out = out.reshape(-1,)
            return out
        
def MODEL(x, params):
    return anm_m(x, params)

def loss_reg(params, x, y, lam = 0.1):
    """
    Loss function for ANM. 
    """
    pred = MODEL(x, params)
    loss_ = (pred.squeeze() - y.squeeze())**2
    return jnp.mean(loss_) + l2_complexity(params, lam = lam)

anm_m = anm(layers = 3, hidden_size = 64)

def causal_score_anm(x_data, y_data, seed):
    """
    Causal score from ANM HSIC.
    """
    PARAMS_INIT = anm_m.params_init(seed = seed, init_weight = 0.5)
    def train_ANM(cause_train, effect_train, cause_test, effect_test, params_init, n_steps = 100, tol = 1e-5, lr = 1e-3):
        """
        Train ANM model and output HSIC GoF score.
        """ 
        def step(params, opt, opt_state, x, y):
            loss, grads = jax.value_and_grad(loss_reg)(params, x, y)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, loss

        opt = optax.adam(lr)
        opt_state = opt.init(params_init)

        best_loss = float('inf')
        loss_prev = float('inf')
        params = params_init.copy()
        best_params = params.copy()
        conv_steps = n_steps

        for i in (range(n_steps)):
            params, loss_val = step(params, opt, opt_state, cause_train, effect_train)
            increment = abs(loss_prev - loss_val)
            loss_prev = loss_val
            if loss_val < best_loss:
                best_loss = loss_val
                best_params = params.copy()
            if increment < tol:
                conv_steps = i+1
                break

        e = effect_test - MODEL(cause_test, best_params)
        best_loss = loss_reg(best_params, cause_test, effect_test, lam = 0)

        e_test = onp.array(e)
        cause_test = onp.array(cause_test)

        indscore = NormalizedHSIC().predict(cause_test, e_test, maxpnt = 500)

        return indscore, best_loss, best_params, conv_steps

    indscore_fwd, mse_fwd, best_params_fwd, conv_steps_fwd = train_ANM(x_data, y_data, x_data, y_data, PARAMS_INIT)
    indscore_bwd, mse_bwd, best_params_bwd, conv_steps_bwd = train_ANM(y_data, x_data, y_data, x_data, PARAMS_INIT)
    score_indtest = indscore_bwd - indscore_fwd
    score_mse = mse_bwd - mse_fwd

    return score_indtest, score_mse, mse_fwd, mse_bwd

if __name__ == "__main__":

    success_weights = []
    total_weights = []
    scores = []
    mse_causal = []
    mse_anticausal = []

    if args.dataset in ["an", "ans", "ls", "lss", "mnu"]:
        meta = np.loadtxt(DATA_PATH + "/pairs_gt.txt")
    else:
        meta = pd.read_csv(DATA_PATH + "/pairmeta.txt",  delim_whitespace=True,
                            header=None,
                            names=['pair_id', 'cause_start', 'cause_end', 'effect_start', 'effect_end', 'weight'],
                            index_col=0)
        discrete_pairs = [47, 70, 107] 
        discrete_pairs_new = [5,6,7,8,9,10,11,13,14,15,16,26,27,28,29,32,33,34,35,36,37,47,70,85,94,95,99,105,107]
        multivariate_pairs = [52, 53, 54, 55, 71, 105]
        tue_blacklist = discrete_pairs + multivariate_pairs
        tue_blacklist_new = discrete_pairs_new + multivariate_pairs

    for iter in range(len(meta)):

        if args.dataset in ["an", "ans", "ls", "lss", "mnu"]:
            pair_id = iter + 1
            effect = meta[iter] + 1
            if effect == 1:
                cause = 2
            else:
                cause = 1
            weight = 1
            dat = np.loadtxt(DATA_PATH + f"/pair_{pair_id}.txt", delimiter = ",", skiprows = 1, usecols = (1,2))
        else: 
            pair_id = meta.index[iter].astype(int)
            if args.dataset=="tuebingen" and pair_id in tue_blacklist:
                continue
            if args.dataset=="tuebingen_new" and pair_id in tue_blacklist_new:
                continue
            cause = meta["cause_start"].values[iter].astype(int)
            effect = meta["effect_start"].values[iter].astype(int)
            weight = meta["weight"].values[iter]
            dat = np.loadtxt(DATA_PATH + f"/pair{pair_id:04d}.txt")

        n = dat.shape[0]

        if FIX_SAMPLES is not None:
            idx_subsample = jax.random.choice(jax.random.PRNGKey(0), n, shape = (FIX_SAMPLES,), replace = True) 
            dat = dat[idx_subsample]
            
        dat, dat_mean, dat_std = standardize_data(dat, return_statistics = True, trim_outliers = args.outlier_trim)
        x_data, x_data_mean, x_data_std = dat[:,0], dat_mean[0], dat_std[0]
        y_data, y_data_mean, y_data_std = dat[:,1], dat_mean[1], dat_std[1] 
        
        if args.model == "hsic":
            score = loci(np.array(x_data), np.array(y_data), n_steps = args.n_steps)
        elif args.model == "lik":
            score = loci(np.array(x_data), np.array(y_data), independence_test = False, n_steps = args.n_steps)
        elif args.model == "hsic_spline":
            score = loci(np.array(x_data), np.array(y_data), neural_network=False, n_steps = args.n_steps)
        elif args.model == "lik_spline":
            score = loci(np.array(x_data), np.array(y_data), independence_test = False, neural_network=False, n_steps = args.n_steps)
        elif args.model == "anm_ind":
            score, _, mse_fwd, mse_bwd = causal_score_anm(x_data, y_data, seed = SEED)
            SEED += 1
            if cause == 1:
                mse_causal = mse_fwd
            else:
                mse_causal = mse_bwd
        elif args.model == "anm_mse":
            _, score, mse_fwd, mse_bwd = causal_score_anm(x_data, y_data, seed = SEED)[1]
            SEED += 1
            if cause == 1:
                mse_causal = mse_fwd
            else:
                mse_causal = mse_bwd
        elif args.model == "cds":
            score = CDS().predict_proba((x_data, y_data))
        elif args.model == "igci_unif":
            score = IGCI().predict_proba((x_data, y_data), ref_measure= "uniform", estimator= "entropy")
        elif args.model == "igci_gauss":
            score = IGCI().predict_proba((x_data, y_data), ref_measure= "gaussian", estimator= "entropy")
        elif args.model == "reci":
            score = RECI().predict_proba((x_data, y_data))
        elif args.model == "ckl":
            score = causal_score("CKL", x_data, y_data)
        elif args.model == "ckm":
            score = causal_score("CKM", x_data, y_data)
        elif args.model == "chd":
            score = causal_score("CHD", x_data, y_data)
        elif args.model == "ccs":
            score = causal_score("CCS", x_data, y_data)
        elif args.model == "ctv":
            score = causal_score("CTV", x_data, y_data)
        
        scores.append(np.abs(score))
        if score > 0: 
            est_cause = 1
        else:
            est_cause = 2

        if est_cause == cause:
            success_weights.append(weight)
        else:
            success_weights.append(0)
        
        total_weights.append(weight)

        if VERBOSE: print(f"Pair ID: {pair_id} \n Running Success Rate: {sum(success_weights)/sum(total_weights)}")

        

    ## AUDRC calculation

    idx = np.flip(np.argsort(scores))
    scores, success_weights, total_weights = np.array(scores)[idx], np.array(success_weights)[idx], np.array(total_weights)[idx]

    AUDRC = 0

    for i in range(len(scores)):
        AUDRC += sum(success_weights[:i+1]) / sum(total_weights[:i+1])
    
    AUDRC /= len(scores)

    if not os.path.exists(OUTFILE):
        with open(OUTFILE, "w") as f:
            f.write("score, kernel, reg, model, bw_factor, bw_factor_joint, dataset, outlier_trim, test_size, AUDRC, success_rate, gof_causal, gof_anticausal, score_time, score_cause_mse, effect_score_mse, joint_score_mse, steps_causal, steps_anticausal, sample_size\n")

    with open(OUTFILE, "a+") as f:
        f.write(f", , ,{args.model}, , ,{args.dataset}, {args.outlier_trim}, , {AUDRC}, {sum(success_weights)/sum(total_weights)}, , , , , , , , {FIX_SAMPLES}\n")

