import jax 
import jax.numpy as jnp
import numpy as onp
import argparse
import pandas as pd
import time 
import os

from training import fit_sm_bd
from utils import standardize_data, nullable_float, count_nparams, nullable_int
from models import parametric_model, nn_velocity_model, anm_model, lsnm_model
from score import stein_score_all, kde_score_all, hybrid_score_all, score_matching_all
from mechanisms import mechanism_from_flow


def parse_arguments():
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument("--seed", metavar = "SEED", type = int, default = 0, help = "Seed for parameter initialization.")
    parser.add_argument("--score", metavar = "SCORE", type = str, default = "stein", help = "Score estimation method (['stein', 'kde', 'hybrid]).")
    parser.add_argument("--kernel", metavar = "KERNEL", type = str, default = "gauss", help = "Kernel for score estimation (['gauss', 'exp']).")
    parser.add_argument("--model", metavar = "MODEL", type = str, default = "parametric-quad", help = "Model to use for training (['nn', 'anm', lsnm', 'additive', 'parametric-lin', 'parametric-quad', 'parametric-cubic', 'parametric-quartic']).")
    parser.add_argument("--layers", metavar = "LAYERS", type = int, default = 2, help = "Number of layers for neural network models.")
    parser.add_argument("--hidden_size", metavar = "HIDDEN_SIZE", type = int, default = 32, help = "Hidden size for neural network models.")
    parser.add_argument("--init_weight", metavar = "INIT_WEIGHT", type = float, default = 0.5, help = "Initialization variance.")
    parser.add_argument("--reg", metavar = "REG", type = float, default = 0.1, help = "Regularization term for score estimator. Suggest 0.1 for stein and 0.01 for kde.")
    parser.add_argument("--bw_factor", metavar = "BW_FACTOR", type = float, default = 1, help = "Multiply the standard heuristic bandwidth by a fixed factor for score estimator.")
    parser.add_argument("--bw_factor_joint", metavar = "BW_FACTOR", type = float, default = 1, help = "Multiply the standard heuristic bandwidth by a fixed factor for score estimator.")
    parser.add_argument("--lr", metavar = "LR", type = float, default = 0.1, help = "Base learning rate for optimizer (actual is scaled by number of parameters in the model).")
    parser.add_argument("--n_steps", metavar = "N_STEPS", type = int, default = 100, help = "Number of steps for optimization.")
    parser.add_argument("--fix_samples", metavar = "MAX_SAMPLES", type = int, default = 1000, help = "Fix a dataset size for benchmarking: if none, use all samples, otherwise, resample if needed to get to this size.")
    parser.add_argument("--dataset", metavar = "DATASET", type = str, default = "tuebingen_new", help = "Dataset to use (['tuebingen', 'tuebingen_new', 'sim', 'simc', 'simg', 'simln', 'an', 'ans', 'ls', 'lss', 'mnu']).")
    parser.add_argument("--outlier_trim", metavar = "OUTLIERS", type = float, default = 0.05, help = "How much to trim the tails for outliers after score estimation.")
    parser.add_argument("--test_size", metavar = "TEST_SIZE", type = nullable_float, help = "Size of the test set.")
    parser.add_argument("--mse_eval", action='store_true', default = False, help = "Numerically integrate to evaluate the MSE of the estimated flow.")
    parser.add_argument("--remote", action='store_true', default = False, help = "Use remote data.")
    parser.add_argument("--verbose", action='store_true', default = False, help = "Verbose mode.")
    parser.add_argument("--out_file", metavar = "OUTFILE", type = str, default = "results.csv", help = "Output file for results.")
    return parser.parse_args()

args = parse_arguments()
SEED = args.seed
LAYERS = args.layers
HIDDEN_SIZE = args.hidden_size
INIT_WEIGHT = args.init_weight
REG = args.reg
LR = args.lr
N_STEPS = args.n_steps
SCORE = args.score
KERNEL = args.kernel
MODEL = args.model
BW_FACTOR = args.bw_factor
BW_FACTOR_JOINT = args.bw_factor_joint
FIX_SAMPLES = args.fix_samples
OUTLIER_TRIM = args.outlier_trim
TEST_SIZE = args.test_size
MSE_EVAL = args.mse_eval
VERBOSE = args.verbose
OUTFILE = args.out_file

if args.remote:
    BASE_PATH = "/scratch/st-benbr-1/xijohnny/causal-velocity/data/"
else:
    BASE_PATH = "data/"

if args.dataset == "tuebingen" or args.dataset == "tuebingen_new":
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

if LAYERS == 1:
    ## linear. 
    HIDDEN_SIZE = 1


if __name__ == "__main__":

    if MODEL=="parametric-lin":
        model = parametric_model(basis_name = "linear")
    elif MODEL=="parametric-quad":
        model = parametric_model(basis_name = "quadratic")
    elif MODEL=="parametric-lin-exp":
        model = parametric_model(basis_name = "linear")
        model.add_exponential_terms = True
        model.nparams += 3
    elif MODEL=="parametric-quad-exp":
        model = parametric_model(basis_name = "quadratic")
        model.add_exponential_terms = True
        model.nparams += 3
    elif MODEL=="parametric-quad-fourier":
        model = parametric_model(basis_name = "quadratic")
        model.add_fourier_terms = True
        model.nparams += 4
    elif MODEL=="nn":
        model = nn_velocity_model(layers = LAYERS, hidden_size = HIDDEN_SIZE)
    elif MODEL=="anm":
        model = anm_model(layers = LAYERS, hidden_size = HIDDEN_SIZE)
    elif MODEL=="lsnm":
        model = lsnm_model(layers = LAYERS, hidden_size = HIDDEN_SIZE)

    success_weights = []
    total_weights = []
    scores = []
    causal_mse = []
    anticausal_mse = []
    times_score = []
    steps_causal = []
    steps_anticausal = []
    gof_causal = []
    gof_anticausal = []

    if args.dataset in ["an", "ans", "ls", "lss", "mnu"]:
        meta = onp.loadtxt(DATA_PATH + "/pairs_gt.txt")
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
            dat = onp.loadtxt(DATA_PATH + f"/pair_{pair_id}.txt", delimiter = ",", skiprows = 1, usecols = (1,2))
        else: 
            pair_id = meta.index[iter].astype(int)
            if args.dataset=="tuebingen" and pair_id in tue_blacklist:
                continue
            if args.dataset=="tuebingen_new" and pair_id in tue_blacklist_new:
                continue
            cause = meta["cause_start"].values[iter].astype(int)
            effect = meta["effect_start"].values[iter].astype(int)
            weight = meta["weight"].values[iter]
            dat = onp.loadtxt(DATA_PATH + f"/pair{pair_id:04d}.txt")

        n = dat.shape[0]
        idx_subsample = jax.random.choice(jax.random.PRNGKey(0), n, shape = (FIX_SAMPLES,), replace = True) 
        dat = dat[idx_subsample]

        dat, dat_mean, dat_std = standardize_data(dat, return_statistics = True, trim_outliers = 0)
        x_data, x_data_mean, x_data_std = dat[:,0], dat_mean[0], dat_std[0]
        y_data, y_data_mean, y_data_std = dat[:,1], dat_mean[1], dat_std[1] 

        t0_score = time.time()
        if SCORE == "stein":
            sx_data, sy_data, sxy_data = stein_score_all(x_data, y_data, reg = REG, score_kernel = KERNEL, bandwidth_factor = BW_FACTOR, bandwidth_factor_joint = BW_FACTOR_JOINT)
        elif SCORE == "kde":    
            sx_data, sy_data, sxy_data = kde_score_all(x_data, y_data, reg = REG, score_kernel = KERNEL, bandwidth_factor = BW_FACTOR, bandwidth_factor_joint = BW_FACTOR_JOINT)
        elif SCORE == "hybrid":
            sx_data, sy_data, sxy_data = hybrid_score_all(x_data, y_data, reg = REG, score_kernel = KERNEL, bandwidth_factor = BW_FACTOR, bandwidth_factor_joint = BW_FACTOR_JOINT)
        elif SCORE == "sm":
            sx_data, sy_data, sxy_data = score_matching_all(x_data, y_data, denoising = True)
        t1_score = time.time()

        times_score.append(t1_score - t0_score)
        if VERBOSE: print(f"Score time: {t1_score - t0_score}")

        dat, dat_mean, dat_std, idx = standardize_data(dat, return_statistics = True, trim_outliers = OUTLIER_TRIM, return_idx = True) ## this is just to trim outliers because data is already standardized.

        ## for experiments, we may want to fix the number of training samples to be the same for all pairs. 

        if FIX_SAMPLES > len(dat):
            samples = len(dat)
        else: 
            samples = FIX_SAMPLES

        idx = jnp.argwhere(idx).flatten()

        if TEST_SIZE is not None:
            idx_train = jax.random.choice(jax.random.PRNGKey(1), idx, shape = (int(samples*TEST_SIZE) - 1,), replace = False) 
            idx_test = jnp.setdiff1d(idx, idx_train)
            idx_test = jax.random.choice(jax.random.PRNGKey(2), idx_test, shape = (int(samples*TEST_SIZE),), replace = False)
        else:
            idx_train = idx
            idx_test = idx

        x_data_train = x_data[idx_train]
        y_data_train = y_data[idx_train]
        sx_data_train = sx_data[idx_train]
        sy_data_train = sy_data[idx_train]
        sxy_data_train = sxy_data[idx_train]
        x_data_test = x_data[idx_test]
        y_data_test = y_data[idx_test]
        sx_data_test = sx_data[idx_test]
        sy_data_test = sy_data[idx_test]
        sxy_data_test = sxy_data[idx_test]


        def V_MODEL(y, t, params):
            return model(y, t, params)
    
        PARAMS_INIT = model.params_init(seed = SEED + iter, init_weight = INIT_WEIGHT)

        gof_fwd, gof_bwd, complexity_fwd, complexity_bwd, params_fwd, params_bwd, steps_fwd, steps_bwd = fit_sm_bd(PARAMS_INIT, V_MODEL, x_data_train, y_data_train, 
                                        lr = LR / jnp.log(count_nparams(PARAMS_INIT)), 
                                        n_steps = N_STEPS,
                                        reg = REG, 
                                        sx_data = sx_data_train,
                                        sy_data = sy_data_train,
                                        sxy_data = sxy_data_train,
                                        x_data_test = x_data_test,
                                        y_data_test = y_data_test,
                                        sx_data_test = sx_data_test,
                                        sy_data_test = sy_data_test,
                                        sxy_data_test = sxy_data_test)

        score = (gof_bwd) - (gof_fwd)
        scores.append(onp.abs(score))
        if score > 0: 
            est_cause = 1
        else:
            est_cause = 2

        if est_cause == cause:
            success_weights.append(weight)
        else:
            success_weights.append(0)
        
        total_weights.append(weight)

        cause_steps = steps_fwd if cause == 1 else steps_bwd
        anticausal_steps = steps_bwd if cause == 1 else steps_fwd
        steps_causal.append(cause_steps)
        steps_anticausal.append(anticausal_steps)

        if cause == 1:
            gof_causal.append(gof_fwd)
            gof_anticausal.append(gof_bwd)
        else:
            gof_causal.append(gof_bwd)
            gof_anticausal.append(gof_fwd)

        if MSE_EVAL:

            class estimated_flow(mechanism_from_flow):

                def __init__(self, model, params):
                    super().__init__(stiff = False)
                    self.model = model
                    self.params = params

                def velocity(self, y, x):
                    return self.model(y, x, self.params)

            est_fwd = estimated_flow(V_MODEL, params_fwd)
            est_bwd = estimated_flow(V_MODEL, params_bwd)

            get_mse_flow_fwd = lambda x, y: ((est_fwd.flow(times = x_data_test, x0 = x, y0 = y) - y_data_test)**2).mean()
            get_mse_flow_bwd = lambda x, y: ((est_bwd.flow(times = y_data_test, x0 = x, y0 = y) - x_data_test)**2).mean()

            ## set starting time as median of the cause. 

            x_0 = jnp.argsort(x_data_test)[len(x_data_test)//2] 
            y_0 = jnp.argsort(y_data_test)[len(y_data_test)//2]

            try:
                fwd_mse = get_mse_flow_fwd(x_data_test[x_0].squeeze(), y_data_test[x_0].squeeze())
            except:
                print("Numerical error in mse calculation, likely the numerical integration failed.")
                fwd_mse = jnp.nan
            try:
                bwd_mse = get_mse_flow_bwd(y_data_test[y_0].squeeze(), x_data_test[y_0].squeeze())
            except:
                print("Numerical error in mse calculation, likely the numerical integration failed.")
                bwd_mse = jnp.nan

            cause_mse = fwd_mse if cause == 1 else bwd_mse
            anticause_mse = bwd_mse if cause == 1 else fwd_mse

            causal_mse.append(cause_mse)
            anticausal_mse.append(anticause_mse)
        
            if VERBOSE: print(f"Pair ID: {pair_id} \n Running Success Rate: {sum(success_weights)/sum(total_weights)}, \n Causal MSE: {cause_mse} \n Anticausal MSE: {anticause_mse} \n Causal Steps: {steps_fwd} \n Anticausal Steps: {steps_bwd}")

        else:
            if VERBOSE: print(f"Pair ID: {pair_id} \n Running Success Rate: {sum(success_weights)/sum(total_weights)} \n Causal Steps: {steps_fwd} \n Anticausal Steps: {steps_bwd}")

        print(sum(success_weights))
        print(sum(total_weights))


    ## AUDRC calculation

    idx = onp.flip(onp.argsort(scores))
    scores, success_weights, total_weights = onp.array(scores)[idx], onp.array(success_weights)[idx], onp.array(total_weights)[idx]

    AUDRC = 0

    for i in range(len(scores)):
        AUDRC += sum(success_weights[:i+1]) / sum(total_weights[:i+1])
    
    AUDRC /= len(scores)

    args_comma_separated = [args.score, args.kernel, args.reg, args.model, args.bw_factor, args.bw_factor_joint, args.dataset, args.outlier_trim, args.test_size]
    args_comma_separated = [str(arg) for arg in args_comma_separated]
    args_comma_separated = ",".join(args_comma_separated)

    if not os.path.exists(OUTFILE):
        with open(OUTFILE, "w") as f:
            f.write("score, kernel, reg, model, bw_factor, bw_factor_joint, dataset, outlier_trim, test_size, AUDRC, success_rate, causal_mse, anticausal_mse, score_time, score_cause_mse, effect_score_mse, joint_score_mse, gof_causal, gof_anticausal, sample_size\n")

    with open(OUTFILE, "a+") as f:
        f.write(f"{args_comma_separated}, {AUDRC}, {sum(success_weights)/sum(total_weights)},  {onp.nanmedian(causal_mse)}, {onp.nanmedian(anticausal_mse)}, {onp.median(times_score[1:])}, , , , {onp.median(gof_causal)}, {onp.median(gof_anticausal)}, {FIX_SAMPLES}\n")



