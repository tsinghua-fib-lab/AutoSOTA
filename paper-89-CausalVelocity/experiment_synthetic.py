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
from mechanisms import mechanism_from_flow

from score import stein_score_all, kde_score_all, hybrid_score_all, score_matching_all

def parse_arguments():
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument("--seed", metavar = "SEED", type = int, default = 0, help = "Seed for parameter initialization.")
    parser.add_argument("--score", metavar = "SCORE", type = str, default = "stein", help = "Score estimation method (['stein', 'kde', 'hybrid]).")
    parser.add_argument("--kernel", metavar = "KERNEL", type = str, default = "gauss", help = "Kernel for score estimation (['gauss', 'exp']).")
    parser.add_argument("--model", metavar = "MODEL", type = str, default = "parametric-quad", help = "Model to use for training (['nn', 'anm', lsnm', 'additive', 'parametric-lin', 'parametric-quad', 'parametric-cubic', 'parametric-quartic']).")
    parser.add_argument("--verbose", action="store_true", default = False, help = "Print additional information.")
    parser.add_argument("--layers", metavar = "LAYERS", type = int, default = 2, help = "Number of layers for neural network models.")
    parser.add_argument("--hidden_size", metavar = "HIDDEN_SIZE", type = int, default = 32, help = "Hidden size for neural network models.")
    parser.add_argument("--init_weight", metavar = "INIT_WEIGHT", type = float, default = 0.5, help = "Initialization variance.")
    parser.add_argument("--reg", metavar = "REG", type = float, default = 0.1, help = "Regularization term for score estimator. Suggest 0.1 for stein and 0.01 for kde.")
    parser.add_argument("--bw_factor", metavar = "BW_FACTOR", type = float, default = 1, help = "Multiply the standard heuristic bandwidth by a fixed factor for marginal score estimator.")
    parser.add_argument("--bw_factor_joint", metavar = "BW_FACTOR", type = float, default = 1, help = "Multiply the standard heuristic bandwidth by a fixed factor for joint score estimator.")
    parser.add_argument("--lr", metavar = "LR", type = float, default = 0.1, help = "Base learning rate for optimizer (actual is scaled by number of parameters in the model).")
    parser.add_argument("--n_steps", metavar = "N_STEPS", type = int, default = 200, help = "Number of steps for velocity optimization.")
    parser.add_argument("--fix_samples", metavar = "MAX_SAMPLES", type = int, default = 1000, help = "Fix a dataset size for benchmarking: if none, use all samples, otherwise, resample if needed to get to this size.")
    parser.add_argument("--test_size", metavar = "TEST_SIZE", type = nullable_float, help = "Size of the test set as a fraction of the training set. If None, no test set is used.")
    parser.add_argument("--dataset", metavar = "DATASET", type = str, default = "anm", help = "Dataset to use.")
    parser.add_argument("--outlier_trim", metavar = "OUTLIER_TRIM", type = float, default = 0, help = "How much to trim the tails for outliers after score estimation.")
    parser.add_argument("--remote", action='store_true', default = False, help = "Use remote data path.")
    parser.add_argument("--outfile", metavar = "OUTFILE", type = str, default = "results.csv", help = "Output file to save results.")
    parser.add_argument("--mse_eval", action='store_true', default = False, help = "Evaluate the MSE of the estimated flow.")
    parser.add_argument("--no_score", action='store_true', default = False, help = "Flag for when there is no score available for validation.")
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
TEST_SIZE = args.test_size
VERBOSE = args.verbose
OUTLIER_TRIM = args.outlier_trim
OUTFILE = args.outfile
MSE_EVAL = args.mse_eval

if args.remote:
    BASE_PATH = "/scratch/st-benbr-1/xijohnny/causal-velocity/data/synthetic/"
else:
    BASE_PATH = "data/synthetic/"

DATA_PATH = BASE_PATH + args.dataset

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
    sx_mse = [] 
    sy_mse = []
    sxy_mse = []
    gof_causal = []
    gof_anticausal = []
    times_score = []
    times_velocity = []
    steps_causal = []
    steps_anticausal = []
    causal_mse = []
    anticausal_mse = []

    meta = pd.read_csv(DATA_PATH + "/pairmeta.txt",  delim_whitespace=True,
                        header=None,
                        names=['pair_id', 'cause_start', 'cause_end', 'effect_start', 'effect_end', 'weight'],
                        index_col=0)

    for iter in range(len(meta)):

        pair_id = meta.index[iter].astype(int)
        cause = meta["cause_start"].values[iter].astype(int)
        effect = meta["effect_start"].values[iter].astype(int)
        weight = meta["weight"].values[iter]
        dat = jnp.array(onp.loadtxt(DATA_PATH + f"/pair{pair_id:04d}.txt"))


        n = dat.shape[0]

        idx = jax.random.choice(jax.random.PRNGKey(0), n, shape = ((FIX_SAMPLES),), replace = False) 
        dat = dat[idx]

        if not args.no_score:
            s_cause_true = jnp.array(onp.loadtxt(DATA_PATH + f"/s_cause{pair_id:04d}.txt"))
            s_effect_true = jnp.array(onp.loadtxt(DATA_PATH + f"/s_effect{pair_id:04d}.txt"))
            s_joint_true = jnp.array(onp.loadtxt(DATA_PATH + f"/s_joint{pair_id:04d}.txt"))
            s_cause_true = s_cause_true[idx]
            s_effect_true = s_effect_true[idx]
            s_joint_true = s_joint_true[idx]

        n = dat.shape[0]

        ## note synthetic data is already standardized. 

        t0_score = time.time()
        if SCORE == "stein":
            sx_data, sy_data, sxy_data = stein_score_all(dat[:, 0], dat[:, 1], reg = REG, score_kernel = KERNEL, bandwidth_factor = BW_FACTOR, bandwidth_factor_joint = BW_FACTOR_JOINT)
        elif SCORE == "kde":
            sx_data, sy_data, sxy_data = kde_score_all(dat[:, 0], dat[:, 1], reg = REG, score_kernel = KERNEL, bandwidth_factor = BW_FACTOR, bandwidth_factor_joint = BW_FACTOR_JOINT)
        elif SCORE == "hybrid":
            sx_data, sy_data, sxy_data = hybrid_score_all(dat[:, 0], dat[:, 1], reg = REG, score_kernel = KERNEL, bandwidth_factor = BW_FACTOR, bandwidth_factor_joint = BW_FACTOR_JOINT)
        elif SCORE == "gt":
            if not args.no_score:
                if cause == 1:
                    sx_data = s_cause_true
                    sy_data = s_effect_true
                else:
                    sy_data = s_cause_true
                    sx_data = s_effect_true
                sxy_data = s_joint_true
            else:
                raise ValueError("Ground truth score not available. Please use a score estimation method")
                
        t1_score = time.time()
        times_score.append(t1_score - t0_score)
        if VERBOSE: print(f"Score time: {t1_score - t0_score}")
        
        dat, dat_mean, dat_std, trim_idx = standardize_data(dat, return_statistics = True, trim_outliers = OUTLIER_TRIM, return_idx = True)
        sx_data, sy_data, sxy_data = sx_data[trim_idx].squeeze(), sy_data[trim_idx].squeeze(), sxy_data[trim_idx].squeeze()
        
        if FIX_SAMPLES > len(dat):
            samples = len(dat)
        else: 
            samples = FIX_SAMPLES

        idx = jnp.argwhere(idx).flatten()

        if TEST_SIZE is not None:
            idx_train = jax.random.choice(jax.random.PRNGKey(1), idx, shape = (int(FIX_SAMPLES*TEST_SIZE) - 1,), replace = False) 
            idx_test = jnp.setdiff1d(idx, idx_train)
            idx_test = jax.random.choice(jax.random.PRNGKey(2), idx_test, shape = (int(FIX_SAMPLES*TEST_SIZE),), replace = False)
        else:
            idx_train = idx
            idx_test = idx

        dat_train = dat[idx_train]
        sx_data_train = sx_data[idx_train]
        sy_data_train = sy_data[idx_train]
        sxy_data_train = sxy_data[idx_train]
        dat_test = dat[idx_test]
        sx_data_test = sx_data[idx_test]
        sy_data_test = sy_data[idx_test]
        sxy_data_test = sxy_data[idx_test]

        if not args.no_score:
            s_cause_true, s_effect_true, s_joint_true = s_cause_true[trim_idx], s_effect_true[trim_idx], s_joint_true[trim_idx]
            s_cause_true_train = s_cause_true[idx_train]
            s_effect_true_train = s_effect_true[idx_train]
            s_joint_true_train = s_joint_true[idx_train]
            s_cause_true_test = s_cause_true[idx_test]
            s_effect_true_test = s_effect_true[idx_test]
            s_joint_true_test = s_joint_true[idx_test]

        x_data_train, y_data_train = dat_train[:,0], dat_train[:,1]
        x_data_test, y_data_test = dat_test[:,0], dat_test[:,1] if FIX_SAMPLES is not None else (None, None)

        def V_MODEL(y, t, params):
            return model(y, t, params)
    
        PARAMS_INIT = model.params_init(seed = SEED + iter, init_weight = INIT_WEIGHT)

        t0_velocity = time.time()

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
        
        t1_velocity = time.time()
        times_velocity.append(t1_velocity - t0_velocity)
        if VERBOSE: print(f"Velocity time: {t1_velocity - t0_velocity}")

        steps_causal.append(steps_fwd) if cause == 1 else steps_causal.append(steps_bwd)
        steps_anticausal.append(steps_bwd) if cause == 1 else steps_anticausal.append(steps_fwd)

        score = (gof_bwd) - (gof_fwd)
        if score > 0: 
            est_cause = 1
        else:
            est_cause = 2

        if est_cause == cause:
            success_weights.append(weight)
        else:
            success_weights.append(0)
        
        ## store the score of the causal direction (negative indicates an incorrect conclusion).

        if cause == 1:
            scores.append(score)
        else:
            scores.append(-score)

        total_weights.append(weight)
        
        s_cause_train= onp.array(sx_data_train) if cause == 1 else onp.array(sy_data_train)
        s_effect_train = onp.array(sy_data_train) if cause == 1 else onp.array(sx_data_train)

        if not args.no_score:
            s_cause_mse = onp.mean((s_cause_train - s_cause_true_train)**2)
            s_effect_mse = onp.mean((s_effect_train - s_effect_true_train)**2)
            s_joint_mse = onp.mean((sxy_data_train - s_joint_true_train)**2)

        else:
            s_cause_mse, s_effect_mse, s_joint_mse = jnp.nan, jnp.nan, jnp.nan

        sx_mse.append(s_cause_mse)
        sy_mse.append(s_effect_mse)
        sxy_mse.append(s_joint_mse)

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

            x_0 = jnp.argsort(x_data_test)[len(x_data_test)//2]
            y_0 = jnp.argsort(y_data_test)[len(y_data_test)//2]

            fwd_mse = get_mse_flow_fwd(x_data_test[x_0].squeeze(), y_data_test[x_0].squeeze())
            bwd_mse = get_mse_flow_bwd(y_data_test[y_0].squeeze(), x_data_test[y_0].squeeze())

            cause_mse = fwd_mse if cause == 1 else bwd_mse
            anticause_mse = bwd_mse if cause == 1 else fwd_mse

            causal_mse.append(cause_mse)
            anticausal_mse.append(anticause_mse)

        if VERBOSE: print(f"Pair ID: {pair_id} \n Success Rate: {sum(success_weights)/sum(total_weights)} \n Cause Score MSE: {s_cause_mse} \n Effect Score MSE: {s_effect_mse} \n Joint Score MSE: {s_joint_mse} \n GoF Causal: {gof_causal[-1]} \n GoF Anticausal: {gof_anticausal[-1]}, \n Score: {scores[-1]} \n Causal MSE: {causal_mse} \n Anticausal MSE: {anticausal_mse} \n Velocity time: {times_velocity[-1]} \n Score time: {times_score[-1]} \n Steps Causal: {steps_causal[-1]} \n Steps Anticausal: {steps_anticausal[-1]}")

    ## AUDRC calculation

    idx = onp.flip(onp.argsort(onp.abs(scores)))  
    scores_abs, success_weights, total_weights = onp.array(onp.abs(scores))[idx], onp.array(success_weights)[idx], onp.array(total_weights)[idx]

    AUDRC = 0

    for i in range(len(scores_abs)):
        AUDRC += sum(success_weights[:i+1]) / sum(total_weights[:i+1])
    
    AUDRC /= len(scores_abs)

    args_comma_separated = [args.score, args.kernel, args.reg, args.model, args.bw_factor, args.bw_factor_joint, args.dataset, args.outlier_trim]
    args_comma_separated = [str(arg) for arg in args_comma_separated]
    args_comma_separated = ",".join(args_comma_separated)

    if not os.path.exists(OUTFILE):
        with open(OUTFILE, "w") as f:
            f.write("score, kernel, reg, model, bw_factor, bw_factor_joint, dataset, outlier_trim, test_size, AUDRC, success_rate, causal_mse, anticausal_mse, score_time, score_cause_mse, effect_score_mse, joint_score_mse, gof_causal, gof_anticausal, sample_size, score_cause_mse_sd, score_effect_mse_sd, score_joint_mse_sd, gof_anticausal_sd, gof_causal_sd\n")

    with open(OUTFILE, "a+") as f:
        f.write(f"{args_comma_separated}, , {AUDRC}, {sum(success_weights)/sum(total_weights)},  {onp.nanmedian(causal_mse)}, {onp.nanmedian(anticausal_mse)}, {onp.median(times_score[1:])}, {onp.median(sx_mse)}, {onp.median(sy_mse)}, {onp.median(sxy_mse)}, {onp.median(gof_causal)}, {onp.median(gof_anticausal)}, {FIX_SAMPLES}, {onp.nanquantile(sx_mse, 0.25)}, {onp.nanquantile(sx_mse, 0.75)}, {onp.nanquantile(sy_mse, 0.25)}, {onp.nanquantile(sy_mse, 0.75)},  {onp.nanquantile(sxy_mse, 0.25)}, {onp.nanquantile(sxy_mse, 0.75)}, {onp.nanquantile(gof_causal, 0.25)}, {onp.nanquantile(gof_causal, 0.75)}, {onp.nanquantile(gof_anticausal, 0.25)}, {onp.nanquantile(gof_anticausal, 0.75)},   \n")
