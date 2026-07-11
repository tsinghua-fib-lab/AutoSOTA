import numpy as np
import pandas as pd
import pickle
import argparse
import multiprocessing

from datetime import datetime

from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, mean_squared_error, mean_absolute_error, r2_score, hamming_loss

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import repo_paths  # noqa: F401  -- adds utils/, packages/, lime-master/, experiments/ to sys.path

from Data_Generation import generate_data


def int_or_none(x):
    return None if x.lower() == "none" else int(x)

def float_or_none(x):
    return None if x.lower() == "none" else float(x)

def str_or_none(x):
    return None if x.lower() == "none" else str(x)

def str_to_bool(x):
    if isinstance(x, bool):
        return x
    value = x.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected")

def int_or_use_gtruth(x):
    if isinstance(x, int):
        return x
    value = str(x).strip()
    if value.lower() == "use_gtruth":
        return "use_gtruth"
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Value must be an integer or 'use_gtruth'") from exc

def _detect_outer_parallel_context():
    """Best-effort detection of whether this call is already inside a worker process."""
    proc_name = multiprocessing.current_process().name
    if proc_name != "MainProcess":
        return True
    for env_key in ("JOBLIB_PARENT_PID", "LOKY_PARENT_PID", "JOBLIB_NESTED_PARALLELISM"):
        if os.environ.get(env_key):
            return True
    return False

def _resolve_ensemble_parallel_policy(
    n_ensemble,
    ensemble_parallel,
    ensemble_n_jobs,
    ensemble_backend,
):
    """
    Aggressive default policy:
    - auto-enable inner parallelism when n_ensemble > 1
    - clamp to serial if nested outer parallel context is detected
    """
    if n_ensemble is None or n_ensemble <= 1:
        return False, 1, 'sequential'

    if ensemble_parallel is False:
        return False, 1, 'sequential'

    if ensemble_backend not in {'loky', 'threading', 'sequential'}:
        raise ValueError("ensemble_backend must be one of {'loky', 'threading', 'sequential'}")

    if ensemble_n_jobs is None:
        cpu_count = os.cpu_count() or 1
        desired_n_jobs = min(int(n_ensemble), int(cpu_count))
    else:
        if ensemble_n_jobs == -1:
            desired_n_jobs = os.cpu_count() or 1
        elif ensemble_n_jobs > 0:
            desired_n_jobs = ensemble_n_jobs
        else:
            raise ValueError("ensemble_n_jobs must be None, -1, or a positive integer")

    desired_n_jobs = max(1, int(desired_n_jobs))
    desired_n_jobs = min(desired_n_jobs, int(n_ensemble))

    if ensemble_backend == 'sequential':
        desired_n_jobs = 1

    if _detect_outer_parallel_context() and desired_n_jobs > 1:
        print("[WARN] Outer parallel context detected; clamping inner ensemble parallelism to n_jobs=1")
        desired_n_jobs = 1

    use_parallel = desired_n_jobs > 1
    effective_backend = ensemble_backend if use_parallel else 'sequential'
    return use_parallel, desired_n_jobs, effective_backend


def _run_hide_and_seek_ensemble_member(
    ensemble_idx,
    member_col_indices,
    x_train,
    x_test,
    y_train,
    lmbda,
    epochs,
    train_seed,
    test_seed,
    task,
    hide_hidden_dim,
    seek_hidden_dim,
    hide_num_hidden_layers,
    seek_num_hidden_layers,
    batch_size,
    data_type,
    batchnorm_hs,
    num_classes,
    lmbda_exponent,
    return_losses_on_val,
    class_weight_alpha,
    perturbation_method,
    full_num_features,
):
    from hide_and_seek.model import train_nn, pred_nn

    iter_train_seed = train_seed + ensemble_idx
    iter_test_seed = test_seed + ensemble_idx

    if member_col_indices is None:
        x_train_member = x_train
        x_test_member = x_test
    else:
        x_train_member = x_train[:, member_col_indices]
        x_test_member = x_test[:, member_col_indices]

    output = train_nn(
        X_train=x_train_member,
        y_train=y_train,
        lmbda=lmbda,
        n_epochs=epochs,
        seed=iter_train_seed,
        task=task,
        hide_hidden_dim=hide_hidden_dim,
        seek_hidden_dim=seek_hidden_dim,
        hide_num_hidden_layers=hide_num_hidden_layers,
        seek_num_hidden_layers=seek_num_hidden_layers,
        batch_size=batch_size,
        print_description=f"{data_type}_ens={ensemble_idx}",
        batchnorm=batchnorm_hs,
        num_classes=num_classes,
        lmbda_exponent=lmbda_exponent,
        return_losses_on_val=return_losses_on_val,
        class_weight_alpha=class_weight_alpha,
        perturbation_method=perturbation_method,
    )

    model = output['model']
    iter_y_test_pred, iter_mask = pred_nn(
        model=model,
        X_test=x_test_member,
        X_train=x_train_member,
        return_masks=True,
        seed=iter_test_seed,
        task=task,
        perturbation_method=perturbation_method,
    )

    if member_col_indices is not None:
        full_mask = np.full((iter_mask.shape[0], full_num_features), np.nan, dtype=float)
        full_mask[:, member_col_indices] = iter_mask
        iter_mask = full_mask

    iter_binary_mask = 1. * (iter_mask > 0.5)

    return {
        'ensemble_idx': ensemble_idx,
        'iter_mask': iter_mask,
        'iter_binary_mask': iter_binary_mask,
        'iter_y_test_pred': iter_y_test_pred,
        'iter_losses_on_val': output.get('losses_on_val') if return_losses_on_val else None,
    }

def parse_args(lmbda_default=0.3, epochs_default=None, batch_size_default=None,
               seed_default=0, syn_idx_default=3, num_syn_features_default=11,
               train_N_default=10_000, test_N_default=10_000,
               hide_hidden_dim_default=32, seek_hidden_dim_default=32,
               hide_num_hidden_layers_default=2, seek_num_hidden_layers_default=2,
               lmbda_exponent_default=2,
               rho_default=0.0,
               data_mode_default='synthetic',
               folder_for_pickle_default='ICML_experiments/unallocated',
               model_type_default='hide_and_seek',
               single_data_set_default=False,
               num_important_features_default='use_gtruth'):
    parser = argparse.ArgumentParser()

    # model hyperparams
    parser.add_argument("--lmbda", type=float, default=lmbda_default)
    parser.add_argument("--epochs", type=int_or_none, default=epochs_default) #500 for hide_and_seek, 10_000 for invase
    parser.add_argument("--batch-size", type=int_or_none, default=batch_size_default) #None for hide_and_seek. 1_000 for invase
    parser.add_argument("--model-type", type=str, default=model_type_default, choices=["hide_and_seek", "invase", "realx", "l2x", "lime", "shap_xgboost"])
    parser.add_argument("--perturbation-method", type=str, default="draw_marginal", choices=["draw_marginal", "knock_off", "conditional_rf"])
    parser.add_argument("--n-ensemble", type=int_or_none, default=None)
    parser.add_argument("--colsample", type=float_or_none, default=None)
    parser.add_argument("--ensemble-parallel", type=str_to_bool, default=None)
    parser.add_argument("--ensemble-n-jobs", type=int_or_none, default=None)
    parser.add_argument("--ensemble-backend", type=str, default="loky", choices=["loky", "threading", "sequential"])
    parser.add_argument("--folder-for-pickle", type=str_or_none, default=folder_for_pickle_default)

    # syn params
    parser.add_argument("--seed", type=int, default=seed_default)
    parser.add_argument("--syn-idx", type=int_or_none, default=syn_idx_default)
    parser.add_argument("--single-data-set", type=str_to_bool, default=single_data_set_default)
    parser.add_argument("--num-syn-features", type=int, default=num_syn_features_default) #11 for all syn experiments except 100 for large dataset
    parser.add_argument("--num-important-features", type=int_or_use_gtruth, default=num_important_features_default)
    parser.add_argument("--train-N", type=int, default=train_N_default)
    parser.add_argument("--test-N", type=int, default=test_N_default)
    parser.add_argument("--rho", type=float_or_none, default=rho_default)
    
    parser.add_argument("--data_mode", type=str, default=data_mode_default)

    # model-structure hyperparams
    parser.add_argument("--hide-hidden-dim", type=int, default=hide_hidden_dim_default)
    parser.add_argument("--seek-hidden-dim", type=int, default=seek_hidden_dim_default)
    parser.add_argument("--hide-num-hidden-layers", type=int, default=hide_num_hidden_layers_default)
    parser.add_argument("--seek-num-hidden-layers", type=int, default=seek_num_hidden_layers_default)
    parser.add_argument("--lmbda-exponent", type=float, default=lmbda_exponent_default)
    
    parser.add_argument("--task", type=str, default="multiclass",
                        choices=["regression", "classification", "multiclass", "multilabel"])
    parser.add_argument("--batchnorm-hs", action="store_true", help="Use batchnorm. Default is False")
    parser.add_argument("--return_losses_on_val", action="store_true", help="Default is False")

    import json
    parser.add_argument("--xgb-params", type=json.loads, default=None, help="JSON string for XGBoost params")

    parser.add_argument("--syn-switch-quantile", type=float, default=None,
                        help="Quantile in (0,1) for X[:,10] split, only used by Syn4Q/Syn5Q/Syn6Q")

    parser.add_argument("--data-set", type=str, default=None,
                        help="Name of the single data set to run, e.g. Syn1 or Syn4Q.")

    parser.add_argument("--metrics-out", type=str, default=None,
                        help="Optional path to write per-run metrics JSON "
                             "({data_set, TPR_mean, FDR_mean, f1}). Used by run_multiple_tests.py.")

    args = parser.parse_args() #use parser.parse_args(args=[]) when copying to jupyter
    return args

def create_data(data_type,
                data_out,
                train_N,
                test_N,
                train_seed,
                test_seed,
                num_features,
                data_mode,
                rho,
                syn_switch_quantile=None):
    """
    Generate training and testing data for INVASE.
    
    Args:
        data_type (str): Name of dataset to generate
        data_out (str): 'Y' for binary output, 'Prob' for probability output
        train_N (int): Number of training samples
        test_N (int): Number of testing samples
        train_seed (int): Random seed for training set
        test_seed (int): Random seed for testing set
        data_mode (str): should be 'synthetic', 'credit_data_val' or 'credit_data_test'
        rho (float): Pairwise correlation coefficient for synthetic data generation (only used if data_mode is 'synthetic')
    Returns:
        (x_train, y_train, g_train, x_test, y_test, g_test)
    """

    if data_mode == 'synthetic':
        data_detail_train = data_mode
        data_detail_test = data_mode
    elif 'credit_data' in data_mode:
        data_detail_train = 'credit_data_train'
        if 'val' in data_mode:
            data_detail_test = 'credit_data_val'
        elif 'test' in data_mode:
            data_detail_test = 'credit_data_test'


    x_train, y_train, g_train = generate_data(
        n=train_N, data_type=data_type, seed=train_seed, out=data_out,
        num_features=num_features, data_mode_detail=data_detail_train,
        rho=rho, syn_switch_quantile=syn_switch_quantile
    )

    x_test, y_test, g_test = generate_data(
        n=test_N, data_type=data_type, seed=test_seed, out=data_out,
        num_features=num_features, data_mode_detail=data_detail_test,
        rho=rho, syn_switch_quantile=syn_switch_quantile
    )

    return x_train, y_train, g_train, x_test, y_test, g_test

def compute_f1(binary_mask, g_truth):
    
    g = g_truth        # shape (n, p)
    pred = binary_mask
    
    # True Positives, False Positives, False Negatives per row
    TP = np.sum((pred == 1) & (g == 1), axis=1)
    FP = np.sum((pred == 1) & (g == 0), axis=1)
    FN = np.sum((pred == 0) & (g == 1), axis=1)
    
    # Compute metrics per row
    # TPR = TP / (TP + FN + 1e-10)
    # FDR = FP / (TP + FP + 1e-10)
    F1  = 2 * TP / (2 * TP + FP + FN + 1e-10)
    
    # return pd.Series({
    #             'TPR': np.mean(TPR),
    #             'FDR': np.mean(FDR),
    #             'F1' : np.mean(F1)
    #         })

    # Take mean across all rows in this experiment
    return np.mean(F1)*100


def prediction_metrics(y_true, y_pred_probs, model_type,
                      verbose=False):
    """
    y_true: 1D array of labels (N,) or one-hot (N, C)
    y_pred_probs: 2D array of probabilities (N, C)
    """
    # If one-hot encoded, convert to class indices
    y_true = y_true.argmax(axis=1) if y_true.ndim == 2 else y_true
    # Predicted labels
    y_pred_labels = y_pred_probs.argmax(axis=1)
    acc = accuracy_score(y_true, y_pred_labels)

    # Check if we are in a binary or multiclass scenario
    num_classes = y_pred_probs.shape[1]
    if num_classes == 2:
        # For Binary: use the probability of the positive class (column 1)
        p_pos = y_pred_probs[:, 1]
        roc_auc = roc_auc_score(y_true, p_pos)
        pr_auc  = average_precision_score(y_true, p_pos)
    else:
        roc_auc = roc_auc_score(y_true, y_pred_probs, multi_class='ovr') #alternative: 'ovo'
        
        if model_type not in ['l2x']:
            pr_auc = average_precision_score(y_true, y_pred_probs)
        else:
            pr_auc = None

    if verbose == True:
        print("Accuracy:", acc)
        print("ROC-AUC:", roc_auc)
        print("PR-AUC:", pr_auc)

    return acc, roc_auc, pr_auc       

#%% Performance Metrics
def performance_metric(binary_mask, g_truth):

    n = len(binary_mask)
    Temp_TPR = np.zeros([n,])
    Temp_FDR = np.zeros([n,])
    
    for i in range(n):

        # TPR    
        TPR_Nom = np.sum(binary_mask[i,:] * g_truth[i,:])
        TPR_Den = np.sum(g_truth[i,:])
        Temp_TPR[i] = 100 * float(TPR_Nom)/float(TPR_Den+1e-8)
    
        # FDR
        FDR_Nom = np.sum(binary_mask[i,:] * (1-g_truth[i,:]))
        FDR_Den = np.sum(binary_mask[i,:])
        Temp_FDR[i] = 100 * float(FDR_Nom)/float(FDR_Den+1e-8)

    return np.mean(Temp_TPR), np.mean(Temp_FDR), np.std(Temp_TPR), np.std(Temp_FDR)


def shuffle_numpy_cols(X, replace=False, random_state=None):
    """
    Shuffle each column of a NumPy array independently.

    Parameters:
        X (np.ndarray): Input 2D array.
        replace (bool): Whether to sample with replacement. Defaults to False.
        random_state (int or None): Seed for reproducibility.

    Returns:
        np.ndarray: Array with columns independently shuffled.
    """

    if hasattr(np.random, "default_rng"):  # NumPy >= 1.17
        rng = np.random.default_rng(random_state)
    else:  # Older NumPy
        rng = np.random.RandomState(random_state)

    X = np.asarray(X)  # Ensure input is a NumPy array
    X_shuffled = np.empty_like(X)

    for i in range(X.shape[1]):
        X_shuffled[:, i] = rng.choice(X[:, i], size=X.shape[0], replace=replace)

    return X_shuffled

def save_results_as_pickle(results,
                          syn_type,
                           model_type,
                           folder,
                          name_end,
                          timestamp='notimestamp'):
    name = f'results_{timestamp}_{syn_type}_{model_type}_{name_end}'
    save_path = f'{os.path.expanduser("~/Data")}/{folder}/{name}.pkl'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    print(save_path)

def find_n_largest_values(arr, n):
    """
    Finds the n largest values in each row of a 2D NumPy array
    and returns a binary array indicating their positions.
    """
    
    arr = np.abs(arr)
    top_idx = np.argsort(arr, axis=1)[:, -n:] #note: no random tiebreaker but shap values are continuous. Perhaps edit in future.
    
    binary_mask = np.zeros_like(arr, dtype=int)
    rows = np.arange(len(binary_mask))[:, None]
    binary_mask[rows, top_idx] = 1
    return binary_mask

def _build_run_type(model_type_text, epochs, lmbda, batch_size, seed,
                    num_important_features, num_syn_features, batchnorm_hs,
                    lmbda_exponent, return_losses_on_val, data_mode,
                    n_ensemble, colsample, rho, task, perturbation_method,
                    model_type, syn_switch_quantile=None):
    _TASK_SUFFIX = {'regression': 't=r', 'classification': 't=mc',
                    'multiclass': 't=mc', 'multilabel': 't=ml'}
    _PERTURBATION_SUFFIX = {
        'draw_marginal': 'mrgl',
        'knock_off': 'ko',
        'conditional_rf': 'crf',
    }
    run_str = (
        f"m={model_type_text}_e={epochs}_l={lmbda}_"
        f"b={batch_size}_seed={seed}_k={num_important_features}_"
        f"f={num_syn_features}_bn={batchnorm_hs}_"
        f"p={lmbda_exponent}_"
        f"vl={return_losses_on_val}_"
        f"dm={data_mode}_"
    )
    if model_type == 'hide_and_seek':
        pm_short = _PERTURBATION_SUFFIX.get(perturbation_method, perturbation_method)
        run_str += (
            f"en={n_ensemble}_"
            f"cs={colsample}_"
            f"pm={pm_short}_"
        )
    run_str += (
        f"rho={rho}_"
        f"sq={syn_switch_quantile}_"
        f"{_TASK_SUFFIX.get(task, f't={task}')}"
    )
    return run_str

def run_feature_selection_model(data_type='experiment', #if synthetic, should be e.g. "Syn#" for # in {1,...,6}. Otherwise, can be any name.
                                num_important_features=None, #not needed for hide_and_seek, invase, realx. For other models specify an integer number of features or 'use_gtruth'
                                full_data_dict=None, #option to provide x_train, y_train, x_test, y_test, g_test for non-synthetic data
                                model_type='hide_and_seek',
                                batch_size=None,
                                epochs=500,
                                lmbda=0.3,
                                task='classification',
                                hide_hidden_dim=32, #only used if model_type == 'hide_and_seek'
                                seek_hidden_dim=32, #only used if model_type == 'hide_and_seek',
                                hide_num_hidden_layers=2, #only used if model_type == 'hide_and_seek'
                                seek_num_hidden_layers=2, #only used if model_type == 'hide_and_seek'
                                folder_for_pickle=None, #location to save pickled results. if None won't pickle. If not none, will save in "~/Data/{folder}". Edit 'save_results_as_pickle' function to change path
                                return_results=True,
                                include_model=False,
                                xgb_params = None,
                                use_custom_nn_for_lime=False,
                                train_N = 10_000,
                                test_N = 10_000,
                                seed = 0,
                                include_y_test = True,
                                num_syn_features = 11,
                                batchnorm_hs = False,
                                save_experiment_data = True,
                                lmbda_exponent = 2,
                                return_losses_on_val=False,
                                data_mode='synthetic', #used for synthetic data. can be 'synthetic' or 'credit_data_val' or 'credit_data_test'. Future improvement: have this work within 'data_type' for simplicity
                                scale_data=True,
                                class_weight_alpha=None, #alpha parameter for balancing class weights. in [0,1]. 0 is unweighted, 1 is full weighting, based on class prevalence in train. Set to None (default) to disable — this parameter was not investigated or validated in experiments.
                                column_names=None, #supply if want to remember them later
                                perturbation_method='draw_marginal',
                                n_ensemble=None,
                                colsample=None,
                                ensemble_parallel=None,
                                ensemble_n_jobs=None,
                                ensemble_backend='loky',
                                rho=0.0,
                                save_train_masks=False, #if True, saves x_train, x_test, mask_train in results for use by a downstream stage
                                syn_switch_quantile=None, #only used by Syn4Q/Syn5Q/Syn6Q
                                warmup_epochs=0
                                ):

    Q_TYPES = {"Syn4Q", "Syn5Q", "Syn6Q"}
    if data_type in Q_TYPES:
        assert syn_switch_quantile is not None, \
            f"{data_type} requires --syn-switch-quantile"
    effective_quantile = syn_switch_quantile if data_type in Q_TYPES else None

    timestamp_start = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    if model_type == 'hide_and_seek' and (n_ensemble is not None and n_ensemble > 1):
        model_type_text = f"{model_type}_ens"
    else:
        model_type_text = model_type

    run_type = _build_run_type(model_type_text, epochs, lmbda, batch_size, seed,
                    num_important_features, num_syn_features, batchnorm_hs,
                    lmbda_exponent, return_losses_on_val, data_mode,
                    n_ensemble, colsample, rho, task, perturbation_method,
                    model_type, syn_switch_quantile=effective_quantile)
    print(run_type)

    if n_ensemble is not None and n_ensemble <= 0:
        raise ValueError("n_ensemble must be None or a positive integer")

    if colsample is not None:
        if n_ensemble is None or n_ensemble <= 1:
            raise ValueError("colsample is only supported when n_ensemble > 1")
        if not (0 < colsample <= 1):
            raise ValueError("colsample must be in (0, 1]")

    train_seed = seed
    test_seed = seed + 1

    if full_data_dict is not None:
        x_train = full_data_dict['x_train']
        y_train = full_data_dict['y_train']
        x_test = full_data_dict['x_test']
        y_test = full_data_dict['y_test']
        g_test = full_data_dict['g_test'] #this should be None if ground truth feature importance is unknwn
    else:
        # Data output can be either binary (Y) or Probability (Prob)
        data_out_sets = ['Y','Prob']
        data_out = data_out_sets[0]

        if (model_type == 'l2x') and (train_N != 1_000_000): #needs more data to work
            print('WARNING: SHOULD USE 1_000_000 TRAINING SAMPLES FOR l2x FOR REASONABLE RESULTS')
    
        #%% Data Generation (Train/Test)
        
        x_train, y_train, _, x_test, y_test, g_test = create_data(
                                                data_type=data_type,
                                                data_out=data_out,
                                                train_N=train_N,
                                                test_N=test_N,
                                                train_seed=train_seed,
                                                test_seed=test_seed,
                                                num_features=num_syn_features,
                                                data_mode=data_mode,
                                                rho=rho,
                                                syn_switch_quantile=effective_quantile
                                            )
    
    if y_train.ndim == 2:
        num_classes = y_train.shape[1]
    elif y_train.ndim == 1:
        num_classes = int(y_train.max()) + 1
    else:
        raise ValueError("y_train has invalid number of dimensions")

    if scale_data == True:
        print("scaling data")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
    else:
        print("not scaling data")

    #baseline model - no feature masking
    if ((model_type == 'lime') and (use_custom_nn_for_lime == False)) or (model_type == 'shap'):
        y_train = y_train.argmax(axis=1) if y_train.ndim == 2 else y_train

        from hide_and_seek.model import train_nn
        baseline_model = train_nn(X_train=x_train,
                                y_train= y_train,
                                lmbda=None,
                                n_epochs=2*epochs,
                                seed=train_seed,
                                task=task,
                                hide_hidden_dim=hide_hidden_dim,
                                seek_hidden_dim=seek_hidden_dim,
                                hide_num_hidden_layers=hide_num_hidden_layers,
                                seek_num_hidden_layers=seek_num_hidden_layers,
                                batch_size=None,
                                train_baseline=True,
                                print_description=data_type,
                                batchnorm=batchnorm_hs,
                                num_classes=num_classes,
                                lmbda_exponent=lmbda_exponent,
                                class_weight_alpha=class_weight_alpha,
                                perturbation_method=perturbation_method
                                )
        baseline_model = baseline_model['model']
        baseline_model = baseline_model.cpu()

    if num_important_features == 'use_gtruth':
        #num_important_features is needed for all models except hide_and_seek, invase, realx
        num_important_features = np.max(g_test.astype(bool).sum(axis=1)) #max possible number of important features across all instances
        print("num_important_features: ",num_important_features)

    if model_type == 'invase':
        assert y_train.ndim == 2
        from INVASE_master.INVASE import PVS
        import tensorflow as tf
        model = PVS(x_train, data_type,
                    batch_size=batch_size,
                    epochs=epochs,
                    lamda=lmbda,
                    num_classes=num_classes,
                    task=task)
        
        tf.config.run_functions_eagerly(True) 
        tf.data.experimental.enable_debug_mode() 
        
        # 2. Algorithm training
        model.train(x_train, y_train)

        # 3. Get the selection probability on the testing set
        mask = model.output(x_test)
        binary_mask = 1.*(mask > 0.5)

        mask_train = None
        if save_train_masks:
            mask_train = model.output(x_train)

    elif model_type == 'realx':
        assert y_train.ndim == 2
        from realx_main.realx import REALX
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.layers import Dense, Input, BatchNormalization
        from tensorflow.keras.models import Model
        from tensorflow.keras import regularizers
        from tensorflow.keras import backend as K

        realx_lmbda = lmbda
        realx_epochs = epochs
        realx_batch_size = batch_size
        realx_optimizer = Adam(1e-4)
        realx_loss = 'binary_crossentropy' if task == 'multilabel' else 'categorical_crossentropy'
        realx_metrics = ['acc', 'AUC']

        if num_classes > 2:
            current_auc = tf.keras.metrics.AUC(multi_label=True, num_labels=num_classes, name='auc')
        else:
            assert num_classes == 2
            current_auc = 'AUC'

        realx_metrics = ['acc', current_auc]

        input_shape = x_train.shape[1]

        model_input = Input(shape=(input_shape,), dtype='float32')
        out = Dense(100, activation='relu', kernel_regularizer=regularizers.l2(1e-3))(model_input)
        out = Dense(100, activation='relu', kernel_regularizer=regularizers.l2(1e-3))(out)
        select_prob = Dense(input_shape, kernel_regularizer=regularizers.l2(1e-3))(out)

        
        selector_model = Model(model_input, select_prob)
        model_input = Input(shape=(input_shape,), dtype='float32')
        out= Dense(200, activation='relu', kernel_regularizer=regularizers.l2(1e-3))(model_input)
        out = BatchNormalization()(out)
        out= Dense(200, activation='relu', kernel_regularizer=regularizers.l2(1e-3))(out)
        out = BatchNormalization()(out)
        out_activation = 'sigmoid' if task == 'multilabel' else 'softmax'
        prob = Dense(num_classes, activation=out_activation, kernel_regularizer=regularizers.l2(1e-3))(out)

        predictor_model = Model(model_input, prob)
        realx = REALX(selector_model, predictor_model, lamda=realx_lmbda, task=task)

        realx.predictor.compile(loss=realx_loss,
                        optimizer=realx_optimizer,
                        metrics=realx_metrics)
        realx.predictor.fit(x_train,
                            y_train,
                            epochs=realx_epochs,
                            batch_size=realx_batch_size,
                            verbose=0)
        realx.build_selector()

        # Train
        realx.selector.compile(loss=None,
                            optimizer=realx_optimizer,
                            metrics=realx_metrics)
        realx.selector.fit(x_train,
                        y_train,
                        epochs=realx_epochs,
                        batch_size=realx_batch_size,
                        verbose=0)
        
        #1. Get Selections 
        score = realx.select(x_test, realx_batch_size, False)

        #2. Get Predictions
        y_score = realx.predict(x_test, realx_batch_size)

        #custom additions
        realx_y_test_pred = y_score
        mask = score.copy()
        binary_mask = 1.*(mask > 0.5)

        mask_train = None
        if save_train_masks:
            mask_train = realx.select(x_train, realx_batch_size, False)

    elif model_type == 'hide_and_seek':
        from hide_and_seek.model import train_nn, pred_nn

        if task in ('classification', 'multiclass'):
            y_train = y_train.argmax(axis=1) if y_train.ndim == 2 else y_train
        # multilabel: keep y_train as 2D float array (N, num_classes)

        

        if n_ensemble is None or n_ensemble == 1:
            #this is the main implementation of hide_and_seek

            output = train_nn(X_train=x_train,
                                y_train=y_train,
                                lmbda=lmbda,
                                n_epochs=epochs,
                                seed=train_seed,
                                task=task,
                                hide_hidden_dim=hide_hidden_dim,
                                seek_hidden_dim=seek_hidden_dim,
                                hide_num_hidden_layers=hide_num_hidden_layers,
                                seek_num_hidden_layers=seek_num_hidden_layers,
                                batch_size=batch_size,
                                print_description=data_type,
                                batchnorm=batchnorm_hs,
                                num_classes=num_classes,
                                lmbda_exponent=lmbda_exponent,
                                return_losses_on_val=return_losses_on_val,
                                class_weight_alpha=class_weight_alpha,
                                perturbation_method=perturbation_method,
                                warmup_epochs=warmup_epochs
                                )
            model = output['model']
            
            if return_losses_on_val == True:
                losses_on_val = output['losses_on_val']

            hide_and_seek_y_test_pred, mask = pred_nn(model=model,
                                            X_test=x_test,
                                            X_train=x_train,
                                            return_masks=True,
                                            seed=test_seed,
                                            task=task,
                                            perturbation_method=perturbation_method
                                            )
            binary_mask = 1.*(mask > 0.5)

            mask_train = None
            
            if save_train_masks:
                _, mask_train = pred_nn(model=model,
                                        X_test=x_train,
                                        X_train=x_train,
                                        return_masks=True,
                                        seed=test_seed, #different seed for training and predicting
                                        task=task,
                                        perturbation_method=perturbation_method
                                        )
        else: 
        #this is the ensembling implementation of hide_and_seek, used in one of the experiments in the appendix
            ensemble_masks = []
            ensemble_binary_masks = []
            ensemble_preds = []
            ensemble_losses_on_val = []
            ensemble_feature_indices = []
            sampled_feature_indices = []
            num_features = x_train.shape[1]

            if colsample is not None:
                subset_size = max(1, int(np.ceil(colsample * num_features)))
                covered_features = np.zeros(num_features, dtype=bool)
                sampled_feature_indices = []

                for ensemble_idx in range(n_ensemble):
                    iter_train_seed = train_seed + ensemble_idx
                    member_rng = np.random.RandomState(iter_train_seed)
                    col_indices = np.sort(member_rng.choice(num_features, size=subset_size, replace=False))
                    sampled_feature_indices.append(col_indices)
                    covered_features[col_indices] = True

                if not np.all(covered_features):
                    raise ValueError(
                        "Some features were not included by any ensemble member; "
                        "try increasing n_ensemble or colsample so that all features are included"
                    )
            else:
                sampled_feature_indices = [None] * n_ensemble

            for idx_arr in sampled_feature_indices:
                if idx_arr is None:
                    ensemble_feature_indices.append(np.arange(num_features))
                else:
                    ensemble_feature_indices.append(idx_arr)

            use_parallel, effective_n_jobs, effective_backend = _resolve_ensemble_parallel_policy(
                n_ensemble=n_ensemble,
                ensemble_parallel=ensemble_parallel,
                ensemble_n_jobs=ensemble_n_jobs,
                ensemble_backend=ensemble_backend,
            )

            if use_parallel:
                from joblib import Parallel, delayed
                member_outputs = Parallel(n_jobs=effective_n_jobs, backend=effective_backend)(
                    delayed(_run_hide_and_seek_ensemble_member)(
                        ensemble_idx=ensemble_idx,
                        member_col_indices=sampled_feature_indices[ensemble_idx],
                        x_train=x_train,
                        x_test=x_test,
                        y_train=y_train,
                        lmbda=lmbda,
                        epochs=epochs,
                        train_seed=train_seed,
                        test_seed=test_seed,
                        task=task,
                        hide_hidden_dim=hide_hidden_dim,
                        seek_hidden_dim=seek_hidden_dim,
                        hide_num_hidden_layers=hide_num_hidden_layers,
                        seek_num_hidden_layers=seek_num_hidden_layers,
                        batch_size=batch_size,
                        data_type=data_type,
                        batchnorm_hs=batchnorm_hs,
                        num_classes=num_classes,
                        lmbda_exponent=lmbda_exponent,
                        return_losses_on_val=return_losses_on_val,
                        class_weight_alpha=class_weight_alpha,
                        perturbation_method=perturbation_method,
                        full_num_features=num_features,
                    )
                    for ensemble_idx in range(n_ensemble)
                )
            else:
                member_outputs = [
                    _run_hide_and_seek_ensemble_member(
                        ensemble_idx=ensemble_idx,
                        member_col_indices=sampled_feature_indices[ensemble_idx],
                        x_train=x_train,
                        x_test=x_test,
                        y_train=y_train,
                        lmbda=lmbda,
                        epochs=epochs,
                        train_seed=train_seed,
                        test_seed=test_seed,
                        task=task,
                        hide_hidden_dim=hide_hidden_dim,
                        seek_hidden_dim=seek_hidden_dim,
                        hide_num_hidden_layers=hide_num_hidden_layers,
                        seek_num_hidden_layers=seek_num_hidden_layers,
                        batch_size=batch_size,
                        data_type=data_type,
                        batchnorm_hs=batchnorm_hs,
                        num_classes=num_classes,
                        lmbda_exponent=lmbda_exponent,
                        return_losses_on_val=return_losses_on_val,
                        class_weight_alpha=class_weight_alpha,
                        perturbation_method=perturbation_method,
                        full_num_features=num_features,
                    )
                    for ensemble_idx in range(n_ensemble)
                ]

            member_outputs = sorted(member_outputs, key=lambda x: x['ensemble_idx'])
            for member_out in member_outputs:
                ensemble_masks.append(member_out['iter_mask'])
                ensemble_binary_masks.append(member_out['iter_binary_mask'])
                ensemble_preds.append(member_out['iter_y_test_pred'])
                if return_losses_on_val == True:
                    ensemble_losses_on_val.append(member_out['iter_losses_on_val'])

            mask = np.nanmean(np.stack(ensemble_masks, axis=0), axis=0)
            hide_and_seek_y_test_pred = np.mean(np.stack(ensemble_preds, axis=0), axis=0)
            #hard vote
            # binary_mask = (np.sum(np.stack(ensemble_binary_masks, axis=0), axis=0) > (n_ensemble / 2)).astype(float)
            #soft vote:
            binary_mask = (mask > 0.5).astype(float)
            if return_losses_on_val == True:
                losses_on_val = ensemble_losses_on_val

            #save_train_masks=True is not yet implemented for ensemble
    
    elif model_type == 'l2x':

        #note - this requires env 'l2x2018'
        from L2X.l2x_for_testing import L2X
        PARENT_DIR_L2X = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'l2x_models') #saves the models

        data_dict = {'x_train':x_train,
                     'y_train':y_train,
                     'x_val':x_test} #mismatch is ok

        l2x_activation = 'relu' if data_type in ['Syn1','Syn2'] else 'selu'
        print(l2x_activation)
        binary_mask, _, l2x_y_test_pred = L2X(datatype=data_type,
                    num_important_features=num_important_features,
                    train=True,
                    parent_dir=PARENT_DIR_L2X,
                    data_dict=data_dict,
                    activation=l2x_activation, #matches up with INVASE choice
                    num_classes=num_classes,
                    task=task,
                    return_pred_and_mask=True)
        mask = binary_mask.copy()
        print(data_type, ': ', num_important_features)

    elif model_type == 'lime':
        from lime import lime_tabular
        import torch
        import pandas as pd
        # from lime code: "As opposed to lime_text.TextExplainer, tabular explainers need a training set. The reason for this is because we compute statistics on each feature (column). If the feature is numerical, we compute the mean and std, and discretize it into quartiles.""
        
        #y_train already made class vector in baseline
        model = lime_tabular.LimeTabularExplainer(x_train, 
                                                   feature_names=None, 
                                                   class_names=None, 
                                                   discretize_continuous=True)
        
        if use_custom_nn_for_lime == True: #True for mnist test - using custom nn classifier as baseline
            from tests_mnist.classifier_for_lime import run_model_lime_nn_classifier
            baseline_model, device = run_model_lime_nn_classifier(x_train, 
                                                          y_train, 
                                                          x_test, 
                                                          y_test,
                                                          epochs=epochs)
        full_data_explanations = {}
        print('starting lime explanations')
        # x_test = x_test[:100,:]
        # for i in [189, 486, 1127, 220, 825, 264]: #mnist images in paper
        for i in range(x_test.shape[0]):
            
            # exp = model.explain_instance(x_test[i], baseline_model.predict_proba, num_features=x_test.shape[1], top_labels=1)
            exp = model.explain_instance(x_test[i], baseline_model.predict_proba, num_features=num_important_features, top_labels=1) #could update to give explanation for all classes, and then aggregate
            
            full_data_explanations[i]={int(k):float(v) for k,v in iter(list(exp.as_map().values())[0])}
        
        lime_explanations = pd.DataFrame(full_data_explanations).T
        lime_explanations = lime_explanations.reindex(columns=range(x_test.shape[1]))
        
        binary_mask = lime_explanations.notna().astype(int).values
        mask = binary_mask.copy()

        lime_y_test_pred = baseline_model.predict_proba(x=x_test)

    # elif model_type == 'shap': #shap_xgboost had better results so that was used instead for our experiments.
    #     import shap

    #     background = shap.sample(x_train, 100)

    #     def safe_predict(X):
    #         return baseline_model.predict_proba(X, clip_eps=1e-7)
        
    #     model = shap.KernelExplainer(model=safe_predict, 
    #                                      data=background, 
    #                                      link="logit")

    #     shap_values = model.shap_values(x_test, nsamples=100)[:,:,0] #takes class0 as baseline_model.predict_proba returns probabilities for both classes
        
    #     binary_mask = find_n_largest_values(shap_values, num_important_features)
    #     mask = np.abs(shap_values)

    #     shap_y_test_pred = safe_predict(x_test)
    #     # model.predict(xgb.DMatrix(x_test))
    #     # xgb_y_pred = np.vstack([xgb_y_pred,1-xgb_y_pred]).T

    elif model_type == 'shap_xgboost':
        #note - this requires env 'xgboost'
        import xgboost as xgb
        import shap

        # #for syn1-5, 100 trees was better than early stopping. So not doing it. Syn6 difference was negligble 
        # # create a val set for early stopping
        # x_train_new, x_val, y_train_new, y_val = train_test_split(
        #             x_train, y_train[:, 0], test_size=0.1, random_state=train_seed
        #         ) 

        # # Train an XGBoost model
        # dtrain = xgb.DMatrix(x_train_new, label=y_train_new)
        # dval = xgb.DMatrix(x_val, label=y_val)
        # watchlist = [(dtrain, 'train'), (dval, 'val')]
        y_train = y_train.argmax(axis=1) if y_train.ndim == 2 else y_train
        y_train = y_train.astype(int)

        dtrain = xgb.DMatrix(x_train, label=y_train)
        watchlist = [(dtrain, 'train')]

        xgb_train_params = {}
        if xgb_params is None:
            # Preserve historical binary defaults exactly.
            if num_classes == 2:
                xgb_train_params = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'max_depth': 5,
                    'colsample_bytree': 0.9,
                    'eta': 0.1,
                }
            else:
                xgb_train_params = {
                    'objective': 'multi:softprob',
                    'eval_metric': 'mlogloss',
                    'num_class': num_classes,
                    'colsample_bytree': 0.9,
                    'max_depth': 5,
                    'eta': 0.1,
                }
            num_boost_round = 100
        else:
            # Never mutate caller's dict; sweep scripts may reuse it after this call.
            xgb_train_params = xgb_params.copy()
            num_boost_round = int(xgb_train_params.pop('num_boost_round', 100))

            if num_classes == 2:
                xgb_train_params.setdefault('objective', 'binary:logistic')
                xgb_train_params.setdefault('eval_metric', 'logloss')
            else:
                xgb_train_params.setdefault('objective', 'multi:softprob')
                xgb_train_params.setdefault('eval_metric', 'mlogloss')
                xgb_train_params['num_class'] = num_classes

        xgb_train_params['seed'] = train_seed
        evals_result = {}
        
        model = xgb.train(
                        xgb_train_params,
                        dtrain,
                        num_boost_round=num_boost_round,
                        evals=watchlist,
                        evals_result=evals_result,
                        verbose_eval=num_boost_round // 10
                        # early_stopping_rounds=10
                    )

        # Use SHAP to explain the model
        explainer = shap.TreeExplainer(model)
        shap_values_raw = explainer(x_test).values

        # Get the binary_masks based on SHAP values.
        if num_classes == 2:
            shap_values = shap_values_raw
            binary_mask = find_n_largest_values(shap_values, num_important_features)
            mask = np.abs(shap_values)

            # Predictions for saving.
            xgb_y_test_pred = model.predict(xgb.DMatrix(x_test))
            xgb_y_test_pred = np.vstack([1-xgb_y_test_pred, xgb_y_test_pred]).T
        else: #note that for shap_xgboost, we are looking at explanations for all classes. In lime, we are just doing the predicted class. 
            # Aggregate per-feature importance across class dimension.
            if isinstance(shap_values_raw, list):
                shap_arr = np.stack([np.abs(sv) for sv in shap_values_raw], axis=-1)
            else:
                shap_arr = np.abs(np.asarray(shap_values_raw))

            if shap_arr.ndim != 3:
                raise ValueError(f"Unexpected SHAP output shape for multiclass: {shap_arr.shape}")

            if shap_arr.shape[1] == x_test.shape[1]:
                # (n_samples, n_features, n_classes)
                shap_feature_scores = shap_arr.mean(axis=2)
            elif shap_arr.shape[2] == x_test.shape[1]:
                # (n_samples, n_classes, n_features)
                shap_feature_scores = shap_arr.mean(axis=1)
            else:
                raise ValueError(f"Could not infer feature axis from SHAP output shape: {shap_arr.shape}")

            binary_mask = find_n_largest_values(shap_feature_scores, num_important_features)
            mask = shap_feature_scores

            # multi:softprob returns (n_samples, n_classes)
            xgb_y_test_pred = model.predict(xgb.DMatrix(x_test))

        # Save effective xgboost params used by training.
        xgb_params = xgb_train_params.copy()
        xgb_params['num_boost_round'] = num_boost_round

    elif model_type == 'lasso':
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        model = make_pipeline( #moved scaling above
                        LogisticRegression(
                            penalty='l1',
                            solver='liblinear',   # supports L1
                            C=1                 # inverse of regularisation strength
                        )
                    )
        y_train = y_train.argmax(axis=1) if y_train.ndim == 2 else y_train

        model.fit(x_train, y_train)
        
        coefs = model.named_steps['logisticregression'].coef_
        
        coefs = np.tile(coefs, (len(x_test),1))
        binary_mask = find_n_largest_values(coefs, num_important_features)
        mask = np.abs(coefs)
        lasso_y_test_pred = model.predict_proba(x_test)

    elif model_type == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(criterion='gini', 
                                    n_estimators=100,
                                    max_depth=5, 
                                    random_state=train_seed)
        model.fit(x_train, y_train)
        
        rf_y_test_pred = model.predict_proba(x_test)[0]
        
        from sklearn.metrics import log_loss
        loss = log_loss(y_test, rf_y_test_pred)
        print(f'loss for {data_type} with random forest: {loss}')

        importance = model.feature_importances_
        
        importance = np.tile(importance, (len(x_test),1))
        binary_mask = find_n_largest_values(importance, num_important_features)
        
        mask = np.abs(importance)

    else:
        raise ValueError("Unsupported model type.")
    
    # 5. Prediction
    if model_type == 'invase':
        y_test_pred, dis_predict = model.get_prediction(x_test, binary_mask)
    elif model_type == 'hide_and_seek':
        y_test_pred = hide_and_seek_y_test_pred.copy()
    elif model_type == 'l2x':
        y_test_pred = l2x_y_test_pred
    elif model_type == 'shap_xgboost':
        y_test_pred = xgb_y_test_pred
    # elif model_type == 'shap':
    #     y_test_pred = shap_y_test_pred
    elif model_type == 'lime':
        y_test_pred = lime_y_test_pred
    elif model_type == 'random_forest':
        y_test_pred = rf_y_test_pred
    elif model_type == 'lasso':
        y_test_pred = lasso_y_test_pred
    elif model_type == 'realx':
        y_test_pred = realx_y_test_pred
            
    #%% Output
    results = {}
    pct_sig = binary_mask.mean()

    if g_test is not None:
        TPR_mean, FDR_mean, TPR_std, FDR_std = performance_metric(binary_mask=binary_mask,
                                                              g_truth=g_test)
        f1 = compute_f1(binary_mask=binary_mask, g_truth=g_test)

        print(f'{data_type}: ' + 'TPR mean: ' + str(np.round(TPR_mean,1)) + '%') # + 'TPR std: ' + str(np.round(TPR_std,1)) + '\%, '  
        print(f'{data_type}: ' + 'FDR mean: ' + str(np.round(FDR_mean,1)) + '%') # + 'FDR std: ' + str(np.round(FDR_std,1)) + '\%, '  
        print(f'{data_type}: ' + 'F1 mean: ' + str(np.round(f1,1)) + '%')

        results['TPR_mean']=TPR_mean
        results['FDR_mean']=FDR_mean
        results['TPR_std']=TPR_std
        results['FDR_std']=FDR_std
        
        results['f1'] = f1
    
    if task in ('classification', 'multiclass'):
        acc, roc_auc, pr_auc = prediction_metrics(y_true=y_test,
                                       y_pred_probs=y_test_pred,
                                        model_type=model_type,
                                      verbose=False)

        results['accuracy']=acc
        results['roc_auc']=roc_auc
        results['pr_auc'] = pr_auc

        print(f'{data_type}: ' + 'pct_sig: ' + str(np.round(pct_sig,4)))
        # print(f'{data_type}: ' + 'accuracy: ' + str(np.round(acc,4)))
        print(f'{data_type}: ' + 'roc_auc: ' + str(np.round(roc_auc,4)))
        # print(f'{data_type}: ' + 'pr_auc: ' + str(np.round(pr_auc,4)))

    elif task == 'regression':
        mse = mean_squared_error(y_test, y_test_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_test_pred)
        R2 = r2_score(y_test, y_test_pred)

        results['rmse'] = rmse
        results['mae'] = mae
        results['R2'] = R2
        results['accuracy'] = None
        results['roc_auc'] = None
        results['pr_auc'] = None

    elif task == 'multilabel':
        y_test_binary = (y_test_pred > 0.5).astype(int)
        acc = 1.0 - hamming_loss(y_test, y_test_binary)
        roc_auc = roc_auc_score(y_test, y_test_pred, average='micro') #using micro as when using this for stage 2 prediction (whether something is a switch feature, we expect many columns with all 0 and some columns with all 1)
        pr_auc = average_precision_score(y_test, y_test_pred, average='micro') #using micro as when using this for stage 2 prediction (whether something is a switch feature, we expect many columns with all 0 and some columns with all 1)

        results['accuracy'] = acc
        results['roc_auc'] = roc_auc
        results['pr_auc'] = pr_auc

        print(f'{data_type}: ' + 'pct_sig: ' + str(np.round(pct_sig, 4)))
        print(f'{data_type}: ' + 'hamming_accuracy: ' + str(np.round(acc, 4)))
        print(f'{data_type}: ' + 'roc_auc (macro): ' + str(np.round(roc_auc, 4)))
        print(f'{data_type}: ' + 'pr_auc (macro): ' + str(np.round(pr_auc, 4)))

    else:
        raise ValueError("Unsupported task type. Use 'regression', 'classification', 'multiclass', or 'multilabel'.")

    results['batch_size'] = batch_size
    results['save_experiment_data'] = save_experiment_data
    results['syn_switch_quantile'] = effective_quantile
    
    if save_experiment_data == True:
        results['binary_mask'] = binary_mask
        results['pct_sig'] = pct_sig
        results['g_test'] = g_test
        results['mask'] = mask
        results['y_test_pred'] = y_test_pred

        if include_y_test == True:
            results['y_test'] = y_test
    else:
        results['binary_mask'] = None
        results['g_test'] = None
        results['mask'] = None
        results['y_test_pred'] = None
        results['y_test'] = None
    

    if include_model == True:
        results['model'] = model if n_ensemble in [None, 1] else None

    if model_type == 'invase':
        results['latent_dim1'] = model.latent_dim1
        results['latent_dim2'] = model.latent_dim2    
        results['activation'] = model.activation
        results['input_shape'] = model.input_shape
        results['input_shape0'] = model.input_shape0    

    if model_type == 'hide_and_seek':
        results['hide_hidden_dim'] = hide_hidden_dim
        results['seek_hidden_dim'] = seek_hidden_dim
        results['hide_num_hidden_layers'] = hide_num_hidden_layers
        results['seek_num_hidden_layers'] = seek_num_hidden_layers
        results['n_ensemble'] = n_ensemble
        results['colsample'] = colsample
        results['ensemble_parallel'] = ensemble_parallel
        results['ensemble_n_jobs'] = ensemble_n_jobs
        results['ensemble_backend'] = ensemble_backend
        results['perturbation_method'] = perturbation_method

        if n_ensemble is not None and n_ensemble > 1:
            results['ensemble_feature_indices'] = [idx.tolist() for idx in ensemble_feature_indices]
        if task == 'regression' and 'y_scaler' in output:
            results['y_scaler'] = output['y_scaler']
        if return_losses_on_val == True:
            results['losses_on_val'] = losses_on_val
            

    if save_train_masks:
        results['x_train'] = x_train
        results['x_test'] = x_test
        results['mask_train'] = mask_train

    if model_type == 'lime':
        results['lime_explanations'] = lime_explanations

    results['epochs'] = epochs
    results['lmbda'] = lmbda
    results['seed'] = seed
    results['rho'] = rho
    results['model_type'] = model_type_text
    results['num_important_features'] = num_important_features
    results['lmbda_exponent'] = lmbda_exponent
    results['return_losses_on_val'] = return_losses_on_val
    results['batchnorm_hs'] = batchnorm_hs
    results['num_classes'] = num_classes
    results['num_syn_features'] = num_syn_features
    results['train_N'] = len(x_train)
    results['test_N'] = len(x_test)
    results['data_mode'] = data_mode
    results['scale_data'] = scale_data
    results['class_weight_alpha'] = class_weight_alpha
    results['column_names'] = column_names
    results['xgb_params'] = xgb_params


    
    results['run_type']=run_type
    results['time_run']=timestamp_start
    timestamp_end = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results['time_end']=timestamp_end

    if folder_for_pickle is not None:
        print("saving results as pickle")
        save_results_as_pickle(results=results,
                            syn_type=data_type,
                            model_type=model_type,
                            folder=folder_for_pickle,
                            name_end=run_type,
                            timestamp=timestamp_start)
    if return_results == True:
        return results


    
