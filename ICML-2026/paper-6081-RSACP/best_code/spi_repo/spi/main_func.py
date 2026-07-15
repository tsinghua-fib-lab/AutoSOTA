import random
from tqdm import tqdm
import pandas as pd
import numpy as np
from multiprocessing import Pool
from utils import set_seed, get_data, is_classification_dataset, get_classes_by_dataset_name
from calibration import MajSplitConformal, SplitConformal, SplitConformalRegression, \
                        MajSplitConformalRegression
from transporter import SPPredictiveInference, SPPredictiveInferenceRegression, SPPredictiveInferenceClustered
from algo import get_results

classification_method_objects = {
    'SC': SplitConformal,
    'SC (synth)': MajSplitConformal,
    'SPI': SPPredictiveInference,
    'SPI [subset]': SPPredictiveInferenceClustered,
}

regression_method_objects = {
    'SC': SplitConformalRegression,
    'SC (synth)': MajSplitConformalRegression,
    'SPI': SPPredictiveInferenceRegression,
}


def map_run_comparison(kwargs):
    return run_comparison(**kwargs)


def run_comparison(dataset='',
                   n_cal=50, n_test=1000,
                   n_cal_maj=1000,
                   X_minority=None, y_minority=None,
                   X_majority=None, y_majority=None,
                   alpha=0.1, beta=0.4,
                   seed=42, n_seeds=100, methods=None,
                   **kwargs):
    if methods is None:
        methods = ['SC', 'SC (synth)', 'SPI']
    if isinstance(alpha, float):
        alpha = [alpha]
    results = pd.DataFrame({})
    set_seed(seed)
    if n_seeds == 1:
        seed_list = [seed]
    else:
        seed_list = random.sample(range(1, 999999), n_seeds)

    for seed_ in tqdm(seed_list):
        X_minority_calib, y_minority_calib, X_minority_test, \
        y_minority_test, X_majority_calib, y_majority_calib = get_data(dataset=dataset,
                                                                     n_cal=n_cal, n_test=n_test,
                                                                     n_cal_maj=n_cal_maj,
                                                                     random_state=seed_,
                                                                     X_minority=X_minority, y_minority=y_minority,
                                                                     X_majority=X_majority, y_majority=y_majority,
                                                                     exact=kwargs.get('class_conditional', False),
                                                                     only_maj_exact='SPI [subset]' in methods and not kwargs.get('class_conditional', False),
                                                                     **kwargs
                                                                     )
        classes_min, classes_maj = get_classes_by_dataset_name(dataset, X_minority_calib.shape)
        n_classes_min, n_classes_maj = len(classes_min), len(classes_maj)
        # draw uniform variables for calib and test
        rng = np.random.default_rng(seed_)
        epsilon_min_calib = rng.uniform(low=0.0, high=1.0, size=len(y_minority_calib))
        epsilon_min_test = rng.uniform(low=0.0, high=1.0, size=len(y_minority_test))
        epsilon_maj_calib = rng.uniform(low=0.0, high=1.0, size=len(y_majority_calib))
        for method in methods:
            method_objects = classification_method_objects if is_classification_dataset(dataset) else regression_method_objects
            calibrator = method_objects[method](method_name=method, random_state=seed_,
                                                X_calib=X_minority_calib, y_calib=y_minority_calib,
                                                X_maj_calib=X_majority_calib, y_maj_calib=y_majority_calib,
                                                alpha=alpha, beta=beta,
                                                epsilon=epsilon_min_calib, epsilon_maj=epsilon_maj_calib,
                                                dataset=dataset, **kwargs)
            pred_sets = calibrator.predict(X_minority_test, epsilon=epsilon_min_test)
            curr_result = get_results(pred_sets, X_minority_test, y_minority_test, method,
                                      alpha, beta, n_cal, n_cal_maj, n_test, dataset,
                                      n_classes_min, n_classes_maj,
                                      class_conditional=kwargs.get('class_conditional', False),
                                      k=kwargs.get('k', 20),
                                      dist_criterion=kwargs.get('dist_criterion', 'cramer-von-mises'),
                                      age_range=kwargs.get('age_range', None),
                                      )
            results = pd.concat([results, curr_result])
    return results


def run_comparison_parallel(X_minority=None, y_minority=None, X_majority=None, y_majority=None, **kwargs):
    distributed = kwargs.get('distribute', False)
    if distributed == "False":
        distributed = False
    if distributed:
        set_seed(kwargs['seed'])
        seed_list = random.sample(range(1, 999999), kwargs['n_seeds'])
        base_params = kwargs.copy()
        base_params['n_seeds'] = 1
        base_params['X_minority'] = X_minority
        base_params['y_minority'] = y_minority
        base_params['X_majority'] = X_majority
        base_params['y_majority'] = y_majority
        try:
            del base_params['distribute']     # not necessary
            del base_params['num_processes']
        except:
            pass
        params_list = []
        for seed_ in seed_list:
            curr_params = base_params.copy()
            curr_params['seed'] = seed_
            params_list.append(curr_params)
        with Pool(int(kwargs.get('num_processes', 16))) as pool:
            results_list = list(tqdm(pool.imap(map_run_comparison, params_list), total=kwargs['n_seeds']))

        results = pd.DataFrame({})
        for result_ in results_list:
            results = pd.concat([results, result_])
    else:
        results = run_comparison(X_minority=X_minority, y_minority=y_minority,
                                 X_majority=X_majority, y_majority=y_majority,
                                 **kwargs)
    return results
