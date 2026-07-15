import random
import torch
import numpy as np
import os.path
import tqdm
import parse
imagenet_dataset_template = 'ImageNet_min_{min_class}_maj_{maj_class}'


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def get_run_description(**kwargs):
    run_name = ''
    run_name += kwargs['dataset'] + '_'
    if 'meps' in kwargs['dataset'].lower():
        run_name += 'age_range_' + kwargs['age_range'] + '_'
    run_name += kwargs['model_name'] + '_'
    run_name += 'n_cal_real_' + str(kwargs['n_cal']) + '_n_cal_maj_' + str(kwargs['n_cal_maj']) + '_n_test_min_' \
               + str(kwargs['n_test'])
    if kwargs['methods'] is None or any(['SPI' in m for m in kwargs['methods']]):
        run_name += '_beta_' + str(kwargs['beta'])
        if 'SPI [subset]' in kwargs['methods']:
            run_name += '_k_' + str(kwargs['k']) + '_dist_' + kwargs['dist_criterion']
    run_name += '_q_' + str(kwargs['alpha']) + '_n_seeds_' + str(kwargs['n_seeds'])
    return run_name


def is_classification_dataset(dataset):
    return dataset.startswith('ImageNet')


def get_classes_by_dataset_name(dataset, data_shape):
    maj_classes = []
    if not dataset.startswith('ImageNet'):
        classes = [0] if len(data_shape) == 1 else range(data_shape[1])
    else:
        args = parse.parse(imagenet_dataset_template, dataset)
        classes, maj_classes = get_class_numbers_imagenet(args['min_class'], args['maj_class'])
    return classes, maj_classes


def get_data(dataset='', n_cal=50, n_test=1000, n_cal_maj=1000, random_state=2020,
             X_minority=None, y_minority=None, X_majority=None, y_majority=None, exact=True, only_maj_exact=False, **kwargs):
    # generate data
    if X_minority is None and y_minority is None and X_majority is None and y_majority is None:
        raise ValueError(f'The get_data function currently does not support data generation for dataset {dataset}.')
    X_minority_calib, y_minority_calib = None, None
    X_minority_test, y_minority_test = None, None
    X_majority_calib, y_majority_calib = None, None

    rng = np.random.default_rng(random_state)
    if X_minority is not None and n_cal + n_test > len(X_minority):
        raise ValueError(f'Not enough real samples available: requested -> {n_cal+n_test} out of {len(X_minority)} available.')
    if X_majority is not None and n_cal_maj > len(X_majority):
        raise ValueError(f'Not enough synthetic samples available: requested -> {n_cal_maj} out of {len(X_majority)} available.')
    if exact:
        X_minority_calib, y_minority_calib, X_minority_test, y_minority_test = None, None, None, None
        X_majority_calib, y_majority_calib = None, None
        # draw the same number of samples from each class
        classes = np.unique(y_minority)
        classes_maj = np.unique(y_majority)
        n_cal_class = n_cal // len(classes)
        n_test_class = n_test // len(classes)
        n_cal_maj_class = n_cal_maj // len(classes_maj)
        for class_ in classes:
            curr_X_minority = X_minority[y_minority == class_]
            curr_y_minority = y_minority[y_minority == class_]
            
            X_minority_calib_, y_minority_calib_, X_minority_test_, y_minority_test_, \
            _, __ = get_data(dataset=dataset, n_cal=n_cal_class, n_test=n_test_class, n_cal_maj=n_cal_maj_class, random_state=random_state,
                                                            X_minority=curr_X_minority, y_minority=curr_y_minority, X_majority=None, y_majority=None, exact=False)
            # concatenate
            X_minority_calib = np.concatenate([X_minority_calib, X_minority_calib_], axis=0) if X_minority_calib is not None else X_minority_calib_
            y_minority_calib = np.concatenate([y_minority_calib, y_minority_calib_], axis=0) if y_minority_calib is not None else y_minority_calib_
            X_minority_test = np.concatenate([X_minority_test, X_minority_test_], axis=0) if X_minority_test is not None else X_minority_test_
            y_minority_test = np.concatenate([y_minority_test, y_minority_test_], axis=0) if y_minority_test is not None else y_minority_test_
        for class_ in classes_maj:
            curr_X_majority = X_majority[y_majority == class_]
            curr_y_majority = y_majority[y_majority == class_]
            _, _, _, _, X_majority_calib_, y_majority_calib_ = get_data(dataset=dataset, n_cal=n_cal_class, n_test=n_test_class, n_cal_maj=n_cal_maj_class, random_state=random_state,
                                                            X_minority=None, y_minority=None, X_majority=curr_X_majority, y_majority=curr_y_majority, exact=False)
            X_majority_calib = np.concatenate([X_majority_calib, X_majority_calib_], axis=0) if X_majority_calib is not None else X_majority_calib_
            y_majority_calib = np.concatenate([y_majority_calib, y_majority_calib_], axis=0) if y_majority_calib is not None else y_majority_calib_
        return X_minority_calib, y_minority_calib, X_minority_test, y_minority_test, X_majority_calib, y_majority_calib
    if only_maj_exact:
        classes_maj = np.unique(y_majority)
        n_cal_maj_class = n_cal_maj // len(classes_maj)
        for class_ in classes_maj:
            curr_X_majority = X_majority[y_majority == class_]
            curr_y_majority = y_majority[y_majority == class_]
            _, _, _, _, X_majority_calib_, y_majority_calib_ = get_data(dataset=dataset, n_cal=n_cal, n_test=n_test, n_cal_maj=n_cal_maj_class, random_state=random_state,
                                                            X_minority=None, y_minority=None, X_majority=curr_X_majority, y_majority=curr_y_majority, exact=False)
            X_majority_calib = np.concatenate([X_majority_calib, X_majority_calib_], axis=0) if X_majority_calib is not None else X_majority_calib_
            y_majority_calib = np.concatenate([y_majority_calib, y_majority_calib_], axis=0) if y_majority_calib is not None else y_majority_calib_
    if X_minority is not None:
        minority_indices = rng.permutation(len(X_minority)).tolist()
        minority_calib = minority_indices[:n_cal]
        X_minority_calib, y_minority_calib = X_minority[minority_calib], y_minority[minority_calib]
        minority_test = minority_indices[n_cal:n_cal+n_test]
        X_minority_test, y_minority_test = X_minority[minority_test], y_minority[minority_test]
    if X_majority is not None and not only_maj_exact:
        majority_indices = rng.permutation(len(X_majority)).tolist()
        majority_calib = majority_indices[:n_cal_maj]
        X_majority_calib, y_majority_calib = X_majority[majority_calib], y_majority[majority_calib]

    return X_minority_calib, y_minority_calib, X_minority_test, y_minority_test, X_majority_calib, y_majority_calib


def load_meps_scores(dataset, dataset_path, dataset_maj_path):
    X_minority = np.load(os.path.join(dataset_path, 'pred.npy')).squeeze()
    y_minority = np.load(os.path.join(dataset_path, 'true.npy')).squeeze()
    X_majority = np.load(os.path.join(dataset_maj_path, 'pred.npy')).squeeze()
    y_majority = np.load(os.path.join(dataset_maj_path, 'true.npy')).squeeze()
    return X_minority, y_minority, X_majority, y_majority


def get_class_numbers_imagenet(min_class, maj_class):
    if min_class == 'all':
        min_classes = list(range(1000))
    elif min_class == 'subset':
        min_classes = [16, 207, 250, 626, 852, 862, 444, 17, 676, 217, 880, 337, 336, 208, 222, 18, 13, 270, 20, 15, 321, 392, 157, 326, 993, 991, 994, 389, 395, 0]
    else:
        try:
            min_classes = [int(min_class)]
        except:
            raise ValueError(f'ImageNet: The following subset/class is not supported {min_class}.')
    if maj_class == 'all_100':
        maj_classes = set(list(range(1000))) - set(min_classes)
        maj_classes = list(maj_classes)
        maj_classes = maj_classes[:100]
    elif maj_class == 'gen':
        maj_classes = min_classes
    return min_classes, maj_classes


def load_imagenet_scores(dataset, dataset_path, dataset_maj_path=None):
    X_ = {}
    args = parse.parse(imagenet_dataset_template, dataset)
    min_classes, maj_classes = get_class_numbers_imagenet(args['min_class'], args['maj_class'])
    X_minority, y_minority = [], []
    for label in tqdm.tqdm(min_classes):
        # Format the path for the current class
        np_file = os.path.join(dataset_path, f"{label:04d}", "probs.npy")

        # Check if file exists
        if not os.path.exists(np_file):
            print(f"Missing file for class {label:04d}")
            continue
        probs = np.load(np_file)
        for example_id in range(probs.shape[0]):
            X_minority.append(probs[example_id].reshape((-1,)))
            y_minority.append(label)
    X_minority = np.array(X_minority)
    y_minority = np.array(y_minority)

    X_majority, y_majority = [], []
    if dataset_maj_path is None and (args['maj_class'].endswith('_gen') or args['maj_class'] == 'gen'):
        raise ValueError('The majority dataset is supposed to contain generated images, but dataset_maj_path is None.')
    dataset_maj_path = dataset_path if dataset_maj_path is None else dataset_maj_path
    for label in tqdm.tqdm(maj_classes):
        # Format the path for the current class
        np_file = os.path.join(dataset_maj_path, f"{label:04d}", "probs.npy")

        # Check if file exists
        if not os.path.exists(np_file):
            print(f"Missing file for class {label:04d}")
            continue
        probs = np.load(np_file)
        for example_id in range(probs.shape[0]):
            X_majority.append(probs[example_id].reshape((-1,)))
            y_majority.append(label)
    X_majority = np.array(X_majority)
    y_majority = np.array(y_majority)
    return X_minority, y_minority, X_majority, y_majority


def get_minority_majority_data(dataset, model_name, dataset_path, dataset_maj_path):
    if dataset.startswith('ImageNet'):
        X_minority, y_minority, X_majority, y_majority = load_imagenet_scores(dataset, dataset_path, dataset_maj_path)
    elif dataset.startswith('MEPS'):
        X_minority, y_minority, X_majority, y_majority = load_meps_scores(dataset, dataset_path, dataset_maj_path)
    else:
        raise ValueError(f'The following data set is not supported - {dataset}')
    return X_minority, y_minority, X_majority, y_majority


def init_dict_of_lists(keys):
    dict_ = {}
    for k in keys:
        dict_[k] = []
    return dict_


def construct_prediction_sets(X, alpha_list, quantiles, classes, class_conditional=False, is_aps_score=False, epsilon=None, random_state=2020):
    from transporter.utils import get_aps_score
    S_hat = init_dict_of_lists(keys=alpha_list)
    n = X.shape[0]
    if epsilon is None:
        rng = np.random.default_rng(random_state)
        epsilon = rng.uniform(low=0.0, high=1.0, size=n)
    for i in range(len(X)):
        curr_S = init_dict_of_lists(keys=alpha_list)
        for y in classes:
            if is_aps_score:
                test_score = get_aps_score(X[i].reshape((1,-1)), np.array([y]), epsilon=epsilon[i])
            else:
                test_score = X[i]
            for alpha_ in alpha_list:
                curr_quantile = quantiles[y][alpha_] if class_conditional else quantiles[alpha_]
                if test_score <= curr_quantile:
                    curr_S[alpha_].append(y)
        for alpha_ in alpha_list:
            S_hat[alpha_].append(np.array(curr_S[alpha_]))
    return S_hat
