import torch

# Cross Validation
from sklearn.model_selection import KFold

import numpy as np


# Timing
from time import perf_counter as pc


def dict_conversion(original_dict):
    """
    Helper function so trainin params can be comfortably passed in the form {"training_param" : [x, y, z, ...]}
    which is then transformed into a list [{"training_param" : [x]}, {"training_param" : [y]}, {"training_param" : [z]}, {"training_param" : [...]}, ... ]

    Parameters
    ----------
    original_dict : dictionary
        dictionary of the form {"training_param" : [x, y, z, ...]}

    Returns
    ----------
    converted_list : list
        list of the form [{"training_param" : [x]}, {"training_param" : [y]}, {"training_param" : [z]}, {"training_param" : [...]}, ... ]

    """
    converted_list = []
    for key, values in original_dict.items():
        for value in values:
            converted_list.append({key: value})

    return converted_list


def validate_model(
    model_cls, model_params, train_params, X, y, p, indices, eval_handle
):
    """
    Validates a single model for given fold but all train_params.
    Should not be used directly but with "cross_validate".

    Parameters
    ----------
    model : see "cross_validate"

    model_params : see "cross_validate"

    train_params : see "cross_validate"

    X : see "cross_validate"

    y : see "cross_validate"

    p : see "cross_validate"

    indices : tuple
        Tuple of train/test indices of sklearn e.g. (train_index, test_index) = indices

    eval_handle : see "cross_validate"

    Returns
    ----------
    tmp_res : np.ndarray
        Shape PARAM x MEASURES
        For all passed params all measures are evaluated for the provided fold.

    """

    tmp_res = []

    (train_index, test_index) = indices

    for train_param in train_params:
        # Create new class instance with keyword arguments
        model = model_cls(**model_params)

        model.train(X[train_index], y[train_index], p[train_index], **train_param)

        y_pred = model.predict(X[test_index]).flatten()

        tmp_res.append(eval_handle(y_pred, y[test_index], p[test_index]))

    return np.array(tmp_res)


def cross_validate(
    model, model_params, train_params, X, y, p, folds=5, eval_handle=None, seed=0
):
    """
    Cross validation handle that takes a custom model class (!, not instance) and its constructore arguments
    to run a cross validation procedure on the provided data. Evaluation is donefor each train_param which
    is in train_params as key value argument.


    Parameters
    ----------
    model : class
        Model to benchmark (not a class instance)

    model_params : dictionary
        Parameters passed to the constructor of model for init.

    train_params : list of two elements dicts of the form {"param" : value}
        For every key value pair in train params
        Example form: [{"penalty", 0.5}, {"penalty", 0.75}]
        A dictionary of the form {"penalty", [0.5, 1 , 2, ...]} can be converted
        using the "dict_conversion" function.

    X : np.ndarray
        Data of the form (n_samples, n_features)

    y : np.ndarray
        Targets of the form (n_samples, n_features)

    p : np.ndarray
        Protected attribute of the from (n_samples, n_features)

    folds : int
        Number of cross val. folds

    eval_handle : function
        A function that takes three values: ''predicted targets, true targets, protected attribute''
        Function should return scors of interests: MAE, GDP, ...


    Returns
    ----------
    res : np.ndarray
        Shape FOLDS x PARAM x MEASURES
        i.e. res[i,j,k] corresponds to the i-th fold for the j-th training
        param and k-th measure (the measure(s) are given by eval_handle
        so we need to retrieve the shape on the fly)

    run_time : float
        Runtime of the whole CV procedure

    """

    # Also measure exectuion time of the whole CV
    t0 = pc()

    # Error handling
    if eval_handle is None:
        raise Exception("Eval hande must be provided")

    # We convert to torch float 32 as the torch based approaches require this
    # but we ensure to convert back to float 64 in the non torch based methods
    X = torch.tensor(X.astype(np.float32))
    y = torch.tensor(y.astype(np.float32))
    p = torch.tensor(p.astype(np.float32))

    # Cross validation based on sklearn
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    # For every cross validation split the model is to be evaluated
    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        print("Fold " + str(i))
        tmp = validate_model(
            model,
            model_params,
            train_params,
            X,
            y,
            p,
            indices=(train_index, test_index),
            eval_handle=eval_handle,
        )

        if i == 0:
            res = torch.empty((kf.n_splits, *tmp.shape))

        res[i, :] = torch.tensor(tmp)

    run_time = pc() - t0

    return res.numpy(), run_time
