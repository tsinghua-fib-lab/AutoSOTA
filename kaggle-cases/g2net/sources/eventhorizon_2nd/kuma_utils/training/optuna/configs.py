
def catboost_reg_params(trial, params):
    _params = {
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
        "depth": trial.suggest_int("depth", 1, 12),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        )
    }
    if _params["bootstrap_type"] == "Bayesian":
        _params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif _params["bootstrap_type"] == "Bernoulli":
        _params["subsample"] = trial.suggest_float("subsample", 0.1, 1)
    
    _params.update(params)
    return _params


def catboost_cls_params(trial, params):
    _params = {
        "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
    }
    _params = catboost_reg_params(trial, params)
    _params.update(params)
    return _params


def xgboost_params(trial, params):
    _params = {
        # "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
    }
    if _params["booster"] == "gbtree" or _params["booster"] == "dart":
        _params["max_depth"] = trial.suggest_int("max_depth", 1, 9)
        _params["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        _params["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        _params["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    if _params["booster"] == "dart":
        _params["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        _params["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        _params["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        _params["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
    _params.update(params)
    return _params


def random_forest_params(trial, params):
    _params = {
        'max_depth': trial.suggest_int("max_depth", 2, 32, log=True),
        'n_estimators': trial.suggest_categorical('n_estimators', [5, 10, 20, 30, 50, 100]),
        'max_features': trial.suggest_float('max_features', 2, 32, log=True)
    }
    _params.update(params)
    return _params


def svm_params(trial, params):
    _params = {
        'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly']),
        'C': trial.suggest_float("C", 1e-3, 1e3, log=True)
    }
    if _params['kernel'] in ['rbf', 'poly']:
        _params['gamma'] = trial.suggest_int('gamma', 1, 1e3, log=True)
    if _params['kernel'] == 'poly':
        _params['degree'] = trial.suggest_int('degree', 0, 5)
    _params.update(params)
    return _params
