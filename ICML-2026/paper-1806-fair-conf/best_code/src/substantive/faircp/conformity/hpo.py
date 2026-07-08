import optuna

from substantive.faircp.conformity.utils import (
    compute_nonconformity_score,
    get_conformal_set,
    coverage_size,
)


def run_hpo(objective_str, return_obj_fn, direction, init_h_params, cfg):
    hpo_cfg = init_h_params.copy()

    # Parameter-specific bounds
    param_bounds = {
        'T': (0.1, 2.0),
        'lamda': (0.01, 0.5),
    }
    
    def objective(trial):
        h_params = {}
        for param, value in hpo_cfg.items():
            if isinstance(value, int):
                new_val = trial.suggest_int(param, value - 1, value + 1)
            elif isinstance(value, float):
                lo, hi = param_bounds.get(param, (value / 10, value * 10))
                new_val = trial.suggest_float(param, lo, hi, log=(param == 'lamda'))
            else:
                raise ValueError(
                    f"Hyper to tune is type {type(value)}, expected int or float"
                )
            h_params[param] = new_val

        return return_obj_fn(h_params)

    if not cfg["verbose"]:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=cfg["seed"])
    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(objective, n_trials=cfg["hpo_iterations"])

    best_value = study.best_trial.values
    best_h_params = study.best_params
    print(f"Result of HPO tuning for {objective_str} is {best_value}")

    return best_h_params


def run_hpo_conformal(
    logits_calib,
    targets_calib,
    logits_val,
    targets_val,
    used_labels,
    init_h_params,
    cfg,
    conformal_category,
    bins_calib=None,
    bins_test=None,
):
    # Define tuning objective
    def return_size(h_params):
        non_conf_scores_calib, non_conf_scores_val = compute_nonconformity_score(
            logits_calib, targets_calib, logits_val, h_params, cfg
        )
        conformal_preds_val = get_conformal_set(
            non_conf_scores_calib,
            non_conf_scores_val,
            labels=used_labels,
            confidence=1 - cfg["alpha"],
            conformal_category=conformal_category,
            bins_cal=bins_calib,
            bins_test=bins_test,
        )
        _, size = coverage_size(conformal_preds_val, targets_val)
        return size

    best_h_params = run_hpo("set size", return_size, "minimize", init_h_params, cfg)

    return best_h_params
