from sweep_engine import run_optimizer_sweep

def cd_params(trial):
    return {
        "h": trial.suggest_categorical("h", [0.0990343687426758]),
        "gamma": trial.suggest_categorical("gamma", [0.]),
        "c": trial.suggest_categorical("c", [1365156.227108403])
    }

if __name__ == "__main__":
    run_optimizer_sweep("sweep_cubic0", "cubic_damping_opt", cd_params, n_trials=10)