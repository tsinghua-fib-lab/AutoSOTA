from sweep_engine import run_optimizer_sweep

def cd_params(trial):
    return {
        "h": trial.suggest_categorical("h", [0.0997457866866461]),
        "gamma": trial.suggest_categorical("gamma", [0.0013778506032957284]),
        "c": trial.suggest_categorical("c", [1429405.2839974575])
    }

if __name__ == "__main__":
    run_optimizer_sweep("sweep_cubic", "cubic_damping_opt", cd_params, n_trials=10)