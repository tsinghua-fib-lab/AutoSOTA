from sweep_engine import run_optimizer_sweep

def cd_params(trial):
    return {
        "h": trial.suggest_categorical("h", [0.4992407518812977]),
        "gamma": trial.suggest_categorical("gamma", [0.0037534843667180505]),
        "c": trial.suggest_categorical("c", [186961.17913090085])
    }

if __name__ == "__main__":
    run_optimizer_sweep("sweep_cubic", "cubic_damping_opt", cd_params, n_trials=10)