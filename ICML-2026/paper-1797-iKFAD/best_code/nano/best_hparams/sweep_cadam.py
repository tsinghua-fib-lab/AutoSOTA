from sweep_engine import run_optimizer_sweep

def cadam_params(trial):
    return {
        "h": trial.suggest_categorical("h", [0.008275804955597058]),
        "gamma": trial.suggest_categorical("gamma", [9.47376051032096]),
        "c": trial.suggest_categorical("c", [0.9520101838407309]),
        "alpha": trial.suggest_categorical("alpha", [8.828765178485884]),
        "eps": 1e-8
    }

if __name__ == "__main__":
    run_optimizer_sweep("sweep_cadam", "Cadam", cadam_params, n_trials=10)