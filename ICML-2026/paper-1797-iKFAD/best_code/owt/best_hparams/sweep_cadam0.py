from sweep_engine import run_optimizer_sweep

def cadam_params(trial):
    return {
        "h": trial.suggest_categorical("h", [1.3238940495127003e-05]),
        "gamma": trial.suggest_categorical("gamma", [0.]),
        "c": trial.suggest_categorical("c", [142783501.9232131]),
        "alpha": trial.suggest_categorical("alpha", [0.490932807956427]),
        "eps": 1e-8
    }

if __name__ == "__main__":
    run_optimizer_sweep("sweep_cadam0", "Cadam", cadam_params, n_trials=10)