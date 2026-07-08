from sweep_engine import run_optimizer_sweep

def cadam_params(trial):
    return {
        "h": trial.suggest_categorical("h", [0.005244120925537811]),
        "gamma": trial.suggest_categorical("gamma", [0.]),
        "c": trial.suggest_categorical("c", [702368.867276113]),
        "alpha": trial.suggest_categorical("alpha", [0.011978038934567422]),
        "eps": 1e-8
    }

if __name__ == "__main__":
    run_optimizer_sweep("sweep_cadam0", "Cadam", cadam_params, n_trials=10)