from sweep_engine import run_optimizer_sweep

def ikfad_params(trial):
    return {
        "h": trial.suggest_categorical("h", [0.09956249615058001]),
        "alpha": trial.suggest_categorical("alpha", [0.04756111942848695]),
        "mu": trial.suggest_categorical("mu", [1.0429248312626405e-05]),
        "gamma": trial.suggest_categorical("gamma", [0.0])
    }

if __name__ == "__main__":
    run_optimizer_sweep("sweep_ikfad", "iKFAD", ikfad_params, n_trials=10)