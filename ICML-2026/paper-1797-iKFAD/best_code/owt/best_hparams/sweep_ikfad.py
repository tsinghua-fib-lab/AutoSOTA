from sweep_engine import run_optimizer_sweep

def ikfad_params(trial):
    return {
        "h": trial.suggest_categorical("h", [0.07634604793861383]),
        "alpha": trial.suggest_categorical("alpha", [0.6736885406163495]),
        "mu": trial.suggest_categorical("mu", [1.3856703363453663e-06]),
        "gamma": trial.suggest_categorical("gamma", [0.0009137551607410188])
    }

if __name__ == "__main__":
    run_optimizer_sweep("sweep_ikfad", "iKFAD", ikfad_params, n_trials=10)