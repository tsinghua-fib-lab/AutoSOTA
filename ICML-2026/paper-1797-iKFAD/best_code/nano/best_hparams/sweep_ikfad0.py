from sweep_engine import run_optimizer_sweep

def ikfad_params(trial):
    return {
        "h": trial.suggest_categorical("h", [0.49412914916046896]),
        "alpha": trial.suggest_categorical("alpha", [2.1955477450604213]),
        "mu": trial.suggest_categorical("mu", [4.552149543615035e-06]),
        "gamma": trial.suggest_categorical("gamma", [0.])
    }

if __name__ == "__main__":
    run_optimizer_sweep("sweep_ikfad0", "iKFAD", ikfad_params, n_trials=10)