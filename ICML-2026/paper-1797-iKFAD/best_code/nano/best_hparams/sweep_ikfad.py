from sweep_engine import run_optimizer_sweep

def ikfad_params(trial):
    return {
        "h": trial.suggest_categorical("h", [0.39054591977460545]),
        "alpha": trial.suggest_categorical("alpha", [0.03473546977454452]),
        "mu": trial.suggest_categorical("mu", [0.00017139736181773782]),
        "gamma": trial.suggest_categorical("gamma", [2.5840911396174533e-05])
    }

if __name__ == "__main__":
    run_optimizer_sweep("sweep_ikfad", "iKFAD", ikfad_params, n_trials=10)