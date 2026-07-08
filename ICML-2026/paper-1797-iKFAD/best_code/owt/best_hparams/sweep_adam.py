# sweep_adam.py
from sweep_engine import run_optimizer_sweep

def adam_params(trial):
    return {
        "lr": trial.suggest_categorical("lr", [0.0006472299797549248]),
        "beta_1": trial.suggest_categorical("beta_1", [0.894474606218001]),
        "beta_2": trial.suggest_categorical("beta_2", [0.9945372904233089]),
        "eps": 1e-8
    }
if __name__ == "__main__":
    run_optimizer_sweep("sweep_adam", "ADAM", adam_params, n_trials=10)