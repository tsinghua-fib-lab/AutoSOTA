# sweep_adam.py
from sweep_engine import run_optimizer_sweep

def adam_params(trial):
    return {
        "lr": trial.suggest_categorical("lr", [0.0016802603535769646]),
        "beta_1": trial.suggest_categorical("beta_1", [0.8875699258194036]),
        "beta_2": trial.suggest_categorical("beta_2", [0.9265285254320775]),
        "eps": 1e-8
    }
if __name__ == "__main__":
    run_optimizer_sweep("sweep_adam", "ADAM", adam_params, n_trials=10)