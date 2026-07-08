# sweep_msgd.py
from sweep_engine import run_optimizer_sweep

def msgd_params(trial):
    return {
        "lr": trial.suggest_categorical("lr", [0.09790562385806308]),
        "momentum": trial.suggest_categorical("momentum", [0.900536843782432])
    }

if __name__ == "__main__":
    run_optimizer_sweep("sweep_msgd", "mSGD", msgd_params, n_trials=10)