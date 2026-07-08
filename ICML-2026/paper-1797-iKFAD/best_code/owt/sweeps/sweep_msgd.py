# sweeps/sweep_msgd.py
from sweep_engine import run_optimizer_sweep

def msgd_params(trial):
    return {
        "lr": trial.suggest_float("lr", 1e-6, 1e-1, log=True),
        "momentum": trial.suggest_float("momentum", 0.5, 0.99, log=True)
    }

if __name__ == "__main__":
    run_optimizer_sweep("sweep_msgd", "mSGD", msgd_params)