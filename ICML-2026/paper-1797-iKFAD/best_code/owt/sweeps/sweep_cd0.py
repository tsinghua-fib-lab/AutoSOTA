# sweeps/sweep_cd0.py
from sweep_engine import run_optimizer_sweep

def cd_params(trial):
    return {
        "h": trial.suggest_float("h", 1e-4, 1e-1, log=True),
        "gamma": trial.suggest_categorical("gamma", [0.]),
        "c": trial.suggest_float("c", 1e4, 1e12, log=True)
    }

if __name__ == "__main__":
    run_optimizer_sweep("sweep_cubic_0", "cubic_damping_opt", cd_params)