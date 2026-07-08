# sweeps/sweep_cd.py
from sweep_engine import run_optimizer_sweep

def cd_params(trial):
    return {
        "h": trial.suggest_float("h", 1e-4, 5e-1, log=True),
        "gamma": trial.suggest_float("gamma", 1e-6, 1e1),
        "c": trial.suggest_float("c", 1e-1, 1e6, log=True)
    }

if __name__ == "__main__":
    run_optimizer_sweep("sweep_cubic", "cubic_damping_opt", cd_params)