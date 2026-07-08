# sweeps/sweep_cadam..py
from sweep_engine import run_optimizer_sweep

def cadam_params(trial):
    return {
        "h": trial.suggest_float("h", 1e-6, 1e-2, log=True),
        "gamma": trial.suggest_categorical("gamma", [0.]),
        "c": trial.suggest_float("c", 1e-1, 1e6, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 1, log=True),
        "eps": 1e-8
    }

if __name__ == "__main__":
    run_optimizer_sweep("sweep_cadam0", "Cadam", cadam_params)