# sweeps/sweep_cadam..py
from sweep_engine import run_optimizer_sweep

def cadam_params(trial):
    return {
        "h": trial.suggest_float("h", 1e-6, 1e-2, log=True),
        "gamma": trial.suggest_float("gamma", 1e-5, 1e1, log=True),
        "c": trial.suggest_float("c", 1e4, 1e10, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 1, log=True),
        "eps": 1e-8
    }

if __name__ == "__main__":
    run_optimizer_sweep("sweep_cadam", "Cadam", cadam_params)