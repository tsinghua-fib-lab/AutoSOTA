from sweep_engine import run_optimizer_sweep

def cadam_params(trial):
    return {
        "h": trial.suggest_categorical("h", [0.006775022516072726]),
        "gamma": trial.suggest_categorical("gamma", [7.532394852955041]),
        "c": trial.suggest_categorical("c", [3106413.359882668]),
        "alpha": trial.suggest_categorical("alpha", [0.44019686135920516]),
        "eps": 1e-8
    }

if __name__ == "__main__":
    run_optimizer_sweep("sweep_cadam", "Cadam", cadam_params, n_trials=10)