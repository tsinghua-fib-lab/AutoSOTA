# sweep.py
import os
import argparse
import optuna
import sys
import math

# Import the training logic from the modified train.py
from train import train_engine

def adam_params(trial):
    return {
        "lr": trial.suggest_float("lr", 1e-6, 1e-2, log=True),
        "beta_1": trial.suggest_float("beta_1", 0.85, 0.999, log=True),
        "beta_2": trial.suggest_float("beta_2", 0.85, 0.999, log=True),
        "eps": 1e-8
    }

def cadam_params(trial):
    return {
        "h": trial.suggest_float("h", 1e-6, 1e-2, log=True),
        "gamma": trial.suggest_float("gamma", 1e-5, 1e1, log=True),
        "c": trial.suggest_float("c", 1e4, 1e10, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 1, log=True),
        "eps": 1e-8
    }

def cadam0_params(trial):
    return {
        "h": trial.suggest_float("h", 1e-6, 1e-2, log=True),
        "gamma": trial.suggest_categorical("gamma", [0.]),
        "c": trial.suggest_float("c", 1e4, 1e10, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 1, log=True),
        "eps": 1e-8
    }

def cd_params(trial):
    return {
        "h": trial.suggest_float("h", 1e-4, 1e-1, log=True),
        "gamma": trial.suggest_float("gamma", 1e-5, 1e1, log=True),
        "c": trial.suggest_float("c", 1e4, 1e10, log=True)
    }

def cd0_params(trial):
    return {
        "h": trial.suggest_float("h", 1e-4, 1e-1, log=True),
        "gamma": trial.suggest_categorical("gamma", [0.]),
        "c": trial.suggest_float("c", 1e4, 1e10, log=True)
    }

def ikfad_params(trial):
    return {
        "h": trial.suggest_float("h", 1e-6, 1e-1, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 1, log=True),
        "mu": trial.suggest_float("mu", 1e-8, 10, log=True),
        "gamma": trial.suggest_float("gamma", 1e-5, 10., log=True)
    }

def ikfad0_params(trial):
    return {
        "h": trial.suggest_float("h", 1e-6, 1e-1, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 1, log=True),
        "mu": trial.suggest_float("mu", 1e-8, 10, log=True),
        "gamma": trial.suggest_categorical("gamma", [0.])
    }

def msgd_params(trial):
    return {
        "lr": trial.suggest_float("lr", 1e-6, 1e-1, log=True),
        "momentum": trial.suggest_float("momentum", 0.5, 0.99, log=True)
    }

def get_hp_from_trial(trial, optimizer_name):
    """
    Maps Optuna trial parameters to the 'hp' dictionary format 
    expected by train.py
    """
    hp = {}
    
    if optimizer_name == 'adam':
        p = adam_params(trial)
        hp['lr'] = p['lr']
        hp['h'] = p['lr'] 
        hp['betas'] = (p['beta_1'], p['beta_2'])
        
    elif optimizer_name == 'cadam':
        p = cadam_params(trial)
        hp = p 
    
    elif optimizer_name == 'cadam0':
        p = cadam0_params(trial)
        hp = p 
        
    elif optimizer_name == 'cd':
        p = cd_params(trial)
        hp = p 
    
    elif optimizer_name == 'cd0':
        p = cd0_params(trial)
        hp = p 
        
    elif optimizer_name == 'ikfad':
        p = ikfad_params(trial)
        hp = p 

    elif optimizer_name == 'ikfad0':
        p = ikfad0_params(trial)
        hp = p 
        
    elif optimizer_name == 'msgd':
        p = msgd_params(trial)
        hp = p 
    
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported in sweep.")
    
    return hp

def objective(trial, args):
    # 1. Generate Hyperparameters
    hp = get_hp_from_trial(trial, args.optimizer)
    
    # 2. Run Training Engine
    try:
        # train_engine now returns best_val_loss always
        result = train_engine(
            optim_name=args.optimizer,
            hp=hp,
            num_epochs=args.epochs,
            device=args.device,
            metric='loss', 
            seed=args.seed
        )
        
        # Prune NaNs
        if math.isnan(result):
            print(f"Trial {trial.number} resulted in NaN. Pruning...")
            raise optuna.TrialPruned()
            
    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        # Return infinity to penalize failure in minimization
        return float('inf')

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='QNLI Optuna Sweep')
    parser.add_argument('--optimizer', type=str, required=True, 
                        choices=['adam', 'cadam', 'cadam0', 'cd', 'cd0', 'ikfad', 'ikfad0', 'msgd'],
                        help='Optimizer to sweep')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--epochs', type=int, default=2, help='Epochs per trial')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--study_name', type=str, default="CD", help='Optuna study name')
    parser.add_argument('--storage', type=str, default=None, help='Database URL for distributed storage')

    args = parser.parse_args()

    if args.study_name and not args.storage:
        print("Warning: Study name provided without storage URL. Study will be in-memory only.")

    print(f"Starting sweep for optimizer: {args.optimizer}")
    
    study = optuna.create_study(
        direction="minimize", # Minimize Validation Loss
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True
    )
    
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)

    print("-" * 50)
    print("Sweep Complete.")
    print(f"Best value (Min Valid Loss): {study.best_value:.4f}")
    print("Best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("-" * 50)