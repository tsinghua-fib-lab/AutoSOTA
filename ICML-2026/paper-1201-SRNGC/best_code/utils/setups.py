from model.models import LSTMModel, MLPModel, ResidualMLP, TSMixer, CausalSimpleMamba, SRUModel
from model.imp_measure import Shapley_Value, Layer_Weight
from model.penalty import Shapley_Penalty, Weight_Penalty
import torch
import numpy as np
import random
import os


def build_model(model_type, dim, lag, params, device):
    """Instantiate the requested model without modifying experiment semantics."""

    hidden_dim = params['hidden_dim']
    num_layers = params['layers']
    dropout = params['dropout']

    if model_type == 'MLP':
        model = MLPModel(dim=dim, lag=lag, hidden_dim=hidden_dim, num_layers=num_layers, componentwise=False, dropout=dropout)
    elif model_type == 'cMLP':
        model = MLPModel(dim=dim, lag=lag, hidden_dim=hidden_dim, num_layers=num_layers, componentwise=True, dropout=dropout)
    elif model_type == 'LSTM':
        model = LSTMModel(dim=dim, lag=lag, hidden_dim=hidden_dim, num_layers=num_layers, componentwise=False, dropout=dropout)
    elif model_type == 'cLSTM':
        model = LSTMModel(dim=dim, lag=lag, hidden_dim=hidden_dim, num_layers=num_layers, componentwise=True, dropout=dropout)
    elif model_type == 'ResidualMLP':
        model = ResidualMLP(input_dim=dim*lag, output_dim=dim, layers=num_layers, hidden_dim=hidden_dim, dropout=dropout)
    elif model_type == 'TSMixer':
        model = ResidualMLP(input_dim=dim*lag, output_dim=dim, layers=0, hidden_dim=hidden_dim, dropout=dropout)
    elif model_type == 'SimpleMamba':
        model = ResidualMLP(input_dim=dim*lag, output_dim=dim, layers=0, hidden_dim=hidden_dim, dropout=dropout)
    elif model_type == 'SRU':
        A = [0.0, 0.5, 0.9, 0.99]
        model = SRUModel(dim=dim, lag=lag, dim_stats=hidden_dim, dim_rec_stats_feedback=hidden_dim, dim_final_stats=hidden_dim, A=A, componentwise=False, dropout=dropout)
    else:
        model = None
        raise ValueError(f"Unknown model_type: {model_type}")
    
    return model.to(device)


def setup_penalties(model_type, penalty_type, device):
    """Build the structural regularizer used during training."""

    if penalty_type.startswith('Fast_Shap'):
        # Fast_Shap_3/Fast_Shap_5 are used by the sensitivity script. The
        # original Fast_Shap name remains the one-projection variant.
        num_proj = int(penalty_type.rsplit('_', 1)[-1]) if penalty_type != 'Fast_Shap' else 1
        penalty = Shapley_Penalty(num_proj=num_proj, approx=True, individual_effect_only=False, device=device)
    elif penalty_type == 'Shapley':
        penalty = Shapley_Penalty(num_proj=-1, approx=False, individual_effect_only=False, device=device)
    elif penalty_type == 'Jacob_F':
        penalty = Shapley_Penalty(num_proj=1,approx=True, individual_effect_only=True, device=device)
    elif penalty_type == 'Jacob_L1':
        penalty = Shapley_Penalty(num_proj=-1, approx=False, individual_effect_only=True, device=device)
    elif penalty_type == 'Layer_Weight':
        penalty = Weight_Penalty(model_type)
    else:
        raise ValueError(f"Unknown penalty_type: {penalty_type}")

    return penalty.to(device)


def setup_importance(model_type, importance_type, device):
    """Build the post-training importance estimator."""

    if importance_type == 'Shapley':
        importance = Shapley_Value(individual_effect_only=False, device=device)
    elif importance_type == 'Jacobian':
        importance = Shapley_Value(individual_effect_only=True, device=device)
    elif importance_type == 'Layer_Weight':
        importance = Layer_Weight(model_type)
    else:
        raise ValueError(f"Unknown importance_type: {importance_type}")
    
    return importance


def set_seed(seed):
    """Seed Python, NumPy, PyTorch, and cuDNN for reproducible training."""

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
