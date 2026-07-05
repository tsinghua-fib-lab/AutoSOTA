"""
Baseline Model Fitters - Alternative models for comparison with cognitive model

Implements:
1. Autoregressive Logistic Regression (AR-k)
2. k-order Hidden Markov Model (HMM-k)
3. LSTM / GRU neural networks

All models follow the same interface as CognitiveFitter for easy comparison.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch not available. LSTM/GRU models will be disabled.")

from ..core import CONFIG
from .cognitive_fitter import FitResult


# =============================================================================
# 1. Autoregressive Logistic Regression
# =============================================================================

class AutoregressiveLogisticFitter:
    """
    Autoregressive Logistic Regression (AR-k)
    
    Predicts action based on k previous actions:
    P(a_t | a_{t-1}, ..., a_{t-k}) = sigmoid(w_0 + w_1*a_{t-1} + ... + w_k*a_{t-k})
    """
    
    def __init__(self, df: pd.DataFrame, order: int = 1):
        """
        Args:
            df: DataFrame with columns ['action', 'group', 'trial', 'file_id']
            order: Number of previous actions to consider (k)
        """
        self.df = df[df['action'].isin(['Compliance', 'Refusal'])].copy()
        if self.df.empty:
            raise ValueError("DataFrame is empty after filtering")
        
        self.df['a_idx'] = self.df['action'].apply(lambda x: 1 if x == 'Compliance' else 0)
        self.order = order
        
    def _prepare_sequences(self, group_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare input sequences for a specific group
        
        Returns:
            X: (N, order) array of previous actions
            y: (N,) array of current actions
        """
        subset = self.df[self.df['group'] == group_name].copy()
        
        if 'file_id' in subset.columns and 'trial' in subset.columns:
            subset = subset.sort_values(['file_id', 'trial'])
        
        actions = subset['a_idx'].values
        file_ids = subset.get('file_id', pd.Series(np.zeros(len(subset)))).values
        
        X_list = []
        y_list = []
        
        prev_file = None
        history = []
        
        for i in range(len(actions)):
            if file_ids[i] != prev_file:
                history = []
                prev_file = file_ids[i]
            
            if len(history) >= self.order:
                X_list.append(history[-self.order:])
                y_list.append(actions[i])
            
            history.append(actions[i])
        
        if not X_list:
            return None, None
        
        return np.array(X_list), np.array(y_list)
    
    def _nll_loss(self, weights: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """
        Negative log-likelihood for logistic regression
        
        Args:
            weights: [w_0, w_1, ..., w_k] where w_0 is bias
            X: (N, k) previous actions
            y: (N,) current actions
        """
        logits = weights[0] + np.dot(X, weights[1:])
        
        log_probs_1 = logits - np.logaddexp(0, logits)
        log_probs_0 = -np.logaddexp(0, logits)
        
        nll = -np.mean(y * log_probs_1 + (1 - y) * log_probs_0)
        
        return nll
    
    def _behavior_entropy(self, y: np.ndarray) -> float:
        """Calculate binary action entropy"""
        eps = 1e-8
        p = y.mean()
        return -p * np.log(p + eps) - (1 - p) * np.log(1 - p + eps)
    
    def fit_scenario(self, group_name: str) -> Optional[FitResult]:
        """
        Fit AR-k model for a specific scenario group
        
        Returns:
            FitResult with parameters and metrics
        """
        X, y = self._prepare_sequences(group_name)
        
        if X is None or len(X) < 5:
            return None
        
        n_params = self.order + 1
        x0 = np.zeros(n_params)
        
        best_res = None
        best_nll = np.inf
        
        for _ in range(3):
            x0_random = x0 + 0.1 * np.random.randn(n_params)
            
            res = minimize(
                self._nll_loss,
                x0_random,
                args=(X, y),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-3}
            )
            
            if res.success and res.fun < best_nll:
                best_nll = res.fun
                best_res = res
        
        if best_res is None:
            return None
        
        nll = best_res.fun * len(y)
        K = n_params
        bic = np.log(len(y)) * K + 2 * nll
        
        H = self._behavior_entropy(y)
        
        param_names = ['bias'] + [f'w_lag{i}' for i in range(1, self.order + 1)]
        params = {name: val for name, val in zip(param_names, best_res.x)}
        
        return FitResult(
            params=params,
            nll=nll,
            bic=bic,
            entropy=H,
            count=len(y),
            group=group_name
        )


# =============================================================================
# 2. Hidden Markov Model (k-order)
# =============================================================================

class HMMFitter:
    """
    k-order Hidden Markov Model
    
    Models behavior as transitions between k hidden states,
    each with emission probabilities for actions.
    """
    
    def __init__(self, df: pd.DataFrame, n_states: int = 2):
        """
        Args:
            df: DataFrame with columns ['action', 'group', 'trial', 'file_id']
            n_states: Number of hidden states (k)
        """
        self.df = df[df['action'].isin(['Compliance', 'Refusal'])].copy()
        if self.df.empty:
            raise ValueError("DataFrame is empty after filtering")
        
        self.df['a_idx'] = self.df['action'].apply(lambda x: 1 if x == 'Compliance' else 0)
        self.n_states = n_states
        
    def _prepare_sequence(self, group_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare action sequences grouped by file_id
        
        Returns:
            sequences: List of action arrays
            file_ids: Array of file IDs
        """
        subset = self.df[self.df['group'] == group_name].copy()
        
        if 'file_id' in subset.columns and 'trial' in subset.columns:
            subset = subset.sort_values(['file_id', 'trial'])
        
        sequences = []
        file_ids = []
        
        for fid in subset['file_id'].unique() if 'file_id' in subset.columns else [0]:
            seq = subset[subset['file_id'] == fid if 'file_id' in subset.columns else True]['a_idx'].values
            if len(seq) > 0:
                sequences.append(seq)
                file_ids.append(fid)
        
        return sequences, np.array(file_ids)
    
    def _forward_backward(self, seq: np.ndarray, pi: np.ndarray, 
                          A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Forward-backward algorithm for HMM
        
        Args:
            seq: Action sequence (T,)
            pi: Initial state distribution (k,)
            A: Transition matrix (k, k)
            B: Emission matrix (k, 2) - prob of action 0/1 in each state
            
        Returns:
            alpha: Forward probabilities (T, k)
            beta: Backward probabilities (T, k)
            log_likelihood: Log P(sequence)
        """
        T = len(seq)
        k = self.n_states
        
        alpha = np.zeros((T, k))
        beta = np.zeros((T, k))
        
        # Forward pass (in log space for numerical stability)
        log_alpha = np.zeros((T, k))
        log_alpha[0] = np.log(pi + 1e-10) + np.log(B[:, seq[0]] + 1e-10)
        
        for t in range(1, T):
            for j in range(k):
                log_alpha[t, j] = logsumexp(log_alpha[t-1] + np.log(A[:, j] + 1e-10)) + \
                                  np.log(B[j, seq[t]] + 1e-10)
        
        log_likelihood = logsumexp(log_alpha[-1])
        
        # Backward pass
        log_beta = np.zeros((T, k))
        log_beta[-1] = 0  # log(1) = 0
        
        for t in range(T-2, -1, -1):
            for i in range(k):
                log_beta[t, i] = logsumexp(
                    np.log(A[i, :] + 1e-10) + 
                    np.log(B[:, seq[t+1]] + 1e-10) + 
                    log_beta[t+1]
                )
        
        return np.exp(log_alpha - logsumexp(log_alpha, axis=1, keepdims=True)), \
               np.exp(log_beta - logsumexp(log_beta, axis=1, keepdims=True)), \
               log_likelihood
    
    def _baum_welch(self, sequences: List[np.ndarray], n_iterations: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Baum-Welch algorithm (EM for HMM)
        
        Returns:
            pi: Initial state distribution
            A: Transition matrix
            B: Emission matrix
        """
        k = self.n_states
        
        # Initialize parameters
        pi = np.ones(k) / k
        A = np.ones((k, k)) / k
        B = np.random.dirichlet(np.ones(2), size=k)
        
        for iteration in range(n_iterations):
            pi_acc = np.zeros(k)
            A_acc = np.zeros((k, k))
            B_acc = np.zeros((k, 2))
            
            for seq in sequences:
                T = len(seq)
                if T < 2:
                    continue
                
                alpha, beta, ll = self._forward_backward(seq, pi, A, B)
                
                # Compute gamma (state occupancy)
                gamma = alpha * beta
                gamma = gamma / (gamma.sum(axis=1, keepdims=True) + 1e-10)
                
                # Compute xi (transition occupancy)
                xi = np.zeros((T-1, k, k))
                for t in range(T-1):
                    for i in range(k):
                        for j in range(k):
                            xi[t, i, j] = alpha[t, i] * A[i, j] * B[j, seq[t+1]] * beta[t+1, j]
                    xi[t] = xi[t] / (xi[t].sum() + 1e-10)
                
                # Accumulate statistics
                pi_acc += gamma[0]
                for t in range(T-1):
                    A_acc += xi[t]
                for t in range(T):
                    B_acc[:, seq[t]] += gamma[t]
            
            # Update parameters
            pi = pi_acc / (pi_acc.sum() + 1e-10)
            A = A_acc / (A_acc.sum(axis=1, keepdims=True) + 1e-10)
            B = B_acc / (B_acc.sum(axis=1, keepdims=True) + 1e-10)
        
        return pi, A, B
    
    def _compute_nll(self, sequences: List[np.ndarray], 
                     pi: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
        """Compute total negative log-likelihood"""
        total_nll = 0.0
        
        for seq in sequences:
            _, _, ll = self._forward_backward(seq, pi, A, B)
            total_nll -= ll
        
        return total_nll
    
    def _behavior_entropy(self, sequences: List[np.ndarray]) -> float:
        """Calculate entropy across all sequences"""
        all_actions = np.concatenate(sequences)
        eps = 1e-8
        p = all_actions.mean()
        return -p * np.log(p + eps) - (1 - p) * np.log(1 - p + eps)
    
    def fit_scenario(self, group_name: str, n_iterations: int = 10) -> Optional[FitResult]:
        """
        Fit HMM for a specific scenario group
        
        Args:
            group_name: Name of the scenario group
            n_iterations: Number of EM iterations (default: 10)
        
        Returns:
            FitResult with parameters and metrics
        """
        sequences, _ = self._prepare_sequence(group_name)
        
        if len(sequences) == 0 or sum(len(s) for s in sequences) < 5:
            return None
        
        pi, A, B = self._baum_welch(sequences, n_iterations=n_iterations)
        
        nll = self._compute_nll(sequences, pi, A, B)
        
        n_params = (self.n_states - 1) + self.n_states * (self.n_states - 1) + self.n_states * (2 - 1)
        total_length = sum(len(s) for s in sequences)
        bic = np.log(total_length) * n_params + 2 * nll
        
        H = self._behavior_entropy(sequences)
        
        params = {
            'pi_0': pi[0],
            **{f'A_{i}_{j}': A[i, j] for i in range(self.n_states) for j in range(self.n_states)},
            **{f'B_{s}_compliance': B[s, 1] for s in range(self.n_states)}
        }
        
        return FitResult(
            params=params,
            nll=nll,
            bic=bic,
            entropy=H,
            count=total_length,
            group=group_name
        )


# =============================================================================
# 3. LSTM / GRU Neural Networks
# =============================================================================

if HAS_TORCH:
    class LSTMModel(nn.Module):
        """LSTM model for binary action prediction"""
        
        def __init__(self, input_size: int = 1, hidden_size: int = 16, 
                     num_layers: int = 1, dropout: float = 0.0):
            super(LSTMModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                               batch_first=True, dropout=dropout if num_layers > 1 else 0)
            self.fc = nn.Linear(hidden_size, 1)
            
        def forward(self, x):
            # x: (batch, seq_len, input_size)
            lstm_out, _ = self.lstm(x)
            out = self.fc(lstm_out[:, -1, :])
            return out
    
    class GRUModel(nn.Module):
        """GRU model for binary action prediction"""
        
        def __init__(self, input_size: int = 1, hidden_size: int = 16, 
                     num_layers: int = 1, dropout: float = 0.0):
            super(GRUModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
            self.fc = nn.Linear(hidden_size, 1)
            
        def forward(self, x):
            gru_out, _ = self.gru(x)
            out = self.fc(gru_out[:, -1, :])
            return out

    class LSTMFitter:
        """
        LSTM/GRU neural network fitter
        """
        
        def __init__(self, df: pd.DataFrame, model_type: str = 'lstm', 
                     hidden_size: int = 16, num_layers: int = 1,
                     seq_length: int = 5, device: str = 'auto'):
            """
            Args:
                df: DataFrame with columns ['action', 'group', 'trial', 'file_id']
                model_type: 'lstm' or 'gru'
                hidden_size: Hidden state size
                num_layers: Number of recurrent layers
                seq_length: Input sequence length
                device: 'cpu', 'cuda', or 'auto'
            """
            self.df = df[df['action'].isin(['Compliance', 'Refusal'])].copy()
            if self.df.empty:
                raise ValueError("DataFrame is empty after filtering")
            
            self.df['a_idx'] = self.df['action'].apply(lambda x: 1 if x == 'Compliance' else 0)
            self.model_type = model_type
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.seq_length = seq_length
            
            if device == 'auto':
                try:
                    if torch.cuda.is_available():
                        test_tensor = torch.zeros(1).cuda()
                        del test_tensor
                        self.device = torch.device('cuda')
                    else:
                        self.device = torch.device('cpu')
                except Exception:
                    self.device = torch.device('cpu')
            else:
                self.device = torch.device(device)
        
        def _prepare_sequences(self, group_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Prepare sequences for PyTorch
            
            Returns:
                X: (N, seq_length, 1) tensor
                y: (N,) tensor
            """
            subset = self.df[self.df['group'] == group_name].copy()
            
            if 'file_id' in subset.columns and 'trial' in subset.columns:
                subset = subset.sort_values(['file_id', 'trial'])
            
            actions = subset['a_idx'].values
            file_ids = subset.get('file_id', pd.Series(np.zeros(len(subset)))).values
            
            X_list = []
            y_list = []
            
            prev_file = None
            history = []
            
            for i in range(len(actions)):
                if file_ids[i] != prev_file:
                    history = []
                    prev_file = file_ids[i]
                
                if len(history) >= self.seq_length:
                    X_list.append(history[-self.seq_length:])
                    y_list.append(actions[i])
                
                history.append(actions[i])
            
            if not X_list:
                return None, None
            
            X = torch.FloatTensor(np.array(X_list)).unsqueeze(-1)
            y = torch.FloatTensor(np.array(y_list))
            
            return X, y
        
        def _train_model(self, X: torch.Tensor, y: torch.Tensor, 
                        n_epochs: int = 50, lr: float = 0.01) -> Tuple[nn.Module, float]:
            """
            Train LSTM/GRU model
            
            Returns:
                model: Trained model
                final_loss: Final training loss
            """
            if self.model_type == 'lstm':
                model = LSTMModel(input_size=1, hidden_size=self.hidden_size, 
                                 num_layers=self.num_layers)
            else:
                model = GRUModel(input_size=1, hidden_size=self.hidden_size, 
                                num_layers=self.num_layers)
            
            model = model.to(self.device)
            X = X.to(self.device)
            y = y.to(self.device)
            
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            
            dataset = TensorDataset(X, y)
            loader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            model.train()
            for epoch in range(n_epochs):
                for batch_X, batch_y in loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X).squeeze(-1)  # Only squeeze last dimension
                    # Ensure outputs has same shape as batch_y
                    if outputs.dim() == 0:
                        outputs = outputs.unsqueeze(0)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            model.eval()
            with torch.no_grad():
                outputs = model(X).squeeze(-1)  # Only squeeze last dimension
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                final_loss = criterion(outputs, y).item()
            
            return model, final_loss
        
        def _compute_nll(self, model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
            """Compute negative log-likelihood"""
            model.eval()
            with torch.no_grad():
                logits = model(X).squeeze()
                probs = torch.sigmoid(logits)
                
                log_probs = torch.where(y == 1, 
                                       torch.log(probs + 1e-10),
                                       torch.log(1 - probs + 1e-10))
                
                nll = -log_probs.sum().item()
            
            return nll
        
        def _count_parameters(self, model: nn.Module) -> int:
            """Count trainable parameters"""
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        def _behavior_entropy(self, y: torch.Tensor) -> float:
            """Calculate binary action entropy"""
            y_np = y.cpu().numpy()
            eps = 1e-8
            p = y_np.mean()
            return -p * np.log(p + eps) - (1 - p) * np.log(1 - p + eps)
        
        def fit_scenario(self, group_name: str, n_epochs: int = 50) -> Optional[FitResult]:
            """
            Fit LSTM/GRU for a specific scenario group
            
            Returns:
                FitResult with parameters and metrics
            """
            X, y = self._prepare_sequences(group_name)
            
            if X is None or len(X) < 10:
                return None
            
            model, final_loss = self._train_model(X, y, n_epochs=n_epochs)
            
            nll = self._compute_nll(model, X, y)
            
            n_params = self._count_parameters(model)
            bic = np.log(len(y)) * n_params + 2 * nll
            
            H = self._behavior_entropy(y)
            
            params = {
                'n_params': n_params,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'seq_length': self.seq_length,
                'model_type': self.model_type,
                'final_train_loss': final_loss
            }
            
            return FitResult(
                params=params,
                nll=nll,
                bic=bic,
                entropy=H,
                count=len(y),
                group=group_name
            )

else:
    class LSTMFitter:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for LSTMFitter. Install with: pip install torch")


# =============================================================================
# Batch Fitting Function
# =============================================================================

def batch_fit_baseline_models(
    df: pd.DataFrame,
    models: List[str] = ['ar', 'hmm', 'lstm'],
    ar_order: int = 1,
    hmm_states: int = 2,
    hmm_iterations: int = 10,
    lstm_hidden: int = 16,
    lstm_layers: int = 1,
    lstm_seq_length: int = 5,
    lstm_epochs: int = 50,
    lstm_device: str = 'auto',
    n_workers: int = 1,
    config: dict = None
) -> Dict[str, Dict[str, FitResult]]:
    """
    Fit multiple baseline models for all scenario groups
    
    Args:
        df: DataFrame with behavior data
        models: List of models to fit ('ar', 'hmm', 'lstm', 'gru')
        ar_order: Order for autoregressive model
        hmm_states: Number of states for HMM
        hmm_iterations: Number of EM iterations for HMM
        lstm_hidden: Hidden size for LSTM/GRU
        lstm_layers: Number of layers for LSTM/GRU
        lstm_seq_length: Sequence length for LSTM/GRU
        lstm_epochs: Number of training epochs for LSTM/GRU
        lstm_device: Device for LSTM/GRU ('cpu', 'cuda', or 'auto')
        n_workers: Number of parallel workers (default: 1, use >1 for parallel)
        config: Optional configuration dict
        
    Returns:
        Dict mapping model_name -> {group_name -> FitResult}
    """
    results = {}
    
    groups = df['group'].unique().tolist()
    
    def _fit_single_group_ar(args):
        """Helper function to fit AR for a single group"""
        fitter, group = args
        return group, fitter.fit_scenario(group)
    
    def _fit_single_group_hmm(args):
        """Helper function to fit HMM for a single group"""
        fitter, group, n_iterations = args
        return group, fitter.fit_scenario(group, n_iterations=n_iterations)
    
    def _fit_single_group_lstm(args):
        """Helper function to fit LSTM for a single group"""
        fitter, group, n_epochs = args
        return group, fitter.fit_scenario(group, n_epochs=n_epochs)
    
    if 'ar' in models:
        print("Fitting Autoregressive Logistic Regression...")
        fitter = AutoregressiveLogisticFitter(df, order=ar_order)
        results['AR'] = {}
        
        if n_workers > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                args_list = [(fitter, group) for group in groups]
                futures = list(tqdm(executor.map(_fit_single_group_ar, args_list), 
                                   total=len(groups), desc="AR Fitting (Parallel)"))
                for group, res in futures:
                    if res is not None:
                        results['AR'][group] = res
        else:
            for group in tqdm(groups, desc="AR Fitting"):
                res = fitter.fit_scenario(group)
                if res is not None:
                    results['AR'][group] = res
    
    if 'hmm' in models:
        print("Fitting Hidden Markov Model...")
        fitter = HMMFitter(df, n_states=hmm_states)
        results['HMM'] = {}
        
        if n_workers > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                args_list = [(fitter, group, hmm_iterations) for group in groups]
                futures = list(tqdm(executor.map(_fit_single_group_hmm, args_list), 
                                   total=len(groups), desc="HMM Fitting (Parallel)"))
                for group, res in futures:
                    if res is not None:
                        results['HMM'][group] = res
        else:
            for group in tqdm(groups, desc="HMM Fitting"):
                res = fitter.fit_scenario(group, n_iterations=hmm_iterations)
                if res is not None:
                    results['HMM'][group] = res
    
    if 'lstm' in models and HAS_TORCH:
        print("Fitting LSTM...")
        fitter = LSTMFitter(df, model_type='lstm', hidden_size=lstm_hidden, 
                           num_layers=lstm_layers, seq_length=lstm_seq_length,
                           device=lstm_device)
        results['LSTM'] = {}
        
        # Note: LSTM parallelization is tricky with GPU due to memory
        # Use parallel only for CPU mode
        if n_workers > 1 and lstm_device == 'cpu':
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                args_list = [(fitter, group, lstm_epochs) for group in groups]
                futures = list(tqdm(executor.map(_fit_single_group_lstm, args_list), 
                                   total=len(groups), desc="LSTM Fitting (Parallel)"))
                for group, res in futures:
                    if res is not None:
                        results['LSTM'][group] = res
        else:
            for group in tqdm(groups, desc="LSTM Fitting"):
                res = fitter.fit_scenario(group, n_epochs=lstm_epochs)
                if res is not None:
                    results['LSTM'][group] = res
    
    if 'gru' in models and HAS_TORCH:
        print("Fitting GRU...")
        fitter = LSTMFitter(df, model_type='gru', hidden_size=lstm_hidden, 
                           num_layers=lstm_layers, seq_length=lstm_seq_length,
                           device=lstm_device)
        results['GRU'] = {}
        
        # Note: GRU parallelization is tricky with GPU due to memory
        # Use parallel only for CPU mode
        if n_workers > 1 and lstm_device == 'cpu':
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                args_list = [(fitter, group, lstm_epochs) for group in groups]
                futures = list(tqdm(executor.map(_fit_single_group_lstm, args_list), 
                                   total=len(groups), desc="GRU Fitting (Parallel)"))
                for group, res in futures:
                    if res is not None:
                        results['GRU'][group] = res
        else:
            for group in tqdm(groups, desc="GRU Fitting"):
                res = fitter.fit_scenario(group, n_epochs=lstm_epochs)
                if res is not None:
                    results['GRU'][group] = res
    
    return results
