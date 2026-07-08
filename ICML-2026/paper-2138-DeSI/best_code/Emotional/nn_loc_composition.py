# nn_loc_composition.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from loc_sphere import loc_sphe_geo_reg, sphe_geo_dist

# Only import generate_data if running as main (for testing)
try:
    from generate_data import generate_compositional_dataset
except ImportError:
    generate_compositional_dataset = None

# =========================================
#      Numerics & Geodesic Distance
# =========================================

def geodesic_distance(y1, y2):
    """
    Geodesic distance on sphere: d_g(y, y^*) = arccos(y^T y^*)
    y1: (..., m), y2: (..., m) unit vectors
    returns: (...) geodesic distance
    """
    return sphe_geo_dist(y1, y2)

def frechet_variance_geodesic(Y_list):
    """Compute Fréchet variance using geodesic distance"""
    n = len(Y_list)
    if n <= 1:
        return torch.tensor(0.0, device=Y_list[0].device)
    
    # Stack all Y vectors
    Y_stack = torch.stack(Y_list, dim=0)  # (n, m)
    
    # Compute Fréchet mean (geodesic mean on sphere)
    # For small sets, use iterative method or approximate with normalized arithmetic mean
    Y_mean = safe_normalize_sphere(Y_stack.mean(dim=0))
    
    # Compute distances to mean
    dists = torch.stack([geodesic_distance(Y_mean.unsqueeze(0), Y.unsqueeze(0)) for Y in Y_list])
    
    return dists.mean()

def safe_normalize_sphere(v):
    """Normalize to unit sphere"""
    norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return v / norm

def lfr_loss_selffit(X, Y_list, theta, h, requires_grad=True):
    """LFR loss with self-fitting using geodesic distance"""
    device = X.device
    n = X.shape[0]
    
    # Compute Z = <X_i, theta_i> for each sample
    Z = (X * theta).sum(dim=1)  # (n,)
    
    # Convert Y_list to tensor if needed
    if isinstance(Y_list, list):
        Y_tensor = torch.stack(Y_list, dim=0)  # (n, m)
    else:
        Y_tensor = Y_list
    
    # Pass h directly as tensor if it's trainable, otherwise as float
    if isinstance(h, torch.Tensor):
        h_val = h if requires_grad else h.item()
    else:
        h_val = h
    
    # Use local spherical geodesic regression
    # Pass h as tensor to preserve gradients during training
    # Note: loc_sphe_geo_reg takes bw as positional parameter
    # Set diff_through_solver based on requires_grad flag
    Y_hat = loc_sphe_geo_reg(
        Z, Y_tensor, Z, bw=h_val, 
        kernel='gauss',
        opt_steps=20,  # Number of optimization steps
        opt_step_size=0.2,  # Step size for Riemannian gradient descent
        diff_through_solver=requires_grad  # True for training, False for validation
    )  # (n, m)
    
    # Compute geodesic distances
    distances = []
    for i in range(n):
        dist = geodesic_distance(Y_tensor[i:i+1], Y_hat[i:i+1])
        distances.append(dist)
    
    return torch.stack(distances).mean()

# =========================================
#      Neural Network Model
# =========================================

class ThetaMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout_rate=0.3):
        super(ThetaMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.LeakyReLU()
        self.drop1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.LeakyReLU()
        self.drop2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden_dim, input_dim)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='leaky_relu')
        for layer in (self.fc1, self.fc2, self.fc3):
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.act1(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.act2(x)
        x = self.drop2(x)
        
        x = self.fc3(x)
        
        # Normalize to unit vector and fix sign ambiguity
        norm = torch.norm(x, dim=1, keepdim=True)
        theta = x / (norm + 1e-8)
        sign = torch.where(theta[:, :1] < 0, -1.0, 1.0)
        theta = theta * sign
        
        return theta

# =========================================
#      Main Training Function
# =========================================

def train_compositional_model(
    X_train, Y_train, X_val, Y_val, X_test, Y_test,
    batch_size=64,
    lr=0.5,
    num_epochs=1000,
    patience=20,
    delta=1e-4,
    lambda_reg=0.01,
    hidden_dim=64,
    dropout_rate=0.2,
    device=None
):
    """
    Train compositional model with local spherical geodesic regression.
    
    Parameters
    ----------
    X_train : torch.Tensor, shape (n_train, p)
        Training input features
    Y_train : list of torch.Tensor or torch.Tensor, shape (n_train, m)
        Training output vectors on sphere
    X_val : torch.Tensor, shape (n_val, p)
        Validation input features
    Y_val : list of torch.Tensor or torch.Tensor, shape (n_val, m)
        Validation output vectors on sphere
    X_test : torch.Tensor, shape (n_test, p)
        Test input features
    Y_test : list of torch.Tensor or torch.Tensor, shape (n_test, m)
        Test output vectors on sphere
    batch_size : int, default=64
        Batch size for training
    lr : float, default=0.5
        Learning rate
    num_epochs : int, default=1000
        Maximum number of training epochs
    patience : int, default=20
        Early stopping patience
    delta : float, default=1e-4
        Minimum improvement threshold for early stopping
    lambda_reg : float, default=0.01
        Regularization hyperparameter for 1/h term
    hidden_dim : int, default=64
        Hidden dimension for neural network
    dropout_rate : float, default=0.2
        Dropout rate
    device : torch.device, optional
        Device to use. If None, uses CUDA if available, else CPU.
    
    Returns
    -------
    theta_pred : torch.Tensor, shape (n_test, p)
        Predicted theta values for test set
    mspe : float
        Mean Squared Prediction Error on test set
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Ensure inputs are tensors and on correct device
    X_train = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    X_val = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    X_test = torch.as_tensor(X_test, dtype=torch.float32, device=device)
    
    # Convert Y to list of tensors if needed
    if isinstance(Y_train, torch.Tensor):
        Y_train = [Y_train[i] for i in range(Y_train.shape[0])]
    if isinstance(Y_val, torch.Tensor):
        Y_val = [Y_val[i] for i in range(Y_val.shape[0])]
    if isinstance(Y_test, torch.Tensor):
        Y_test = [Y_test[i] for i in range(Y_test.shape[0])]
    
    # Ensure Y tensors are on correct device
    Y_train = [torch.as_tensor(y, dtype=torch.float32, device=device) for y in Y_train]
    Y_val = [torch.as_tensor(y, dtype=torch.float32, device=device) for y in Y_val]
    Y_test = [torch.as_tensor(y, dtype=torch.float32, device=device) for y in Y_test]
    
    p = X_train.shape[1]
    
    # Normalize predictors using training statistics
    X_mean = X_train.mean(dim=0, keepdim=True)
    X_std = X_train.std(dim=0, keepdim=True).clamp_min(1e-8)
    X_train = (X_train - X_mean) / X_std
    X_val = (X_val - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std
    
    # Compute global Fréchet variance for normalization
    with torch.no_grad():
        frechet_var = frechet_variance_geodesic(Y_train)
        frechet_var = frechet_var + 1e-8
        # Ensure frechet_var is not NaN or zero
        if torch.isnan(frechet_var) or frechet_var < 1e-8:
            frechet_var = torch.tensor(1.0, device=device)
    
    # Create model
    model = ThetaMLP(input_dim=p, hidden_dim=hidden_dim, dropout_rate=dropout_rate).to(device)
    # Learnable bandwidth parameter, clamped to [0.2, 50.0] - minimum larger than 0.1
    # Use a reasonable initial value
    initial_h = 0.5  # Start with moderate bandwidth
    h_param = nn.Parameter(torch.tensor(initial_h, device=device))
    optimizer = optim.Adam(list(model.parameters()) + [h_param], lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    best_h = None
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            end_idx = min(i + batch_size, len(X_train))
            Xb = X_train[i:end_idx]
            Yb = Y_train[i:end_idx]
            
            optimizer.zero_grad()
            
            # Forward pass
            theta_b = model(Xb)
            h_value = torch.clamp(h_param, min=0.2, max=50.0)
            lfr = lfr_loss_selffit(Xb, Yb, theta_b, h_value, requires_grad=True)
            
            # Check for NaN values
            if torch.isnan(lfr):
                print(f"    Warning: NaN detected in LFR loss at batch {i}, skipping...")
                continue
            
            norm_loss = lfr / frechet_var
            # Regularization term - h_value is a tensor, so this preserves gradients
            loss = norm_loss + (1.0 / h_value) * lambda_reg
            
            # Check for NaN in final loss
            if torch.isnan(loss):
                print(f"    Warning: NaN detected in total loss at batch {i}, skipping...")
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        # Handle case where all batches were skipped
        if num_batches == 0:
            print(f"    Warning: All batches produced NaN in epoch {epoch+1}, skipping epoch...")
            avg_loss = float('inf')
        else:
            avg_loss = epoch_loss / num_batches
        
        # Validation evaluation
        with torch.no_grad():
            model.eval()
            theta_val = model(X_val)
            h_eval = torch.clamp(h_param.detach(), min=0.2, max=50.0)
            val_lfr = lfr_loss_selffit(X_val, Y_val, theta_val, h_eval, requires_grad=False)
            
            # Check for NaN in validation
            if torch.isnan(val_lfr):
                val_loss = torch.tensor(float('inf'), device=device)
            else:
                val_loss = (val_lfr / frechet_var) + (1.0 / h_eval.item()) * lambda_reg
                if torch.isnan(val_loss):
                    val_loss = torch.tensor(float('inf'), device=device)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping criterion
        current_val_loss = val_loss.item()
        
        # Print epoch information every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            h_current = torch.clamp(h_param, min=0.2, max=50.0).item()
            print(f"  Epoch {epoch+1:4d}/{num_epochs} | Train Loss: {avg_loss:.6f} | Val Loss: {current_val_loss:.6f} | LR: {current_lr:.6f} | h: {h_current:.4f} | Patience: {patience_counter}/{patience}")
        
        if current_val_loss < best_val_loss - delta:
            best_val_loss = current_val_loss
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_h = h_param.detach().cpu().clone()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)
        if best_h is not None:
            h_param.data = best_h.to(device)
    
    # Final evaluation on test set
    model.eval()
    with torch.no_grad():
        theta_test = model(X_test)
        h_eval = torch.clamp(h_param.detach(), min=0.2, max=50.0)
        
        # Calculate predictions and MSPE
        Y_train_tensor = torch.stack(Y_train, dim=0)
        theta_train_final = model(X_train)
        theta_test_final = model(X_test)
        
        Z_train = (X_train * theta_train_final).sum(dim=1)
        Z_test = (X_test * theta_test_final).sum(dim=1)
        
        Y_pred_test = loc_sphe_geo_reg(
            Z_train, Y_train_tensor, Z_test,
            bw=h_eval.item(), kernel='gauss',
            opt_steps=20, opt_step_size=0.2,
            diff_through_solver=False  # Test mode - no gradients needed
        )
        
        Y_test_tensor = torch.stack(Y_test, dim=0)
        
        # Calculate geodesic distances (for MSPE, we use squared distances)
        distances_squared = []
        for i in range(len(Y_test)):
            dist = geodesic_distance(Y_test_tensor[i:i+1], Y_pred_test[i:i+1])
            distances_squared.append(dist.item() ** 2)
        
        mspe = np.mean(distances_squared)  # Root Mean Squared Prediction Error
        
        # Print mean of predicted thetas
        theta_mean = theta_test.mean(dim=0)
        print(f"Mean of predicted thetas: {theta_mean.cpu().numpy()}")
        print(f"Individual predicted thetas shape: {theta_test.shape}")
        print(f"Final bandwidth (h): {h_eval.item():.6f}")
        print(f"MSPE: {mspe:.6f}")
    
    return theta_test, mspe


# =========================================
#      Example Usage / Testing
# =========================================

if __name__ == "__main__":
    # Example: Generate data and train
    if generate_compositional_dataset is None:
        print("Error: generate_data module not found. Cannot run example.")
        exit(1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate data
    n = 100
    X_np, Y_np, Z_np, theta_true = generate_compositional_dataset(n=n, seed=123)
    
    # Convert to torch tensors
    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    Y_list = [torch.tensor(Y_np[i], dtype=torch.float32, device=device) for i in range(n)]
    
    # Split data
    perm_cpu = torch.randperm(n)
    perm = perm_cpu.to(device)
    X = X[perm]
    Y_list = [Y_list[idx] for idx in perm_cpu.tolist()]
    
    train_size = int(0.4 * n)
    val_size = int(0.4 * n)
    test_size = n - train_size - val_size
    
    X_train = X[:train_size]
    X_val = X[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    
    Y_train = Y_list[:train_size]
    Y_val = Y_list[train_size:train_size + val_size]
    Y_test = Y_list[train_size + val_size:]
    
    # Train model
    theta_pred, mspe = train_compositional_model(
        X_train, Y_train, X_val, Y_val, X_test, Y_test,
        batch_size=64,
        lr=0.5,
        num_epochs=1000,
        patience=20,
        device=device
    )
    
    print(f"Test MSPE: {mspe:.6f}")
    print(f"Predicted theta shape: {theta_pred.shape}")
