import numpy as np
import xg as xgb
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class GPUQuantileRegressor(BaseEstimator, RegressorMixin):
    """
    A GPU-accelerated Quantile Regressor using XGBoost.
    
    This fits a separate regressor for each requested quantile.
    """
    def __init__(self, quantiles=[0.05, 0.5, 0.95], n_estimators=100, 
                 max_depth=6, learning_rate=0.1, device="cuda",
                 n_jobs=-1, random_state=42):
        self.quantiles = quantiles
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.device = device
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.models_ = {}

    def fit(self, X, y):
        """
        Fits one XGBoost model per quantile on the GPU.
        """
        X, y = check_X_y(X, y)
        
        # XGBoost requires DMatrix for optimized GPU training
        # We construct it once to save memory if possible, though 
        # distinct quantiles require distinct training loops.
        # Note: 'quantiles' support in XGBoost is via the objective param.
        
        for q in self.quantiles:
            # Configure parameters for GPU training
            params = {
                'objective': 'reg:quantileerror',
                'quantile_alpha': q,
                'tree_method': 'hist',       # Required for efficient GPU usage
                'device': self.device,       # 'cuda' for GPU
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'n_jobs': self.n_jobs,
                'random_state': self.random_state,
                'verbosity': 0
            }
            
            # If you strictly want "Random Forest" behavior (bagging) instead of Boosting:
            # params['num_parallel_tree'] = self.n_estimators
            # params['subsample'] = 0.8
            # params['colsample_bytree'] = 0.8
            # num_boost_round = 1
            
            # Standard Boosting approach (usually superior for regression accuracy):
            num_boost_round = self.n_estimators

            dtrain = xgb.DMatrix(X, label=y)
            model = xgb.train(params, dtrain, num_boost_round=num_boost_round)
            self.models_[q] = model
            
        return self

    def predict(self, X):
        """
        Predicts requested quantiles.
        Returns array of shape (n_samples, n_quantiles)
        """
        check_is_fitted(self, ['models_'])
        X = check_array(X)
        dtest = xgb.DMatrix(X)
        
        preds = []
        for q in self.quantiles:
            preds.append(self.models_[q].predict(dtest))
            
        return np.column_stack(preds)

# --- Usage Example ---

if __name__ == "__main__":
    # 1. Generate Synthetic Data (Sine wave with heteroscedastic noise)
    print("Generating data...")
    np.random.seed(42)
    X = np.sort(5 * np.random.rand(10000, 1), axis=0)
    y = np.sin(X).ravel()
    # Add noise that scales with X (heteroscedasticity)
    y += (0.5 + 0.5 * X.ravel()) * np.random.normal(size=len(X))
    
    # 2. Train GPU Quantile Regressor
    # We want the 5th, 50th (median), and 95th percentiles
    quantiles = [0.05, 0.5, 0.95]
    print(f"Training on GPU ({len(X)} samples)...")
    
    qrf = GPUQuantileRegressor(
        quantiles=quantiles,
        n_estimators=200, 
        max_depth=6, 
        device="cuda" # Ensure you have a GPU available
    )
    
    try:
        qrf.fit(X, y)
    except xgb.core.XGBoostError as e:
        print("\n[Error] GPU not found or XGBoost not compiled with GPU support.")
        print("Falling back to CPU for demonstration...")
        qrf.device = "cpu"
        qrf.fit(X, y)

    # 3. Predict
    print("Predicting...")
    y_pred = qrf.predict(X)

    # 4. Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(X, y, 'k.', markersize=3, alpha=0.3, label='Data')
    plt.plot(X, y_pred[:, 1], 'r-', lw=2, label='Median (50%)')
    plt.fill_between(X.ravel(), y_pred[:, 0], y_pred[:, 2], 
                     color='red', alpha=0.3, label='90% Prediction Interval')
    
    plt.title("GPU Accelerated Quantile Regression")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()