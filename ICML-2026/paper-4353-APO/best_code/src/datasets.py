import numpy as np
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.interpolate import splev, splrep 

class BaseSyntheticDataset:
    """ Base Dataset Class """
    
    # --- MODIFICATION: Added n_plots parameter ---
    def __init__(self, n_pool=5000, test_ratio=0.3, noise_std=0.1, seed=42, n_plots=5):
        self.rng = np.random.RandomState(seed)
        self.noise_std = noise_std
        if not hasattr(self, "dim_x") or self.dim_x is None:
            self.dim_x = 1
        
        # Calculate n_test
        n_test = int(n_pool * test_ratio)
        if n_test == 0:
            print(f"Warning: n_pool * test_ratio resulted in 0 test samples. Setting n_test=1.")
            n_test = 1
            
        print(f"Dataset Info: n_pool={n_pool}, test_ratio={test_ratio} -> n_test={n_test}")

        # Create data pools
        self.X_pool, self.X_test = self._create_pools(n_pool, n_test)
        
        # Pre-compute true values for f and t*
        print("Pre-computing true values for pool and test sets...")
        self.t_star_pool = self.get_t_star(self.X_pool)
        self.f_at_t_star_pool = self.get_f(self.X_pool, self.t_star_pool)
        
        self.t_star_test = self.get_t_star(self.X_test)
        self.f_at_t_star_test = self.get_f(self.X_test, self.t_star_test)
        
        # --- MODIFICATION: Automatic plotting during initialization ---
        if n_plots > 0:
            self._plot_samples(n_plots)
            
        print("Initialization complete.")

    def _create_pools(self, n_pool, n_test):
        """
        [Overridable] Default implementation for fully synthetic datasets.
        """
        print("Using default _create_pools (generating new data)...")
        X_pool = self._generate_x(n_pool)
        X_test = self._generate_x(n_test)
        return X_pool, X_test

    def _generate_x(self, n):
        raise NotImplementedError

    def get_f(self, x, t):
        raise NotImplementedError

    def get_t_star(self, x):
        raise NotImplementedError

    def observe(self, x, t):
        """ Observe y = f(x,t) + epsilon """
        f_val = self.get_f(x, t)
        noise = self.rng.normal(0, self.noise_std, size=f_val.shape)
        return f_val + noise
    
    # --- MODIFICATION: New plotting method ---
    def _plot_samples(self, n_plots):
        """
        Randomly select n_plots samples to plot Dose-Response curves and save.
        Path: plots/{DatasetName}/dose_curves.png
        """

        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['mathtext.fontset'] = 'stix'

        dataset_name = self.__class__.__name__
        save_dir = os.path.join("plots", dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Plotting {n_plots} sample curves to '{save_dir}'...")
        
        # Randomly select indices from X_pool
        available_n = self.X_pool.shape[0]
        if n_plots > available_n:
            n_plots = available_n
        indices = self.rng.choice(available_n, n_plots, replace=False)
        
        t_grid = np.linspace(0, 1, 100) # (100,)
        
        plt.figure(figsize=(4.3, 3.7))
        
        for i, idx in enumerate(indices):
            # Get single x: (1, dim_x)
            x_i = self.X_pool[idx : idx+1] 
            
            # Repeat x to match t_grid length
            # x_repeated: (100, dim_x)
            x_repeated = np.repeat(x_i, 100, axis=0)
            
            # Calculate y = f(x, t)
            y_vals = self.get_f(x_repeated, t_grid)
            
            # Find peak of the curve
            peak_idx = np.argmax(y_vals)
            t_star = t_grid[peak_idx]
            y_star = y_vals[peak_idx]
            
            # Plot
            plt.plot(t_grid, y_vals, label=f'Sample {i+1}', alpha=0.7, linewidth=2)
            plt.scatter(t_star, y_star, s=30, zorder=5) # Mark peak
            
        plt.xlabel('Dose (t)', fontsize=14)
        plt.ylabel('Outcome (y)', fontsize=14)
        plt.legend(loc='best') 
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, "dose_curves.pdf")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Plot saved to {save_path}")

    @staticmethod
    def _ensure_2d(x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        return x


class SemiSynthNews(BaseSyntheticDataset):
    """
    Semi-synthetic News Dataset (V15-Fix SparseVC, MonteCarlo t*)
    """
    
    DATA_PATH = "raw/news_pp_V2.npy" 

    # --- MODIFICATION: Passing n_plots ---
    def __init__(self, n_pool=5000, test_ratio=0.3, noise_std=0.1, seed=42, n_plots=5):
        print(f"Initializing SemiSynthNews (d=501, V15-Fix SparseVC, MonteCarlo t*)...")
        
        self.rng = np.random.RandomState(seed) 
        self.X_full_norm, self.dim_x = self._load_data()
        self.total_rows = self.X_full_norm.shape[0] 
        
        rng_weights = np.random.RandomState(seed)
        
        self.w1 = self._normalized_weights(rng_weights, 4) 
        self.w2 = self._normalized_weights(rng_weights, 4) 
        self.w3 = self._normalized_weights(rng_weights, 4) 

        self.cate_idx1 = np.arange(12, 16)
        self.cate_mean1 = np.mean(self.X_full_norm[:, self.cate_idx1])
        self.alpha = 5.0

        # Call parent constructor
        super().__init__(n_pool=n_pool, test_ratio=test_ratio, noise_std=noise_std, seed=seed, n_plots=n_plots)
    
    def _normalized_weights(self, rng, dim):
        w_raw = rng.normal(size=(dim,))
        return w_raw / np.linalg.norm(w_raw)

    def _load_data(self):
        if not os.path.exists(self.DATA_PATH):
            print("="*60)
            print(f"Error: Data file not found '{self.DATA_PATH}'")
            print("Please run 'news_preprocessing.py' first")
            print("="*60)
            raise FileNotFoundError(f"Missing {self.DATA_PATH}.")
        
        try:
            X = np.load(self.DATA_PATH)
            X_log = np.log1p(X)
            X_min = X_log.min(axis=0)
            X_max = X_log.max(axis=0)
            X_norm = (X_log - X_min) / (X_max - X_min + 1e-9)
            dim_x = X_norm.shape[1]
            print(f"Loaded and normalized {X_norm.shape[0]} real covariates (d={dim_x}) from '{self.DATA_PATH}'.")
            return X_norm, dim_x
        except Exception as e:
            print(f"Error processing {self.DATA_PATH}: {e}")
            raise

    def _create_pools(self, n_pool, n_test):
        print(f"Using SemiSynthNews _create_pools (sampling w/o replacement)...")
        print(f"Total available unique samples: {self.total_rows}")

        total_requested = n_pool + n_test
        if total_requested > self.total_rows:
            raise ValueError(
                f"Requested samples ({total_requested}) exceeds available rows ({self.total_rows})."
            )
            
        all_indices = self.rng.permutation(self.total_rows)
        
        pool_indices = all_indices[:n_pool]
        test_indices = all_indices[n_pool : total_requested]
        
        X_pool = self.X_full_norm[pool_indices, :]
        X_test = self.X_full_norm[test_indices, :]
        
        print(f"Created non-overlapping X_pool (shape: {X_pool.shape}) and X_test (shape: {X_test.shape})")
        return X_pool, X_test

    def _generate_x(self, n):
        raise NotImplementedError(
            "_generate_x should not be called directly in SemiSynthNews."
        )
    
    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))

    def _get_coeffs(self, x):
        x = np.atleast_2d(x)
        g1 = 10.0 * (x[:, 0:4] @ self.w1)
        g2_nonlinear = np.cos(2 * np.pi * x[:, 4]) * np.sin(2 * np.pi * x[:, 5]) + \
                       np.cos(2 * np.pi * x[:, 6]) * np.sin(2 * np.pi * x[:, 7])
        g2 = 40.0 * self._sigmoid(g2_nonlinear)
        g3_nonlinear = np.cos(2 * np.pi * x[:, 8]) * np.sin(2 * np.pi * x[:, 9]) + \
                       np.cos(2 * np.pi * x[:, 10]) * np.sin(2 * np.pi * x[:, 11])
        g3 = 20.0 * self._sigmoid(g3_nonlinear) - 10.0 
        return g1, g2, g3

    def get_t_star(self, x):
        x = np.atleast_2d(x)
        n_samples = x.shape[0]
        t_grid = np.linspace(0, 1, 101).reshape(1, -1)
        x_expanded = np.repeat(x, 101, axis=0)
        t_expanded = np.tile(t_grid, (n_samples, 1)).flatten()
        f_values_flat = self.get_f(x_expanded, t_expanded)
        f_values_grid = f_values_flat.reshape(n_samples, 101)
        best_t_indices = np.argmax(f_values_grid, axis=1)
        t_star_batch = t_grid.flatten()[best_t_indices]
        return t_star_batch

    def c(self, x):
        x = np.atleast_2d(x)
        mean_cate1 = np.mean(x[:, self.cate_idx1], axis=1)
        return 0.4 + 0.4 * self._sigmoid((mean_cate1 - self.cate_mean1) * self.alpha)
        
    def get_f(self, x, t):
        t = np.clip(t.flatten(), 0, 1) 
        x = np.atleast_2d(x)
        g1, g2, g3 = self._get_coeffs(x)
        center = self.c(x)
        t_offset = t - center
        concave_term = np.cosh(t_offset) - 1
        f_val = g1 * center - g2 * concave_term + g3 * t_offset
        return f_val


class HardNonLinear8D(BaseSyntheticDataset):
    """
    Hard Non-Linear 8D Dataset
    """

    # --- MODIFICATION: Passing n_plots ---
    def __init__(self, n_pool=5000, test_ratio=0.3, noise_std=0.1, seed=42, n_plots=5):
        print(f"Initializing HardNonLinear8D (d=8, Highly Non-linear, Variable Curvature)...")
        self.dim_x = 8
        
        rng_weights = np.random.RandomState(seed)
        self.W1 = rng_weights.uniform(-1.5, 1.5, size=(self.dim_x, 2)) 
        self.W2 = rng_weights.uniform(-1.0, 1.0, size=(self.dim_x, 2))
        self.bias = rng_weights.uniform(-1, 1, size=(2,))

        self.w_center = rng_weights.uniform(-1, 1, size=(self.dim_x, 1))
        self.w_sharp  = rng_weights.uniform(-1, 1, size=(self.dim_x, 1))
        
        # Normalize weights
        self.w_center /= np.linalg.norm(self.w_center)
        self.w_sharp /= np.linalg.norm(self.w_sharp)

        super().__init__(n_pool=n_pool, test_ratio=test_ratio, noise_std=noise_std, seed=seed, n_plots=n_plots)

    def _generate_x(self, n):
        return self.rng.uniform(-1, 1, size=(n, self.dim_x))

    def _get_params(self, x):
        x = np.atleast_2d(x)
        h1 = np.sin(3.0 * (x @ self.W1) + self.bias) 
        h2 = np.cos(1.5 * (x @ self.W2))
        combined = np.sum(h1 + h2, axis=1) 
        t_star = 0.5 + 0.2 * np.tanh(combined)

        center_logit = np.sin(x @ self.w_center)
        t_star = 0.1 + 0.7 * (1.0 / (1.0 + np.exp(-center_logit * 5))).flatten()
        
        x_norm = np.linalg.norm(x, axis=1)
        k_val = 20.0 + 100 * np.clip((x_norm - 1.0), 0, 1)
        return t_star, k_val

    def get_t_star(self, x):
        t_star, _ = self._get_params(x)
        return t_star

    def get_f(self, x, t):
        t = np.clip(t.flatten(), 0, 1)
        t_star, k_val = self._get_params(x)
        t_offset = t - t_star
        concave_term = np.cosh(t_offset) - 1
        f_val = 10.0 - k_val * concave_term
        return f_val

class ComplexSharpConcave(BaseSyntheticDataset):
    """
    Complex Sharp Concave 8D Dataset
    
    Design Goals:
    1. Difficult to predict t*(x): determined by a 2-layer random NN with Sin activation.
    2. Asymmetric strong concavity: different slopes on left/right of the peak.
    3. Dynamic sharpness: extremely sharp peaks for some x.
    """

    def __init__(self, n_pool=5000, test_ratio=0.3, noise_std=0.1, seed=42, n_plots=5):
        print(f"Initializing ComplexSharpConcave (d=8, Neural t*, Asymmetric)...")
        self.dim_x = 8
        
        # --- 1. Random NN weights for complex t* ---
        rng_weights = np.random.RandomState(seed)
        
        # Layer 1: 8 -> 16
        self.W1 = rng_weights.uniform(-1.0, 1.0, size=(self.dim_x, 16))
        self.b1 = rng_weights.uniform(-1.0, 1.0, size=(16,))
        
        # Layer 2: 16 -> 1
        self.W2 = rng_weights.uniform(-1.0, 1.0, size=(16, 1))
        self.b2 = rng_weights.uniform(-0.5, 0.5, size=(1,))

        self.w_center = rng_weights.uniform(-1, 1, size=(self.dim_x, 1))
        self.w_sharp  = rng_weights.uniform(-1, 1, size=(self.dim_x, 1))
        
        # Normalize
        self.w_center /= np.linalg.norm(self.w_center)
        self.w_sharp /= np.linalg.norm(self.w_sharp)
        
        # --- 2. Asymmetry and Sharpness weights ---
        self.w_skew = rng_weights.uniform(-0.5, 0.5, size=(self.dim_x, 1))
        self.w_sharp = rng_weights.uniform(-0.5, 0.5, size=(self.dim_x, 1))

        super().__init__(n_pool=n_pool, test_ratio=test_ratio, noise_std=noise_std, seed=seed, n_plots=n_plots)

    def _generate_x(self, n):
        # Generate x in [-2, 2]
        return self.rng.uniform(-2, 2, size=(n, self.dim_x))

    def _neural_t_star(self, x):
        """ Highly non-linear t* via random NN """
        h = (x @ self.W1) + self.b1
        out = (h @ self.W2) + self.b2
        # Compress to [0.1, 0.9]
        t_star = 0.1 + 0.8 * (1.0 / (1.0 + np.exp(-out.flatten())))
        return t_star

    def _get_params(self, x):
        x = np.atleast_2d(x)
        
        # 1. Compute t*
        t_star = self._neural_t_star(x)

        center_logit = x @ self.w_center
        
        # 2. Compute Skewness (alpha_L, alpha_R)
        skew_logit = (x @ self.w_skew).flatten()
        base_skew = 1.0 / (1.0 + np.exp(-skew_logit)) + 0.5 
        
        alpha_L = base_skew 
        alpha_R = 2.0 - base_skew
        
        # 3. Compute overall Sharpness [20, 150]
        sharp_logit = (x @ self.w_sharp).flatten()
        sharpness = 20.0 + 130.0 * (1.0 / (1.0 + np.exp(-sharp_logit)))
        
        return t_star, alpha_L, alpha_R, sharpness

    def get_t_star(self, x):
        t_star, _, _, _ = self._get_params(x)
        return t_star

    def get_f(self, x, t):
        """
        f(x,t) = 10 - Sharpness * ( AsymCosh(t - t*) - 1 )
        """
        t = np.clip(t.flatten(), 0, 1)
        t_star, alpha_L, alpha_R, sharpness = self._get_params(x)
        
        t_diff = t - t_star
        
        # Asymmetric Cosh
        alpha_eff = np.where(t_diff < 0, alpha_L, alpha_R)
        scaled_diff = t_diff * alpha_eff
        
        concave_term = np.cosh(scaled_diff) - 1
        f_val = 10.0 - sharpness * concave_term
        
        return f_val

class SimpleWavy8D(BaseSyntheticDataset):
    """
    Simple Wavy Strong Concave 8D Dataset
    
    Formula:
    f(x,t) = 10 - k(x) * (cosh(t - t_center(x)) - 1) + 2.0 * sin(6 * pi * t)
    
    Features:
    1. Dynamic interference between Cosh envelope and Sin wave.
    2. True t* is found via grid search due to sine interference.
    """

    def __init__(self, n_pool=5000, test_ratio=0.3, noise_std=0.1, seed=42, n_plots=5):
        print(f"Initializing SimpleWavy8D (d=8, Formula: Cosh + Sin, MC t*)...")
        self.dim_x = 8
        
        rng_weights = np.random.RandomState(seed)
        self.w_center = rng_weights.uniform(-1, 1, size=(self.dim_x, 1))
        self.w_sharp  = rng_weights.uniform(-1, 1, size=(self.dim_x, 1))
        
        self.w_center /= np.linalg.norm(self.w_center)
        self.w_sharp /= np.linalg.norm(self.w_sharp)

        super().__init__(n_pool=n_pool, test_ratio=test_ratio, noise_std=noise_std, seed=seed, n_plots=n_plots)

    def _generate_x(self, n):
        return self.rng.uniform(-2, 2, size=(n, self.dim_x))

    def _get_params(self, x):
        x = np.atleast_2d(x)
        
        # 1. t_center(x) mapped to [0.1, 0.8]
        center_logit = x @ self.w_center
        t_center = 0.1 + 0.7 * (1.0 / (1.0 + np.exp(-center_logit * 5))).flatten()

        # 2. k(x) sharpness [20, 60]
        sharp_logit = x @ self.w_sharp
        k_val = 20.0 + 40.0 * (1.0 / (1.0 + np.exp(-sharp_logit))).flatten()
        
        return t_center, k_val

    def get_t_star(self, x):
        """
        Grid search for true global maximum t* because sin wave shifts the peak.
        """
        x = np.atleast_2d(x)
        n_samples = x.shape[0]
        
        n_grid = 201
        t_grid = np.linspace(0, 1, n_grid).reshape(1, -1) 
        
        # Batch calculation via expansion
        x_expanded = np.repeat(x, n_grid, axis=0)
        t_expanded = np.tile(t_grid, (n_samples, 1)).flatten()
        
        f_values_flat = self.get_f(x_expanded, t_expanded)
        f_values_grid = f_values_flat.reshape(n_samples, n_grid)
        best_t_indices = np.argmax(f_values_grid, axis=1)
        
        t_star_batch = t_grid.flatten()[best_t_indices]
        return t_star_batch

    def get_f(self, x, t):
        """
        f(x,t) = 10 - k(x)*(cosh(t-t_center) - 1) + 2.0*sin(6*pi*t)
        """
        t = np.clip(t.flatten(), 0, 1)
        t_center, k_val = self._get_params(x)
        
        f_val = 10.0 - k_val * (np.cosh(t - t_center) - 1) + 2.0 * np.sin(6 * np.pi * t)
        return f_val


# --- Dataset Registry ---
DATASET_REGISTRY = {
    "SemiSynthNews": SemiSynthNews,
    "HardNonLinear8D": HardNonLinear8D,
}

DATASET_REGISTRY["ComplexSharpConcave"] = ComplexSharpConcave
DATASET_REGISTRY["SimpleWavy8D"] = SimpleWavy8D