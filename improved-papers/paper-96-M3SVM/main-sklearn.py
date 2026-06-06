import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import argparse
import numpy as np
import scipy.io as scio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import time
from sklearn.utils import check_random_state

from utils.gdp_account import *

class DPLinearSVC:
    """Differentially Private Linear SVC"""
    
    def __init__(self, C=1.0, epsilon=3.0, delta=1e-5, max_iter=10000, random_state=42, multi_class='ovr', dp_method='ours'):
        self.C_over_n = C
        self.epsilon = epsilon
        self.delta = delta
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.dp_method = dp_method
        self.coef_ = None
        self.classes_ = None
        self.random_state = random_state
        
    def fit(self, X, y):
        """Fit the model with differential privacy"""
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        self.n = len(y)
        
        # Use LinearSVC with specified parameters
        base_model = LinearSVC(
            C=self.C_over_n,
            max_iter=self.max_iter, 
            random_state=self.random_state,
            multi_class=self.multi_class,
            fit_intercept=False,
        )
        
        # Store base model for dpkl method
        if self.dp_method == "dpkl":
            self.base_model = base_model
        
        # Fit the model
        base_model.fit(X, y)
        
        # Get the model coefficients
        self.coef_ = base_model.coef_.copy()
        if self.multi_class == "crammer_singer" and self.dp_method == "pmsvm":
            # Calculate sensitivity for the privacy mechanism
            sensitivity = 4 * (2 ** 0.5) * self.C_over_n
            sigma = calibrateAnalyticGaussianMechanism(self.epsilon, self.delta, sensitivity)
            print(f"Using sigma={sigma} for differential privacy")
            noise = np.random.normal(0, sigma, self.coef_.shape)
            self.coef_ += noise
            # Store sigma and base coef for ensemble prediction
            self._sigma = sigma
            self._base_coef = self.coef_ - noise  # clean weights before noise
            
        elif self.multi_class == "ovr":
            if self.dp_method == "dpkl":
                # For dpkl method, we don't add noise to weights
                # Noise will be added at prediction time to outputs
                print(f"Using dpkl method - will add noise to predictions")
            elif self.dp_method == "privatesvm":
                self.epsilon_per_class = self.epsilon / len(self.classes_)
                sensitivity = 4 * self.C_over_n
                sigma = sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon_per_class
                print(f"[privatesvm] sigma={sigma}")
                self.coef_ += np.random.normal(0, sigma, self.coef_.shape)
            
            elif self.dp_method == "opera":
                self.epsilon_per_class = self.epsilon / len(self.classes_)
                sensitivity = 4 * self.C_over_n
                sigma = calibrateAnalyticGaussianMechanism(
                    self.epsilon_per_class, self.delta, sensitivity)
                print(f"[opera] sigma={sigma}")
                self.coef_ += np.random.normal(0, sigma, self.coef_.shape)
            
            elif self.dp_method == "privatesvm-advcom":
                k = len(self.classes_)
                sensitivity = 4 * self.C_over_n
                eps0 = advanced_composition(self.epsilon, self.delta, k)
                delta0 = self.delta / k
                sigma = sensitivity * math.sqrt(2 * math.log(1.25 / delta0)) / eps0
            
                print(f"[privatesvm-advcom] exact ε₀={eps0:.4f}, σ={sigma:.4f}")
                self.coef_ += np.random.normal(0, sigma, self.coef_.shape)
            
            elif self.dp_method == "opera-advcom":
                k = len(self.classes_)
                sensitivity = 4 * self.C_over_n
                eps0 = advanced_composition(self.epsilon, self.delta, k)
                delta0 = self.delta / k
                sigma = calibrateAnalyticGaussianMechanism(eps0, delta0, sensitivity)
            
                print(f"[opera-advcom] exact ε₀={eps0:.4f}, σ={sigma:.4f}")
                self.coef_ += np.random.normal(0, sigma, self.coef_.shape)
        return self
                       
    def predict(self, X, n_ensemble=50):
        """Predict using the noisy model"""
        if self.dp_method == "pmsvm" and self.multi_class == "crammer_singer" and hasattr(self, '_sigma') and n_ensemble > 1:
            # Ensemble: average scores over K noise realizations (pure post-processing, no extra privacy cost)
            scores_sum = np.zeros((X.shape[0], len(self.classes_)))
            for _ in range(n_ensemble):
                noisy_coef = self._base_coef + np.random.normal(0, self._sigma, self._base_coef.shape)
                scores_sum += np.dot(X, noisy_coef.T)
            return np.argmax(scores_sum, axis=1)
        elif self.dp_method == "dpkl":
            # For dpkl method, add noise to predictions instead of weights
            raw_scores = np.dot(X, self.coef_.T)
            def _softmax(z):
                """row-wise softmax"""
                z_shift = z - np.max(z, axis=1, keepdims=True)      # numerical stability
                e = np.exp(z_shift)
                return e / e.sum(axis=1, keepdims=True)
            if raw_scores.ndim == 1 or raw_scores.shape[1] == 1:     # binary
                prob_pos = 1 / (1 + np.exp(-raw_scores.ravel()))
                probs = np.column_stack([1 - prob_pos, prob_pos])    # shape (n, 2)
            else:                                                    # multi-class
                probs = _softmax(raw_scores)
    
            # Noise scale = 1 / epsilon
            eps_per = self.epsilon / len(self.classes_) if self.multi_class == "ovr" else self.epsilon
            noise_scale = 1.0 / eps_per
    
            noisy_probs = probs + np.random.laplace(0, noise_scale, probs.shape)
            return np.argmax(noisy_probs, axis=1)
        else:
            # Standard prediction with noisy weights
            if self.coef_.shape[0] == 1:  # Binary classification
                scores = np.dot(X, self.coef_.T).ravel()
                return np.where(scores > 0, 1, 0)
            else:  # Multi-class
                scores = np.dot(X, self.coef_.T)
                return np.argmax(scores, axis=1)

def R_MLR(para):
    path = f'./dataset/{para.data}.mat'
    X = scio.loadmat(path)['X']
    y = scio.loadmat(path)['Y'].squeeze()
    print(X.shape, y.shape)

    n, d = X.shape[0], X.shape[1]
    num_class = len(np.unique(y))

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    y = y-1  # Convert to 0-indexed

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=para.test_size, random_state=para.state, stratify=y)        

    print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    # Choose the model based on the optimizer parameter
    if para.optimizer == 'standard_svc':
        print(f"Using standard LinearSVC with C={para.C}")
        
        # Create LinearSVC
        model = LinearSVC(
            C=para.C,
            max_iter=10000, 
            random_state=para.state,
            multi_class=para.multi_class,
            fit_intercept=False,
        )
        
        # Start training timer
        start_time = time.time()
        
        # Fit the model
        model.fit(X_train, y_train)
        
        # End training timer
        train_time = time.time() - start_time
        
        print(f"Training completed in {train_time:.2f} seconds")
        print("Note: No differential privacy guarantees with standard LinearSVC")
        
    else:  # Use our custom DP model
        print(f"Using Differentially Private LinearSVC with C={para.C}")
        model = DPLinearSVC(
            C=para.C,
            epsilon=para.eps,
            delta=para.delta,
            max_iter=10000,
            random_state = para.state,
            multi_class=para.multi_class,
            dp_method=para.dp_method,
        )
        
        # Start training timer
        start_time = time.time()
        
        # Fit the model
        model.fit(X_train, y_train)
        
        # End training timer
        train_time = time.time() - start_time
        print(f"Training completed in {train_time:.2f} seconds")
        
        # Print the privacy guarantee
        print(f"Privacy Guarantee: (ε = {para.eps:.2f}, δ = {para.delta})")
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, test_acc

if __name__ == "__main__":
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    parser = argparse.ArgumentParser(
        description='Train with differential privacy using scikit-learn models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', type=str, default='vehicle_uni')
    parser.add_argument('--state', type=int, default=42)
    parser.add_argument('--C', type=float, default=1.0,
                       help='Regularization parameter')
    parser.add_argument('--eps', type=float, default=3.0)
    parser.add_argument('--delta', type=float, default=1e-5)
    parser.add_argument('--optimizer', type=str, default='dp_svc', 
                        choices=['dp_svc', 'standard_svc'], 
                        help='Optimizer to use')
    parser.add_argument('--multi_class', type=str, default='ovr', 
                        choices=['ovr', 'crammer_singer'], 
                        help='Multi-class to use')
    parser.add_argument('--dp_method', type=str, default='pmsvm',
                        choices=['pmsvm', 'privatesvm', 'opera', 'dpkl'],
                        help='DP methods')
    para = parser.parse_args()
    para.test_size = 0.2
    
    model, accuracy = R_MLR(para)
    print(f"Final test accuracy: {accuracy:.4f}")
