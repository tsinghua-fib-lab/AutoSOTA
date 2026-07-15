import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from tqdm import tqdm

def sample_sparse_causal_model(n, p=0.3, num_hidden=3):
    """Sample a sparse causal model with normalized covariance and optional confounding.
    
    Args:
        n (int): Number of variables
        p (float): Probability of keeping each causal coefficient (sparsity parameter)
        num_hidden (int): Number of hidden confounding variables
    
    Returns:
        tuple: (matrix of causal cefficients, covariance matrix)
    """
    # Sample lower triangular causal coefficients (including hidden variables)
    total_n = n + num_hidden
    C = np.tril(np.random.randn(total_n, total_n), -1)
    
    # Create sparsity by setting coefficients to 0 with probability 1-p
    mask = np.random.rand(total_n, total_n) < p
    C = C * mask

    # Sample random error variances
    error_vars = np.random.exponential(1, total_n)

    # Compute covariance matrix from linear Gaussian model
    S = np.linalg.inv(np.eye(total_n)-C)
    cov = S @ np.diag(error_vars) @ S.T

    # Normalize the model so that each variable has variance 1
    inv_std = np.diag(1 / np.sqrt(np.diag(cov)))
    C = inv_std @ C @ np.diag(np.sqrt(np.diag(cov)))
    cov = inv_std @ cov @ inv_std
    
    # Add confounding effects
    if num_hidden > 0:
        # Sample indices of unobserved variables
        hidden_indices = np.random.choice(total_n, num_hidden, replace=False)
        
        # Adjust causal coefficients 
        for l in hidden_indices:
            for i in range(l):
                for j in range(l+1, total_n):
                    C[j, i] += C[l, i] * C[j, l]
        
        # Remove hidden variables from the model
        C = np.delete(C, hidden_indices, axis=0)
        C = np.delete(C, hidden_indices, axis=1)
        cov = np.delete(cov, hidden_indices, axis=0)
        cov = np.delete(cov, hidden_indices, axis=1)
    
    return C, cov

def backdoor_paths(cov, C, k, i, j):
    n = cov.shape[0]
    
    # Validate indices
    if not (0 <= k < i < j < n):
        raise ValueError(f"Invalid indices: must have 0 <= k < i < j < {n}, got k={k}, i={i}, j={j}")
    
    def path_product(start, end, intermediates):
        """Compute product of causal coefficients along a path."""
        path = [start] + sorted(intermediates) + [end]
        prod = 1.0
        for idx in range(len(path) - 1):
            prod *= C[path[idx + 1], path[idx]]
        return prod
    
    def subset_to_bitmask(subset, offset):
        """Convert subset to bitmask with given offset."""
        mask = 0
        for v in subset:
            mask |= (1 << (v - offset))
        return mask
    
    # Vertices available for paths
    vertices_to_i = list(range(k + 1, i))
    vertices_to_j = [v for v in range(k + 1, j) if v != i]  # Exclude i from paths to j
    offset = k + 1  # For bitmask representation
    
    # Precompute all path products and bitmasks for k -> i
    paths_i = []
    for size in range(len(vertices_to_i) + 1):
        for subset in combinations(vertices_to_i, size):
            prod = path_product(k, i, subset)
            mask = subset_to_bitmask(subset, offset)
            paths_i.append((prod, mask))
    
    # Precompute all path products and bitmasks for k -> j
    paths_j = []
    for size in range(len(vertices_to_j) + 1):
        for subset in combinations(vertices_to_j, size):
            prod = path_product(k, j, subset)
            mask = subset_to_bitmask(subset, offset)
            paths_j.append((prod, mask))
    
    # Convert to numpy arrays for vectorization
    prods_i = np.array([p[0] for p in paths_i])
    masks_i = np.array([p[1] for p in paths_i], dtype=np.int64)
    prods_j = np.array([p[0] for p in paths_j])
    masks_j = np.array([p[1] for p in paths_j], dtype=np.int64)
    
    # Compute disjointness matrix using broadcasting (bitwise AND == 0 means disjoint)
    disjoint = (masks_i[:, None] & masks_j[None, :]) == 0
    
    # Compute outer product of path products and sum where disjoint
    product_matrix = np.outer(prods_i, prods_j)
    total = np.sum(product_matrix[disjoint])
    
    # Multiply by variance of k
    return total * cov[k, k]

def bivariate_confounding(A, cov, i, j):
    return (cov[i,j] - A[j,i] * cov[i,i])**2

def multivariate_confounding(A, cov, i, j):
    n = len(cov)
    confounding_score = cov[i,j] - A[j,i] * cov[i,i]
    C = np.eye(n) - np.linalg.inv(A)
    for k in range(i):
        confounding_score -= backdoor_paths(cov, C, k, i, j)
    return confounding_score**2

def compatibility_score(A, cov):
    n = cov.shape[0]
    bivariate_score = sum(bivariate_confounding(A, cov, i, j) for i in range(n) for j in range(i+1, n))
    multivariate_score = sum(multivariate_confounding(A, cov, i, j) for i in range(n) for j in range(i+1, n))
    return bivariate_score - multivariate_score
    

def run_experiment_vary_m(n=10, p=0.5, num_hidden_values=[0, 2, 5, 10], sigma_values=np.linspace(0, 1, 11), num_models=50, num_draws=20, seed=42):
    """Sample causal models, add noise to coefficients, and plot compatibility scores.
    
    Args:
        n (int): Number of observed variables
        p (float): Sparsity parameter for causal model
        num_hidden_values (list): List of different numbers of hidden confounders to test
        sigma_values (array): Noise levels to test (including 0)
        num_models (int): Number of causal models to sample per configuration
        num_draws (int): Number of noisy coefficient draws per model
        seed (int): Random seed for reproducibility
    """
    np.random.seed(seed)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Define distinct line styles and markers for black-and-white printing
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]  # solid, dashed, dash-dot, dotted, dash-dot-dot
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']  # circle, square, triangle, diamond, etc.
    
    total_iterations = len(num_hidden_values) * len(sigma_values) * num_models
    pbar = tqdm(total=total_iterations, desc="Experiment vary_m")
    
    for h_idx, num_hidden in enumerate(num_hidden_values):
        results_main = np.zeros(len(sigma_values))
        
        for s_idx, sigma in enumerate(sigma_values):
            total_main_positive = 0
            total_draws = 0
            
            for _ in range(num_models):
                pbar.set_postfix({'m': num_hidden, 'σ': f'{sigma:.2f}'})
                # Sample true causal model
                true_C, cov = sample_sparse_causal_model(n, p, num_hidden)
                true_A = np.linalg.inv(np.eye(n) - true_C)
                
                # If sigma is 0, only need one draw (no noise)
                draws = 1 if sigma == 0 else num_draws
                
                for _ in range(draws):
                    # Add noise to coefficients
                    if sigma == 0:
                        noisy_A = true_A.copy()
                    else:
                        noisy_A = true_A + sigma * np.tril(np.random.randn(n, n), -1)
                    
                    try:
                        # Our compatibility score
                        score_main = compatibility_score(noisy_A, cov)
                        if score_main > 0:
                            total_main_positive += 1
                        
                        total_draws += 1
                    except (np.linalg.LinAlgError, ValueError):
                        pass  # Handle numerical issues
                pbar.update(1)
            
            results_main[s_idx] = 100 * total_main_positive / total_draws if total_draws > 0 else 0
        
        # Plot with distinct line style and marker for B&W compatibility
        ls = line_styles[h_idx % len(line_styles)]
        mk = markers[h_idx % len(markers)]
        ax.plot(sigma_values, results_main, linestyle=ls, marker=mk, color='black',
                markersize=6, linewidth=1.5, markerfacecolor='white', 
                markeredgecolor='black', markeredgewidth=1.0, label=f'$m={num_hidden}$')
    
    # Configure plot
    ax.set_xlabel('Error variance', fontsize=24)
    ax.set_ylabel('% of draws with positive \n compatibility score', fontsize=24)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=24)
    ax.grid(True)
    ax.tick_params(labelsize=24)
    
    pbar.close()
    plt.tight_layout()
    plt.savefig('linear_synthetic_vary_m.png', dpi=150)
    plt.show()


def run_experiment_vary_n(n_values=[5, 7, 9, 11], p=0.5, num_hidden=3, sigma_values=np.linspace(0, 1, 11), num_models=50, num_draws=20, seed=42):
    """Sample causal models, add noise to coefficients, and plot compatibility scores.
    Varies the number of observed variables (n) across different lines.
    
    Args:
        n_values (list): List of different numbers of observed variables to test
        p (float): Sparsity parameter for causal model
        num_hidden (int): Number of hidden confounders
        sigma_values (array): Noise levels to test (including 0)
        num_models (int): Number of causal models to sample per configuration
        num_draws (int): Number of noisy coefficient draws per model
        seed (int): Random seed for reproducibility
    """
    np.random.seed(seed)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Define distinct line styles and markers for black-and-white printing
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]  # solid, dashed, dash-dot, dotted, dash-dot-dot
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']  # circle, square, triangle, diamond, etc.
    
    total_iterations = len(n_values) * len(sigma_values) * num_models
    pbar = tqdm(total=total_iterations, desc="Experiment vary_n")
    
    for n_idx, n in enumerate(n_values):
        results_main = np.zeros(len(sigma_values))
        
        for s_idx, sigma in enumerate(sigma_values):
            total_main_positive = 0
            total_draws = 0
            
            for _ in range(num_models):
                pbar.set_postfix({'n': n, 'σ': f'{sigma:.2f}'})
                # Sample true causal model
                true_C, cov = sample_sparse_causal_model(n, p, num_hidden)
                true_A = np.linalg.inv(np.eye(n) - true_C)
                
                # If sigma is 0, only need one draw (no noise)
                draws = 1 if sigma == 0 else num_draws
                
                for _ in range(draws):
                    # Add noise to coefficients
                    if sigma == 0:
                        noisy_A = true_A.copy()
                    else:
                        noisy_A = true_A + sigma * np.tril(np.random.randn(n, n), -1)
                    
                    try:
                        # Our compatibility score
                        score_main = compatibility_score(noisy_A, cov)
                        if score_main > 0:
                            total_main_positive += 1
                        
                        total_draws += 1
                    except (np.linalg.LinAlgError, ValueError):
                        pass  # Handle numerical issues
                pbar.update(1)
            
            results_main[s_idx] = 100 * total_main_positive / total_draws if total_draws > 0 else 0
        
        # Plot with distinct line style and marker for B&W compatibility
        ls = line_styles[n_idx % len(line_styles)]
        mk = markers[n_idx % len(markers)]
        ax.plot(sigma_values, results_main, linestyle=ls, marker=mk, color='black',
                markersize=6, linewidth=1.5, markerfacecolor='white', 
                markeredgecolor='black', markeredgewidth=1.0, label=f'$n={n}$')
    
    # Configure plot
    ax.set_xlabel('Error variance', fontsize=24)
    ax.set_ylabel('% of draws with positive \n compatibility score', fontsize=24)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=24)
    ax.grid(True)
    ax.tick_params(labelsize=24)
    
    pbar.close()
    plt.tight_layout()
    plt.savefig('linear_synthetic_vary_n.png', dpi=150)
    plt.show()


def run_experiment_vary_p(n=10, p_values=[0.1, 0.3, 0.7, 1], num_hidden=3, sigma_values=np.linspace(0, 1, 11), num_models=50, num_draws=20, seed=42):
    """Sample causal models, add noise to coefficients, and plot compatibility scores.
    Varies the sparsity parameter (p) across different lines.
    
    Args:
        n (int): Number of observed variables
        p_values (list): List of different sparsity parameters to test
        num_hidden (int): Number of hidden confounders
        sigma_values (array): Noise levels to test (including 0)
        num_models (int): Number of causal models to sample per configuration
        num_draws (int): Number of noisy coefficient draws per model
        seed (int): Random seed for reproducibility
    """
    np.random.seed(seed)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Define distinct line styles and markers for black-and-white printing
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]  # solid, dashed, dash-dot, dotted, dash-dot-dot
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']  # circle, square, triangle, diamond, etc.
    
    total_iterations = len(p_values) * len(sigma_values) * num_models
    pbar = tqdm(total=total_iterations, desc="Experiment vary_p")
    
    for p_idx, p in enumerate(p_values):
        results_main = np.zeros(len(sigma_values))
        
        for s_idx, sigma in enumerate(sigma_values):
            total_main_positive = 0
            total_draws = 0
            
            for _ in range(num_models):
                pbar.set_postfix({'p': p, 'σ': f'{sigma:.2f}'})
                # Sample true causal model
                true_C, cov = sample_sparse_causal_model(n, p, num_hidden)
                true_A = np.linalg.inv(np.eye(n) - true_C)
                
                # If sigma is 0, only need one draw (no noise)
                draws = 1 if sigma == 0 else num_draws
                
                for _ in range(draws):
                    # Add noise to coefficients
                    if sigma == 0:
                        noisy_A = true_A.copy()
                    else:
                        noisy_A = true_A + sigma * np.tril(np.random.randn(n, n), -1)
                    
                    try:
                        # Our compatibility score
                        score_main = compatibility_score(noisy_A, cov)
                        if score_main > 0:
                            total_main_positive += 1
                        
                        total_draws += 1
                    except (np.linalg.LinAlgError, ValueError):
                        pass  # Handle numerical issues
                pbar.update(1)
            
            results_main[s_idx] = 100 * total_main_positive / total_draws if total_draws > 0 else 0
        
        # Plot with distinct line style and marker for B&W compatibility
        ls = line_styles[p_idx % len(line_styles)]
        mk = markers[p_idx % len(markers)]
        ax.plot(sigma_values, results_main, linestyle=ls, marker=mk, color='black',
                markersize=6, linewidth=1.5, markerfacecolor='white', 
                markeredgecolor='black', markeredgewidth=1.0, label=f'$p={p}$')
    
    # Configure plot
    ax.set_xlabel('Error variance', fontsize=24)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=24)
    ax.grid(True) 
    ax.tick_params(labelsize=24)
    
    pbar.close()
    plt.tight_layout()
    plt.savefig('linear_synthetic_vary_p.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    run_experiment_vary_m(n=10, p=0.5, num_hidden_values=[0, 2, 5, 10], sigma_values=np.linspace(0, 1, 11), num_models=50, num_draws=20)
    run_experiment_vary_n(n_values=[5, 7, 9, 11], p=0.5, num_hidden=3, sigma_values=np.linspace(0, 1, 11), num_models=50, num_draws=20)
    run_experiment_vary_p(n=10, p_values=[0.1, 0.3, 0.7, 1], num_hidden=3, sigma_values=np.linspace(0, 1, 11), num_models=50, num_draws=20)