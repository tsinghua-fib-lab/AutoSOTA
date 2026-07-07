"""
Adapted from https://github.com/yuchen-zhu-zyc/MDNS/blob/main/utils_ising.py

Utility functions for the 2D Ising models
"""

import numpy as np
import torch
from tqdm import tqdm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def ising2d_ham(S, J=1.0, h=0.0):
    r"""
    Compute the Hamiltonian for a batch of configurations in a 2D Ising model, with periodic boundary conditions.
    
    Parameters:
    - S: torch.tensor of shape (B, L * L)
        each element is -1 or 1 (not 0 or 1!), representing spin configurations.
    - J: float, interaction strength between neighboring spins (default=1.0).
    - h: float, external magnetic field strength (default=0.0).

    Returns:
    - hamiltonians: torch.tensor of shape (B,) containing the Hamiltonian for each configuration.
        H = -J \sum_{i \sim j} S_{i} S_{j} - h \sum_{i} S_{i}
    (The p.m.f. is given by p(S) \propto e^{-\beta H(S)})
    """
    # assert S.ndim == 2
    # assert torch.all((S == 1) | (S == -1)), "All entries of S must be either 1 or -1"

    S = S.view(S.size(0), int(S.shape[1]**.5), int(S.shape[1]**.5))
    Sx = torch.roll(S, shifts=-1, dims=1)  # Sx[i,j] = S[i+1,j]
    Sy = torch.roll(S, shifts=-1, dims=2)  # Sy[i,j] = S[i,j+1]
    interaction_energy = -J * torch.sum(S * (Sx + Sy), dim=(1, 2))
    magnetic_energy = -h * torch.sum(S, dim=(1, 2))
    return interaction_energy + magnetic_energy


def ising2d_log_discrete_score(S, J=1.0, h=0.0, beta=1.0):
    r"""
    Compute the log of the discrete score for a batch of configurations in a 2D Ising model.
    
    Parameters:
    - S: torch.tensor of shape (B, L * L)
        each element is -1 or 1 (not 0 or 1!), representing spin configurations.
    - J: float, interaction strength between neighboring spins (default=1.0).
    - h: float, external magnetic field strength (default=0.0).
    - beta: float, inverse temperature (default=1.0).

    Returns:
    - scores: torch.tensor of shape (B, L * L, 2):
        scores[b, i, n] = log p(S[b] ^ {i<-n}) / p(S[b])
                        = beta * (n - S[b, i]) * (J * sum_{j ~ i} S[b, j] + h),
        where n in {-1, 1}
    """
    S = S.view(S.shape[0], int(S.shape[1]**.5), int(S.shape[1]**.5))
    neighbor_sum = (torch.roll(S, shifts=1, dims=1) + torch.roll(S, shifts=-1, dims=1) +
                    torch.roll(S, shifts=1, dims=2) + torch.roll(S, shifts=-1, dims=2)) # [B, L, L]
    diff = torch.tensor([-1, 1], device=S.device, dtype=S.dtype) - S.unsqueeze(-1) # [B, L, L, 2]
    scores = beta * diff * (J * neighbor_sum.unsqueeze(-1) + h) # [B, L, L, 2]
    return scores.view(S.shape[0], -1, 2)
    

def ising2d_get_all_configs(L=4, device='cuda:0'):
    """
    Generate all possible Ising configurations for L x L lattice in increasing order
    e.g., [-1, -1], [-1, 1], [1, -1], [1, 1].
    Return: [2 ** (L ** 2), L ** 2], values are in {1, -1}
    """
    B = 2 ** (L ** 2)
    bits = torch.arange(L ** 2 - 1, -1, -1, device=device)
    return (((torch.arange(B, device=device)[:, None] >> bits) & 1) * 2 - 1).to(torch.int8) # [B, L ** 2]


def ising2d_part_func(L=4, beta=.5, J=1.0, h=0.0, device='cuda:0'):
    """
    Compute the partition function of a 2D Ising model with periodic boundary conditions.
    """
    assert 1 <= L <= 4, "Only support L <= 4 due to memory constraints"
    configs = ising2d_get_all_configs(L=L, device=device) # (2 ** D, D)
    H = ising2d_ham(configs, J=J, h=h) # (2 ** D,)
    return torch.exp(-beta * H).sum().item()


def ising2d_mag(S):
    """
    Compute the magnetization for a batch of configurations.

    Parameters:
    S: torch.tensor of shape (B, L * L) representing K configurations on an L x L lattice.
       Each element in S is +1 or -1.

    Returns:
    - torch.tensor of shape (B,) containing the magnetization for each configuration.
    """
    # assert torch.all((S == 1) | (S == -1)), "All entries of S must be either 1 or -1"
    return S.float().mean(dim=1)


def ising2d_2pt_corr(S, r):
    """
    Compute the two-point correlation function for a batch of configurations.
    
    Parameters:
    - S: torch.tensor of shape (B, L * L) representing K configurations on an L x L lattice.
         Each element in S is +1 or -1.
    - r: int, horizontal and vertical distance between points for correlation calculation.
    
    Return:
    - a float: average two-point correlation at distance r over the batch.
    """
    # assert torch.all((S == 1) | (S == -1)), "All entries of S must be either 1 or -1"
    S = S.view(S.size(0), int(S.shape[1]**.5), int(S.shape[1]**.5)).to(dtype=torch.float)
    Sx = torch.roll(S, shifts=-r, dims=1)
    Sy = torch.roll(S, shifts=-r, dims=2)
    corr_x = ((S * Sx).mean(dim=0) - S.mean(dim=0) * Sx.mean(dim=0)).mean().item()
    corr_y = ((S * Sy).mean(dim=0) - S.mean(dim=0) * Sy.mean(dim=0)).mean().item()
    return (corr_x + corr_y) / 2


def ising2d_emp_dist(samples):
    """
    samples: [B, L^2], elements in {-1, 1}
    Output the empirical distribution of the samples as a probability vector of length 2^{L^2}
    The configuations are sorted in increasing order 
    """
    assert torch.all((samples == 1) | (samples == -1)), "All entries of samples must be either 1 or -1"
    B, N = samples.shape  # N = L^2
    bin_samples = ((samples + 1) // 2).to(torch.int32)  # (B, N)
    bits = torch.arange(N - 1, -1, -1, device=samples.device)
    indices = (bin_samples << bits).sum(dim=1)  # (B,)
    counts = torch.bincount(indices, minlength=2 ** N)
    return counts.float() / B


def ising2d_get_pmf(L, J=1.0, h=0.0, beta=1.0, device='cuda:0'):
    """
    Compute the pmf of all configurations (in increasing order), shape [2**(L^2)]
    """
    all_configs = ising2d_get_all_configs(L, device)
    log_pmf = -beta * ising2d_ham(all_configs, J=J, h=h) # [2**D]
    return log_pmf.softmax(dim=0)


def ising2d_visualize(S: torch.tensor, num_per_row: int = 8):
    """
    Visualize multiple Ising configurations in a grid.
    Args:
        S: 2D tensor of shape (N, D=L^2) representing N Ising configurations, values in {-1, 1}.
        num_per_row: Number of configurations to display per row.
    """
    plt.close('all')
    N, D = S.shape; L = int(D**0.5); 
    # assert L**2 == D, "The number of columns must be a perfect square."
    # assert torch.all((S == 1) | (S == -1)), "All entries of S must be either 1 or -1"
    num_rows = (N + num_per_row - 1) // num_per_row
    fig, axes = plt.subplots(num_rows, num_per_row, figsize=(num_per_row * 2, num_rows * 2))
    axes = axes.flatten()
    for i in range(len(axes)):
        if i < N:
            axes[i].imshow(S[i].reshape(L, L).cpu().numpy(), cmap='inferno', interpolation='nearest', vmin=-1, vmax=1)
        axes[i].axis('off')
    fig.tight_layout()
    return fig


def ising2d_row_col_mag(S, axis=0):
    """
    Compute the magnetization for each row or column in a batch of 2D Ising configurations.
    
    Parameters:
    - S: torch.tensor of shape (B, L * L) representing B configurations on an L x L lattice.
         Each element in S is +1 or -1.
    - axis: int, 0 for row-wise magnetization, 1 for column-wise magnetization.
    
    Returns:
    - torch.tensor of shape (B, L) containing the magnetization for each row/column.
    """
    assert torch.all((S == 1) | (S == -1)), "All entries of S must be either 1 or -1"
    L = int(S.shape[1]**.5)
    S = S.view(S.size(0), L, L)
    return S.float().mean(dim=axis+1)  # +1 because first dim is batch


def visualize_ising(S, k_x, k_y):    # Convert to numpy if it's a torch tensor
    S = 2 * S - 1
    if isinstance(S, torch.Tensor):
        S = S.detach().cpu().numpy()
    
    if S.ndim == 2:
        S = S.reshape(-1, int(np.sqrt(S.shape[1])), int(np.sqrt(S.shape[1])))
    assert k_x * k_y == S.shape[0], "k_x * k_y must be equal to the number of samples"
    B = S.shape[0]
    # Check if B is a perfect square
    k = int(np.sqrt(B))
    fig, axes = plt.subplots(k_x, k_y, figsize=(1.5*k_x, 1.5*k_y), constrained_layout=True)
    axes = axes.ravel()  # Flatten the axes array for easy iteration
    
    # Create a colormap with q distinct colors
    # palette = ["#440154", "#FDE725", "#ff7f00"]
    palette = ["#313342", "#DEB4B2"]
    # palette = ["white", "black"]
    cmap = ListedColormap(palette)    
    for i in range(B):
        # Plot heatmap for each sample
        im = axes[i].imshow(S[i], cmap=cmap, vmin=-1, vmax=1, origin='lower',
            interpolation='nearest')
        # Hide ticks and their labels but keep the frame
        # axes[i].set_xticks([])
        # axes[i].set_yticks([])
        # axes[i].set_title(f'Sample {i+1}')
        axes[i].axis('off')
    # Add colorbar
    # cbar = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05)
    # cbar.set_ticks(np.arange(q))
    # cbar.set_ticklabels(np.arange(q))
    plt.tight_layout()
    return fig


def ising_2pt_corr_direction(S, r_x, r_y, use_x = True, use_y = True):
    """
    Compute the two-point correlation function for a batch of configurations.
    
    Parameters:
    - S: torch.tensor of shape (B, L * L) representing K configurations on an L x L lattice.
         Each element in S is +1 or -1.
    - rx, ry: int, horizontal and vertical distance between points for correlation calculation.
    
    Return:
    - torch.tensor of shape (B,) containing the two-point correlation for each configuration.
    """
    
    if not isinstance(S, torch.Tensor):
        S = torch.from_numpy(S)
    
    assert torch.all((S == 1) | (S == -1)), "All entries of S must be either 1 or -1"
    S = S.reshape(-1, 16, 16)
    
    if use_x:
        S_neighbor = 1/2 * (torch.roll(S, shifts=-r_x, dims=1) + torch.roll(S, shifts=r_x, dims=1))
    if use_y:
        S_neighbor = 1/2 * (torch.roll(S, shifts=-r_y, dims=2) + torch.roll(S, shifts=r_y, dims=2))
        
    return (S * S_neighbor).float().mean()

def ising2d_mag_direction(S, use_row = True, use_col = False):
    """
    Compute the magnetization for each row or column in a batch of 2D Ising configurations.
    
    Parameters:
    - S: torch.tensor of shape (B, L * L) representing B configurations on an L x L lattice.
         Each element in S is +1 or -1.
    - axis: int, 0 for row-wise magnetization, 1 for column-wise magnetization.
    
    Returns:
    - torch.tensor of shape (B, L) containing the magnetization for each row/column.
    """
    
    if not isinstance(S, torch.Tensor):
        S = torch.from_numpy(S)
    assert torch.all((S == 1) | (S == -1)), "All entries of S must be either 1 or -1"
    S = S.view(S.size(0), 16, 16)
    
    if use_row:
        return S.float().mean(dim=1).mean(dim = 0)
    if use_col:
        return S.float().mean(dim=2).mean(dim = 0)


def ising2d_plot_2pt_corr(S):
    """
    Plot the 2-point correlation function for a batch of Ising configurations.

    Args:
        S: 2D tensor of shape (B, D=L^2) representing N Ising configurations, values in {-1, 1}.
    """
    L = int(S.shape[1]**0.5)
    # assert torch.all((S == 1) | (S == -1)), "All entries of S must be either 1 or -1"

    plt.close('all')
    fig = plt.figure()
    r = np.arange(-L//2, L//2 + 1)
    corr = [ising2d_2pt_corr(S, i) for i in r]
    plt.plot(r, corr, marker='o')
    plt.xlabel('Distance $r$')
    plt.xticks(r, [f'${i}$' for i in r])
    plt.ylim(-0.05, 1.05)
    plt.ylabel('2-point Correlation')
    for i, c in enumerate(corr):
        plt.text(r[i], c + 0.02, f'{c:.3f}', ha='center', fontsize=8)
    plt.title('2-point Correlation Function')
    plt.grid(True, alpha=0.3)
    return fig


##### sampling algorithms #####

def ising2d_mh(L, beta=0.5, J=1.0, h=0.0,
               batch_size=256, num_collect=20000, burn_in=10000, collect_every=1000, init=None):
    """
    Metropolis-Hastings algorithm to sample from the 2D Ising model's distribution.

    Parameters:
    - L: int, size of the lattice (L * L).
    - beta: float, inverse temperature (default=0.5).
    - J: float, interaction strength between neighboring spins (default=1.0).
    - h: float, external magnetic field (default=0.0).
    - batch_size: int, number of parallel configurations.
    - num_collect: int, number of times to collect.
    - burn_in: int, number of initial steps to discard (burn-in period).
    - collect_every: int, collect a sample every `collect_every` steps.
    - init: numpy.ndarray of shape (batch_size, L, L) or (batch_size, L * L), initial configuration.
            If None, random configurations are used.

    Returns:
    - samples: numpy.ndarray of shape (num_collect * batch_size, L, L), sampled configurations, values in {-1, 1}.
    """
    if init is not None:
        S = init.reshape(batch_size, L, L).astype(np.int16)
    else:
        S = np.random.choice([-1, 1], size=(batch_size, L, L)).astype(np.int16)

    samples = []
    batch_arange = np.arange(batch_size)

    pbar = tqdm(range(num_collect * collect_every + burn_in))
    for step in pbar:
        i, j = np.random.randint(0, L, size=(batch_size,)), np.random.randint(0, L, size=(batch_size,))
        dH = 2 * J * S[batch_arange, i, j] * (
            S[batch_arange, (i - 1) % L, j] + S[batch_arange, (i + 1) % L, j]
            + S[batch_arange, i, (j - 1) % L] + S[batch_arange, i, (j + 1) % L]
            ) + 2 * h * S[batch_arange, i, j]
        flip = np.random.rand(batch_size) < np.exp(-beta * dH)
        S[batch_arange[flip], i[flip], j[flip]] *= -1

        if step >= burn_in and (step - burn_in) % collect_every == 0:
            samples.append(np.copy(S))

    samples = np.array(samples).reshape(-1, L, L)
    np.random.shuffle(samples)
    return samples


def ising2d_wolff(L, beta=0.5, J=1.0, h=0.0,
                  batch_size=256, num_collect=20000, burn_in=10000, collect_every=1000, init=None):
    """
    (Metropolis-adjusted) Wolff cluster algorithm to sample from the 2D Ising model's distribution.

    Parameters:
    - L: int, size of the lattice (L * L).
    - beta: float, inverse temperature (default=0.5).
    - J: float, interaction strength between neighboring spins (default=1.0).
    - h: float, external magnetic field (default=0.0).
    - batch_size: int, number of parallel configurations.
    - num_collect: int, number of times to collect.
    - burn_in: int, number of initial steps to discard (burn-in period).
    - collect_every: int, collect a sample every `collect_every` steps.
    - init: numpy.ndarray of shape (batch_size, L, L) or (batch_size, L * L), initial configuration.
        If None, random configurations are used.

    Returns:
    - samples: numpy.ndarray of shape (num_collect * batch_size, L, L), sampled configurations, values in {-1, 1}.
    """
    if init is not None:
        S = init.reshape(batch_size, L, L).astype(np.int16)
    else:
        S = np.random.choice([-1, 1], size=(batch_size, L, L)).astype(np.int16)
    
    samples = []
    p_add = 1 - np.exp(-2 * beta * J)
    
    def grow_cluster(Sb, start_i, start_j):
        cluster = set([(start_i, start_j)])
        stack = [(start_i, start_j)]
        spin = Sb[start_i, start_j]
        while stack:
            ci, cj = stack.pop()
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = (ci + di) % L, (cj + dj) % L
                if (ni, nj) not in cluster and Sb[ni, nj] == spin:
                    if np.random.rand() < p_add:
                        cluster.add((ni, nj))
                        stack.append((ni, nj))
        return cluster
    
    pbar = tqdm(range(burn_in + num_collect * collect_every))
    for step in pbar:
        for b in range(batch_size):
            # pick a random position and grow a cluster
            i, j = np.random.randint(0, L, size=2)
            cluster = grow_cluster(S[b], i, j)

            if h == 0.0: # no external field, directly flip
                for ci, cj in cluster:
                    S[b, ci, cj] *= -1
            
            else: # with external field, use Metropolis-Hastings criterion
                # compute energy difference, only requires the external field part
                dH = 2 * h * sum(S[b, ci, cj] for ci, cj in cluster)
                if np.random.rand() < np.exp(-beta * dH):
                    for ci, cj in cluster:
                        S[b, ci, cj] *= -1

        if step >= burn_in and (step - burn_in) % collect_every == 0:
            samples.append(np.copy(S))

    samples = np.array(samples).reshape(-1, L, L)
    np.random.shuffle(samples)
    return samples


def ising2d_swendsen_wang(L, beta=0.5, J=1.0, h=0.0,
                          batch_size=256, num_collect=20000, burn_in=10000, collect_every=1000, init=None):
    # TODO: finish the Metropolis adjustment part
    """
    (Metropolis-adjusted) Swendsen-Wang algorithm to sample from the 2D Ising model's distribution.

    Parameters:
    - L: int, size of the lattice (L * L).
    - beta: float, inverse temperature (default=0.5).
    - J: float, interaction strength between neighboring spins (default=1.0).
    - h: float, external magnetic field (default=0.0).
    - batch_size: int, number of parallel configurations.
    - num_collect: int, number of times to collect.
    - burn_in: int, number of initial steps to discard (burn-in period).
    - collect_every: int, collect a sample every `collect_every` steps.
    - init: numpy.ndarray of shape (batch_size, L, L) or (batch_size, L * L), initial configuration.
            If None, random configurations are used.

    Returns:
    - samples: numpy.ndarray of shape (num_collect * batch_size, L, L), sampled configurations.
    """
    if init is not None:
        S = init.reshape(batch_size, L, L)
    else:
        S = np.random.choice([-1, 1], size=(batch_size, L, L))
    
    samples = []
    
    # Pre-compute bond probability
    p_bond = 1 - np.exp(-2 * beta * J)
    
    # Pre-allocate arrays for efficiency
    parent = np.zeros((batch_size, L**2), dtype=np.int32)
    rank = np.zeros((batch_size, L**2), dtype=np.int32)
    
    # Pre-compute neighbor indices for periodic boundary conditions
    indices = np.arange(L**2).reshape(L, L)
    right_neighbors = np.roll(indices, -1, axis=1).ravel()
    down_neighbors = np.roll(indices, -1, axis=0).ravel()
    
    def find(b, x):
        """Path compression find operation"""
        while parent[b, x] != x:
            parent[b, x] = parent[b, parent[b, x]]
            x = parent[b, x]
        return x
    
    def union(b, x, y):
        """Union by rank"""
        root_x = find(b, x)
        root_y = find(b, y)
        if root_x != root_y:
            if rank[b, root_x] < rank[b, root_y]:
                parent[b, root_x] = root_y
            else:
                parent[b, root_y] = root_x
                if rank[b, root_x] == rank[b, root_y]:
                    rank[b, root_x] += 1
    
    def process_configuration(b):
        # Step 1: Create bonds between aligned spins (fully vectorized)
        # Reshape for easier neighbor comparison
        S_flat = S[b].ravel()
        
        # Check horizontal bonds (right neighbors)
        aligned_h = S_flat == S_flat[right_neighbors]
        h_bonds = aligned_h & (np.random.random(L**2) < p_bond)
        
        # Check vertical bonds (down neighbors)
        aligned_v = S_flat == S_flat[down_neighbors]
        v_bonds = aligned_v & (np.random.random(L**2) < p_bond)
        
        # Step 2: Process bonds and identify clusters
        # Reset parent and rank arrays for this configuration
        parent[b] = np.arange(L**2, dtype=np.int32)
        rank[b].fill(0)
        
        # Process horizontal bonds
        h_indices = np.where(h_bonds)[0]
        for idx in h_indices:
            union(b, idx, right_neighbors[idx])
        
        # Process vertical bonds
        v_indices = np.where(v_bonds)[0]
        for idx in v_indices:
            union(b, idx, down_neighbors[idx])
        
        # Step 3: Identify clusters and flip them
        # Get unique cluster roots
        roots = np.array([find(b, i) for i in range(L**2)])
        unique_roots = np.unique(roots)
        
        # Generate flip decisions for each cluster
        flip_decisions = np.random.random(len(unique_roots)) < 0.5
        flip_map = dict(zip(unique_roots, flip_decisions))
        
        # Apply flips (vectorized)
        flip_mask = np.array([flip_map[root] for root in roots]).reshape(L, L)
        S[b] = np.where(flip_mask, -S[b], S[b])
    
    pbar = tqdm(range(burn_in + num_collect * collect_every))
    for step in pbar:
        # Process all configurations
        for b in range(batch_size):
            process_configuration(b)
        
        # Collect samples after burn-in period
        if step >= burn_in and (step - burn_in) % collect_every == 0:
            samples.append(np.copy(S))
    
    samples = np.array(samples).reshape(-1, L, L)
    np.random.shuffle(samples)
    return samples


def ising_2d_exact_observables(L, beta, J=1.0):
    """
    Compute the exact observables of the 2D Ising model with periodic boundary conditions,
    based on the paper: Bounded and Inhomogeneous Ising Models. I. (Ferdinand & Fisher, 1969)

    Output are free energy per spin, internal energy per spin, and entropy per spin
    """
    K = beta * J
    
    l = np.arange(2 * L)
    
    # Eq (2.4): c_l
    c_l = np.cosh(2 * K) / np.tanh(2 * K) - np.cos(l * np.pi / L)
    
    # Eq (2.5): 计算 gamma_l
    gamma_l = np.zeros(2 * L)
    gamma_l[0] = 2 * K + np.log(np.tanh(K))
    gamma_l[1:] = np.arccosh(c_l[1:])
    
    # Derivative of gamma_l with respect to K (for internal energy calculation)
    c_prime = 2 * np.cosh(2 * K) * (1 - 1 / np.sinh(2 * K)**2)
    gamma_l_prime = np.zeros(2 * L)
    gamma_l_prime[0] = 2 * (1 + 1 / np.sinh(2 * K))
    gamma_l_prime[1:] = c_prime / np.sqrt(c_l[1:]**2 - 1)
    
    # Numerical stability helper functions
    def log_2cosh(x):
        return np.abs(x) + np.log1p(np.exp(-2 * np.abs(x)))
    
    def log_2sinh(x):
        return np.abs(x) + np.log1p(-np.exp(-2 * np.abs(x)))
    
    # Prevent division by zero warning when gamma_0 is extremely close to 0 at the critical temperature
    gamma_l_safe = np.copy(gamma_l)
    if np.abs(gamma_l_safe[0]) < 1e-14:
        gamma_l_safe[0] = 1e-14 if gamma_l_safe[0] >= 0 else -1e-14
        
    # Define odd and even indices (corresponding to the product indices of Z1, Z2, Z3, Z4 in the paper)
    idx_odd = 2 * np.arange(L) + 1
    idx_even = 2 * np.arange(L)
    
    # Calculate the log absolute value parts v_i and signs s_i of the four partition function components
    v1 = np.sum(log_2cosh(L * gamma_l[idx_odd] / 2))
    s1 = 1.0
    
    v2 = np.sum(log_2sinh(L * gamma_l[idx_odd] / 2))
    s2 = 1.0
    
    v3 = np.sum(log_2cosh(L * gamma_l[idx_even] / 2))
    s3 = 1.0
    
    v4 = np.sum(log_2sinh(L * gamma_l[idx_even] / 2))
    s4 = np.sign(gamma_l[0]) if gamma_l[0] != 0 else 0.0
    
    # Log-Sum-Exp stable calculation of the core part of the partition function
    v_max = np.max([v1, v2, v3, v4])
    Z_sum = (s1 * np.exp(v1 - v_max) + 
             s2 * np.exp(v2 - v_max) + 
             s3 * np.exp(v3 - v_max) + 
             s4 * np.exp(v4 - v_max))
    
    log_Z_part = v_max + np.log(Z_sum)
    
    # Log of the common prefix of the partition function: log( 1/2 * (2 sinh 2K)^(L^2 / 2) )
    log_2sinh_2K = 2 * K + np.log1p(-np.exp(-4 * K))
    log_prefactor = -np.log(2) + 0.5 * (L**2) * log_2sinh_2K
    
    log_Z = log_prefactor + log_Z_part
    
    # [1] Free energy per spin
    f_exact = - log_Z / (beta * L**2)
    
    # [2] Internal energy per spin
    def compute_R(gamma, gamma_prime, func='tanh'):
        A = L * gamma / 2
        if func == 'tanh':
            return (L / 2) * np.sum(gamma_prime * np.tanh(A))
        else: # coth
            coth_A = 1.0 / np.tanh(A)
            return (L / 2) * np.sum(gamma_prime * coth_A)
            
    R1 = compute_R(gamma_l[idx_odd], gamma_l_prime[idx_odd], 'tanh')
    R2 = compute_R(gamma_l[idx_odd], gamma_l_prime[idx_odd], 'coth')
    R3 = compute_R(gamma_l[idx_even], gamma_l_prime[idx_even], 'tanh')
    R4 = compute_R(gamma_l_safe[idx_even], gamma_l_prime[idx_even], 'coth')
    
    sum_ZR = (s1 * np.exp(v1 - v_max) * R1 + 
              s2 * np.exp(v2 - v_max) * R2 + 
              s3 * np.exp(v3 - v_max) * R3 + 
              s4 * np.exp(v4 - v_max) * R4)
              
    # [2] Internal energy per spin (using 1/tanh instead of coth)
    U_exact = -J / np.tanh(2 * K) - (J / L**2) * (sum_ZR / Z_sum)
    
    # [3] Entropy per spin ( F = U - TS => S = (U - F)/T = beta*U + logZ/N )
    S_exact = beta * U_exact + log_Z / L**2
    
    return {
        "free_energy": f_exact.item(),
        "internal_energy": U_exact.item(),
        "entropy": S_exact.item()
    }


def ising_2d_empirical_observables(samples, beta, log_p=None, J=1.0):
    """
    Compute the empirical observables based on the generated samples.
    samples: torch.Tensor, shape [B, L, L], values should be {-1, 1}
    beta: float, inverse temperature
    log_p: (optional) torch.Tensor, shape [B], the exact log probability of the samples. If not provided, entropy and free energy cannot be estimated.
    """
    if isinstance(samples, np.ndarray):
        samples = torch.from_numpy(samples)
    if samples.dim() == 2:
        L = int(samples.shape[1]**0.5)
        assert samples.shape[1] == L ** 2
        samples = samples.view(-1, L, L)
    elif samples.dim() == 3:
        L = samples.shape[1]
        assert samples.shape[1] == samples.shape[2]
    else:
        raise ValueError("Samples should have shape [B, L*L] or [B, L, L]")

    # Compute the neighboring nodes under periodic boundary conditions
    right_neighbors = torch.roll(samples, shifts=-1, dims=2)
    down_neighbors = torch.roll(samples, shifts=-1, dims=1)
    
    # Compute the interaction for each spin pair only once (right and down)
    interaction = samples * right_neighbors + samples * down_neighbors
    E = -J * torch.sum(interaction, dim=(1, 2))  # Total energy of each sample, shape [B]
    
    # [1] Empirical internal energy per spin
    U_sample = E.float().mean().item() / (L**2)
    
    # [2] Empirical entropy and free energy (requires the log likelihood log_p computed by the model)
    S_sample, F_sample = None, None
    if log_p is not None:
        # S = - E[log p(x)] / N
        S_sample = -torch.mean(log_p).item() / (L**2)
        # F = U - T*S = U - S / beta
        F_sample = U_sample - S_sample / beta
        
    return {
        "free_energy_est": F_sample,
        "internal_energy_est": U_sample,
        "entropy_est": S_sample,
    }
