"""
Adapted from https://github.com/yuchen-zhu-zyc/MDNS/blob/main/utils_potts.py

Utility functions for the 2D Potts models
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def potts2d_ham(S, J=1.0):
    r"""
    Compute the energy of a single 2D Potts configuration.
    
    - S: torch.tensor of shape (B, L * L) with values in range(q)
    - J: float, interaction strength between neighboring spins (default=1.0).

    Returns:
    - hamiltonians: torch.tensor of shape (B,) containing the Hamiltonian for each configuration.
        H = -J * sum_{i ~ j} 1_{S_i == S_j}
    (The p.m.f. is given by p(S) \propto e^{-\beta H(S)})
    """
    # assert S.ndim == 2, "Input tensor must have shape (B, L * L)"
    S = S.view(S.size(0), int(S.shape[1]**.5), int(S.shape[1]**.5))

    S_left = torch.roll(S, shifts=1, dims=2)
    S_top = torch.roll(S, shifts=1, dims=1)
    interaction_per_node = (S == S_left).float() + (S == S_top).float()
    return -J * interaction_per_node.sum(dim = (1,2))


def potts2d_log_discrete_score(S, J=1.0, beta=1.0, q=3):
    r"""
    Compute the log of the discrete score for a batch of configurations in a 2D Potts model.
    
    Parameters:
    - S: torch.tensor of shape (B, L * L), values in range(q)
    - J: float, interaction strength between neighboring spins (default=1.0).
    - beta: float, inverse temperature (default=1.0).
    - q: int, number of states in the Potts model (default=3).

    Returns:
    - scores: torch.tensor of shape (B, L * L, q):
        scores[b, i, n] = log p(S[b] ^ {i<-n}) / p(S[b])
                        = beta * J * \sum_{j ~ i} (1_{S[j] = n} - 1_{S[i] = S[j]})
        where n in range(q)
    """
    S = S.view(S.shape[0], int(S.shape[1]**.5), int(S.shape[1]**.5))
    neighbors = [torch.roll(S, shifts=1, dims=2), torch.roll(S, shifts=-1, dims=2),
                 torch.roll(S, shifts=1, dims=1), torch.roll(S, shifts=-1, dims=1)]
    colors = torch.arange(q, device=S.device)

    eq_neigbors = sum([(S == x).float() for x in neighbors])  # [B, L, L]
    eq_colors = sum([(x.unsqueeze(-1) == colors).float() for x in neighbors])  # [B, L, L, q]
    scores = beta * J * (eq_colors - eq_neigbors.unsqueeze(-1))  # [B, L, L, q]
    return scores.view(S.shape[0], -1, q)  # [B, L*L, q]


def potts2d_visualize(S: torch.tensor, num_per_row: int = 8, q: int = 3):
    """
    Visualize multiple Potts configurations in a grid.
    Args:
        S: 2D tensor of shape (N, D=L^2) representing N Potts configurations, values in range(q)
        num_per_row: Number of configurations to display per row.
    """
    N, D = S.shape; L = int(D**0.5); 
    assert L**2 == D, "The number of columns must be a perfect square."
    num_rows = (N + num_per_row - 1) // num_per_row
    fig, axes = plt.subplots(num_rows, num_per_row, figsize=(num_per_row * 2, num_rows * 2))
    axes = axes.flatten()
    
    for i in range(len(axes)):
        if i < N:
            axes[i].imshow(S[i].reshape(L, L).cpu().numpy(), cmap='inferno',
                           interpolation='nearest', vmin=0, vmax=q-1)
        axes[i].axis('off')
    fig.tight_layout()
    return fig


def potts2d_2pt_corr(S, r, q=3):
    """
    Calculate the 2-point correlation function of Potts model samples.
    Args:
        S (torch.Tensor): Potts model samples of shape (B, D=L^2)
        r (int): Distance between points to compute correlation
        q (int): Number of states (0 to q-1)
    Returns:
        torch.Tensor: average correlation, shape (B,)
    """
    B, D = S.shape; L = int(np.sqrt(D))
    assert L**2 == D, "The number of columns must be a perfect square."
    S = S.view(B, L, L)
    neighbors = [
        torch.roll(S, shifts=r, dims=1), torch.roll(S, shifts=-r, dims=1),
        torch.roll(S, shifts=r, dims=2), torch.roll(S, shifts=-r, dims=2),
    ]
    corr = sum((S == neighbor).int() for neighbor in neighbors) / 4
    return corr.mean() - 1/q


def potts2d_plot_2pt_corr(S, q=3):
    """
    Plot the 2-point correlation function for a batch of Potts configurations.

    Args:
        S: 2D tensor of shape (B, D=L^2) representing N Potts configurations, values in range(q).
    """
    B, D = S.shape; L = int(np.sqrt(D))
    assert L ** 2 == D, "The number of columns must be a perfect square."

    plt.close('all')
    fig = plt.figure()
    r = np.arange(-L//2, L//2 + 1)
    corr = [potts2d_2pt_corr(S, i, q).item() for i in r]
    plt.plot(r, corr, marker='o')
    plt.xlabel('Distance $r$')
    plt.xticks(r, [f'${i}$' for i in r])
    plt.ylim(-0.05, 1)
    plt.ylabel('2-point Correlation')
    for i, c in enumerate(corr):
        plt.text(r[i], c + 0.02, f'{c:.3f}', ha='center', fontsize=8)
    plt.title('2-point Correlation Function')
    plt.grid(True, alpha=0.3)
    return fig


def potts2d_mag(S: torch.Tensor, q: int):
    """
    Compute the magnetization of the 2D Potts model for all sites.
    
    Args:
        S (torch.Tensor): Potts model samples of shape (B, L, L) or (B, L*L), values in range(q)
        q (int): Number of states (0 to q-1)
    
    Returns:
        torch.Tensor: Magnetization values between 0 and 1 for each sample, shape (B,)
    """
    S_flat = S.reshape(S.shape[0], -1)
    S_onehot = F.one_hot(S_flat.long(), num_classes=q)
    counts = S_onehot.sum(dim=1)
    most_frequent = counts.argmax(dim=1)
    return (S_flat == most_frequent.unsqueeze(1)).float().mean(dim=1)


def potts2d_mag_row(S, q, row=None, col=None):
    """
    Compute the magnetization of the 2D Potts model for a specific row or column.
    
    Args:
        S (torch.Tensor or numpy.ndarray): Potts model samples of shape (B, L, L) or (B, L*L)
        q (int): Number of states (0 to q-1)
        row (int, optional): Row index to compute magnetization for
        col (int, optional): Column index to compute magnetization for
    
    Returns:
        float: Magnetization value between 0 and 1 for the specified row or column
    """
    # Convert to numpy if it's a torch tensor
    if isinstance(S, torch.Tensor):
        S = S.detach().cpu().numpy()
    
    # Reshape if needed
    if S.ndim == 2:
        B, L2 = S.shape
        L = int(np.sqrt(L2))
        S = S.reshape(B, L, L)
    else:
        B, L, L = S.shape
    
    
    # Compute magnetization for specific row
    if row is not None:
        # Get the row data for all batches at once
        row_data = S[:, row, :]  # Shape: (B, L)
        # Count occurrences of each state for all batches at once
        counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=q), 1, row_data)  # Shape: (B, q)
        # Get most frequent state for each batch
        most_frequent = np.argmax(counts, axis=1)  # Shape: (B,)
        # Compute magnetization for all batches at once
        magnetization = (q * np.mean(row_data == most_frequent[:, None], axis=1) - 1) / (q - 1)
        return np.mean(magnetization)  # Average over batches
    
    # Compute magnetization for specific column
    elif col is not None:
        # Get the column data for all batches at once
        col_data = S[:, :, col]  # Shape: (B, L)
        # Count occurrences of each state for all batches at once
        counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=q), 1, col_data)  # Shape: (B, q)
        # Get most frequent state for each batch
        most_frequent = np.argmax(counts, axis=1)  # Shape: (B,)
        # Compute magnetization for all batches at once
        magnetization = (q * np.mean(col_data == most_frequent[:, None], axis=1) - 1) / (q - 1)
        return np.mean(magnetization)  # Average over batches
    
    else:
        raise ValueError("Either row or col must be specified")

def potts2d_mag_site(S, q):
    """
    Compute the magnetization for each individual site in the 2D Potts model.
    
    Args:
        S (torch.Tensor or numpy.ndarray): Potts model samples of shape (B, L, L) or (B, L*L)
        q (int): Number of states (0 to q-1)
    
    Returns:
        numpy.ndarray: LxL matrix where each entry (i,j) is the magnetization for that site
    """
    # Convert to numpy if it's a torch tensor
    if isinstance(S, torch.Tensor):
        S = S.detach().cpu().numpy()
    
    # Reshape if needed
    if S.ndim == 2:
        B, L2 = S.shape
        L = int(np.sqrt(L2))
        S = S.reshape(B, L, L)
    else:
        B, L, L = S.shape
    
    # Initialize magnetization matrix
    magnetization = np.zeros((L, L))
    
    # For each site, compute its magnetization
    for i in range(L):
        for j in range(L):
            # Get the state at this site for all batches
            site_states = S[:, i, j]  # Shape: (B,)
            # Count occurrences of each state
            counts = np.bincount(site_states, minlength=q)
            # Get the most frequent state
            most_frequent = np.argmax(counts)
            # Compute magnetization for this site
            magnetization[i, j] = (q * np.mean(site_states == most_frequent) - 1) / (q - 1)
    
    return magnetization

def potts2d_mag_ij(S, q):
    """
    Compute the magnetization for all sites in the 2D Potts model.
    
    Args:
        S (torch.Tensor or numpy.ndarray): Potts model samples of shape (B, L, L) or (B, L*L)
        q (int): Number of states (0 to q-1)
    
    Returns:
        numpy.ndarray: LxL matrix where each entry (i,j) is the magnetization at that site
    """
    # Convert to numpy if it's a torch tensor
    if isinstance(S, torch.Tensor):
        S = S.detach().cpu().numpy()
    
    # Reshape if needed
    if S.ndim == 2:
        B, L2 = S.shape
        L = int(np.sqrt(L2))
        S = S.reshape(B, L, L)
    else:
        B, L, L = S.shape
    
    # Reshape S to (B, L*L) for easier processing
    S_flat = S.reshape(B, L*L)  # Shape: (B, L*L)
    
    # Initialize magnetization array
    magnetization = np.zeros(L*L)
    
    # Process each site
    for i in range(L*L):
        # Get states for this site across all batches
        site_states = S_flat[:, i]  # Shape: (B,)
        # Count occurrences of each state
        counts = np.bincount(site_states, minlength=q)
        # Get most frequent state
        most_frequent = np.argmax(counts)
        # Compute magnetization for this site
        magnetization[i] = (q * np.mean(site_states == most_frequent) - 1) / (q - 1)
    
    # Reshape to (L, L) for the final result
    magnetization = magnetization.reshape(L, L)  # Shape: (L, L)
    
    return magnetization


def potts_2pt_corr_distance(S, q, r = 1):
    """
    Calculate the 2-point correlation function of Potts model samples.
    
    Args:
        S (torch.Tensor): Potts model samples of shape (B, L, L)
        q (int): Number of states (0 to q-1)
        r (int): Distance between points to compute correlation
        return_full (bool): If True, return full correlation map. If False, return average correlation per batch.
    Returns:
        torch.Tensor: 
        returns average correlation of shape (B,)
    """
    # Ensure input is a torch tensor
    if not isinstance(S, torch.Tensor):
        S = torch.from_numpy(S)
    
    B, L, L = S.shape
    corr = torch.zeros_like(S, dtype=torch.float)
    
    # Get neighbors at distance r in all directions using roll
    # Horizontal and vertical neighbors
    neighbors = [
        torch.roll(S, shifts=r, dims=1),  # right
        torch.roll(S, shifts=-r, dims=1),  # left
        torch.roll(S, shifts=r, dims=2),   # down
        torch.roll(S, shifts=-r, dims=2),  # up
    ]
    
    # Compute correlation for each neighbor
    for neighbor in neighbors:
        # For Potts model, correlation is 1 if states match, -1/(q-1) if they don't
        corr += torch.where(S == neighbor, 
                           torch.ones_like(S, dtype=torch.float),
                           torch.zeros_like(S, dtype=torch.float))
    
    # Average over all neighbors
    corr = corr / len(neighbors)

    return corr.mean() - 1/q

def potts_2pt_corr_direction(S, q, r_x = 1, r_y = 0, use_x = True, use_y = False):
    # Ensure input is a torch tensor
    if not isinstance(S, torch.Tensor):
        S = torch.from_numpy(S)
        
    if S.ndim == 2:
        S = S.reshape(S.shape[0], int(np.sqrt(S.shape[1])), int(np.sqrt(S.shape[1])))
    B, L, L = S.shape
    corr = torch.zeros_like(S, dtype=torch.float)
    
    # Get neighbors at distance r in all directions using roll
    # Horizontal and vertical neighbors
    if use_x:
        neighbors = [torch.roll(S, shifts=r_x, dims=1),  torch.roll(S, shifts=-r_x, dims=1)]
    if use_y:
        neighbors = [torch.roll(S, shifts=r_y, dims=2),  torch.roll(S, shifts=-r_y, dims=2)]
    
    # Compute correlation for each neighbor
    for neighbor in neighbors:
        # For Potts model, correlation is 1 if states match, -1/(q-1) if they don't
        corr += torch.where(S == neighbor, 
                           torch.ones_like(S, dtype=torch.float),
                           torch.zeros_like(S, dtype=torch.float))
    corr = corr / len(neighbors)

    return corr.mean() - 1/q


def potts2d_balance(S: torch.Tensor, q: int) -> float:
    """Compute the balance (maximum deviation from uniform distribution) of Potts samples.

    Args:
        S: Tensor of shape (B, L ^ 2) with values in range(q)
        q: Number of Potts states

    Returns:
        balance: float, maximum deviation from uniform distribution across all states
    """
    colors = torch.arange(q, device=S.device)
    freqs = (S[..., None] == colors).float().mean(dim=(0, 1))  # (q,)
    return (freqs - 1 / q).abs().max().item()


def get_all_potts_configs(L, q):
    """
    Generate all possible configurations for a 2D Potts model.
    
    Args:
        L (int): Size of the lattice (L x L)
        q (int): Number of states (0 to q-1)
    
    Returns:
        torch.Tensor: All possible configurations of shape (q^(L^2), L*L)
                     Each row is a flattened configuration with values in {0,1,...,q-1}
    """
    # Total number of configurations

    # Generate all possible configurations using meshgrid
    # First create a list of possible values for each site
    values = [np.arange(q) for _ in range(L * L)]
    
    # Use meshgrid to generate all combinations
    configs = np.array(np.meshgrid(*values)).T.reshape(-1, L * L)
    
    return torch.from_numpy(configs)


def visualize_potts(S, q, k_x, k_y):    # Convert to numpy if it's a torch tensor
    if S.ndim == 2:
        S = S.reshape(S.shape[0], int(np.sqrt(S.shape[1])), int(np.sqrt(S.shape[1])))
    
    if isinstance(S, torch.Tensor):
        S = S.detach().cpu().numpy()
    assert k_x * k_y == S.shape[0], "k_x * k_y must be equal to the number of samples"
    B = S.shape[0]
    # Check if B is a perfect square
    k = int(np.sqrt(B))
    fig, axes = plt.subplots(k_x, k_y, figsize=(1.5*k_x, 1.5*k_y), constrained_layout=True)
    axes = axes.ravel()  # Flatten the axes array for easy iteration
    
    # Create a colormap with q distinct colors
    
    palette = ["#540D6E", "#EE4266", "#FFD23F"]
    # palette = ["#26547c", "#ef476f", "#ffd166"]
    # palette = ["#FED71A", "#BC98C6", "#1B3E76"]
    # palette = ["#440154", "#FDE725", "#ff7f00"]
    cmap = ListedColormap(palette)    
    for i in range(B):
        # Plot heatmap for each sample
        im = axes[i].imshow(S[i], cmap=cmap, vmin=0, vmax=q-1, origin='lower',
            interpolation='nearest')
        # axes[i].set_title(f'Sample {i+1}')
        axes[i].axis('off')
    # Add colorbar
    # cbar = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05)
    # cbar.set_ticks(np.arange(q))
    # cbar.set_ticklabels(np.arange(q))
    plt.tight_layout()
    return fig


def potts2d_mh(L, beta=.5, J=1.0, h=0.0, q=3, batch_size=256, num_collect=20000, 
               burn_in=10000, collect_every=1000, init=None):
    """
    Metropolis-Hastings algorithm to sample from the 2D Potts model's distribution.

    Parameters:
    - L: int, size of the lattice (L * L).
    - beta: float, inverse temperature.
    - J: float, coupling constant
    - h: float, external field. The current version only supports h = 0.
    - q: int, number of states (0 to q-1)
    - batch_size: int, number of parallel configurations.
    - num_collect: int, number of times to collect.
    - burn_in: int, number of initial steps to discard (burn-in period).
    - collect_every: int, collect a sample every `collect_every` steps.
    - init: numpy.ndarray of shape (B, L, L) or (B, L * L), initial configuration.
            If None, random configurations are used.
    Returns:
    - samples: numpy.ndarray of shape (num_collect * B, L * L), sampled configurations.
    """
    if init is None:
        S = np.random.randint(0, q, size=(batch_size, L, L))
    else:
        S = init.reshape(batch_size, L, L) if init.ndim == 2 else init
    
    samples = []
    total_steps = burn_in + num_collect * collect_every
    arange_B = np.arange(batch_size)
    
    for step in tqdm(range(total_steps)):
        i = np.random.randint(0, L, size=batch_size); j = np.random.randint(0, L, size=batch_size)
        
        current_spins = S[arange_B, i, j]
        new_spins = np.random.randint(0, q, size=batch_size)
        while True:
            mask = new_spins == current_spins
            if not np.any(mask):
                break # Ensure new spin is different from current spin
            new_spins[mask] = np.random.randint(0, q, size=np.sum(mask))
        
        left = S[arange_B, i, (j-1)%L]; right = S[arange_B, i, (j+1)%L]
        up = S[arange_B, (i-1)%L, j]; down = S[arange_B, (i+1)%L, j]
        
        H_old = -J * ((current_spins == left) + (current_spins == right) + 
                      (current_spins == up) +  (current_spins == down))
        H_new = -J * ((new_spins == left) + (new_spins == right) + 
                      (new_spins == up) + (new_spins == down))
        accept = np.random.random(size=batch_size) < np.exp(-beta * (H_new - H_old))
        S[arange_B[accept], i[accept], j[accept]] = new_spins[accept]
        
        if step >= burn_in and (step - burn_in) % collect_every == 0:
            samples.append(np.copy(S))
    return np.array(samples).reshape(-1, L*L)


def potts2d_glauber(L, beta=.5, J=1.0, h=0.0, q=3, batch_size=256, num_collect=20000, 
                    burn_in=10000, collect_every=1000, init=None):
    """
    Glauber dynamics algorithm to sample from the 2D Potts model's distribution.
    Optimized version with vectorized operations.
    """
    # Initialize the lattice
    if init is None:
        S = np.random.randint(0, q, size=(batch_size, L, L))
    else:
        S = init.reshape(batch_size, L, L) if init.ndim == 2 else init
    
    samples = []
    total_steps = burn_in + num_collect * collect_every
    batch_arange = np.arange(batch_size)
    
    # Pre-allocate arrays
    local_fields = np.zeros((batch_size, q))
    exp_fields = np.zeros((batch_size, q))
    
    betaJ = -beta * (-J)
    
    for step in tqdm(range(total_steps)):
        # Randomly select sites to update
        i = np.random.randint(0, L, size=batch_size)
        j = np.random.randint(0, L, size=batch_size)
        
        # Get neighbors with periodic boundary conditions
        left = S[batch_arange, i, (j-1)%L]
        right = S[batch_arange, i, (j+1)%L]
        up = S[batch_arange, (i-1)%L, j]
        down = S[batch_arange, (i+1)%L, j]
        
        # Vectorized calculation of local fields for all states at once
        # Create a (B, q) array where each row is [0,1,...,q-1]
        states = np.arange(q)[None, :].repeat(batch_size, axis=0)
        
        # Calculate matching neighbors for all states at once
        # Shape: (B, q) - each element is number of matching neighbors for that state
        matches = ((states == left[:, None]).astype(int) + 
                  (states == right[:, None]).astype(int) + 
                  (states == up[:, None]).astype(int) + 
                  (states == down[:, None]).astype(int))
        
        # Calculate local fields
        local_fields = betaJ * matches
        
        # Calculate probabilities using softmax (vectorized)
        exp_fields = np.exp(local_fields - np.max(local_fields, axis=1, keepdims=True))
        probs = exp_fields / np.sum(exp_fields, axis=1, keepdims=True)
        
        # Sample new states according to probabilities
        # Vectorized sampling using cumsum trick
        cumsum = np.cumsum(probs, axis=1)
        r = np.random.random(size=batch_size)[:, None]
        new_spins = np.argmax(cumsum > r, axis=1)
        
        # Update spins
        S[batch_arange, i, j] = new_spins
        
        # Collect samples after burn-in
        if step >= burn_in and (step - burn_in) % collect_every == 0:
            samples.append(S.reshape(batch_size, L*L).copy())
    
    return np.concatenate(samples, axis=0)


def potts2d_swendsen_wang(L, beta=.5, J=1.0, h=0.0, q=3, batch_size=256, num_collect=20000, 
                         burn_in=10000, collect_every=1000, init=None):
    """
    Swendsen-Wang algorithm to sample from the 2D Potts model's distribution.
    
    Parameters:
    - L: int, size of the lattice (L * L)
    - beta: float, inverse temperature
    - J: float, coupling constant
    - h: float, external magnetic field. The current version only supports h = 0.
    - q: int, number of states (0 to q-1)
    - B: int, number of parallel configurations
    - num_collect: int, number of times to collect
    - burn_in: int, number of initial steps to discard
    - collect_every: int, collect a sample every `collect_every` steps
    - init: numpy.ndarray of shape (B, L, L) or (B, L * L), initial configuration
    
    Returns:
    - samples: numpy.ndarray of shape (num_collect * B, L * L), sampled configurations
    """
    # Initialize the lattice
    if init is None:
        S = np.random.randint(0, q, size=(batch_size, L, L))
    else:
        S = init.reshape(batch_size, L, L) if init.ndim == 2 else init
    
    samples = []
    total_steps = burn_in + num_collect * collect_every
    
    # Pre-compute bond probability
    p = 1 - np.exp(-beta * J)
    
    for step in tqdm(range(total_steps)):
        # For each configuration in the batch
        for b in range(batch_size):
            # Step 1: Identify bonds between same-state neighbors
            # Create arrays for horizontal and vertical bonds
            h_bonds = np.zeros((L, L), dtype=bool)  # horizontal bonds
            v_bonds = np.zeros((L, L), dtype=bool)  # vertical bonds
            
            # Check horizontal bonds
            h_bonds[:, :-1] = (S[b, :, :-1] == S[b, :, 1:])
            h_bonds[:, -1] = (S[b, :, -1] == S[b, :, 0])  # periodic BC
            
            # Check vertical bonds
            v_bonds[:-1, :] = (S[b, :-1, :] == S[b, 1:, :])
            v_bonds[-1, :] = (S[b, -1, :] == S[b, 0, :])  # periodic BC
            
            # Step 2: Activate bonds with probability p
            h_bonds = h_bonds & (np.random.random((L, L)) < p)
            v_bonds = v_bonds & (np.random.random((L, L)) < p)
            
            # Step 3: Identify clusters using Union-Find
            # Initialize parent array for Union-Find
            parent = np.arange(L * L).reshape(L, L)
            rank = np.zeros((L, L), dtype=int)
            
            def find(x, y):
                if parent[x, y] != x * L + y:
                    px, py = parent[x, y] // L, parent[x, y] % L
                    parent[x, y] = find(px, py)
                return parent[x, y]
            
            def union(x1, y1, x2, y2):
                root1 = find(x1, y1)
                root2 = find(x2, y2)
                if root1 != root2:
                    r1, c1 = root1 // L, root1 % L
                    r2, c2 = root2 // L, root2 % L
                    if rank[r1, c1] < rank[r2, c2]:
                        parent[r1, c1] = root2
                    else:
                        parent[r2, c2] = root1
                        if rank[r1, c1] == rank[r2, c2]:
                            rank[r1, c1] += 1
            
            # Process horizontal bonds
            for i in range(L):
                for j in range(L):
                    if h_bonds[i, j]:
                        union(i, j, i, (j + 1) % L)
            
            # Process vertical bonds
            for i in range(L):
                for j in range(L):
                    if v_bonds[i, j]:
                        union(i, j, (i + 1) % L, j)
            
            # Step 4: Identify clusters
            clusters = {}
            for i in range(L):
                for j in range(L):
                    root = find(i, j)
                    if root not in clusters:
                        clusters[root] = []
                    clusters[root].append((i, j))
            
            # Step 5: Flip clusters
            for cluster in clusters.values():
                # Randomly choose new state for the cluster
                new_state = np.random.randint(0, q)
                # Update all spins in the cluster
                for i, j in cluster:
                    S[b, i, j] = new_state
        
        # Collect samples after burn-in
        if step >= burn_in and (step - burn_in) % collect_every == 0:
            samples.append(S.reshape(batch_size, L*L).copy())
    
    return np.concatenate(samples, axis=0)
