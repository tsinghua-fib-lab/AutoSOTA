import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from tqdm import tqdm


def potts2d_ham(S, J=1, q=3):
    """
    Compute the energy of a single 2D Ising configuration.
    S: Tensor of shape (L, L) with values in {0,1,...,q-1}
    J    : coupling constant   (float)
    Returns: energy.
    """
    assert S.ndim == 2, "Input tensor must have shape (B, L * L)"
    S = S.view(S.size(0), int(S.shape[1]**.5), int(S.shape[1]**.5))

    # periodic bcs, shift right on columns and down on rows
    s_left = torch.roll(S, shifts=1, dims=2)
    s_top = torch.roll(S, shifts=1, dims=1)
    s_right = torch.roll(S, shifts=-1, dims=2)
    s_down = torch.roll(S, shifts=-1, dims=1)

    # Count number of edges with same category
    equal_left = (S == s_left).int()
    equal_right = (S == s_right).int()
    equal_top = (S == s_top).int()
    equal_down = (S == s_down).int()
    interaction_per_node = (equal_left + equal_right + equal_top + equal_down)

    return -J * interaction_per_node.sum(dim=(1, 2)) / 2


def potts2d_magnetization_all(S, q):
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

    # Compute the most frequent state for each configuration
    most_frequent = np.zeros(B)
    for b in range(B):
        # Count occurrences of each state
        counts = np.bincount(S[b].flatten(), minlength=q)
        # Get the most frequent state
        most_frequent[b] = np.argmax(counts)

    # Compute magnetization as the fraction of spins in the most frequent state
    magnetization = np.array(
        [np.mean(S[b] == most_frequent[b]) for b in range(B)])
    return magnetization


def potts2d_magnetization(S, q, row=None, col=None):
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
        counts = np.apply_along_axis(lambda x: np.bincount(
            x, minlength=q), 1, row_data)  # Shape: (B, q)
        # Get most frequent state for each batch
        most_frequent = np.argmax(counts, axis=1)  # Shape: (B,)
        # Compute magnetization for all batches at once
        magnetization = (
            q * np.mean(row_data == most_frequent[:, None], axis=1) - 1) / (q - 1)
        return np.mean(magnetization)  # Average over batches

    # Compute magnetization for specific column
    elif col is not None:
        # Get the column data for all batches at once
        col_data = S[:, :, col]  # Shape: (B, L)
        # Count occurrences of each state for all batches at once
        counts = np.apply_along_axis(lambda x: np.bincount(
            x, minlength=q), 1, col_data)  # Shape: (B, q)
        # Get most frequent state for each batch
        most_frequent = np.argmax(counts, axis=1)  # Shape: (B,)
        # Compute magnetization for all batches at once
        magnetization = (
            q * np.mean(col_data == most_frequent[:, None], axis=1) - 1) / (q - 1)
        return np.mean(magnetization)  # Average over batches

    else:
        raise ValueError("Either row or col must be specified")


def potts2d_magnetization_site(S, q):
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
            magnetization[i, j] = (
                q * np.mean(site_states == most_frequent) - 1) / (q - 1)

    return magnetization


def potts2d_magnetization_ij(S, q):
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
        magnetization[i] = (
            q * np.mean(site_states == most_frequent) - 1) / (q - 1)

    # Reshape to (L, L) for the final result
    magnetization = magnetization.reshape(L, L)  # Shape: (L, L)

    return magnetization


def potts2d_mh(L, J=1, h=0, beta=.5, q=3, B=256, num_collect=20000,
               burn_in=10000, collect_every=1000, init=None):
    """
    Metropolis-Hastings algorithm to sample from the 2D Potts model's distribution.

    Parameters:
    - L: int, size of the lattice (L * L).
    - B: int, number of parallel configurations.
    - num_collect: int, number of times to collect.
    - burn_in: int, number of initial steps to discard (burn-in period).
    - collect_every: int, collect a sample every `collect_every` steps.
    - init: numpy.ndarray of shape (B, L, L) or (B, L * L), initial configuration.
            If None, random configurations are used.
    Returns:
    - samples: numpy.ndarray of shape (num_collect * B, L * L), sampled configurations.
    """
    # Initialize the lattice
    if init is None:
        S = np.random.randint(0, q, size=(B, L, L))
    else:
        S = init.reshape(B, L, L) if init.ndim == 2 else init

    samples = []
    total_steps = burn_in + num_collect * collect_every
    batch_arange = np.arange(B)

    for step in tqdm(range(total_steps)):
        # Randomly select sites to update
        i = np.random.randint(0, L, size=B)
        j = np.random.randint(0, L, size=B)

        # Propose new states
        current_spins = S[batch_arange, i, j]
        new_spins = np.random.randint(0, q, size=B)
        # Ensure new spin is different from current spin
        while np.any(new_spins == current_spins):
            mask = new_spins == current_spins
            new_spins[mask] = np.random.randint(0, q, size=np.sum(mask))

        # Get neighbors with periodic boundary conditions
        left = S[batch_arange, i, (j-1) % L]
        right = S[batch_arange, i, (j+1) % L]
        up = S[batch_arange, (i-1) % L, j]
        down = S[batch_arange, (i+1) % L, j]

        # Calculate energy difference with beta included
        current_E = -beta * J * ((current_spins == left) +
                                 (current_spins == right) +
                                 (current_spins == up) +
                                 (current_spins == down))

        new_E = -beta * J * ((new_spins == left) +
                             (new_spins == right) +
                             (new_spins == up) +
                             (new_spins == down))

        delta_E = new_E - current_E

        # Metropolis acceptance criterion (no need for beta here since it's already in delta_E)
        accept = np.log(np.random.random(size=B)) < -delta_E

        # Update accepted spins
        S[batch_arange[accept], i[accept], j[accept]] = new_spins[accept]

        # Collect samples after burn-in
        if step >= burn_in and (step - burn_in) % collect_every == 0:
            samples.append(S.reshape(B, L*L).copy())

    return np.concatenate(samples, axis=0)


def potts_2pt_corr_distance(S, q, r=1):
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


def potts_2pt_corr_direction(S, q, r_x=1, r_y=0, use_x=True, use_y=False):
    # Ensure input is a torch tensor
    if not isinstance(S, torch.Tensor):
        S = torch.from_numpy(S)

    if S.ndim == 2:
        S = S.reshape(S.shape[0], int(np.sqrt(S.shape[1])),
                      int(np.sqrt(S.shape[1])))
    B, L, L = S.shape
    corr = torch.zeros_like(S, dtype=torch.float)

    # Get neighbors at distance r in all directions using roll
    # Horizontal and vertical neighbors
    if use_x:
        neighbors = [torch.roll(S, shifts=r_x, dims=1),
                     torch.roll(S, shifts=-r_x, dims=1)]
    if use_y:
        neighbors = [torch.roll(S, shifts=r_y, dims=2),
                     torch.roll(S, shifts=-r_y, dims=2)]

    # Compute correlation for each neighbor
    for neighbor in neighbors:
        # For Potts model, correlation is 1 if states match, -1/(q-1) if they don't
        corr += torch.where(S == neighbor,
                            torch.ones_like(S, dtype=torch.float),
                            torch.zeros_like(S, dtype=torch.float))
    corr = corr / len(neighbors)

    return corr.mean() - 1/q


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


def potts2d_glauber(L, J=1, h=0, beta=.5, q=3, B=256, num_collect=20000,
                    burn_in=10000, collect_every=1000, init=None):
    """
    Glauber dynamics algorithm to sample from the 2D Potts model's distribution.
    Optimized version with vectorized operations.
    """
    # Initialize the lattice
    if init is None:
        S = np.random.randint(0, q, size=(B, L, L))
    else:
        S = init.reshape(B, L, L) if init.ndim == 2 else init

    samples = []
    total_steps = burn_in + num_collect * collect_every
    batch_arange = np.arange(B)

    # Pre-allocate arrays
    local_fields = np.zeros((B, q))
    exp_fields = np.zeros((B, q))

    betaJ = -beta * (-J)

    for step in tqdm(range(total_steps)):
        # Randomly select sites to update
        i = np.random.randint(0, L, size=B)
        j = np.random.randint(0, L, size=B)

        # Get neighbors with periodic boundary conditions
        left = S[batch_arange, i, (j-1) % L]
        right = S[batch_arange, i, (j+1) % L]
        up = S[batch_arange, (i-1) % L, j]
        down = S[batch_arange, (i+1) % L, j]

        # Vectorized calculation of local fields for all states at once
        # Create a (B, q) array where each row is [0,1,...,q-1]
        states = np.arange(q)[None, :].repeat(B, axis=0)

        # Calculate matching neighbors for all states at once
        # Shape: (B, q) - each element is number of matching neighbors for that state
        matches = ((states == left[:, None]) +
                   (states == right[:, None]) +
                   (states == up[:, None]) +
                   (states == down[:, None]))

        # Calculate local fields
        local_fields = betaJ * matches

        # Calculate probabilities using softmax (vectorized)
        exp_fields = np.exp(
            local_fields - np.max(local_fields, axis=1, keepdims=True))
        probs = exp_fields / np.sum(exp_fields, axis=1, keepdims=True)

        # Sample new states according to probabilities
        # Vectorized sampling using cumsum trick
        cumsum = np.cumsum(probs, axis=1)
        r = np.random.random(size=B)[:, None]
        new_spins = np.argmax(cumsum > r, axis=1)

        # Update spins
        S[batch_arange, i, j] = new_spins

        # Collect samples after burn-in
        if step >= burn_in and (step - burn_in) % collect_every == 0:
            samples.append(S.reshape(B, L*L).copy())

    return np.concatenate(samples, axis=0)


def potts2d_swendsen_wang(L, J=1, beta=.5, q=3, B=256, num_collect=20000,
                          burn_in=10000, collect_every=1000, init=None):
    """
    Swendsen-Wang algorithm to sample from the 2D Potts model's distribution.

    Parameters:
    - L: int, size of the lattice (L * L)
    - J: float, coupling constant
    - beta: float, inverse temperature
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
        S = np.random.randint(0, q, size=(B, L, L))
    else:
        S = init.reshape(B, L, L) if init.ndim == 2 else init

    samples = []
    total_steps = burn_in + num_collect * collect_every

    # Pre-compute bond probability
    p = 1 - np.exp(-beta * J)

    for step in tqdm(range(total_steps)):
        # For each configuration in the batch
        for b in range(B):
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
            samples.append(S.reshape(B, L*L).copy())

    return np.concatenate(samples, axis=0)


def visualize_potts(S, q, k_x, k_y):    # Convert to numpy if it's a torch tensor
    if S.ndim == 2:
        S = S.reshape(S.shape[0], int(np.sqrt(S.shape[1])),
                      int(np.sqrt(S.shape[1])))

    if isinstance(S, torch.Tensor):
        S = S.detach().cpu().numpy()
    assert k_x * \
        k_y == S.shape[0], "k_x * k_y must be equal to the number of samples"
    B = S.shape[0]
    # Check if B is a perfect square
    k = int(np.sqrt(B))
    fig, axes = plt.subplots(k_x, k_y, figsize=(
        1.5*k_x, 1.5*k_y), constrained_layout=True)
    axes = axes.ravel()  # Flatten the axes array for easy iteration

    # Create a colormap with q distinct colors

    palette = ["#540D6E", "#EE4266", "#FFD23F"]
    # palette = ["#26547c", "#ef476f", "#ffd166"]
    # palette = ["#1B3E76", "#BC98C6", "#FED71A"]
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

def project_energy_1d(energy_grid_2d, cv_grid_coords, project_dim, kT):
    """Project 2D energy landscape to 1D by marginalizing over one dimension.
    Similar to plot 'f' in the reference molecular system visualization.

    Formula: F(x) = -kT * log(∫ exp(-F(x,y)/kT) dy)

    Args:
        energy_grid_2d: 2D energy grid [nx, ny] in eV
        cv_grid_coords: List of 1D coordinate arrays [x_coords, y_coords]
        project_dim: Dimension to project onto (0 for x, 1 for y)
        kT: Temperature in energy units (eV). Should match units of energy_grid_2d.
    
    Returns:
        cv_coords: 1D array of coordinates
        f_projected: 1D array of projected free energies
        xlabel: Label for the x-axis
    """
    energy_grid_2d = np.asarray(energy_grid_2d)

    if project_dim == 0:
        # Project onto x: marginalize over y
        if cv_grid_coords[1].ndim == 2:
            y_coords_1d = cv_grid_coords[1][0, :]
        else:
            y_coords_1d = cv_grid_coords[1]

        exp_neg_energy = np.exp(-energy_grid_2d / kT)  # [nx, ny]
        integral = np.trapz(exp_neg_energy, x=y_coords_1d, axis=1)  # [nx]
        f_projected = -kT * np.log(integral + 1e-10)

        if cv_grid_coords[0].ndim == 2:
            cv_coords = cv_grid_coords[0][:, 0]
        else:
            cv_coords = cv_grid_coords[0]
        xlabel = 'CV 1 (x)'
    else:
        # Project onto y: marginalize over x
        if cv_grid_coords[0].ndim == 2:
            x_coords_1d = cv_grid_coords[0][:, 0]
        else:
            x_coords_1d = cv_grid_coords[0]

        exp_neg_energy = np.exp(-energy_grid_2d / kT)  # [nx, ny]
        integral = np.trapz(exp_neg_energy, x=x_coords_1d, axis=0)  # [ny]
        f_projected = -kT * np.log(integral + 1e-10)  # [ny]

        if cv_grid_coords[1].ndim == 2:
            cv_coords = cv_grid_coords[1][0, :]
        else:
            cv_coords = cv_grid_coords[1]
        xlabel = 'CV 2 (y)'

    # Shift to have minimum at 0
    f_projected = f_projected - f_projected.min()

    # Ensure cv_coords and f_projected are 1D arrays
    cv_coords = np.asarray(cv_coords).flatten()
    f_projected = np.asarray(f_projected).flatten()
    return cv_coords, f_projected, xlabel

def plot_1d_projected_energy(energy_grid_2d, cv_grid_coords,
                             project_dim=0, kT=1.0, ax=None, label=None,
                             figsize=(2.28, 2.0), fontsize=8, grid_alpha=0.3, normalize_kT=False):
    """
    Project 2D energy landscape to 1D by marginalizing over one dimension.
    Similar to plot 'f' in the reference molecular system visualization.

    Formula: F(x) = -kT * log(∫ exp(-F(x,y)/kT) dy)

    Args:
        energy_grid_2d: 2D energy grid [nx, ny] in eV
        cv_grid_coords: List of 1D coordinate arrays [x_coords, y_coords]
        project_dim: Dimension to project onto (0 for x, 1 for y)
        kT: Temperature in energy units (eV). Should match units of energy_grid_2d.
        ax: Matplotlib axis
        label: Label for legend
        figsize: Figure size (width, height)
        fontsize: Font size for labels
        grid_alpha: Alpha for grid lines
        normalize_kT: Normalize the free energy by kT
    Returns:
        ax: Matplotlib axis with plotted 1D projected free energy profile.
            Output units are the same as input energy units (eV if input is in eV).
    """

    # Use the extracted helper in main function
    cv_coords, f_projected, xlabel = project_energy_1d(
        energy_grid_2d, cv_grid_coords, project_dim, kT
    )

    if normalize_kT:
        f_projected = f_projected / kT
        ax.set_ylabel(r'Free Energy ($/k_\text{B}T$)', fontsize=fontsize)
    else:
        ax.set_ylabel('Free Energy', fontsize=fontsize)

    ax.plot(cv_coords, f_projected, label=label, linewidth=1.5)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    # Only set title if it hasn't been set yet (for multi-plot cases)
    # if ax.get_title() == '':
    #     ax.set_title(
    #         f"1D Projected Energy Profile (projected onto {xlabel})", fontsize=fontsize)
    ax.grid(True, alpha=grid_alpha)

    # Don't create legend here - let caller handle it for multi-plot cases
    # if label:
    #     ax.legend(fontsize=fontsize)

    return ax
