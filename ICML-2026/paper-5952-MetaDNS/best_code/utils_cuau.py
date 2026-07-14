
# Map from index to atomic number
number_to_element: dict[int, int] = {
    0: 29,  # Cu
    1: 79,  # Au
    2: 1,  # mask
}
element_to_number: dict[int, int] = {v: k for k, v in number_to_element.items()}
num_elements: int = len(number_to_element)


# Mask index for discrete flow
mask_index: int = len(number_to_element) - 1

def get_sublattice_map(atoms, size, lattice_constant=3.8):
    """
    Generate sublattice mapping for FCC supercell using Cartesian coordinates.
    Robust for both primitive and conventional cells.
    
    Args:
        atoms: ASE Atoms object of the supercell.
        size: Tuple (Nx, Ny, Nz) - unused for mapping but kept for compatibility.
        lattice_constant: Lattice parameter 'a' in Angstroms (default 3.8 for CuAu).
    
    Returns:
        torch.LongTensor of shape [num_sites] with values 0-3 (Alpha, Beta, Gamma, Delta).
    """
    import numpy as np
    import torch
    
    # Use Cartesian positions to be independent of cell basis (Primitive vs Conventional)
    # Shape: [N_sites, 3]
    pos = atoms.get_positions()
    
    # We define sublattices on the conventional cubic grid of lattice constant 'a'.
    # Sites are at multiples of a/2.
    # Grid units = pos / (a/2)
    grid_units = pos / (lattice_constant / 2.0)
    
    # Round to nearest integer (handles small relaxations or float errors)
    ints = np.rint(grid_units).astype(int)
    
    x = ints[:, 0]
    y = ints[:, 1]
    z = ints[:, 2]
    
    is_x_odd = (x % 2 != 0)
    is_y_odd = (y % 2 != 0)
    is_z_odd = (z % 2 != 0)
    
    # Initialize map with -1
    sub_map = np.full(len(atoms), -1, dtype=int)
    
    # Alpha: (0, 0, 0) * a/2 -> Even, Even, Even
    mask_alpha = (~is_x_odd) & (~is_y_odd) & (~is_z_odd)
    sub_map[mask_alpha] = 0
    
    # Beta: (1, 1, 0) * a/2  -> Odd, Odd, Even (Face Center XY)
    mask_beta = is_x_odd & is_y_odd & (~is_z_odd)
    sub_map[mask_beta] = 1
    
    # Gamma: (1, 0, 1) * a/2 -> Odd, Even, Odd (Face Center XZ)
    mask_gamma = is_x_odd & (~is_y_odd) & is_z_odd
    sub_map[mask_gamma] = 2
    
    # Delta: (0, 1, 1) * a/2 -> Even, Odd, Odd (Face Center YZ)
    mask_delta = (~is_x_odd) & is_y_odd & is_z_odd
    sub_map[mask_delta] = 3
    
    return torch.tensor(sub_map, dtype=torch.long)

def compute_order_parameter(x, sublattice_map, num_sites):
    """
    Compute Global Order Parameter Q for CuAu L10/L12 ordering.
    
    Args:
        x: [B, num_sites] Tensor with binary occupations (1 for Au, 0 for Cu).
        sublattice_map: [num_sites] Tensor mapping sites to 0-3.
        num_sites: Total number of sites (integer).
    
    Returns:
        Q: [B] Tensor of global order parameters.
    """
    import torch
    
    # x is occupation of Au (1) or Cu (0).
    # We need counts of Au on each sublattice.
    
    # One-hot encode sublattice map: [num_sites, 4]
    # This is static, so could be cached, but for now we compute on fly or assume efficient enough.
    # Actually, scatter_add is better.
    
    B = x.shape[0]
    device = x.device
    
    # Ensure map is on device
    if sublattice_map.device != device:
        sublattice_map = sublattice_map.to(device)
        
    # Prepare storage for counts: [B, 4]
    # We can use index_add_ or scatter_add
    # x: [B, N]
    # sub_map: [N] -> expand to [B, N]
    
    # sub_map_expanded = sublattice_map.unsqueeze(0).expand(B, -1)
    
    # Actually, simpler: matmul if we precompute matrix M [4, N]
    # But N is large.
    # Let's use scatter add.
    
    # We want to sum x_i for all i where map[i] == k
    counts = torch.zeros(B, 4, device=device, dtype=x.dtype)
    
    # x is float usually? If indices, need to convert.
    # Assuming x is binary 0/1 float tensor from relax/sample?
    # Or indices? In utils_train, x is usually indices [B, L]
    # In CuAu, 0=Cu, 1=Au.
    # So if x contains 0/1, we can sum directly.
    # If x contains 29/79, we need to map.
    # The Reward function receives x as 0/1/2 indices.
    # We assume x is [B, L] with 0 or 1.
    
    x_float = x.float()
    
    # Add to counts
    # We can't vectorise easily with scatter without flattening
    # But since N is small (64-256), loop over 4 sublattices is allowed.
    
    # N_alpha
    for k in range(4):
        mask = (sublattice_map == k)
        # Sum x over masked sites
        # mask is [N], x is [B, N]
        counts[:, k] = (x_float[:, mask]).sum(dim=1)
        
    # Normalize to concentrations n_i
    # Each sublattice has num_sites / 4 sites.
    sites_per_sub = num_sites / 4.0
    n = counts / sites_per_sub # [B, 4]
    
    n_alpha = n[:, 0]
    n_beta = n[:, 1]
    n_gamma = n[:, 2]
    n_delta = n[:, 3]
    
    # S_z = 0.5 * |na + nb - ng - nd|
    # S_x = 0.5 * |na - nb + ng - nd|
    # S_y = 0.5 * |na - nb - ng + nd|
    
    S_z = 0.5 * torch.abs(n_alpha + n_beta - n_gamma - n_delta)
    S_y = 0.5 * torch.abs(n_alpha - n_beta + n_gamma - n_delta)
    S_x = 0.5 * torch.abs(n_alpha - n_beta - n_gamma + n_delta)
    
    # Q = sqrt(Sx^2 + Sy^2 + Sz^2)
    Q = torch.sqrt(S_x**2 + S_y**2 + S_z**2)
    
    return Q

