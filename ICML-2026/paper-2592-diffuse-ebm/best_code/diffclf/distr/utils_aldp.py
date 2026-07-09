# Utils for ALDP

# Libraries
import mdtraj as md
import networkx as nx
import numpy as np
import torch
from networkx.algorithms import isomorphism
from tqdm import trange
from ..utils.se3_utils import interatomic_dist

# Mapping from atoms types to indexes
atom_dict = {"C": 0, "H":1, "N":2, "O":3, "S":4}

def torch_to_mdtraj(samples, topology):
    """Convert torch tensor of samples to mdtraj.Trajectory"""
    traj = md.Trajectory(samples.cpu().numpy(), topology=topology)
    return traj

def create_adjacency_list(distance_matrix, atom_types):
    """Building an adjacency list representation of a molecular
    graph based on a distance matrix and a list of atom types.

    Args:
        * distance_matrix (torch.Tensor or numpy.Array of shape (N, N)): Interatomic distances
        * atom_types (torch.Tensor or numpy.Array of int of shape (N,)): Atom types

    Returns:
        * adjacency_list (list): Adjacency list

    """
    adjacency_list = []
    num_nodes = len(distance_matrix)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  # Avoid duplicate pairs
            distance = distance_matrix[i][j]
            element_i = atom_types[i]
            element_j = atom_types[j]
            if 1 in (element_i, element_j):
                distance_cutoff = 0.14
            elif 4 in (element_i, element_j):
                distance_cutoff = 0.22
            elif 0 in (element_i, element_j):
                distance_cutoff = 0.18
            else:
                # elements should not be bonded
                distance_cutoff = 0.0
            # Add edge if distance is below the cutoff
            if distance < distance_cutoff:
                adjacency_list.append([i,j])
    return adjacency_list


def find_chirality_centers(adj_list, atom_types, num_h_atoms=2):
    """Returns the chirality centers for a peptide, e.g. carbon alpha atoms and their bonds.

        TODO: Optimize by precomputing neighbors per atom

    Args:
        * adj_list (list): List of bonds
        * atom_types (list of int): List of atom types
        * num_h_atoms (int): If num_h_atoms or more hydrogen atoms connected to the center,
            it is not reported. (Default is 2, because in this case the mirroring is a simple permutation.)

    Returns:
        chirality_centers
    """
    chirality_centers = []
    candidate_chirality_centers = torch.where(torch.unique(adj_list, return_counts=True)[1] == 4)[0]
    for center in candidate_chirality_centers:
        bond_idx, bond_pos = torch.where(adj_list == center)
        bonded_idxs = adj_list[bond_idx, (bond_pos + 1) % 2].long()
        adj_types = atom_types[bonded_idxs]
        if torch.count_nonzero(adj_types - 1) > num_h_atoms:
            chirality_centers.append([center, *bonded_idxs[:3]])
    return torch.tensor(chirality_centers).to(adj_list).long()


def compute_chirality_sign(coords, chirality_centers):
    """Computes indicator signs for a given configuration. If the signs for two configurations are
        different for the same center, the chirality changed.

    Args:
        * coords (torch.Tensor of shape (batch_size, n_particles, n_dimensions)): Atoms coordinates
        * chirality_centers (torch.Tensor): List of chirality_centers

    Returns:
        * indicator_signs (torch.Tensor of shape (batch_size, n_particles)): Indicator sign
    """
    direction_vectors = (
        coords[:, chirality_centers[:, 1:], :] - coords[:, chirality_centers[:, [0]], :]
    )
    perm_sign = torch.einsum(
        "ijk, ijk->ij",
        direction_vectors[:, :, 0],
        torch.cross(direction_vectors[:, :, 1], direction_vectors[:, :, 2], dim=-1),
    )
    return torch.sign(perm_sign)


def check_symmetry_change(coords, chirality_centers, reference_signs):
    """Check for a batch if the chirality changed with respecto to some reference reference_signs.
    If the signs for two configurations are different for the same center, the chirality changed.

    Args:
        * coords (torch.Tensor of shape (batch_size, n_particles, n_dimensions)): Atoms coordinates
        * chirality_centers (torch.Tensor): List of chirality_centers
        * reference_signs (torch.Tensor of shape (batch_size,)): Indicator sign

    Returns:
        * mask (torch.Tensor of shape (batch_size,)): Mask indicating the changes
    """
    perm_sign = compute_chirality_sign(coords, chirality_centers)
    return (perm_sign != reference_signs.to(coords)).any(dim=-1)

def filter_chirality(samples: torch.Tensor, traj: md.Trajectory):
    assert samples.dim() == 3, "samples should be a batch of configurations, (Batch_size, n_particles, n_dimensions)"
    atom_dict = {"C": 0, "H":1, "N":2, "O":3, "S":4}
    atom_types = []
    for atom_name in traj.topology.atoms:
        atom_types.append(atom_name.name[0])        
    atom_types = torch.from_numpy(np.array([atom_dict[atom_type] for atom_type in atom_types]))
    adj_list = torch.from_numpy(np.array([(b.atom1.index, b.atom2.index) for b in traj.topology.bonds], dtype=np.int32))
    chirality_centers = find_chirality_centers(adj_list, atom_types)
    reference_signs = compute_chirality_sign(torch.from_numpy(traj.xyz.reshape(*samples.shape))[[1]].to(samples.device), chirality_centers)

    samples = align_sample(samples, adj_list, atom_types, scaling=30.)

    symmetry_change = check_symmetry_change(samples, chirality_centers, reference_signs)
    D_form_counter = [symmetry_change.sum().float().item()]
    samples[symmetry_change] *=-1

    symmetry_change = check_symmetry_change(samples, chirality_centers, reference_signs)
    D_form_counter.append(symmetry_change.sum().float().item())
    return samples, D_form_counter

def compute_phi_psi(traj):
    """Compute the diheral angles"""
    phi = md.compute_phi(traj)[1].flatten()
    psi = md.compute_psi(traj)[1].flatten()
    return phi, psi

def align_topology(sample, all_dists, reference_graph, atom_types, scaling):
    """Align the topology of a sample

    Args:
        * sample (torch.Tensor (n_particles, n_dimensions)): Sample
        * all_dists (torch.Tensor of shape (n_particles, n_particles)): Pairwise distances
        * reference_graph (nx.Graph): Reference graph
        * atom_types (np.Array): Different atom types
        * scaling (float): Scaling of the atoms distances

    Returns:
        * new_sample (torch.Tensor of same shape as sample): Permuted sample
        * is_isomorphic (bool): Whether any permutation happened
    """
    # Compute the graph
    adj_list_computed = create_adjacency_list(all_dists / scaling, atom_types)
    sample_graph = nx.Graph(adj_list_computed)
    # not same number of nodes
    if len(sample_graph.nodes) != len(reference_graph.nodes):
        return sample, False
    for i, atom_type in enumerate(atom_types):
        sample_graph.nodes[i]['type']=atom_type
    nm = isomorphism.categorical_node_match("type", -1)
    GM = isomorphism.GraphMatcher(reference_graph, sample_graph, node_match=nm)
    is_isomorphic = GM.is_isomorphic()
    if len(GM.mapping) > 0:
        initial_idx = torch.LongTensor(list(GM.mapping.keys())).to(sample)
        final_idx = torch.LongTensor(list(GM.mapping.values())).to(sample)
        sample[initial_idx] = sample[final_idx]
    return sample, is_isomorphic


def align_sample(samples, adj_list, atom_types, scaling=30):
    """Align the topology of a batch of samples

    Args:
        * sample (torch.Tensor (batch_size, n_particles, n_dimensions)): Sample
        * adj_list (list): List of bonds
        * atom_types (np.Array): Different atom types
        * scaling (float): Scaling of the atoms distances (default is 30)

    Returns:
        * aligned_samples (torch.Tensor of same shape as sample): Permuted samples
            (Warning: some samples could miss but a message indicates it.)
    """
    # Convert to numpy arrays
    adj_list_ = adj_list.int().detach().cpu().numpy().tolist()
    atom_types_np = atom_types.int().detach().cpu().numpy()
    # Build the reference graph
    reference_graph = nx.Graph(adj_list_)
    for i, atom_type in enumerate(atom_types_np):
        reference_graph.nodes[i]['type'] = atom_type
    # Compute the samples distance
    dists = interatomic_dist(samples, keep_only_upper_tri=False).detach().cpu().numpy()
    # Align the samples
    cnt = 0
    aligned_samples = []
    for i in trange(samples.shape[0]):   
        try:
            # Try the align the topologies
            aligned_sample, is_isomorphic = align_topology(samples[i], dists[i],
                reference_graph, atom_types, scaling=scaling)
            # Append the sample
            aligned_samples.append(aligned_sample)
            if is_isomorphic:
                cnt += 1
        except TimeoutError:
            print("Skipping iteration, function call took too long")
            continue  # Skip to the next iteration
    aligned_samples = torch.stack(aligned_samples)
    return aligned_samples