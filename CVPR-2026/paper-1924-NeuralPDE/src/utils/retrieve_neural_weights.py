# retrieve_neural_weights.py 

from scipy.spatial import KDTree
import numpy as np
import torch

from .space_partition import space_partition
from .changing_local_basis import change_local_basis

def prepare_features(surface_feats, surface_indexes, local_band, tangent_plane):
    '''
    Build normalized local inputs for the neural operator.

    Extracts surface points (optionally via `surface_indexes`) and expresses both the local band points
    and the corresponding surface points in the local tangent-frame defined by `tangent_plane`. Surface
    normals are L2-normalized; a ValueError is raised if a near-zero normal is detected.

    Args:
        surface_feats: (N_surface, 6) array/tensor containing [xyz, normals] for surface points.
        surface_indexes: Optional indices selecting the surface points belonging to the current local region.
        local_band: (local_size, 3) array of band points for the current local region.
        tangent_plane: Tuple (center, normal, (t1, t2)) defining the local orthonormal frame.

    Returns:
        local_band_normalise: (local_size, 3) band points in the local frame.
        surface_local_features: (M, 6) surface features in the local frame: [xyz_local, normals_unit].
    '''
    if surface_indexes is not None:
        local_surface_feats = surface_feats[surface_indexes, :]
    else:
        local_surface_feats = surface_feats.detach().cpu().numpy()
    
    local_surface = local_surface_feats[:, :3]
    normals = local_surface_feats[:, 3:]
    
    central_pt = tangent_plane[0].numpy()
    normal = tangent_plane[1].numpy()
    tp1 = tangent_plane[2][0].numpy()
    tp2 = tangent_plane[2][1].numpy()
    local_band_normalise, surface_points_valids_normalise = change_local_basis(
        central_pt, normal, tp1, tp2, local_band, local_surface)

    norms = np.linalg.norm(normals, axis=1, keepdims=True)  
    if np.any(norms < 1e-12):
        raise ValueError(
            "Zero (or near-zero) surface normal detected. "
            "Surface features appear corrupted.")
    normals_valids_normalise = normals / norms

    surface_local_features = np.concatenate([
        surface_points_valids_normalise,           
        normals_valids_normalise                 
    ], axis=1)                         
    
    return local_band_normalise, surface_local_features

def build_surface_features(surface_features, all_local_surface_indexes):
    '''
    Constructs a list of surface features corresponding to local regions defined by provided indexes.

            Parameters:
                    surface_features (np.ndarray): Array of shape (N_surface, F) containing precomputed surface features.
                    all_local_surface_indexes (list of np.ndarray): List of arrays, each containing indices of surface points 
                                                                    within the local region.

            Returns:
                    all_surface_local_features (list of np.ndarray): List of length B, each element is an array of shape (Mi, F)
                                                                     containing surface features for the local region.
    '''
    all_surface_local_features = []

    for indexes in all_local_surface_indexes:
        local_feats = surface_features[indexes, :]
        all_surface_local_features.append(torch.tensor(local_feats, dtype=torch.float32))

    return all_surface_local_features 

def build_band_pos(band_points, all_local_band_indexes):
    '''
    Constructs a list of band positions corresponding to local regions defined by provided indexes.

            Parameters:
                    band_points (np.ndarray): Array of shape (N_band, 3) representing the band points.
                    all_local_band_indexes (list of np.ndarray): List of arrays, each containing indices of local band points.

            Returns:
                    all_local_band_pos (list of np.ndarray): List of length B, each element is an array of shape (local_size, 3)
                                                             containing band points for the local region.
    '''
    all_local_band_pos = []

    for indexes in all_local_band_indexes:
        local_band = band_points[indexes, :]
        all_local_band_pos.append(torch.tensor(local_band, dtype=torch.float32))

    return torch.stack(all_local_band_pos, dim=0) 

def prepare_model_input_after_rot(band_pos_rot, surface_features_rot, tangent_plane_rot):
    band_pos_norm_loc = []
    band_pos_norm = []
    surface_feats_norm_loc = []
    surface_feats_norm = []
    
    for rot_num in range(band_pos_rot.shape[0]):
        for local_num in range(band_pos_rot.shape[1]):
            band, surf = prepare_features(
                surface_features_rot[rot_num][local_num], None, band_pos_rot[rot_num, local_num], 
                tangent_plane = tangent_plane_rot[rot_num][local_num])
            band_pos_norm_loc.append(band.detach().clone().to(torch.float32))
            # band_pos_norm_loc.append(torch.tensor(band, dtype=torch.float32))
            surface_feats_norm_loc.append(torch.tensor(surf, dtype=torch.float32))
        band_pos_norm.append(torch.stack(band_pos_norm_loc, dim=0)) 
        band_pos_norm_loc = []
        surface_feats_norm.append(surface_feats_norm_loc) 
        surface_feats_norm_loc = []
    band_pos_norm = torch.stack(band_pos_norm, dim=0)

    return band_pos_norm, surface_feats_norm 

def random_unit_vector():
    v = np.random.normal(size=3)
    return v / np.linalg.norm(v)

def Rodrigues_matrix(axis_of_rotation, phi):
    axis = axis_of_rotation / np.linalg.norm(axis_of_rotation)

    K_1 = np.outer(axis, axis)

    x, y, z = axis
    K_2 = np.array([[0, -z, y],
                    [z,  0, -x],
                    [-y, x, 0]])
    
    I = np.eye(3)
    return np.cos(phi) * I + (1 - np.cos(phi)) * K_1 + np.sin(phi) * K_2

def orthonormal_basis_from_axis(axis_of_rotation):
    """
    Construct an orthonormal basis (u, v, w) where:
    - u = normalized axis_of_rotation
    - v, w = orthogonal unit vectors
    
    Returns:
        list of np.ndarray: [u, v, w]
    """
    u = axis_of_rotation / np.linalg.norm(axis_of_rotation)

    rand_vec = np.random.randn(3)
    while np.allclose(np.cross(u, rand_vec), 0, atol=1e-6):
        rand_vec = np.random.randn(3)

    v = rand_vec - np.dot(rand_vec, u) * u
    v /= np.linalg.norm(v)

    w = np.cross(u, v)
    w /= np.linalg.norm(w)

    return [u, v, w]

def get_rotations_around_axe(number_of_axis = 1):
    """
    We implement the Rodrigues formula to get rotations around a axe.
    """
    axis_of_rotation = [random_unit_vector() for _ in range(number_of_axis)]
    all_axes = [vec for axis in axis_of_rotation for vec in orthonormal_basis_from_axis(axis)]

    # Adding some rotations can slightly improve the robustness and performance but for the experiments
    # in the paper we only used one rotation.
    # angles = [2*np.pi/3, 4*np.pi/3, np.pi/2, np.pi, 3*np.pi/2]
    # angles = [2*np.pi/3, 4*np.pi/3]
    # angles = [np.pi/2, np.pi, 3*np.pi/2]
    angles = [0]

    rotations = []
    for axes in all_axes:
        for angle in angles:
            rotations.append(Rodrigues_matrix(axes, angle))
    return rotations

def apply_rotations_batch(band_pos, surface_features, rotations, tangent_plane):
    """
    Apply a list of rotations to a batch of points and features.

    Args:
        band_pos: Tensor of shape (B, local_size, 3)
        surface_features: list of size B, each entry is (Mi, 6)
        rotations: list of matrices (nb_rot, 3, 3) (numpy ou torch)
        tangent_plane: list len B, each item: (center(3,), normal(3,), (tp1(3,), tp2(3,)))


    Returns:
        band_pos_rot: Tensor (Nb_rot, B, local_size, 3)
        surface_features_rot: list de taille Nb_rot,
                              chaque élément est une list de taille B,
                              contenant (Mi, 6)
    """
    rotations = [torch.tensor(R, dtype=band_pos.dtype) for R in rotations]
    
    band_pos_rot = torch.stack([band_pos @ R.T for R in rotations], dim=0)

    surface_features_rot = []
    for R in rotations:
        rot_list = []
        for feats in surface_features: 
            coords = feats[:, :3] @ R.T
            normals = feats[:, 3:6] @ R.T
            feats_rot = torch.cat([coords, normals], dim=-1)
            rot_list.append(feats_rot)
        surface_features_rot.append(rot_list)

    tangent_plane_rot = []
    for R in rotations:
        rot_tp_batch = []
        for tup in tangent_plane:
            c, n, princ = tup
            t1 = princ[:3]
            t2 = princ[3:]
            c_r  = c @ R.T
            n_r  = n @ R.T
            t1_r = t1.detach().clone().to(band_pos.dtype) @ R.T 
            t2_r = t2.detach().clone().to(band_pos.dtype) @ R.T
            # t1_r = torch.tensor(t1_r, dtype=band_pos.dtype) @ R.T
            # t2_r = torch.tensor(t2, dtype=band_pos.dtype) @ R.T
            rot_tp_batch.append((c_r, n_r, (t1_r, t2_r)))
        tangent_plane_rot.append(rot_tp_batch)

    return band_pos_rot, surface_features_rot, tangent_plane_rot

def build_tangent_plane(indexes_central_points, surface_feature_TP):
    tangent_planes = []
    for idx in indexes_central_points:
        central_pt = torch.tensor(surface_feature_TP[idx, :3], dtype=torch.float32)
        normal = torch.tensor(surface_feature_TP[idx, 3:6], dtype=torch.float32)
        tp1 = torch.tensor(surface_feature_TP[idx, 6:9], dtype=torch.float32)
        tp2 = torch.tensor(surface_feature_TP[idx, 9:12], dtype=torch.float32)
        tangent_planes.append((central_pt, normal, torch.cat([tp1, tp2], dim=0)))
    return tangent_planes  

def retrieve_neural_weights(surface_points, band_points, local_size, model, Tree_band_points,
                            mask_threshold, surface_feature_TP):
    '''
    Retrieves neural weights from a model based on surface and band points.
            Parameters:
                    surface_points (np.ndarray): Array of shape (N_surface, 3) representing the surface points.
                    band_points (np.ndarray): Array of shape (N_band, 3) representing the band points.
                    local_size (int): Number of nearest band points to consider for each local region.
                    model (torch.nn.Module): Neural network model to extract weights.
                    Tree_band_points (KDTree): KDTree built on band points for efficient nearest neighbor search.
                    mask_threshold (float): Distance threshold to filter band points based on proximity to surface points.
                    surface_feature_TP (array, optional): Array containing points, normals, principal directions of curvatures.

            Returns:
                    neural_weights (torch.Tensor): Neural weights extracted from the model for the band points.
                    all_distances_to_central (list of np.ndarray): List of distances from each local band point to its associated central surface point.
                    all_local_band_indexes (list of np.ndarray): List of indices of local band points in the global band_points array.
    '''
    print("Defining the local parts...")
    tree_surface_points = KDTree(surface_points)
    surface_features = surface_feature_TP[:, :6]

    all_local_band_indexes, all_local_surface_indexes, all_distances_to_central, indexes_central_points = space_partition(
        surface_points, band_points, tree_surface_points, Tree_band_points, local_size, mask_threshold)

    surface_features = build_surface_features(surface_features, all_local_surface_indexes)
    band_pos = build_band_pos(band_points, all_local_band_indexes)  
    tangent_plane = build_tangent_plane(indexes_central_points, surface_feature_TP)
    rotations = get_rotations_around_axe()
    
    print("Applying rotations and extracting neural weights...")
    band_pos_rot, surface_features_rot, tangent_plane_rot = apply_rotations_batch(band_pos, surface_features, rotations, tangent_plane) 

    band_pos_input, surface_features_input = prepare_model_input_after_rot(
        band_pos_rot, surface_features_rot, tangent_plane_rot=tangent_plane_rot)
    
    neural_weights = [] 
    number = len(rotations)

    device = next(model.parameters()).device
    
    for i in range(number):
        band_in = band_pos_input[i].to(device)
        surf_in = [s.to(device) for s in surface_features_input[i]]
        weights_rot = model(band_in, surf_in)
        neural_weights.append(weights_rot)

    return neural_weights, all_distances_to_central, all_local_band_indexes