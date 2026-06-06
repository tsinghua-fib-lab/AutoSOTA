# Solve_poisson_equation.py

import numpy as np
import torch 
from scipy.spatial import KDTree
from model.SurfNO import SurfNO_weights_only
from utils.Laplacian_matrix import Laplacian_matrix
from utils.retrieve_neural_weights import retrieve_neural_weights
from scipy.sparse.linalg import spsolve
from utils.RBF_update import precompute_rbf_data, interpolate_from_precomputed
from utils.rot_update import rot_update
from utils.define_band_points import define_band_points
from utils.draw import plot_3d_point_cloud

if __name__ == "__main__":
    # Put the path of your surface feature array (N,12) as used in the paper.
    surface_features_path = "data/Apple_surface_feature.npy"
    surface_feature_TP = np.load(surface_features_path)
    surface_points = surface_feature_TP[:, :3]
    Tree_surface_points = KDTree(surface_points)
    
    delta_x = 0.05
    dist_to_surface = 0.2
    threshold = 0.8 * dist_to_surface

    band_points, mask_threshold = define_band_points(
        delta_x, surface_points, Tree_surface_points, dist_to_surface, threshold)
    Tree_band_points = KDTree(band_points)

    model = SurfNO_weights_only()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file_location = "model/weights/SurfNO_pretrained_weights.pth"
    model.load_state_dict(torch.load(file_location, map_location=device))
    model.to(device)
    
    local_size = 400
    model.eval()
    with torch.no_grad():
        neural_weights, all_distances_to_central, all_local_band_indexes = retrieve_neural_weights(
            surface_points, band_points, local_size, model, Tree_band_points, mask_threshold, 
            surface_feature_TP=surface_feature_TP)
    
    Lap, denom = Laplacian_matrix(band_points, Tree_band_points, delta_x)

    omega = 10
    lhs_function = np.sin(omega*band_points[:,0]) * np.sin(omega*band_points[:,1]) * np.sin(omega*band_points[:,2])
    lhs_function -= np.mean(lhs_function)

    lhs_function = rot_update(neural_weights, lhs_function, all_local_band_indexes, all_distances_to_central)[0]
    
    rhs = lhs_function * denom
    U = spsolve(Lap, rhs)
    U -= np.mean(U)

    neighbors_indices, factors, phi_vecs = precompute_rbf_data(band_points, Tree_band_points, surface_points,
                                                            k=8, epsilon=1.0)
    
    solution_on_surface = np.zeros(surface_points.shape[0])
    solution_on_surface = interpolate_from_precomputed(U, neighbors_indices, factors, phi_vecs, clipping=True)


    # Saving the solution
    np.save("poisson_solution.npy", solution_on_surface)
    print("Solution saved to poisson_solution.npy")
    
    # Plot the solution
    plot_3d_point_cloud(surface_points, solution_on_surface, "Poisson Solution on Surface", output_file="poisson_solution.html")
