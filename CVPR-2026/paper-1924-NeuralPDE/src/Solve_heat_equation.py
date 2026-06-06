# Solve_heat_equation.py

import numpy as np
import torch
from scipy.spatial import KDTree
from utils.Laplacian_matrix import Laplacian_matrix
from utils.retrieve_neural_weights import retrieve_neural_weights
from utils.rot_update import rot_update
from utils.define_band_points import define_band_points
from utils.RBF_update import precompute_rbf_data, interpolate_from_precomputed
from model.SurfNO import SurfNO_weights_only
from utils.draw import plot_3d_point_cloud
from tqdm import tqdm

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
    Lap /= denom

    # Explicit Euler: dt must satisfy stability constraints; we use dt = 0.1 * dx^2 as in the paper.
    dt = 0.1 * (delta_x**2)
    nsteps = 100

    omega = 10 
    u_init = np.sin(omega*band_points[:, 0]) * np.sin(omega*band_points[:, 1]) * np.sin(omega*band_points[:, 2])
    
    u_init = rot_update(neural_weights, u_init, all_local_band_indexes, all_distances_to_central)[0]
   
   # You can also just store the function values only at the time steps you want
    U_over_t = np.zeros((nsteps + 1, u_init.shape[0]))
    U_over_t[0] = u_init
    u_band = u_init.copy()

    for t in range(nsteps):
        print(f'Diffusion step: {t+1}/{nsteps}')
        
        u_band_new = u_band + dt * (Lap @ u_band)
        
        u_band = rot_update(neural_weights, u_band_new, all_local_band_indexes, all_distances_to_central, temperature=0.0423)[0]
       
        U_over_t[t+1] = u_band

    # Project the solution on the surface for the time steps needed (e.g, the last one here)       
    neighbors_indices, factors, phi_vecs = precompute_rbf_data(band_points, Tree_band_points, surface_points, k=8, epsilon=1.0)

    # Interpolate for each time step
    surface_solutions = np.zeros((nsteps + 1, surface_points.shape[0]))
    
    for t in tqdm(range(nsteps + 1), desc="Interpolating to surface"):
        surface_solutions[t] = interpolate_from_precomputed(U_over_t[t], neighbors_indices, factors, phi_vecs, clipping=True)
    
    # Save the final solution as before
    surface_solution = surface_solutions[-1]
    np.save("diffusion_solution.npy", surface_solution)
    print("Diffusion solution saved to diffusion_solution.npy")
    # Plot the animation
    plot_3d_point_cloud(surface_points, surface_solutions, title="Heat Diffusion on Apple", output_file="heat_diffusion.html", frame_duration=300)