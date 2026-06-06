# training_model.py

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from itertools import product
import torch.nn as nn
import os
from tqdm import tqdm
from torch.utils.data import random_split
from model.SurfNO import SurfNO
from utils.changing_local_basis import change_local_basis
from utils.normal_consistency_loss import normal_consitency_loss, normal_consitency_loss_eval

def generate_polynomial_functions(local_band: np.ndarray, max_degree: int = 5) -> torch.Tensor:
    """
    Generate monomials x^i y^j z^k evaluate on local_band for i + j + k <= max_degree.

    Args:
        local_band (np.ndarray): array (N, 3)
        max_degree (int): max degree

    Returns:
        torch.Tensor: (K, N) where K is the number of monomials with degree <= max_degree.
    """
    local_band = torch.from_numpy(local_band).float()

    x, y, z = local_band[:, 0], local_band[:, 1], local_band[:, 2]

    x_powers = [x**i for i in range(max_degree + 1)]
    y_powers = [y**j for j in range(max_degree + 1)]
    z_powers = [z**k for k in range(max_degree + 1)]

    polys = []
    for i, j, k in product(range(max_degree + 1), repeat=3):
        if i + j + k <= max_degree:
            polys.append(x_powers[i] * y_powers[j] * z_powers[k])

    return torch.stack(polys, dim=1).T  

def get_min_max(arr1, arr2):
    '''
    Returns the global min and max of the three arrays.
    '''
    min1 = arr1.min()
    max1 = arr1.max()
    
    min2 = arr2.min()
    max2 = arr2.max()
    
    return np.min([min1, min2]), np.max([max1, max2])

def common_min_max_normalisation(local_band, local_surface, eps=1e-8):
    '''
    Normalize the local_band and local_surface arrays jointly.
    '''
    norm_band = np.zeros_like(local_band)
    norm_surf = np.zeros_like(local_surface)
    
    for i in range(3):
        min_val, max_val = get_min_max(local_band[:, i], local_surface[:, i])
        denom = max_val - min_val

        if abs(denom) < eps:
            norm_band[:, i] = local_band[:, i]
            norm_surf[:, i] = local_surface[:, i]
        else:
            norm_band[:, i] = (local_band[:, i] - min_val) / denom
            norm_surf[:, i] = (local_surface[:, i] - min_val) / denom

    return norm_band, norm_surf
    
def general_normalisation(local_sf, local_band, tangent_plane):
    '''
    Normalize the local band and local surface features.

            Parameters:
                    local_sf (np.ndarray): Array of shape (Mi, 6) containing local surface features.
                    local_band (np.ndarray): Array of shape (nb, 3) representing the local band.
                    tangent_plane (tuple): Tuple containing the central point, normal vector, and tangent plane vectors.
            Returns:
                    local_band_norm (np.ndarray): Normalized local band of shape (nb, 3).
                    local_surface_features (np.ndarray): Normalized local surface features of shape (Mi, 6).
    '''
    local_surface = local_sf[:, :3]  
    normals = local_sf[:, 3:6]       

    norms = np.linalg.norm(normals, axis=1, keepdims=True)  
    if np.any(norms < 1e-12):
        raise ValueError(
            "norm of normal vector is too small, cannot normalize. Check the input data for correctness."
        )
    normals_norm = normals / norms
    
    central_pt = tangent_plane[0]
    normal = tangent_plane[1]
    tp1 = tangent_plane[2][:3]
    tp2 = tangent_plane[2][3:]
    local_band_norm, local_surface_norm = change_local_basis(central_pt, normal, tp1, tp2, local_band, local_surface)

    local_surface_features = np.concatenate([
        local_surface_norm,
        normals_norm
    ], axis=1)       
    
    return local_band_norm, local_surface_features

# Dataset class
class MultiResoSpikeDataset(Dataset):
    def __init__(self, voxels_path, surface_features_path, max_degree=5):        
        voxel_loading = np.load(voxels_path, allow_pickle=True)
        sf_loading = np.load(surface_features_path)
        
        self.surface_features = sf_loading[:, :6]
        self.tangent_plane = sf_loading[:, -6:]
        self.normals = sf_loading[:, 3:6]
        
        self.sf_indexes = voxel_loading["surface_features"]
        self.closest_points = voxel_loading["closest"]
        self.voxels = voxel_loading["voxels"]
        self.center_point_idx = voxel_loading["center_point_idx"]

        diff = self.voxels - self.closest_points            
        norm = np.linalg.norm(diff, axis=-1, keepdims=True) 

        self.normals_gt = np.zeros_like(diff)
        np.divide(diff, norm, out=self.normals_gt, where=(norm > 1e-12))

        self.max_degree = max_degree

    def __len__(self):
        return self.sf_indexes.shape[0]

    def __getitem__(self, idx):
        # nb is the size of the local band. Adjust as needed, but ensure it matches the model's expected input size.
        # Can only be increse to 512 due to the limit of the dataset provided.
        nb = 400 
        local_sf = self.surface_features[self.sf_indexes[idx]]  
        local_cp = self.closest_points[idx][:nb]                                    
        local_band = self.voxels[idx][:nb]                                          
        normals_gt = self.normals_gt[idx][:nb]   

        cp_idx = self.center_point_idx[idx]
        central_pt = self.surface_features[cp_idx, :3]  
        normal = self.normals[cp_idx, :]            
        tp = self.tangent_plane[cp_idx, :] 
        tangent_plane = (central_pt, normal, tp)                        

        input_functions = generate_polynomial_functions(local_band, max_degree=self.max_degree)
        target_functions = generate_polynomial_functions(local_cp, max_degree=self.max_degree) 

        local_band_norm, local_surface_features = general_normalisation(local_sf, local_band, tangent_plane)

        return {
            'local band': torch.from_numpy(local_band_norm).float(),
            'input u': input_functions,
            'target u': target_functions,
            'local surface features': torch.from_numpy(local_surface_features).float(),
            'normals_gt': torch.from_numpy(normals_gt).float()
        }

def custom_collate(batch):
    batch_out = {}
    for key in batch[0].keys():
        if key == 'local surface features':  
            batch_out[key] = [item[key] for item in batch]
        else:
            batch_out[key] = torch.stack([item[key] for item in batch])
    return batch_out

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_mse  = 0.0
    total_nc   = 0.0

    bar = tqdm(total=len(dataloader), desc="Training")
    for batch in dataloader:
        grid_points = batch['local band'].to(device)              
        surface_feats_list = [sf.to(device) for sf in batch['local surface features']]  
        function_values = batch['input u'].to(device)             
        target_values = batch['target u'].to(device)              
        normals_gt = batch['normals_gt'].to(device)              

        optimizer.zero_grad()

        grid_points = grid_points.detach().requires_grad_(True)

        preds = model(grid_points, surface_feats_list, function_values) 

        L_mse = criterion(preds, target_values)

        L_nc = normal_consitency_loss(preds, grid_points, normals_gt, k_select=None, create_graph=True)

        alpha = 1e-2
        loss = L_mse + alpha * L_nc

        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item()
        total_mse  += L_mse.detach().item()
        total_nc   += L_nc.detach().item()

        bar.update(1)
        bar.set_description(f"Train loss={loss.detach().item():.4g}")

    return total_loss / len(dataloader)

def eval_one_epoch(model, dataloader, criterion, device):
    model.eval()
    total_mse, total_nc = 0.0, 0.0 

    for batch in tqdm(dataloader, desc="Evaluating"):
        grid_points = batch['local band'].to(device)              
        surface_feats_list = [sf.to(device) for sf in batch['local surface features']] 
        function_values = batch['input u'].to(device)             
        target_values = batch['target u'].to(device)           
        normals_gt = batch['normals_gt'].to(device)                

        with torch.no_grad():
            preds_mse = model(grid_points, surface_feats_list, function_values)
            loss_mse = criterion(preds_mse, target_values)
            total_mse += loss_mse.item()

        with torch.enable_grad():
            grid_points = grid_points.detach().requires_grad_(True)
            preds = model(grid_points, surface_feats_list, function_values)
            L_nc = normal_consitency_loss_eval(preds, grid_points, normals_gt,
                                          k_select=None, create_graph=False)
            total_nc += L_nc.item()

    alpha = 1e-2
    return (total_mse + alpha * total_nc) / len(dataloader)

def train_model(model, dataset, num_epochs=10, batch_size=16, lr=1e-3, file_name='best_model.pth', num_workers=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, 
                              persistent_workers=True if num_workers > 0 else False, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False, 
                            persistent_workers=True if num_workers > 0 else False, collate_fn=custom_collate)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, min_lr=0.00005)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.2e}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Training Loss: {train_loss:.4f}")

        val_loss = eval_one_epoch(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            loss_str = f"{best_val_loss:.4f}".replace(".", "_")
            file_name_with_loss = f"{file_name}_loss_{loss_str}.pth"

            save_path = os.path.join("pretrained", "multi_reso_spike_training", file_name_with_loss)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

if __name__ == "__main__":
    dataset = MultiResoSpikeDataset('data/dataset_multi_reso_spike/voxels.npz', 'data/dataset_multi_reso_spike/surface_features.npy')

    model = SurfNO()
    
    train_model(model, dataset, num_epochs=1500, batch_size=64, lr=5e-4, file_name='Surf_NO.pth')