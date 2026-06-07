import torch
import numpy as np
from skimage import measure
import trimesh
import os
import argparse
from scipy import ndimage

def setup_device(gpu_id=None):
    if torch.cuda.is_available() and gpu_id is not None:
        try:
            device = torch.device(f"cuda:{gpu_id}")
            print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        except RuntimeError:
            print(f"Unable to access GPU {gpu_id}, falling back to CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        if gpu_id is not None:
            print("CUDA unavailable or GPU not specified, using CPU")
        else:
            print("No GPU specified, using CPU")
    return device

def volume_to_mesh(volume_path, output_path=None, threshold=0.5, voxel_size=0.1, gpu_id=None,
                   enable_smoothing=True, smooth_iterations=15, smooth_lambda=0.5, smooth_nu=0.53,
                   enable_enhanced_smoothing=True, enable_volume_preprocessing=True):

    device = setup_device(gpu_id)
    
    
    if output_path is None:
        input_basename = os.path.basename(volume_path)
        input_name = os.path.splitext(input_basename)[0]
        output_dir = os.path.join("meshes", "illusion", input_name)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{input_name}.obj")
        print(f"No output path provided, defaulting to {output_path}")
    
    
    volume_data = torch.load(volume_path, map_location=device)
    
    
    print(f"Volume data keys: {list(volume_data.keys())}")
    
    
    densities = None
    colors = None
    
    densities_converted = False
    colors_converted = False
    
    
    if 'state_dict' in volume_data:
        state_dict = volume_data['state_dict']
        print(f"State dict keys: {list(state_dict.keys())}")
        
        
        for key in state_dict.keys():
            if 'log_densities' in key:
                densities = state_dict[key]
                print(f"Found density logits in state_dict key: {key}")
                densities = torch.sigmoid(densities)
                print("Converted density logits to [0,1] range")
                densities_converted = True
                break
            elif 'densities' in key:
                densities = state_dict[key]
                print(f"Found density tensor in state_dict key: {key}")
                break
        
        
        color_keys = []
        for key in state_dict.keys():
            if 'log_colors' in key:
                color_keys.append(key)

        if len(color_keys) == 1:
            colors = state_dict[color_keys[0]]
            print(f"Found color logits in key: {color_keys[0]}")
        elif len(color_keys) >= 2:
            print(f"Detected multiple color tensors: {color_keys}, averaging the first two")
            cols = [torch.sigmoid(state_dict[k]) for k in color_keys[:2]]
            colors = (cols[0] + cols[1]) / 2.0
    else:
        
        if 'log_densities' in volume_data:
            densities = volume_data['log_densities']
        else:
            for key in ['densities', 'density', 'volume', 'values']:
                if key in volume_data:
                    densities = volume_data[key]
                    print(f"Using key '{key}' to load density data")
                    break
        if 'log_colors' in volume_data:
            colors = volume_data['log_colors']
    
    
    if densities is None:
        raise KeyError("Density tensor not found in the volume data. Please verify the checkpoint structure.")
    
    
    densities = densities.to(device)
    if colors is not None:
        colors = colors.to(device)
    
    
    print(f"Original density shape: {densities.shape}")
    if colors is not None:
        print(f"Original color shape: {colors.shape}")
    
    density_range = (densities.min().item(), densities.max().item())
    print(f"Density value range: {density_range}")
    
    if not densities_converted:
        if density_range[0] < -1.0 or density_range[1] > 3.0:
            print("Detected log-space densities, applying sigmoid")
            densities = torch.sigmoid(densities)
            densities_converted = True
        else:
            print("Using raw densities without conversion")
    
    if colors is not None:
        color_range = (colors.min().item(), colors.max().item())
        print(f"Color value range: {color_range}")
        
        if color_range[0] < -1.0 or color_range[1] > 3.0:
            print("Detected log-space colors, applying sigmoid")
            colors = torch.sigmoid(colors)
            colors_converted = True
        else:
            print("Using raw colors without conversion")
    
    if densities.dim() == 4 and densities.shape[0] == 1:
        densities = densities.squeeze(0)
        print(f"Density shape after squeezing batch dim: {densities.shape}")
    elif densities.dim() != 3:
        raise ValueError(f"Unsupported density shape {densities.shape}, expected (D,H,W) or (1,D,H,W)")
    
    if colors is not None:
        if colors.dim() == 4 and colors.shape[0] == 3:
            print("Color tensor shape OK: (3, D, H, W)")
        elif colors.dim() == 5 and colors.shape[0] == 1 and colors.shape[1] == 3:
            colors = colors.squeeze(0)
            print(f"Color shape after squeezing batch dim: {colors.shape}")
        else:
            print(f"Warning: unexpected color shape {colors.shape}, export may be affected")
    
    
    densities_np = densities.cpu().numpy()
    
    densities_np_raw = densities_np.copy()

    if enable_volume_preprocessing:
        print("Starting voxel preprocessing smoothing...")

        original_mean = np.mean(densities_np)
        original_std = np.std(densities_np)
        original_min = np.min(densities_np)
        original_max = np.max(densities_np)

        print(f"  - Original mean={original_mean:.6f}, std={original_std:.6f}")
        print(f"  - Original range: [{original_min:.6f}, {original_max:.6f}]")

        try:
            from skimage.restoration import denoise_bilateral

            sigma_spatial = 1.0
            sigma_color = 0.1
            print(f"  - Applying bilateral filter (sigma_spatial={sigma_spatial}, sigma_color={sigma_color})")

            densities_np_smoothed = denoise_bilateral(
                densities_np,
                sigma_color=sigma_color,
                sigma_spatial=sigma_spatial,
                channel_axis=None
            )
        except Exception as _e:
            print(f"  - Bilateral filter unavailable ({str(_e)}), falling back to Gaussian smoothing")
            sigma = 0.8
            densities_np_smoothed = ndimage.gaussian_filter(densities_np, sigma=sigma)
        
        blend_ratio = 0.3
        densities_np = (1.0 - blend_ratio) * densities_np + blend_ratio * densities_np_smoothed

        new_min = np.min(densities_np)
        new_max = np.max(densities_np)
        if new_max > new_min:
            densities_np = (densities_np - new_min) / (new_max - new_min) * (original_max - original_min) + original_min

        processed_mean = np.mean(densities_np)
        processed_std = np.std(densities_np)

        print(f"  - Post-processed mean={processed_mean:.6f}, std={processed_std:.6f}")
        if 'sigma' in locals():
            print(f"  - Gaussian sigma={sigma}, blend ratio={blend_ratio}")
        else:
            print(f"  - Blend ratio={blend_ratio}")
        print("  - Expected effect: smoother voxel boundaries for cleaner marching cubes")

    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        base_dir = os.path.dirname(output_path)
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        raw_npy_path = os.path.join(base_dir, base_name + "_densities_raw.npy")
        processed_npy_path = os.path.join(base_dir, base_name + "_densities_processed.npy")
        np.save(raw_npy_path, densities_np_raw)
        np.save(processed_npy_path, densities_np)
        print(f"Saved density volumes to {raw_npy_path} (raw) and {processed_npy_path} (processed)")
    except Exception as e:
        print(f"Failed to save .npy volumes: {e}")
    
    
    print(f"Processed density shape: {densities_np.shape}")
    print(f"Processed range: [{np.min(densities_np):.6f}, {np.max(densities_np):.6f}], mean={np.mean(densities_np):.6f}")
    
    
    data_min = np.min(densities_np)
    data_max = np.max(densities_np)
    data_mean = np.mean(densities_np)
    data_std = np.std(densities_np)
    
    print("Density statistics:")
    print(f"  - Min: {data_min:.6f}")
    print(f"  - Max: {data_max:.6f}")
    print(f"  - Mean: {data_mean:.6f}")
    print(f"  - Std: {data_std:.6f}")
    
    
    original_threshold = threshold
    
    if densities_converted:
        
       
        percentile_75 = np.percentile(densities_np, 75)
        percentile_90 = np.percentile(densities_np, 90)
        percentile_95 = np.percentile(densities_np, 95)
        
        print("Density percentiles:")
        print(f"  - P75: {percentile_75:.6f}")
        print(f"  - P90: {percentile_90:.6f}")
        print(f"  - P95: {percentile_95:.6f}")
        

        adaptive_threshold_1 = data_mean + 1.5 * data_std
        adaptive_threshold_2 = data_mean + 2.0 * data_std
        
        print("Adaptive thresholds:")
        print(f"  - mean + 1.5σ: {adaptive_threshold_1:.6f}")
        print(f"  - mean + 2.0σ: {adaptive_threshold_2:.6f}")
        

        target_voxel_ratio = 0.02
        max_target_ratio = 0.08
        

        candidate_thresholds = [
            percentile_75, percentile_90, percentile_95,
            adaptive_threshold_1, adaptive_threshold_2,
            data_mean + 0.5 * data_std,
            data_mean + 1.0 * data_std
        ]
        
        best_threshold = threshold
        best_ratio_diff = float('inf')
        
        for candidate in candidate_thresholds:
            if data_min <= candidate <= data_max:
                ratio = (densities_np > candidate).mean()
                ratio_diff = abs(ratio - target_voxel_ratio)
                
                print(f"  Threshold {candidate:.6f}: voxel ratio {ratio*100:.3f}%")
                
                if ratio_diff < best_ratio_diff and target_voxel_ratio <= ratio <= max_target_ratio:
                    best_threshold = candidate
                    best_ratio_diff = ratio_diff
        

        if best_threshold == threshold:
            if data_max > 0.1:
                best_threshold = max(data_mean + 0.5 * data_std, 0.05)
            else:
                best_threshold = max(percentile_75, data_mean + 0.2 * data_std)
        
        threshold = best_threshold
        
        print("Adaptive threshold selection:")
        print(f"  - Original: {original_threshold:.6f}")
        print(f"  - Selected: {threshold:.6f}")
        print(f"  - Expected ratio: {(densities_np > threshold).mean()*100:.3f}%")
        
    else:

        if threshold < data_min or threshold > data_max:
            if data_std > 0:
                adjusted_threshold = data_mean + data_std
            else:
                adjusted_threshold = data_min + (data_max - data_min) * 0.5
            
            print(f"Warning: threshold {threshold} outside [{data_min:.6f}, {data_max:.6f}]")
            print(f"Adjusting to {adjusted_threshold:.6f}")
            threshold = adjusted_threshold
    

    final_voxel_ratio = (densities_np > threshold).mean()
    print("\nFinal threshold report:")
    print(f"  - Threshold: {threshold:.6f}")
    print(f"  - Retained ratio: {final_voxel_ratio*100:.3f}%")
    print(f"  - Retained voxels: {int(final_voxel_ratio * densities_np.size):,}")


    if final_voxel_ratio < 0.005:
        print(f"\nWarning: voxel ratio too low ({final_voxel_ratio*100:.3f}%).")
        print("Consider lowering the threshold, checking supervision, or improving training.")
        alt_threshold = max(data_mean, 0.01)
        alt_ratio = (densities_np > alt_threshold).mean()
        print(f"  Suggested alternative: {alt_threshold:.6f} (ratio {alt_ratio*100:.3f}%)")
    
    elif final_voxel_ratio > 0.1:
        print(f"\nNote: voxel ratio is high ({final_voxel_ratio*100:.3f}%).")
        print("This may produce a large mesh; increasing the threshold could help.")
    

    try:

        data_min = float(np.min(densities_np))
        data_max = float(np.max(densities_np))
        if not (data_min < threshold < data_max):
            print(f"Warning: threshold {threshold} outside ({data_min:.6f}, {data_max:.6f}); clamping.")
            mu = float(np.mean(densities_np))
            sigma = float(np.std(densities_np))
            threshold = min(max(mu + 0.5 * sigma, data_min + 1e-6), data_max - 1e-6)
            print(f"  - Adjusted threshold: {threshold:.6f}")
        print(f"Running marching cubes with level {threshold}")
        vertices, faces, normals, _ = measure.marching_cubes(
            densities_np, 
            level=threshold,
            spacing=(voxel_size, voxel_size, voxel_size)
        )

        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_normals=normals
        )
        

        if colors is not None:
            try:
                if colors.dim() == 4 and colors.shape[0] == 3:
                    colors_np = colors.permute(1, 2, 3, 0).cpu().numpy()
                    print(f"Converted colors to numpy array with shape {colors_np.shape}")
                    
                    vertex_indices = np.floor(vertices / voxel_size).astype(int)
                    
                    D, H, W = densities_np.shape
                    vertex_indices[:, 0] = np.clip(vertex_indices[:, 0], 0, D - 1)
                    vertex_indices[:, 1] = np.clip(vertex_indices[:, 1], 0, H - 1)
                    vertex_indices[:, 2] = np.clip(vertex_indices[:, 2], 0, W - 1)
                    
                    vertex_colors = colors_np[
                        vertex_indices[:, 0],
                        vertex_indices[:, 1],
                        vertex_indices[:, 2]
                    ]
                    
                    vertex_colors = np.clip(vertex_colors, 0.0, 1.0)
                    mesh.visual.vertex_colors = (vertex_colors * 255).astype(np.uint8)
                    print(f"Assigned vertex colors in range [{vertex_colors.min():.3f}, {vertex_colors.max():.3f}]")
                else:
                    print(f"Unexpected color tensor shape {colors.shape}, skipping color bake")
                    
            except Exception as e:
                print(f"Failed to assign vertex colors: {e}")
                print("Continuing without colors")


        if enable_smoothing:
            print("Applying Taubin surface smoothing...")
            print(f"  - Iterations: {smooth_iterations}")
            print(f"  - lambda: {smooth_lambda}")
            print(f"  - nu: {smooth_nu}")

            if not (0.0 < smooth_lambda < 1.0):
                print(f"  Warning: lambda {smooth_lambda} out of range, clamping to 0.5")
                smooth_lambda = 0.5

            if not (0.0 < smooth_nu < 1.0):
                print(f"  Warning: nu {smooth_nu} out of range, clamping to 0.53")
                smooth_nu = 0.53

            constraint_value = 1.0/smooth_lambda - 1.0/smooth_nu
            print(f"  - Constraint value: {constraint_value:.6f} (target 0.0-0.1)")

            if not (0.0 < constraint_value < 0.1):
                print("  Warning: Taubin constraint violated, auto-adjusting params")
                target_constraint = 0.03
                smooth_nu = 1.0 / (1.0/smooth_lambda - target_constraint)

                if smooth_nu >= 1.0:
                    smooth_nu = 0.95
                    smooth_lambda = 1.0 / (1.0/smooth_nu + target_constraint)
                elif smooth_nu <= 0.0:
                    smooth_nu = 0.05
                    smooth_lambda = 1.0 / (1.0/smooth_nu + target_constraint)

                print(f"  Adjusted lambda={smooth_lambda:.6f}, nu={smooth_nu:.6f}")
                print(f"  Updated constraint={1.0/smooth_lambda - 1.0/smooth_nu:.6f}")

            mesh_complexity = len(mesh.faces) / len(mesh.vertices) if len(mesh.vertices) > 0 else 0
            print(f"  - Face/vertex ratio: {mesh_complexity:.2f}")

            if mesh_complexity > 2.5:
                print("  High complexity mesh detected, using conservative smoothing")
                smooth_lambda = min(smooth_lambda, 0.4)
                smooth_nu = max(smooth_nu, 0.55)
            elif mesh_complexity < 1.5:
                print("  Low complexity mesh detected, increasing smoothing strength")
                smooth_lambda = min(smooth_lambda + 0.1, 0.6)

            original_vertices = len(mesh.vertices)
            original_faces = len(mesh.faces)
            original_volume = mesh.volume if mesh.is_volume else 0
            original_area = mesh.area

            try:
                from trimesh import smoothing

                if enable_enhanced_smoothing:
                    stage1_lambda = min(smooth_lambda + 0.1, 0.6)
                    stage1_nu = max(smooth_nu - 0.03, 0.5)
                    stage1_iterations = max(5, smooth_iterations // 3)
                    smoothing.filter_taubin(mesh, lamb=stage1_lambda, nu=stage1_nu, iterations=stage1_iterations)

                    stage2_iterations = max(5, smooth_iterations // 3)
                    smoothing.filter_taubin(mesh, lamb=smooth_lambda, nu=smooth_nu, iterations=stage2_iterations)

                    stage3_lambda = max(smooth_lambda - 0.1, 0.3)
                    stage3_nu = min(smooth_nu + 0.05, 0.6)
                    stage3_iterations = max(8, smooth_iterations // 2)
                    smoothing.filter_taubin(mesh, lamb=stage3_lambda, nu=stage3_nu, iterations=stage3_iterations)

                    laplacian_lambda = 0.03
                    laplacian_iterations = 10
                    smoothing.filter_laplacian(mesh, lamb=laplacian_lambda, iterations=laplacian_iterations)
                else:
                    smoothing.filter_taubin(mesh, lamb=smooth_lambda, nu=smooth_nu, iterations=smooth_iterations)

                try:
                    hc_iter = max(10, smooth_iterations // 2)
                    smoothing.filter_humphrey(mesh, alpha=0.1, beta=0.4, iterations=hc_iter)
                except AttributeError:
                    print("  Warning: HC-Laplacian unavailable, skipping final pass")
                except Exception as e:
                    print(f"  HC-Laplacian smoothing failed: {e}")

                new_volume = mesh.volume if mesh.is_volume else 0
                new_area = mesh.area

                print("  Smoothing summary:")
                print(f"    Vertices: {original_vertices} -> {len(mesh.vertices)}")
                print(f"    Faces: {original_faces} -> {len(mesh.faces)}")
                if original_volume > 0 and new_volume > 0:
                    volume_change = ((new_volume - original_volume) / original_volume) * 100
                    print(f"    Volume change: {volume_change:+.2f}%")
                if original_area > 0:
                    area_change = ((new_area - original_area) / original_area) * 100
                    print(f"    Surface area change: {area_change:+.2f}%")

                try:
                    face_angles = mesh.face_angles
                    if len(face_angles) > 0:
                        angle_std = np.std(face_angles)
                        print(f"    Face angle std: {angle_std:.4f}")
                except Exception:
                    pass

                if mesh.is_valid:
                    print("    Mesh validity: OK")
                else:
                    print("    Mesh validity: WARNING")

                if mesh.is_watertight:
                    print("    Watertight: yes")
                else:
                    print("    Watertight: no")

                if enable_volume_preprocessing:
                    print("    Voxel preprocessing: enabled")
                else:
                    print("    Voxel preprocessing: disabled")

            except ImportError:
                print("  Error: trimesh.smoothing not available, skipping smoothing")
            except Exception as e:
                print(f"  Smoothing failed: {e}")
                print("  Proceeding with unsmoothed mesh")
        else:
            print("Surface smoothing disabled")

       
        try:
            mesh.export(output_path)
        except Exception as e:
            #
            tmp_ply = output_path + "._tmp_export.ply"
            print(f"OBJ ({e}), PLY -> OBJ process failed")
            mesh.export(tmp_ply)
            mesh = trimesh.load(tmp_ply)
            mesh.export(output_path)
            try:
                os.remove(tmp_ply)
            except Exception:
                pass
        print(f"Mesh saved to: {output_path}")

        
        base_path, current_ext = os.path.splitext(output_path)
        ply_path = base_path + ".ply"

       
        if current_ext.lower() != ".ply" or not os.path.exists(ply_path):
            try:
                mesh.export(ply_path)
                print(f"Mesh also saved as: {ply_path}")
            except Exception as e:
                print(f"Failed to export .ply: {e}")
        else:
            obj_path = base_path + ".obj"
            if current_ext.lower() != ".obj" and not os.path.exists(obj_path):
                try:
                    mesh.export(obj_path)
                    print(f"Mesh also saved as: {obj_path}")
                except Exception as e:
                    print(f"Failed to export .obj: {e}")
        
        print("Mesh statistics:")
        print(f"  - Vertices: {len(vertices)}")
        print(f"  - Faces: {len(faces)}")
        print(f"  - Ratio above threshold: {(densities_np > threshold).mean()*100:.2f}%")
        
    except Exception as e:
        print(f"Mesh export failed: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Convert a volume checkpoint into a mesh')
    parser.add_argument('--volume_path', type=str, required=True, help='Path to the volume checkpoint (.pt)')
    parser.add_argument('--output_path', type=str, help='Destination .obj path (default: auto)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Marching cubes density threshold')
    parser.add_argument('--voxel_size', type=float, default=0.1, help='Physical voxel size for spacing')
    parser.add_argument('--gpu', type=int, help='Optional GPU index (CPU fallback if unavailable)')

    parser.add_argument('--no_smoothing', action='store_true', help='Disable all surface smoothing')
    parser.add_argument('--smooth_iterations', type=int, default=15, help='Taubin smoothing iterations')
    parser.add_argument('--smooth_lambda', type=float, default=0.5, help='Taubin contraction coefficient')
    parser.add_argument('--smooth_nu', type=float, default=0.53, help='Taubin expansion coefficient')

    parser.add_argument('--no_enhanced_smoothing', action='store_true', help='Disable multi-stage smoothing')
    parser.add_argument('--no_volume_preprocessing', action='store_true', help='Disable voxel preprocessing')

    args = parser.parse_args()

    volume_to_mesh(
        args.volume_path,
        args.output_path,
        args.threshold,
        args.voxel_size,
        args.gpu,
        enable_smoothing=not args.no_smoothing,
        smooth_iterations=args.smooth_iterations,
        smooth_lambda=args.smooth_lambda,
        smooth_nu=args.smooth_nu,
        enable_enhanced_smoothing=not args.no_enhanced_smoothing,
        enable_volume_preprocessing=not args.no_volume_preprocessing
    )

if __name__ == "__main__":
    main() 