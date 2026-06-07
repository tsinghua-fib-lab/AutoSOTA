"""
Evaluation script: Run KASAL v1 axis localization on DSRSTO dataset
with ground truth symmetry types, and measure eADI/d and runtime.
"""
import sys
sys.path.insert(0, '/repo')

import json
import os
import time
import glob
import numpy as np
from scipy import spatial

from kasal.symmetry_lab.symmetry_axis_template import get_sym_axis_temp
from kasal.symmetry_lab.symmetry_axis_localization import cal_model_sym
from kasal.utils.io_ply import load_ply_model
import kasal.config.config as config

# Suppress polyscope import
config.save_2_fold_a = False

def compute_chamfer_distance(pts1, pts2):
    """Compute Chamfer distance between two point clouds."""
    dist1 = spatial.distance.cdist(pts1, pts2, metric='euclidean')
    d1 = np.mean(np.min(dist1, axis=1))
    d2 = np.mean(np.min(dist1, axis=0))
    return (d1 + d2) / 2.0

def compute_eADI(model_i_, diameter):
    """Compute eADI/d given the localized symmetry axes.

    eADI = average pointwise distance between original model and
    its symmetric instances, divided by object diameter.
    """
    vertices = model_i_['vertices']

    symmetries_discrete = model_i_.get('symmetries_discrete', [])
    symmetries_continuous = model_i_.get('symmetries_continuous', [])

    # Build rotation set from discrete symmetries
    rotation_set = [np.eye(4)]  # identity

    if len(symmetries_discrete) > 0:
        for sym_mat in symmetries_discrete:
            mat = np.array(sym_mat).reshape(4, 4)
            rotation_set.append(mat)
    elif len(symmetries_continuous) > 0:
        # For continuous symmetries, sample some rotations
        if symmetries_continuous[0].get('axis') and len(symmetries_continuous[0]['axis']) > 0:
            axis = np.array(symmetries_continuous[0]['axis'])
            offset = np.array(symmetries_continuous[0].get('offset', [0, 0, 0]))
            for ang in [45, 90, 135, 180, 225, 270, 315]:
                theta = np.radians(ang)
                c = np.cos(theta)
                s = np.sin(theta)
                # Rodrigues rotation formula
                a_norm = axis / np.linalg.norm(axis)
                K = np.array([[0, -a_norm[2], a_norm[1]],
                              [a_norm[2], 0, -a_norm[0]],
                              [-a_norm[1], a_norm[0], 0]])
                R = np.eye(3) + s * K + (1 - c) * np.dot(K, K)
                mat = np.eye(4)
                mat[:3, :3] = R
                mat[:3, 3] = offset - np.dot(R, offset)
                rotation_set.append(mat)
        elif 'offset' in symmetries_continuous[0]:
            # Spherical symmetry (C>>1)
            offset = np.array(symmetries_continuous[0]['offset'])
            for axis_idx in range(3):
                axis = np.zeros(3)
                axis[axis_idx] = 1.0
                for ang in [45, 90, 135, 180, 225, 270, 315]:
                    theta = np.radians(ang)
                    c = np.cos(theta)
                    s = np.sin(theta)
                    K = np.array([[0, -axis[2], axis[1]],
                                  [axis[2], 0, -axis[0]],
                                  [-axis[1], axis[0], 0]])
                    R = np.eye(3) + s * K + (1 - c) * np.dot(K, K)
                    mat = np.eye(4)
                    mat[:3, :3] = R
                    mat[:3, 3] = offset - np.dot(R, offset)
                    rotation_set.append(mat)

    if len(rotation_set) <= 1:
        return 0.0  # No symmetry found

    # Compute eADI: average over all non-identity rotations
    total_adi = 0.0
    count = 0
    for mat in rotation_set[1:]:  # Skip identity
        R = mat[:3, :3]
        t = mat[:3, 3]
        rotated_pts = np.dot(vertices, R.T) + t
        adi = compute_chamfer_distance(vertices, rotated_pts)
        total_adi += adi
        count += 1

    if count == 0:
        return 0.0
    eADI = total_adi / count
    return eADI / diameter  # Normalize by diameter


def main():
    shape_dir = '/repo/DSRSTO/DSRSTO dataset/shape_meshes'
    tex_dir = '/repo/DSRSTO/DSRSTO dataset/tex_meshes'

    results = {'shape': [], 'tex': []}
    total_runtime = 0.0
    total_objects = 0

    for category, data_dir in [('shape', shape_dir), ('tex', tex_dir)]:
        # Find all _sym_type.json files
        if category == 'shape':
            json_files = sorted(glob.glob(os.path.join(data_dir, '*_ours_sym_type.json')))
            ply_suffix = '_ours.ply'
        else:
            json_files = sorted(glob.glob(os.path.join(data_dir, '*_sym_type.json')))
            ply_suffix = '_sym.ply'  # For texture, the ground truth is in _sym.ply

        for json_file in json_files:
            # Determine the PLY file
            if category == 'shape':
                base_name = json_file.replace('_ours_sym_type.json', '')
                ply_file = base_name + ply_suffix
            else:
                base_name = json_file.replace('_sym_type.json', '')
                # For texture, find the correct ply
                possible_ply = base_name + '.ply'
                if not os.path.exists(possible_ply):
                    possible_ply = base_name + '_t2_sym.ply'
                    if not os.path.exists(possible_ply):
                        possible_ply = base_name + '_t_sym.ply'
                        if not os.path.exists(possible_ply):
                            print(f"  WARNING: No PLY found for {json_file}")
                            continue
                ply_file = possible_ply

            if not os.path.exists(ply_file):
                print(f"  WARNING: PLY not found: {ply_file}")
                continue

            # Read ground truth
            with open(json_file, 'r') as f:
                gt = json.load(f)

            sym_type = gt.get('sym_type', '')
            n_fold = gt.get('n-fold', 2)

            # Map symmetry type to sym_op
            if sym_type == "C(>1): Cylindrical Item":
                sym_op = 'symmetries_continuous_2'
            elif sym_type == "C(=1): Circular Item":
                sym_op = 'symmetries_continuous'
            elif sym_type == "C(>>1): Spherical Item":
                sym_op = 'symmetries_continuous_3'
            elif sym_type in ["D(>1): n-fold Prismatic Item",
                              "D(=1): n-fold Pyramidal Item",
                              "P(4): Tetrahedral Item",
                              "P(8): Octahedral Item",
                              "P(20): Icosahedral Item"]:
                sym_op = 'symmetries_discrete'
            elif sym_type == 'None' or sym_type is None:
                print(f"  SKIP {os.path.basename(json_file)}: No symmetry")
                continue
            else:
                print(f"  SKIP {os.path.basename(json_file)}: Unknown type {sym_type}")
                continue

            # Get axis template
            step_path = get_sym_axis_temp(sym_type, n_fold)

            obj_name = os.path.basename(json_file)
            print(f"Processing {obj_name}: type={sym_type}, n={n_fold}")

            try:
                # Load model
                model_i_ = load_ply_model(ply_file, color_op=False)

                # Time the axis localization
                start_time = time.time()
                model_i_ = cal_model_sym(
                    model_i_,
                    step_path=step_path,
                    sym_op=sym_op,
                    sym_aware=False,
                    op='pts',
                    sample_num=10001,
                    fpsample_num=1500,
                    icp_op=True,
                    xyz_op=None
                )
                elapsed = time.time() - start_time

                # Compute eADI/d
                diameter = model_i_['diameter']
                eadi_d = compute_eADI(model_i_, diameter)

                results[category].append({
                    'name': obj_name,
                    'sym_type': sym_type,
                    'eadi_d': eadi_d,
                    'runtime': elapsed,
                    'diameter': diameter
                })

                total_runtime += elapsed
                total_objects += 1

                print(f"  eADI/d={eadi_d:.6f}, runtime={elapsed:.3f}s, diameter={diameter:.2f}")

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for category in ['shape', 'tex']:
        if results[category]:
            eadi_vals = [r['eadi_d'] for r in results[category] if r['eadi_d'] > 0]
            runtime_vals = [r['runtime'] for r in results[category]]
            print(f"\n{category.upper()} ({len(results[category])} objects):")
            if eadi_vals:
                print(f"  eADI/d: mean={np.mean(eadi_vals):.6f}, median={np.median(eadi_vals):.6f}")
            if runtime_vals:
                print(f"  Runtime: mean={np.mean(runtime_vals):.3f}s, median={np.median(runtime_vals):.3f}s")

    all_eadi = []
    all_runtime = []
    for cat in ['shape', 'tex']:
        all_eadi.extend([r['eadi_d'] for r in results[cat] if r['eadi_d'] > 0])
        all_runtime.extend([r['runtime'] for r in results[cat]])

    print(f"\nALL ({len(all_runtime)} objects):")
    print(f"  eADI/d: mean={np.mean(all_eadi):.6f}")
    print(f"  Runtime: mean={np.mean(all_runtime):.4f}s")

    # Detailed results
    print("\nDetailed results:")
    for cat in ['shape', 'tex']:
        for r in results[cat]:
            print(f"  {r['name']}: type={r['sym_type']}, eADI/d={r['eadi_d']:.6f}, runtime={r['runtime']:.3f}s")

if __name__ == '__main__':
    main()
