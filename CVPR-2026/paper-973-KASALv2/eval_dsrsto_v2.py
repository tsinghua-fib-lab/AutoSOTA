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

# Suppress GUI by setting environment
os.environ["DISPLAY"] = ""

from kasal.symmetry_lab.symmetry_axis_template import get_sym_axis_temp
from kasal.symmetry_lab.symmetry_axis_localization import cal_model_sym
from kasal.utils.io_ply import load_ply_model
import kasal.config.config as config

config.save_2_fold_a = False


def compute_chamfer_distance(pts1, pts2):
    """Compute Chamfer distance between two point clouds."""
    dist1 = spatial.distance.cdist(pts1, pts2, metric='euclidean')
    d1 = np.mean(np.min(dist1, axis=1))
    d2 = np.mean(np.min(dist1, axis=0))
    return (d1 + d2) / 2.0


def compute_eADI(vertices, symmetries_discrete, symmetries_continuous, diameter):
    """Compute eADI/d given the localized symmetry axes."""
    rotation_set = [np.eye(4)]  # identity

    if symmetries_discrete and len(symmetries_discrete) > 0:
        for sym_mat in symmetries_discrete:
            mat = np.array(sym_mat).reshape(4, 4)
            rotation_set.append(mat)

    if symmetries_continuous and len(symmetries_continuous) > 0:
        for sc in symmetries_continuous:
            axis = sc.get('axis', [])
            offset = np.array(sc.get('offset', [0, 0, 0]))
            if axis and len(axis) == 3:
                a = np.array(axis)
                a_norm = a / np.linalg.norm(a)
                for ang_deg in [45, 90, 135, 180, 225, 270, 315]:
                    theta = np.radians(ang_deg)
                    c = np.cos(theta)
                    s = np.sin(theta)
                    K = np.array([[0, -a_norm[2], a_norm[1]],
                                  [a_norm[2], 0, -a_norm[0]],
                                  [-a_norm[1], a_norm[0], 0]])
                    R = np.eye(3) + s * K + (1 - c) * np.dot(K, K)
                    mat = np.eye(4)
                    mat[:3, :3] = R
                    mat[:3, 3] = offset - np.dot(R, offset)
                    rotation_set.append(mat)

    if len(rotation_set) <= 1:
        return 0.0

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
    return total_adi / count / diameter


def map_sym_type(sym_type, n_fold):
    """Map JSON sym_type to sym_op and step_path."""
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
    else:
        return None, None

    step_path = get_sym_axis_temp(sym_type, n_fold)
    return sym_op, step_path


def main():
    shape_dir = '/repo/DSRSTO/DSRSTO dataset/shape_meshes'
    tex_dir = '/repo/DSRSTO/DSRSTO dataset/tex_meshes'

    all_results = []
    total_runtime = 0.0
    total_objects = 0
    errors = []

    # Process shape meshes
    json_files = sorted(glob.glob(os.path.join(shape_dir, '*_ours_sym_type.json')))
    for json_file in json_files:
        base_name = json_file.replace('_ours_sym_type.json', '')
        ply_file = base_name + '_ours.ply'

        if not os.path.exists(ply_file):
            print(f"SKIP: PLY not found: {ply_file}")
            continue

        with open(json_file, 'r') as f:
            gt = json.load(f)

        sym_type = gt.get('sym_type', '')
        n_fold = gt.get('n-fold', 2)
        sym_op, step_path = map_sym_type(sym_type, n_fold)

        if sym_op is None:
            print(f"SKIP {os.path.basename(json_file)}: No symmetry or unknown type '{sym_type}'")
            continue

        obj_name = os.path.basename(json_file).replace('_ours_sym_type.json', '')
        print(f"\n[{obj_name}] type={sym_type}, n={n_fold}")

        try:
            model_i_ = load_ply_model(ply_file, color_op=False)
            diameter = model_i_['diameter']

            start_time = time.time()
            model_i_ = cal_model_sym(
                model_i_, step_path=step_path, sym_op=sym_op,
                sym_aware=False, op='pts', sample_num=10001,
                fpsample_num=1500, icp_op=True, xyz_op=None
            )
            elapsed = time.time() - start_time

            vertices = model_i_['vertices']
            symmetries_discrete = model_i_.get('symmetries_discrete', [])
            symmetries_continuous = model_i_.get('symmetries_continuous', [])
            eadi_d = compute_eADI(vertices, symmetries_discrete, symmetries_continuous, diameter)

            all_results.append({
                'name': obj_name, 'category': 'shape',
                'sym_type': sym_type, 'eadi_d': eadi_d,
                'runtime': elapsed, 'diameter': diameter
            })
            total_runtime += elapsed
            total_objects += 1
            print(f"  eADI/d={eadi_d:.6f}, runtime={elapsed:.1f}s")

        except Exception as e:
            print(f"  ERROR: {e}")
            errors.append((obj_name, str(e)))
            import traceback
            traceback.print_exc()

    # Process texture meshes
    tex_json_files = sorted(glob.glob(os.path.join(tex_dir, '*_sym_type.json')))
    for json_file in tex_json_files:
        with open(json_file, 'r') as f:
            gt = json.load(f)

        sym_type = gt.get('sym_type', '')
        n_fold = gt.get('n-fold', 2)
        sym_op, step_path = map_sym_type(sym_type, n_fold)

        if sym_op is None:
            print(f"SKIP tex {os.path.basename(json_file)}: No symmetry")
            continue

        base_name = json_file.replace('_sym_type.json', '')
        # Try different possible PLY/OBJ files
        found = False
        for suffix in ['_t2_sym.ply', '_t_sym.ply', '_t3_sym.ply', '.obj', '_t.obj', '_t2.obj']:
            ply_file = base_name + suffix
            if os.path.exists(ply_file):
                found = True
                break
        if not found:
            print(f"SKIP tex {os.path.basename(json_file)}: No mesh file found")
            continue

        obj_name = os.path.basename(json_file).replace('_sym_type.json', '')
        print(f"\n[TEX:{obj_name}] type={sym_type}, n={n_fold}, file={os.path.basename(ply_file)}")

        try:
            color_op = ('ADI-C' in gt and gt['ADI-C']) or 'tex' in sym_type.lower()
            model_i_ = load_ply_model(ply_file, color_op=color_op)
            diameter = model_i_['diameter']

            op_mode = 'colors' if color_op else 'pts'
            start_time = time.time()
            model_i_ = cal_model_sym(
                model_i_, step_path=step_path, sym_op=sym_op,
                sym_aware=False, op=op_mode, sample_num=10001,
                fpsample_num=1500, icp_op=True, xyz_op=None
            )
            elapsed = time.time() - start_time

            vertices = model_i_['vertices']
            symmetries_discrete = model_i_.get('symmetries_discrete', [])
            symmetries_continuous = model_i_.get('symmetries_continuous', [])
            eadi_d = compute_eADI(vertices, symmetries_discrete, symmetries_continuous, diameter)

            all_results.append({
                'name': obj_name, 'category': 'tex',
                'sym_type': sym_type, 'eadi_d': eadi_d,
                'runtime': elapsed, 'diameter': diameter
            })
            total_runtime += elapsed
            total_objects += 1
            print(f"  eADI/d={eadi_d:.6f}, runtime={elapsed:.1f}s")

        except Exception as e:
            print(f"  ERROR: {e}")
            errors.append((obj_name, str(e)))

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    valid_eadi = [r['eadi_d'] for r in all_results if r['eadi_d'] > 0]
    all_runtimes = [r['runtime'] for r in all_results]

    print(f"\nTotal objects processed: {len(all_results)}")
    print(f"Errors: {len(errors)}")
    for e in errors:
        print(f"  - {e[0]}: {e[1]}")

    print(f"\n=== REPRODUCTION RESULTS ===")
    if valid_eadi:
        mean_eadi = np.mean(valid_eadi)
        print(f"Metric: eADI/d, Dataset: DSRSTO (shape+tex)")
        print(f"Paper reported value: 0.00212, CI: [0.00172, 0.00216]")
        print(f"Reproduced value: {mean_eadi:.6f}")
        within_ci = 0.00172 <= mean_eadi <= 0.00216
        print(f"Within CI: {'Yes' if within_ci else 'No'}")
        print(f"---")

    if all_runtimes:
        mean_rt = np.mean(all_runtimes)
        print(f"Metric: Runtime (s) per object, Dataset: DSRSTO")
        print(f"Paper reported value: 1.46, CI: [1.4308, 1.4892]")
        print(f"Reproduced value: {mean_rt:.4f}")
        within_ci = 1.4308 <= mean_rt <= 1.4892
        print(f"Within CI: {'Yes' if within_ci else 'No'}")

    # Breakdown by type
    from collections import defaultdict
    by_type = defaultdict(list)
    for r in all_results:
        by_type[r['sym_type']].append(r)

    print(f"\nResults by symmetry type:")
    for stype, results in sorted(by_type.items()):
        eadi_vals = [r['eadi_d'] for r in results if r['eadi_d'] > 0]
        rt_vals = [r['runtime'] for r in results]
        print(f"  {stype}: count={len(results)}, eADI/d={np.mean(eadi_vals):.6f}, runtime={np.mean(rt_vals):.1f}s")


if __name__ == '__main__':
    main()
