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

os.environ["DISPLAY"] = ""

# Import directly from modules to avoid namespace issues
from kasal.symmetry_lab.symmetry_axis_template import get_sym_axis_temp
from kasal.symmetry_lab.symmetry_axis_localization import cal_model_sym
from kasal.utils.io_ply import load_ply_model

# Set config value directly
import kasal.config
kasal.config.config.save_2_fold_a = False


def compute_adi(pts1, pts2):
    """Compute ADI (average distance of model points) between two point clouds."""
    dist = np.linalg.norm(pts1 - pts2, axis=1)
    return np.mean(dist)


def compute_eADI(vertices, symmetries_discrete, symmetries_continuous, diameter):
    """Compute eADI/d given the localized symmetry axes.

    eADI is the average ADI over all symmetry transformations.
    ADI for a transformation T: mean pointwise distance between original and T(original).
    """
    rotation_set = [np.eye(4)]

    if symmetries_discrete and len(symmetries_discrete) > 0:
        for sym_mat in symmetries_discrete:
            mat = np.array(sym_mat).reshape(4, 4)
            rotation_set.append(mat)

    if symmetries_continuous and len(symmetries_continuous) > 0:
        for sc in symmetries_continuous:
            axis = sc.get('axis', [])
            offset = np.array(sc.get('offset', [0, 0, 0]))
            if axis and len(axis) == 3 and not np.allclose(axis, 0):
                a = np.array(axis)
                a_norm = a / np.linalg.norm(a)
                for ang_deg in range(0, 360, 30):  # Sample every 30 degrees
                    if ang_deg == 0:
                        continue
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
    for mat in rotation_set[1:]:
        R = mat[:3, :3]
        t = mat[:3, 3]
        rotated_pts = np.dot(vertices, R.T) + t
        adi = compute_adi(vertices, rotated_pts)
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


def process_object(ply_file, json_file, category):
    """Process a single object and return metrics."""
    with open(json_file, 'r') as f:
        gt = json.load(f)

    sym_type = gt.get('sym_type', '')
    n_fold = gt.get('n-fold', 2)
    sym_op, step_path = map_sym_type(sym_type, n_fold)

    if sym_op is None:
        return None

    obj_name = os.path.basename(json_file)
    print(f"\n[{obj_name}] type={sym_type}, n={n_fold}")

    # For texture objects, use color-based mode
    if category == 'tex':
        adi_c = gt.get('ADI-C', False)
        op_mode = 'colors' if adi_c else 'pts'
    else:
        op_mode = 'pts'

    model_i_ = load_ply_model(ply_file, color_op=(op_mode == 'colors'))
    diameter = model_i_['diameter']

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

    print(f"  eADI/d={eadi_d:.6f}, runtime={elapsed:.1f}s, diameter={diameter:.2f}")
    return {
        'name': obj_name, 'category': category,
        'sym_type': sym_type, 'eadi_d': eadi_d,
        'runtime': elapsed, 'diameter': diameter
    }


def main():
    shape_dir = '/repo/DSRSTO/DSRSTO dataset/shape_meshes'
    tex_dir = '/repo/DSRSTO/DSRSTO dataset/tex_meshes'

    all_results = []
    errors = []

    # Process shape meshes
    json_files = sorted(glob.glob(os.path.join(shape_dir, '*_ours_sym_type.json')))
    print(f"Found {len(json_files)} shape objects with annotations")

    for json_file in json_files:
        base_name = json_file.replace('_ours_sym_type.json', '')
        ply_file = base_name + '_ours.ply'
        if not os.path.exists(ply_file):
            print(f"SKIP: PLY not found: {ply_file}")
            continue
        try:
            result = process_object(ply_file, json_file, 'shape')
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            errors.append((os.path.basename(json_file), str(e)))

    # Process texture meshes
    tex_json_files = sorted(glob.glob(os.path.join(tex_dir, '*_sym_type.json')))
    print(f"\nFound {len(tex_json_files)} texture objects with annotations")

    # Map each tex JSON to its PLY file
    for json_file in tex_json_files:
        base_name = json_file.replace('_sym_type.json', '')
        found = False
        for suffix in ['_t2_sym.ply', '_t_sym.ply', '_t3_sym.ply', '_t4_sym.ply']:
            ply_candidate = base_name + suffix
            if os.path.exists(ply_candidate):
                ply_file = ply_candidate
                found = True
                break
        # If sym.ply not found, try the .obj file
        if not found:
            for suffix in ['_t2.obj', '_t.obj', '_t3.obj', '.obj']:
                ply_candidate = base_name + suffix
                if os.path.exists(ply_candidate):
                    ply_file = ply_candidate
                    found = True
                    break
        if not found:
            print(f"SKIP: no mesh for {os.path.basename(json_file)}")
            continue
        try:
            result = process_object(ply_file, json_file, 'tex')
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            errors.append((os.path.basename(json_file), str(e)))

    # Summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(f"\nTotal objects processed: {len(all_results)}")
    print(f"Errors: {len(errors)}")

    valid_eadi = [r['eadi_d'] for r in all_results if r['eadi_d'] > 0]
    all_runtimes = [r['runtime'] for r in all_results]

    # Print results per object
    print(f"\n{'Name':<30} {'Type':<35} {'eADI/d':<12} {'Runtime(s)':<12}")
    print("-" * 90)
    for r in sorted(all_results, key=lambda x: x['category'] + x['name']):
        print(f"{r['name']:<30} {r['sym_type']:<35} {r['eadi_d']:<12.6f} {r['runtime']:<12.2f}")

    # Final report
    print(f"\n=== REPRODUCTION RESULTS ===")
    if valid_eadi:
        mean_eadi = np.mean(valid_eadi)
        median_eadi = np.median(valid_eadi)
        print(f"Metric: eADI/d (normalized by object diameter), Dataset: DSRSTO")
        print(f"Paper reported value: 0.00212, CI: [0.00172, 0.00216]")
        print(f"Reproduced value (mean): {mean_eadi:.6f}")
        print(f"Reproduced value (median): {median_eadi:.6f}")
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

    # By type breakdown
    from collections import defaultdict
    by_type = defaultdict(list)
    for r in all_results:
        by_type[r['sym_type']].append(r)

    print(f"\nResults by symmetry type:")
    for stype in sorted(by_type.keys()):
        results = by_type[stype]
        eadi_vals = [r['eadi_d'] for r in results if r['eadi_d'] > 0]
        rt_vals = [r['runtime'] for r in results]
        if eadi_vals:
            print(f"  {stype}: n={len(results)}, eADI/d={np.mean(eadi_vals):.6f}, runtime={np.mean(rt_vals):.1f}s")


if __name__ == '__main__':
    main()
