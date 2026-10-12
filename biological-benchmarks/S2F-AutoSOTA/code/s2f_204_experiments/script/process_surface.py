import os
import sys
import numpy as np
import pickle
import argparse

import torch
from torch.nn import functional as F

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from s3f import surface


ele2num = {"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "SE": 5}


def process(input_dir, output_dir, pkl_file):
    with open(os.path.join(input_dir, pkl_file), "rb") as fin:
        data_dict = pickle.load(fin)
    output_fname = os.path.join(output_dir, pkl_file)

    num_residue = len(data_dict["aatype"])
    atom_position = torch.as_tensor(data_dict["atom_positions"][:, :3]).float()     # take backbone atoms
    atom_type = torch.tensor([3, 0, 0], dtype=torch.long)[None, :].repeat(num_residue, 1)   # (N, CA, C)
    atom_position = atom_position.flatten(0, 1).cuda()
    atom_type = F.one_hot(atom_type.flatten(0, 1), num_classes=6).cuda()
    num_atom = len(atom_position)
    batch = torch.zeros((num_atom,), dtype=torch.long).cuda()

    surf_points, surf_normals, _ = surface.atoms_to_points_normals(atom_position, batch, atomtypes=atom_type)
    num_surf_points = len(surf_points)

    # Surface -> residue graph correspondence   (num_residue, 21) each element is a surface point index
    res2surf, _ = surface.knn_atoms(atom_position, surf_points, k=20)
    res2surf = res2surf.view(num_residue, 3, -1)

    batch_surf = torch.zeros((num_surf_points,), dtype=torch.long) #.cuda()
    surf_curvatures = surface.compute_curvatures(surf_points, surf_normals, batch=batch_surf, curvature_scales=[1.0, 2.0, 3.0, 5.0, 10.0])

    surf_points = surf_points.cpu().detach().numpy()
    eigs_ratio = 0.01 if num_surf_points > 20000 else 0.06
    surf_eig_vals, surf_eig_vecs, surf_eig_vecs_inv = surface.compute_eigens(num_surf_points, surf_points, min_n_eigs=50, eigs_ratio=eigs_ratio)

    surf_hks = surface.compute_HKS(surf_eig_vecs, surf_eig_vals, num_t=32, t_min=0.1, t_max=1000, scale=1000)

    surf_data_dict = {
        "surf_points": surf_points.astype(np.float32),
        "surf_normals": surf_normals.cpu().detach().numpy().astype(np.float32),
        "surf_hks": surf_hks.astype(np.float32),
        "surf_curvatures": surf_curvatures.cpu().detach().numpy().astype(np.float32),
        "res2surf": res2surf.cpu().detach().numpy(),
    }

    with open(output_fname, "wb") as fout:
        pickle.dump(surf_data_dict, fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str)
    parser.add_argument("-o", "--output_dir", type=str)
    args, unparsed = parser.parse_known_args()

    input_dir = os.path.expanduser(args.input_dir)
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pkl_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".pkl")])
    for pkl_file in tqdm(pkl_files):
        process(input_dir, output_dir, pkl_file)