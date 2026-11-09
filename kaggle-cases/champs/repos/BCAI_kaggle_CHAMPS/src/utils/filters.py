## Copyright (c) 2017 Robert Bosch GmbH
## All rights reserved.
##
## This source code is licensed under the MIT license found in the
## LICENSE file in the root directory of this source tree.

from graph_transformer import sqdist
import torch
import torch.nn.functional as F
import numpy as np

def tta(x_atom, x_atom_pos, x_bond, x_bond_dist, x_triplet, x_triplet_angle, x_quad, x_quad_angle, args):
    elem_drop = args.elem_drop
    if elem_drop == 0.0:
        return x_atom, x_atom_pos, x_bond, x_bond_dist, x_triplet, x_triplet_angle, x_quad, x_quad_angle
    dev = x_atom.device
    bsz = x_atom.shape[0]
    N = x_atom.shape[1]
    M = x_bond.shape[1]
    P = x_triplet.shape[1]
    Q = x_quad.shape[1] if args.use_quad else 0
    atom_mask = (torch.zeros(bsz, N, 1).bernoulli_(1-elem_drop) / (1-elem_drop)).to(dev)
    bond_mask = (torch.zeros(bsz, M, 1).bernoulli_(1-elem_drop) / (1-elem_drop)).to(dev)
    trip_mask = (torch.zeros(bsz, P, 1).bernoulli_(1-elem_drop) / (1-elem_drop)).to(dev)
    x_atom = x_atom * atom_mask.long()
    x_atom_pos = x_atom_pos * atom_mask
    x_bond = x_bond * bond_mask.long()
    x_bond_dist = x_bond_dist * bond_mask[:,:,0]
    x_triplet = x_triplet * trip_mask.long()
    x_triplet_angle = x_triplet_angle * trip_mask[:,:,0]
    if args.use_quad:
        quad_mask = torch.zeros(bsz, Q, 1).bernoulli_(1-elem_drop) / (1-elem_drop)
        x_quad = x_quad * quad_mask
        x_quad_angle = x_quad_angle * quad_mask
    return x_atom, x_atom_pos, x_bond, x_bond_dist, x_triplet, x_triplet_angle, x_quad, x_quad_angle


def subgraph_filter(x_atom, x_atom_pos, x_bond, x_bond_dist, x_triplet, x_triplet_angle, args):
    D = sqdist(x_atom_pos[:,:,:3], x_atom_pos[:,:,:3])
    x_atom, x_atom_pos, x_bond, x_bond_dist, x_triplet, x_triplet_angle = \
        x_atom.clone().detach(), x_atom_pos.clone().detach(), x_bond.clone().detach(), x_bond_dist.clone().detach(), x_triplet.clone().detach(), x_triplet_angle.clone().detach()
    bsz = x_atom.shape[0]
    bonds_mask = torch.ones(bsz, x_bond.shape[1], 1).to(x_atom.device)
    for mol_id in range(bsz):
        if np.random.uniform(0,1) > args.cutout:
            continue
        assert not args.use_quad, "Quads are NOT cut out yet"
        atom_dists = D[mol_id]
        atoms = x_atom[mol_id, :, 0]
        n_valid_atoms = (atoms > 0).sum().item()
        if n_valid_atoms < 10:
            continue
        idx_to_drop = np.random.randint(n_valid_atoms-1)
        dist_row = atom_dists[idx_to_drop]
        neighbor_to_drop = torch.argmin((dist_row[dist_row>0])[:n_valid_atoms-1]).item()
        if neighbor_to_drop >= idx_to_drop: 
            neighbor_to_drop += 1
        x_atom[mol_id, idx_to_drop] = 0
        x_atom[mol_id, neighbor_to_drop] = 0
        x_atom_pos[mol_id, idx_to_drop] = 0
        x_atom_pos[mol_id, neighbor_to_drop] = 0
        bond_pos_to_drop = (x_bond[mol_id, :, 3] == idx_to_drop) | (x_bond[mol_id, :, 3] == neighbor_to_drop) \
                         | (x_bond[mol_id, :, 4] == idx_to_drop) | (x_bond[mol_id, :, 4] == neighbor_to_drop)
        trip_pos_to_drop = (x_triplet[mol_id, :, 2] == idx_to_drop) | (x_triplet[mol_id, :, 2] == neighbor_to_drop) \
                         | (x_triplet[mol_id, :, 3] == idx_to_drop) | (x_triplet[mol_id, :, 3] == neighbor_to_drop) \
                         | (x_triplet[mol_id, :, 4] == idx_to_drop) | (x_triplet[mol_id, :, 4] == neighbor_to_drop)
        x_bond[mol_id, bond_pos_to_drop] = 0
        x_bond_dist[mol_id, bond_pos_to_drop] = 0
        bonds_mask[mol_id, bond_pos_to_drop] = 0
        x_triplet[mol_id, trip_pos_to_drop] = 0
        x_triplet_angle[mol_id, trip_pos_to_drop] = 0
    return x_atom, x_atom_pos, x_bond, x_bond_dist, x_triplet, x_triplet_angle, bonds_mask
