#!/usr/bin/env python

## Copyright (c) 2017 Robert Bosch GmbH
## All rights reserved.
##
## This source code is licensed under the MIT license found in the
## LICENSE file in the root directory of this source tree.

import collections
import gzip
import itertools
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import rdkit
import xyz2mol as x2m

# Due to some compatibility issues between rdkit/pybel and torch, we have to load them as needed.
# Rules are meant to be broken, including best-programming practices :)


bond_order_dict = { rdkit.Chem.rdchem.BondType.SINGLE: 1,
                    rdkit.Chem.rdchem.BondType.AROMATIC: 1.5,
                    rdkit.Chem.rdchem.BondType.DOUBLE: 2,
                    rdkit.Chem.rdchem.BondType.TRIPLE: 3}
root = '../'  # This should be the root of the archive
with open(os.path.join(root,'SETTINGS.json')) as f:
    settings = json.load(f)
with open(os.path.join(root,settings['CONFIG_DIR'],'manual_bond_order_fix.json')) as f:
    manual_bond_order_dict = json.load(f)

atomic_num_dict = { 'H':1, 'C':6, 'N':7, 'O':8, 'F':9 }
# These were mistaken or too small datasets, so we are relabeling them.
classification_corrections = {
                      '1JHN_2_2_1_1':'1JHN_3_2_2_1',
                      '3JHN_4.5_3_1.5_1.5':'3JHN_4_3_1.5_1.5',
                      '2JHC_3_3_1_1':'2JHC_4_3_2_1',
                      '3JHC_3_3_1_1':'3JHC_4_3_2_1',
                      '3JHC_4_2_2_2':'3JHC_4_2_3_1'}
# These have less than 1000 between train and test, so we will drop the subtypes
small_longtypes = {'2JHN_4.5_2_3_1.5', '3JHN_4_2_3_1', '2JHN_4_2_3_1',
                   '2JHN_4.5_3_1.5_1.5', '2JHN_4_3_2_1', '3JHN_4_4_1_1',
                   '3JHN_4_3_2_1', '2JHN_4_4_1_1', '3JHN_4.5_2_3_1.5',
                   '2JHN_4_2_2_2', '3JHN_4_2_2_2', '1JHN_4_3_2_1',
                   '1JHN_4_4_1_1', '2JHN_3_1_3_0'}
(MAX_ATOM_COUNT,MAX_BOND_COUNT,MAX_TRIPLET_COUNT,MAX_QUAD_COUNT) = (29, 406, 54, 117)


def make_structure_dict(atoms_dataframe):
    """Convert from structures.csv output to a dictionary data storage.

    Args:
        atoms_dataframe: The dataframe corresponding to structures.csv

    Returns:
        dict: Mapping of molecule name to molecule properties.

    """
    atoms = atoms_dataframe.sort_values(["molecule_name", "atom_index"])  # ensure ordering is consistent
    # Make a molecule-based dictionary of the information
    structure_dict = collections.defaultdict(lambda: {"symbols":[],"positions":[]})
    for index,row in atoms.iterrows():
        structure_dict[row["molecule_name"]]["symbols"].append(row["atom"])
        structure_dict[row["molecule_name"]]["positions"].append([row["x"],row["y"],row["z"]])
    return structure_dict


def enhance_structure_dict(structure_dict):
    """Add derived information to the structure dictionary.

    Args:
        structure_dict: Output of :func:`make_structure_dict`.

    Returns:
        dict: The same, modified in-place, with derived information (e.g. atom distances).

    Caution: If torch is imported at the same time as this is run, you may get a segmentation fault. Complain to pybel or rdkit, I suppose.
    """
    import pybel
    for molecule_name in structure_dict:

        # positions - array (N,3) of Cartesian positions
        molecule = structure_dict[molecule_name]
        positions = np.array(molecule['positions'])
        n_atom = positions.shape[0]
        molecule['positions'] = positions

        # distances - array (N,N) of distances between atoms
        pos1 = np.tile(positions, (n_atom,1,1) )
        pos2 = np.transpose(pos1, (1,0,2) )
        dist = np.linalg.norm(pos1 - pos2, axis=-1)
        molecule['distances'] = dist

        # angle - array (N,) of angles to the 2 closest atoms
        sorted_j = np.argsort(dist, axis=-1)
        relpos1 = positions[sorted_j[:,1],:] - positions[sorted_j[:,0],:]
        relpos2 = positions[sorted_j[:,2],:] - positions[sorted_j[:,0],:]
        cos = np.sum(relpos1*relpos2,axis=1) / (np.linalg.norm(relpos1,axis=1) * np.linalg.norm(relpos2,axis=1))
        angle = np.arccos( np.clip(cos,-1.0,1.0) ).reshape((n_atom,1)) / np.pi
        molecule['angle'] = angle[:,0]

        # bond orders - array (N,N) of the bond order (0 for no chemical bond)
        # Note this relies on a few manual corrections
        molecule['bond_orders'] = np.zeros((n_atom,n_atom))
        atomicNumList = [atomic_num_dict[symbol] for symbol in molecule['symbols']]
        if molecule_name in manual_bond_order_dict:
            molecule['bond_orders'] = np.array(manual_bond_order_dict[molecule_name],dtype=float)
        else:
            mol = x2m.xyz2mol(atomicNumList,0,positions,True,True)
            for bond in mol.GetBonds():
                atom0, atom1 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                bond_order = bond.GetBondType()
                molecule['bond_orders'][atom0,atom1] = bond_order_dict[bond_order]
                molecule['bond_orders'][atom1,atom0] = bond_order_dict[bond_order]

        # Supplementary information for tagging:
        # top_bonds: (N,4 or less) bond orders of the top 4 bonds, for each atom
        # bond_ids: (N,4): Label the atom with the following 4 linear transform of top_bonds:
        #   * total num bonds (valence), counting double as 2
        #   * total num bonded neighbors, counting double as 1
        #   * largest order
        #   * second largest order.
        molecule['top_bonds'] = np.sort(molecule['bond_orders'],axis=-1)[:,-1:-5:-1]
        molecule['bond_ids'] = np.hstack((molecule['top_bonds'].sum(axis=-1)[:,np.newaxis],
                                          np.sum(molecule['top_bonds']>1e-3,axis=-1)[:,np.newaxis],
                                          molecule['top_bonds'][:,:2]))
        # long_symbols (N,) string relabel of the symbol straight from bond_ids
        molecule['long_symbols'] = ['_'.join([
            molecule['symbols'][i]]+[str(x) for x in molecule['bond_ids'][i]])
                                    for i in range(n_atom)]
        chem_bond_atoms = [sorted([molecule['symbols'][i] for i in molecule['bond_orders'][atom_index].nonzero()[0]])
                           for atom_index in range(n_atom)]
        molecule['sublabel_atom'] = ['-'.join([molecule['long_symbols'][atom_index]]+chem_bond_atoms[atom_index])
                                    for atom_index in range(n_atom)]

        # pybel information. I think we only end up using Gastiger charges.
        # Each of these is (N,) arrays
        # Convert to xyz string for pybel's I/O
        xyz = str(n_atom)+'\n\n' + '\n'.join([ ' '.join( [
                str(molecule['symbols'][i]),
                str(molecule['positions'][i,0]),
                str(molecule['positions'][i,1]),
                str(molecule['positions'][i,2])] )
                for i in range(n_atom)])
        mol = pybel.readstring('xyz',xyz)
        molecule['charges'] = [mol.atoms[i].partialcharge for i in range(n_atom)]
        molecule['spins'] = [mol.atoms[i].spin for i in range(n_atom)]
        molecule['heavyvalences'] = [mol.atoms[i].heavyvalence for i in range(n_atom)]
        molecule['heterovalences'] = [mol.atoms[i].heterovalence for i in range(n_atom)]
        molecule['valences'] = [mol.atoms[i].valence for i in range(n_atom)]
        molecule['hyb_types'] = [mol.atoms[i].type for i in range(n_atom)]
    return structure_dict


def enhance_atoms(atoms_dataframe,structure_dict):
    """Enhance the atoms dataframe by including derived information.

    Args:
        atoms_dataframe: Pandas dataframe read from structures.csv.
        structure_dict: Output of :func:`make_structure_dict`, after running :func:`enhance_structure_dict`.

    Returns:
        pandas.DataFrame: Same dataframe, modified in-place, with derived information added.

    """
    assert int(atoms_dataframe.groupby("molecule_name").count().max()[0]) <= MAX_ATOM_COUNT
    for key in ['distances','angle', 'bond_orders', 'top_bonds', 'bond_ids', 'long_symbols','sublabel_atom',
                'charges', 'spins', 'heavyvalences', 'heterovalences', 'valences', 'hyb_types']:
        newkey = key if key[-1]!='s' else key[:-1]
        atoms_dataframe[newkey] = atoms_dataframe.apply(lambda x:
                                                        structure_dict[x['molecule_name']][key][x['atom_index']],
                                                        axis=1)
        atoms_dataframe.rename(columns={'long_symbol':'labeled_atom'},inplace=True)
    return atoms_dataframe


def enhance_bonds(bond_dataframe,structure_dict):
    """Enhance the bonds dataframe by including derived information.

    Args:
        bond_dataframe: Pandas dataframe read from train.csv or test.csv.
        structure_dict: Output of :func:`make_structure_dict`, after running :func:`enhance_structure_dict`.

    Returns:
        pandas.DataFrame: Same dataframe, modified in-place, with derived information added.

    """
    bond_dataframe.sort_values(['molecule_name','atom_index_0','atom_index_1'],inplace=True)
    assert int(bond_dataframe.groupby("molecule_name").count().max()[0]) <= MAX_BOND_COUNT
    new_columns = collections.defaultdict(list)
    for index,row in bond_dataframe.iterrows():
        molecule_name, iatom0, iatom1 = row['molecule_name'],row['atom_index_0'],row['atom_index_1']
        if 'predict' not in structure_dict[molecule_name]:
            structure_dict[molecule_name]['predict'] = structure_dict[molecule_name]['bond_orders'] * 0
        structure_dict[molecule_name]['predict'][iatom0,iatom1] = 1
        structure_dict[molecule_name]['predict'][iatom1,iatom0] = 1
        long_symbols = [structure_dict[molecule_name]['long_symbols'][x] for x in [iatom0,iatom1]]

        # labeled_type
        if all([x[0]=='H' for x in long_symbols]):
            lt = row['type']
        elif not any([x[0]=='H' for x in long_symbols]):
            raise ValueError("No hydrogen found in {}".format(row))
        else:
            ls = [x for x in long_symbols if x[0]!='H'][0]
            lt = row["type"] + ls[1:].replace('.0','')
            if lt in classification_corrections:
                lt = classification_corrections[lt]
            if lt in small_longtypes:
                lt = lt.split('_')[0]
        new_columns["labeled_type"].append(lt)

        # sublabeled type
        new_columns["sublabel_type"].append(row['type'] + '-'+ '-'.join(sorted(long_symbols)))
        # bond order
        new_columns["bond_order"].append(structure_dict[molecule_name]['bond_orders'][iatom0,iatom1])
        new_columns["predict"].append(1)
    for key in new_columns:
        bond_dataframe[key] = new_columns[key]
    return bond_dataframe


def add_all_pairs(bond_dataframe,structure_dict):
    """Add all pairs of atoms, including those without coupling and without chemical bonds.

    Args:
        bond_dataframe: Pandas dataframe read from train.csv or test.csv, after running :func:`enhance_bonds`.
        structure_dict: Output of :func:`make_structure_dict`, after running :func:`enhance_structure_dict`.

    Returns:
        pandas.DataFrame: New dataframe, with new bonds added.

    """
    # NOTE: The convention for id used to be very large numbers for new bonds; now it is negative.
    iadd = -1
    new_data = collections.defaultdict(list)
    for molecule_name in bond_dataframe["molecule_name"].unique():
        n_atom = len(structure_dict[molecule_name]["symbols"])
        # for backwards compatibility, this is iatom1,iatom0. See make_new_csv.py, write_pairs.
        for iatom1,iatom0 in itertools.combinations(range(n_atom),r=2):
            if 'predict' not in structure_dict[molecule_name]:
                raise KeyError('{} has no "predict" value'.format(molecule_name))
            if structure_dict[molecule_name]['predict'][iatom0,iatom1]:
                continue  # already got it
            symbols = [structure_dict[molecule_name]['symbols'][i] for i in [iatom0,iatom1]]
            bond_order = structure_dict[molecule_name]['bond_orders'][iatom0,iatom1]
            nottype = '-'.join(sorted(symbols)) + '_' + str(bond_order)

            row = {'id':iadd,'molecule_name':molecule_name,'atom_index_0':iatom0,'atom_index_1':iatom1,
                   'type':nottype,'labeled_type':nottype,'sublabel_type':nottype,
                   'bond_order': bond_order,
                   'predict':0}
            if 'scalar_coupling_constant' in bond_dataframe:
                row['scalar_coupling_constant'] = 0.
            for k,v in row.items():
                new_data[k].append(v)
            iadd -= 1
    new_data = pd.DataFrame(new_data)
    if bond_dataframe.index.name!='id':
        bond_dataframe = bond_dataframe.set_index('id')
    new_data.set_index('id',inplace=True)
    all_data = bond_dataframe.append(new_data,verify_integrity=True,sort=False)
    return all_data


def make_triplets(molecule_list,structure_dict):
    """Make the triplet dataframe.

    Args:
        molecule_list: List of molecules to generate.
        structure_dict: Output of :func:`make_structure_dict`, after running :func:`enhance_structure_dict`.

    Returns:
        pandas.DataFrame: New dataframe, with triplets and related information. The convention is the bond looks like 1-0-2, where 0 is the central atom.

    """
    new_data = collections.defaultdict(list)
    for molecule_name in molecule_list:
        molecule = structure_dict[molecule_name]
        bond_orders = molecule['bond_orders']
        short = molecule['symbols']
        long = molecule['long_symbols']
        for i, atom_bond_order in enumerate(bond_orders):
            connection_indices = atom_bond_order.nonzero()[0]
            pairs = itertools.combinations(connection_indices,2)
            for pair in pairs:
                j, k = pair[0], pair[1]
                atom0_short = short[i] + long[i].split('_')[2]
                atom1_short = short[j] + long[j].split('_')[2]
                atom2_short = short[k] + long[k].split('_')[2]
                atom0_long = long[i]
                atom1_long = long[j]
                atom2_long = long[k]
                #labels = ['-'.join([atom1_short,str(atom_bond_order[j])]),
                #          '-'.join([atom2_short,str(atom_bond_order[k])])]
                labels = [atom1_short,atom2_short]
                labels.sort()
                label = '-'.join([atom0_short]+labels)
                #sublabels = ['-'.join([atom1_long,str(atom_bond_order[j])]),
                #             '-'.join([atom2_long,str(atom_bond_order[k])])]
                sublabels = [atom1_long,atom2_long]
                sublabels.sort()
                sublabel = '-'.join([atom0_long]+sublabels)
                r10 = molecule['positions'][j] - molecule['positions'][i]
                r20 = molecule['positions'][k] - molecule['positions'][i]
                angle = np.sum(r10*r20) / (np.linalg.norm(r10)*np.linalg.norm(r20))
                angle = np.arccos( np.clip(angle,-1.0,1.0) )
                row = {'molecule_name':molecule_name,'atom_index_0':i,'atom_index_1':j,'atom_index_2':k,
                      'label':label,'sublabel':sublabel,'angle':angle}
                for k,v in row.items():
                    new_data[k].append(v)
    ans = pd.DataFrame(new_data)
    ans.sort_values(['molecule_name','atom_index_0','atom_index_1','atom_index_2'])
    assert int(ans.groupby("molecule_name").count().max()[0]) <= MAX_TRIPLET_COUNT
    return ans


def make_quadruplets(molecule_list,structure_dict):
    """Make the quadruplet dataframe.

    Args:
        molecule_list: List of molecules to generate.
        structure_dict: Output of :func:`make_structure_dict`, after running :func:`enhance_structure_dict`.

    Returns:
        pandas.DataFrame: New dataframe, with quadruplets and related information. Make quadruplets. Convention is that they are connected 2-0-1-3, where 0,1 are the central atoms and 0-2 is a bond.

    """
    new_data = collections.defaultdict(list)
    icount = 0  # for debugging
    for molecule_name in molecule_list:
        molecule = structure_dict[molecule_name]
        bond_orders = molecule['bond_orders']
        short = molecule['symbols']
        long = molecule['long_symbols']
        pos = molecule['positions']
        for i,j in zip(*bond_orders.nonzero()):
            if i > j:
                continue  # we will get it the other way
            for i_nei,j_nei in itertools.product(
                    bond_orders[i].nonzero()[0],bond_orders[j].nonzero()[0]):
                if j_nei==i or i_nei==j:
                    continue  # no self
                # But we could have i_nei==j_nei, which is a triangle
                # Atomic structure looks like i_nei-i-j-j_nei
                # There's an easy way and a quick way.
                mode = 'fast'
                assert ['test','fast','slow'].count(mode),'Mode must be one of: test, fast, slow'
                if ['test','slow'].count(mode):
                    plane_1 = np.cross( pos[i_nei]-pos[i], pos[j]-pos[i])
                    plane_2 = np.cross( pos[i]-pos[j],pos[j_nei]-pos[j])
                    if np.allclose(plane_1,0.) or np.allclose(plane_2,0.):
                        # Planar; not really a dihedral
                        continue
                    # Compute the dihedral in radians
                    costheta = np.dot(plane_1,plane_2) / (
                        np.linalg.norm(plane_1)*np.linalg.norm(plane_2))
                    costheta1 = costheta
                if ['test','fast'].count(mode):  # this way is much faster
                    # Uses some clever algebra
                    ijpos = np.array([
                            pos[i_nei] - pos[i],
                            pos[j] - pos[i],
                            pos[j_nei] - pos[j],
                            ])
                    # For simplicity, call these a,b,c
                    dots = np.dot(ijpos,ijpos.T)
                    # numerator = (a x b).(-b x c)
                    # denominator = |a x b| |b x c|
                    # So:
                    # -(axb).(bxc) = (b.b)(a.c) - (a.b)(b.c)
                    numerator = dots[1,1]*dots[0,2] - dots[0,1]*dots[1,2]
                    # |axb|^2=|a|^2|b|^2-(a.b)^2
                    denominator = np.sqrt( (
                            dots[0,0]*dots[1,1]-dots[0,1]**2) * (
                            dots[2,2]*dots[1,1]-dots[2,1]**2 ))
                    if abs(denominator) < 1e-7:
                        # Planar, not really a dihedral
                        continue
                    costheta = numerator / denominator
                if mode=='test':
                    assert abs(costheta-costheta1)<1e-4,"Fancy algebra failed"
                    icount += 1
                    if icount > 50000:
                        raise Exception("50K counts confirmed.")
                assert abs(costheta)<1.0001,'Cos theta too large'
                dihedral = np.arccos( np.clip(costheta,-1.0,1.0) )
                # Start labeling
                label = '_'.join(sorted([
                    '_'.join([short[i],short[i_nei]]),
                    '_'.join([short[j],short[j_nei]]),
                    ]))

                # This definition finds several unique labels in the test set, e.g. 'C3_C4_C4_N4'
                #sublabel = '_'.join(sorted([
                #    '_'.join([short[i]+long[i].split('_')[1],short[i_nei]+long[i_nei].split('_')[1]]),
                #    '_'.join([short[j]+long[j].split('_')[1],short[j_nei]+long[j_nei].split('_')[1]]),
                #    ])).replace('.0','')

                # This definition finds several unique labels in the test set, e.g. C_3_3_1_1_C_C_4_4_1_1_N
                #sublabel2 = '_'.join(sorted([
                #    '_'.join([long[i],short[i_nei]]),
                #    '_'.join([long[j],short[j_nei]]),
                #    ])).replace('.0','')

                # This definition finds several unique labels in the test set, {'C_O_1_N_C_2_2',
                # 'N_C_1_N_O_1_2', 'N_N_2_O_C_1_1'}
                sublabel4 = '_'.join(sorted([
                    '_'.join([short[i],short[i_nei],str(bond_orders[i,i_nei].round(1))]),
                    '_'.join([short[j],short[j_nei],str(bond_orders[j,j_nei].round(1))]),
                    ]) + [str(bond_orders[i,j].round(1))]
                    ).replace('.0','')

                # This definition finds several unique labels in the test set, e.g. C3_C4_1_C4_N4_1_1'
                #sublabel4 = '_'.join(sorted([
                #    '_'.join([short[i]+long[i].split('_')[1],short[i_nei]+long[i_nei].split('_')[1],
                #        str(bond_orders[i,i_nei].round(1))]),
                #    '_'.join([short[j]+long[j].split('_')[1],short[j_nei]+long[j_nei].split('_')[1],
                #        str(bond_orders[j,j_nei].round(1))]),
                #    ]) + [str(bond_orders[i,j].round(1))]
                #    ).replace('.0','')

                sublabel = '_'.join(sorted([
                    '_'.join([short[i],short[i_nei]]),
                    '_'.join([short[j],short[j_nei]]),
                    ]) + [str(bond_orders[i,j].round(1))]
                    ).replace('.0','')

                sublabel2 = '_'.join(sorted([
                    '_'.join([short[i]+long[i].split('_')[1],short[i_nei]]),
                    '_'.join([short[j]+long[j].split('_')[1],short[j_nei]]),
                    ]) + [str(bond_orders[i,j].round(1))]
                    ).replace('.0','')

                sublabel3 = '_'.join(sorted([
                    '_'.join([short[i]+long[i].split('_')[1],short[i_nei]]),
                    '_'.join([short[j]+long[j].split('_')[1],short[j_nei]]),
                    ])).replace('.0','')
                row = {'molecule_name':molecule_name,
                       'atom_index_0':i,'atom_index_1':j,'atom_index_2':i_nei,'atom_index_3':j_nei,
                      'label':label,'sublabel':sublabel,'sublabel2':sublabel2,'sublabel3':sublabel3,
                       'sublabel4':sublabel4,'angle':dihedral}
                for k,v in row.items():
                    new_data[k].append(v)
    ans = pd.DataFrame(new_data)
    ans.sort_values(['molecule_name','atom_index_0','atom_index_1','atom_index_2','atom_index_3'])
    assert int(ans.groupby("molecule_name").count().max()[0]) <= MAX_QUAD_COUNT
    return ans


def write_csv(directory,label,atoms,bonds,triplets,quadruplets):
    """Write the relevant dataframes to a CSV file.

    Args:
        directory: Directory to write to.
        label (str): How to label the files, e.g. test or train.
        atoms: Pandas dataframe read from structures.csv, after running :func:`enhance_atoms`.
        bonds: Pandas dataframe read from train.csv or test.csv, after running :func:`enhance_bonds`.
        triplets: Pandas dataframe created by :func:`make_triplets`.
        quadruplets: Pandas dataframe created by :func:`make_quadruplets`.

    Returns:
        None

    """
    filename = os.path.join(directory,'new_big_{}.csv.bz2')
    if atoms is not None and len(atoms):
        atoms = atoms.sort_values(["molecule_name",'atom_index'])
        for i in range(4):
            atoms["top_bond_{}".format(i)] = [x[i] if len(x)>i else 0.0 for x in atoms["top_bond"].values]
        for i in ["x","y","z"]:
            atoms[i] = atoms[i].values.round(10)
        renames = {k:k[:-1] for k in atoms.columns if k[-1]=='s'}
        renames.update({'long_symbols':'labeled_atom'})
        atoms = atoms.rename(columns=renames)
        atoms.to_csv(filename.format('structures'),index=False,columns=
            'molecule_name,atom_index,atom,x,y,z,labeled_atom,angle,top_bond_0,top_bond_1,top_bond_2,top_bond_3,sublabel_atom,charge,spin,heavyvalence,heterovalence,valence,hyb_type'.split(','))
    if bonds is not None and len(bonds):
        bonds = bonds.reset_index()
        bond_columns = 'id,molecule_name,atom_index_0,atom_index_1,type,scalar_coupling_constant,labeled_type,sublabel_type,bond_order,predict'.split(',')
        if 'scalar_coupling_constant' not in bonds.columns:
            bond_columns = [x for x in bond_columns if x!='scalar_coupling_constant']
        bonds = bonds.sort_values(["predict","molecule_name",'atom_index_0','atom_index_1'],
                                  ascending=[False,True,True,True])
        bonds.to_csv(filename.format(label),index=False,columns=bond_columns)
    if triplets is not None and len(triplets):
        triplets = triplets.sort_values(["molecule_name",'atom_index_0','atom_index_1','atom_index_2'])
        triplets.to_csv(filename.format(label+'_triplets'),index=False,columns=
            'molecule_name,atom_index_0,atom_index_1,atom_index_2,label,sublabel,angle'.split(','))
    if quadruplets is not None and len(quadruplets):
        quadruplets = quadruplets.sort_values(["molecule_name",'atom_index_0','atom_index_1',
                                               'atom_index_2','atom_index_3'])
        quadruplets.to_csv(filename.format(label+'_quadruplets'),index=False,columns=
            'molecule_name,atom_index_0,atom_index_1,atom_index_2,atom_index_3,label,sublabel,sublabel2,sublabel3,sublabel4,angle'.split(','))


def _create_embedding(series):
    """Create a one-hot encoding embedding.

    Args:
        series: A DataFrame series (column).

    Returns:
        dict: Mapping of the entries (or "<None>") to the index number.

    """
    types = sorted(series.unique().tolist())
    assert "<None>" not in types
    emb_index = dict(zip(["<None>"] + types , range(len(types)+1)))
    return emb_index


def add_embedding(atoms,bonds,triplets,quadruplets,embeddings=None):
    """Add embedding indices to the dataframes.

    Args:
        atoms: Pandas dataframe read from structures.csv, after running :func:`enhance_atoms`.
        bonds: Pandas dataframe read from train.csv or test.csv, after running :func:`enhance_bonds`.
        triplets: Pandas dataframe created by :func:`make_triplets`.
        quadruplets: Pandas dataframe created by :func:`make_quadruplets`.
        embeddings (dict or None): If None, we create a new embedding (e.g. train data), otherwise we use the given embeddigns thar are output by :func:`add_embedding` (e.g. test data).

    Returns:
        dict: The embedding dictionary that can be passed to this function for using the same embedding on a new dataset.

    """
    # Add the embedding info to the dataframes.
    atoms["type_0"] = atoms["atom"]
    atoms["type_1"] = atoms["labeled_atom"].apply(lambda x : x[:5])
    atoms["type_2"] = atoms["labeled_atom"]
    bonds["type_0"] = bonds["type"]
    bonds["type_1"] = bonds["labeled_type"]
    bonds["type_2"] = bonds["sublabel_type"]
    triplets["type_0"] = triplets["label"].apply(lambda x : x[0] + x[5] + x[10])
    triplets["type_1"] = triplets["label"]
    quadruplets["type_0"] = quadruplets["label"]
    if embeddings is None:
        embeddings = {}
        embeddings.update({('atom',t):_create_embedding(atoms["type_" + str(t)]) for t in range(3)})
        embeddings.update({('bond',t):_create_embedding(bonds["type_" + str(t)]) for t in range(3)})
        embeddings.update({('triplet',t):_create_embedding(triplets["type_" + str(t)]) for t in range(2)})
        embeddings.update({('quadruplet',t):_create_embedding(quadruplets["type_" + str(t)]) for t in range(1)})
    for t in range(3):
        atoms["type_index_" + str(t)] = atoms["type_" + str(t)].apply(lambda x : embeddings[('atom',t)][x])
    for t in range(3):
        bonds["type_index_" + str(t)] = bonds["type_" + str(t)].apply(lambda x : embeddings[('bond',t)][x])
    for t in range(2):
        triplets["type_index_" + str(t)] = triplets["type_" + str(t)].apply(lambda x : embeddings[('triplet',t)][x])
    for t in range(1):
        quadruplets["type_index_" + str(t)] = quadruplets["type_" + str(t)].apply(lambda x : embeddings[('quadruplet',t)][x])
    return embeddings


def get_scaling(bonds_train):
    """Get the mean/std scaling factors for each ``labeled_type``.

    Args:
        bonds_train: The training data that we can use to set the values.

    Returns:
        tuple: Mean and std dicts, mapping labeled_type to scalar_coupling_constant mean/std.

    """
    # Get the mean/std scaling factors
    means = bonds_train.groupby("labeled_type").mean()["scalar_coupling_constant"].to_dict()
    stds = bonds_train.groupby("labeled_type").std()["scalar_coupling_constant"].to_dict()
    return means,stds


def add_scaling(bonds,means,stds):
    """Add the scaling information to the bonds dataframe.

    Args:
        bonds (pd.DataFrame): The dataframe of the bonds, after :func:`enhance_bonds`.
        means (dict): Output of :func:`get_scaling`.
        stds (dict): Output of :func:`get_scaling`.

    Returns:
        pd.DataFrame: Same dataframe, with added columns.

    """
    # Add mean/std scaling factors to bonds dataframe
    bonds["sc_mean"] = bonds["labeled_type"].apply(lambda x : means[x])
    bonds["sc_std"] = bonds["labeled_type"].apply(lambda x : stds[x])
    if "scalar_coupling_constant" in bonds.columns:
        bonds["sc_scaled"] = (bonds["scalar_coupling_constant"] - bonds["sc_mean"]) / bonds["sc_std"]
    return bonds


def create_dataset(atoms, bonds, triplets, quads, labeled = True, max_count = 10**10):
    """Create the python loaders, which we can pkl to a file for batching.

    Args:
        atoms: Pandas dataframe read from structures.csv, after running :func:`enhance_atoms`.
        bonds: Pandas dataframe read from train.csv or test.csv, after running :func:`enhance_bonds`.
        triplets: Pandas dataframe created by :func:`make_triplets`.
        quads: Pandas dataframe created by :func:`make_quadruplets`.
        labeled (bool): Whether this is train data, labeled with the y value.
        max_count (int): Maximum number of entries; useful for testing.

    Returns:
        tuple: With the following entries

            * x_index: (M,) Index of the molecule.
            * x_atom: (M,N,3) Atom type index.
            * x_atom_pos: (M,N,5) Atom position (3), closest-atom angle (1), and partial charge (1).
            * x_bond: (M,B,5) Bond type index (3), Atom index (2) corresponding to the bond.
            * x_bond_dist: (M,B) Distance of the bond.
            * x_triplet: (N,P,7): Triplet type (2), Atom index (3), Bond index (2) corresponding to the triplet.
            * x_triplet_angle: (N,P) Triplet angle.
            * x_quad: (N,Q,10) Quadruplet type (1), Atom index (4), Bond index (3), and triplet index (2) corresponding to the quadruplet.
            * x_quad_angle: (N,Q) Quadruplet dihedral angle.
            * y_bond_scalar_coupling: (N,M,4) of the scalar coupling constant, type mean, type std, and whether it should be predicted.

    """
    import torch
    from tqdm import tqdm
    # create mapping from molecule names to indices
    mol_unique = sorted(bonds["molecule_name"].unique().tolist())
    index = dict(zip(mol_unique, range(len(mol_unique))))
    atoms = atoms.set_index("molecule_name")
    bonds = bonds.set_index("molecule_name")
    triplets = triplets.set_index("molecule_name")
    quads = quads.set_index("molecule_name")
    quad_mols = set(quads.index)

    max_count = M = min(max_count, len(index))

    x_index = torch.arange(M, dtype=torch.long)
    x_atom = torch.zeros(M, MAX_ATOM_COUNT, 3, dtype=torch.long)
    x_atom_pos = torch.zeros(M, MAX_ATOM_COUNT, 5)
    x_bond = torch.zeros(M, MAX_BOND_COUNT, 5, dtype=torch.long)
    x_bond_dist = torch.zeros(M, MAX_BOND_COUNT)
    x_triplet = torch.zeros(M, MAX_TRIPLET_COUNT, 7, dtype=torch.long)
    x_triplet_angle = torch.zeros(M, MAX_TRIPLET_COUNT)
    x_quad = torch.zeros(M, MAX_QUAD_COUNT, 10, dtype=torch.long)
    x_quad_angle = torch.zeros(M, MAX_QUAD_COUNT)

    y_bond_scalar_coupling = torch.zeros(M, MAX_BOND_COUNT, 4)

    for k,i in tqdm(index.items()):
        if i >= M:
            break
        mol_atoms = atoms.loc[[k]]
        mol_bonds = bonds.loc[[k]]
        mol_real_bonds = mol_bonds[(mol_bonds["predict"]==1) | (mol_bonds["bond_order"]>0)]
        mol_fake_bonds = mol_bonds[(mol_bonds["predict"]==0) & (mol_bonds["bond_order"]==0)]
        mol_triplets = triplets.loc[[k]]

        n = mol_atoms.shape[0]
        m = mol_bonds.shape[0]
        mr = mol_real_bonds.shape[0]
        mf = mol_fake_bonds.shape[0]
        p = mol_triplets.shape[0]
        assert mr + mf == m, "Real + fake bonds != number of bonds?"
        assert mr < MAX_BOND_COUNT, "The number of real bonds is SMALLER than the MAX_BOND_COUNT"

        # STEP 1: Atoms
        for t in range(3):
            x_atom[i,:n,t] = torch.tensor(mol_atoms["type_index_" + str(t)].values)
        x_atom_pos[i,:n,:3] = torch.tensor(mol_atoms[["x", "y", "z"]].values)
        x_atom_pos[i,:n,3] = torch.tensor(mol_atoms["angle"].values)
        x_atom_pos[i,:n,4] = torch.tensor(mol_atoms["charge"].values)

        # STEP 2: Real bonds
        for t in range(3):
            x_bond[i,:mr,t] = torch.tensor(mol_real_bonds["type_index_" + str(t)].values)
        x_bond[i,:mr,3] = torch.tensor(mol_real_bonds["atom_index_0"].values)
        x_bond[i,:mr,4] = torch.tensor(mol_real_bonds["atom_index_1"].values)

        idx1 = torch.tensor(mol_real_bonds["atom_index_0"].values)
        idx2 = torch.tensor(mol_real_bonds["atom_index_1"].values)
        x_bond_dist[i,:mr] = ((x_atom_pos[i,idx1,:3] - x_atom_pos[i,idx2,:3])**2).sum(1)

        if mf > 0:
            # STEP 3: Fake bonds
            fidx1 = torch.tensor(mol_fake_bonds["atom_index_0"].values)
            fidx2 = torch.tensor(mol_fake_bonds["atom_index_1"].values)
            fdists = ((x_atom_pos[i,fidx1,:3] - x_atom_pos[i,fidx2,:3])**2).sum(1)   # Length mf
            argsort_fdists = torch.argsort(fdists)
            top_count = min(MAX_BOND_COUNT - mr, mf)

            for t in range(3):
                x_bond[i,mr:mr+top_count,t] = torch.tensor(mol_fake_bonds["type_index_" + str(t)].values)[argsort_fdists][:top_count]
            x_bond[i,mr:mr+top_count,3] = torch.tensor(mol_fake_bonds["atom_index_0"].values)[argsort_fdists][:top_count]
            x_bond[i,mr:mr+top_count,4] = torch.tensor(mol_fake_bonds["atom_index_1"].values)[argsort_fdists][:top_count]
            x_bond_dist[i,mr:mr+top_count] = fdists[argsort_fdists][:top_count]

        # STEP 4: Triplets
        for t in range(2):
            x_triplet[i,:p,t] = torch.tensor(mol_triplets["type_index_" + str(t)].values)
        x_triplet[i,:p,2] = torch.tensor(mol_triplets["atom_index_0"].values)
        x_triplet[i,:p,3] = torch.tensor(mol_triplets["atom_index_1"].values)
        x_triplet[i,:p,4] = torch.tensor(mol_triplets["atom_index_2"].values)

        x_triplet_angle[i,:p] = torch.tensor(mol_triplets["angle"].values)
        lookup = dict(zip(mol_real_bonds["atom_index_0"].apply(str) + "_" + mol_real_bonds["atom_index_1"].apply(str),
                          range(mol_real_bonds.shape[0])))
        lookup.update(dict(zip(mol_real_bonds["atom_index_1"].apply(str) + "_" + mol_real_bonds["atom_index_0"].apply(str),
                           range(mol_real_bonds.shape[0]))))

        b_idx1 = (mol_triplets["atom_index_0"].apply(str) + "_" +
                  mol_triplets["atom_index_1"].apply(str)).apply(lambda x : lookup[x])
        b_idx2 = (mol_triplets["atom_index_0"].apply(str) + "_" +
                  mol_triplets["atom_index_2"].apply(str)).apply(lambda x : lookup[x])

        x_triplet[i,:p,5] = torch.tensor(b_idx1.values)
        x_triplet[i,:p,5] = torch.tensor(b_idx2.values)

        # STEP 5: Quadruplets
        if k in quad_mols:
            mol_quads = quads.loc[[k]]
            q = mol_quads.shape[0]

            x_quad[i,:q,0] = torch.tensor(mol_quads["type_index_0"].values)
            x_quad[i,:q,1] = torch.tensor(mol_quads["atom_index_0"].values)
            x_quad[i,:q,2] = torch.tensor(mol_quads["atom_index_1"].values)
            x_quad[i,:q,3] = torch.tensor(mol_quads["atom_index_2"].values)
            x_quad[i,:q,4] = torch.tensor(mol_quads["atom_index_3"].values)

            x_quad_angle[i,:q] = torch.tensor(mol_quads["angle"].values)
            # Triplet convention is 1-0-2, so only 1/2 are exchangeable
            # Quadruplet convention is 2-0-1-3
            lookup3 = dict(zip(mol_triplets["atom_index_0"].apply(str) + "_" +
                               mol_triplets["atom_index_1"].apply(str) + "_" +
                               mol_triplets["atom_index_2"].apply(str),
                              range(mol_triplets.shape[0])))
            lookup3.update(dict(zip(mol_triplets["atom_index_0"].apply(str) + "_" +
                               mol_triplets["atom_index_2"].apply(str) + "_" +
                               mol_triplets["atom_index_1"].apply(str),
                              range(mol_triplets.shape[0]))))
            b_idx1 = (mol_quads["atom_index_0"].apply(str) + "_" +
                      mol_quads["atom_index_1"].apply(str)).apply(lambda x : lookup[x])
            b_idx2 = (mol_quads["atom_index_0"].apply(str) + "_" +
                      mol_quads["atom_index_2"].apply(str)).apply(lambda x : lookup[x])
            b_idx3 = (mol_quads["atom_index_1"].apply(str) + "_" +
                      mol_quads["atom_index_3"].apply(str)).apply(lambda x : lookup[x])
            t_idx1 = (mol_quads["atom_index_0"].apply(str) + "_" +
                      mol_quads["atom_index_1"].apply(str) + "_" +
                      mol_quads["atom_index_2"].apply(str)).apply(lambda x : lookup3[x])
            t_idx2 = (mol_quads["atom_index_1"].apply(str) + "_" +
                      mol_quads["atom_index_0"].apply(str) + "_" +
                      mol_quads["atom_index_3"].apply(str)).apply(lambda x : lookup3[x])

            x_quad[i,:q,5] = torch.tensor(b_idx1.values)
            x_quad[i,:q,6] = torch.tensor(b_idx2.values)
            x_quad[i,:q,7] = torch.tensor(b_idx3.values)
            x_quad[i,:q,8] = torch.tensor(t_idx1.values)
            x_quad[i,:q,9] = torch.tensor(t_idx2.values)

            x_quad_angle[i,:q] = torch.tensor(mol_quads["angle"].values)

        if labeled:
            y_bond_scalar_coupling[i,:mr, 0] = torch.tensor(mol_real_bonds["scalar_coupling_constant"].values)
        else:
            y_bond_scalar_coupling[i,:mr, 0] = torch.tensor(mol_real_bonds["id"].values)
        y_bond_scalar_coupling[i,:mr, 1] = torch.tensor(mol_real_bonds["sc_mean"].values)
        y_bond_scalar_coupling[i,:mr, 2] = torch.tensor(mol_real_bonds["sc_std"].values)
        y_bond_scalar_coupling[i,:mr, 3] = torch.tensor(mol_real_bonds["predict"].values).float()  # binary tensor (1s to be predicted)

    return x_index, x_atom, x_atom_pos, x_bond, x_bond_dist, x_triplet, x_triplet_angle, x_quad, x_quad_angle, y_bond_scalar_coupling


def auto_preproc_stage1():
    """Stage 1: Read and process csv files to new csv files."""
    print('Reading structures...')
    atoms = pd.read_csv(os.path.join(root,settings['RAW_DATA_DIR'],'structures.csv'))
    print('Parsing structures...')
    structure_dict = make_structure_dict(atoms)
    print('Adding structure features...')
    enhance_structure_dict(structure_dict)
    print('Updating atoms dataframe...')
    enhance_atoms(atoms,structure_dict)
    print('Writing structures...')
    write_csv(os.path.join(root,settings['PROCESSED_DATA_DIR']),'',atoms,None,None,None)

    print('Reading bonds for train...')
    bonds = pd.read_csv(os.path.join(root,settings['RAW_DATA_DIR'],'train.csv'))
    print('Parsing bonds...')
    enhance_bonds(bonds,structure_dict)
    bonds = add_all_pairs(bonds,structure_dict)
    triplets = make_triplets(bonds["molecule_name"].unique(),structure_dict)
    quadruplets = make_quadruplets(bonds["molecule_name"].unique(),structure_dict)
    print('Writing bonds...')
    write_csv(os.path.join(root,settings['PROCESSED_DATA_DIR']),'train',None,bonds,triplets,quadruplets)

    print('Reading bonds for test...')
    bonds = pd.read_csv(os.path.join(root,settings['RAW_DATA_DIR'],'test.csv'))
    print('Parsing bonds...')
    enhance_bonds(bonds,structure_dict)
    bonds = add_all_pairs(bonds,structure_dict)
    triplets = make_triplets(bonds["molecule_name"].unique(),structure_dict)
    quadruplets = make_quadruplets(bonds["molecule_name"].unique(),structure_dict)
    print('Writing bonds...')
    write_csv(os.path.join(root,settings['PROCESSED_DATA_DIR']),'test',None,bonds,triplets,quadruplets)


def auto_preproc_stage2():
    import torch
    print("Loading data...")
    os.chdir(os.path.join(root,settings['PROCESSED_DATA_DIR']))
    atoms = pd.read_csv('new_big_structures.csv.bz2')
    bonds = pd.read_csv('new_big_train.csv.bz2')
    triplets = pd.read_csv('new_big_train_triplets.csv.bz2')
    quadruplets = pd.read_csv('new_big_train_quadruplets.csv.bz2')

    print('Sorting...')
    atoms.sort_values(['molecule_name','atom_index'],inplace=True)
    bonds.sort_values(['molecule_name','atom_index_0','atom_index_1'],inplace=True)
    triplets.sort_values(['molecule_name','atom_index_0','atom_index_1','atom_index_2'],inplace=True)
    quadruplets.sort_values(['molecule_name','atom_index_0','atom_index_1','atom_index_2','atom_index_3'],inplace=True)

    assert int(atoms.groupby("molecule_name").count().max()[0]) <= MAX_ATOM_COUNT
    assert int(bonds.groupby("molecule_name").count().max()[0]) <= MAX_BOND_COUNT
    assert int(triplets.groupby("molecule_name").count().max()[0]) <= MAX_TRIPLET_COUNT
    assert int(quadruplets.groupby("molecule_name").count().max()[0]) <= MAX_QUAD_COUNT

    print("Adding embeddings and scaling...")
    embeddings = add_embedding(atoms,bonds,triplets,quadruplets)
    means,stds = get_scaling(bonds)
    bonds = add_scaling(bonds,means,stds)

    print("Creating train dataset...")
    D = create_dataset(atoms, bonds, triplets, quadruplets, labeled=True)

    print('Splitting train dataset...')
    #Split the training data into train (80%) and validation (20%) for model selection.
    np.random.seed(0)
    p = np.random.permutation(D[0].shape[0])

    idx_train = torch.cat([torch.tensor(p[:int(0.6*len(p))]), torch.tensor(p[int(0.8*len(p)):])])
    idx_val = torch.tensor(p[int(0.6*len(p)):int(0.8*len(p))])

    D_train = tuple([d[idx_train] for d in D])
    D_val = tuple([d[idx_val] for d in D])

    print('Saving train (80%)/validation (20%) datasets...')
    # If too large, save the two parts (just so that we can push to github)
    if sum([d.nelement() for d in D_train]) > 3e8:
        # Split D_train into 2 parts
        print("Splitting the 80% training data into part 1 and 2...")
        total_len = D_train[0].size(0)
        D_train_part1 = tuple([d[:total_len//2].clone().detach() for d in D_train])
        D_train_part2 = tuple([d[total_len//2:].clone().detach() for d in D_train])
        with gzip.open("torch_proc_train_p1.pkl.gz", "wb") as f:
            pickle.dump(D_train_part1, f, protocol=4)
        with gzip.open("torch_proc_train_p2.pkl.gz", "wb") as f:
            pickle.dump(D_train_part2, f, protocol=4)
    else:
        with gzip.open("torch_proc_train.pkl.gz", "wb") as f:
            pickle.dump(D_train, f, protocol=4)
    with gzip.open("torch_proc_val.pkl.gz", "wb") as f:
        pickle.dump(D_val, f, protocol=4)

    print("Saving the full train dataset. Splitting into part 1 and 2...")
    total_len = D[0].size(0)
    D_part1 = tuple([d[:total_len//2].clone().detach() for d in D])
    D_part2 = tuple([d[total_len//2:].clone().detach() for d in D])
    with gzip.open("torch_proc_train_full_p1.pkl.gz", "wb") as f:
        pickle.dump(D_part1, f, protocol=4)
    with gzip.open("torch_proc_train_full_p2.pkl.gz", "wb") as f:
        pickle.dump(D_part2, f, protocol=4)

    # ## Test
    print('Loading test data...')
    bonds = pd.read_csv('new_big_test.csv.bz2')
    triplets = pd.read_csv('new_big_test_triplets.csv.bz2')
    quadruplets = pd.read_csv('new_big_test_quadruplets.csv.bz2')

    print('Sorting...')
    bonds.sort_values(['molecule_name','atom_index_0','atom_index_1'],inplace=True)
    triplets.sort_values(['molecule_name','atom_index_0','atom_index_1','atom_index_2'],inplace=True)
    quadruplets.sort_values(['molecule_name','atom_index_0','atom_index_1','atom_index_2','atom_index_3'],inplace=True)

    assert int(atoms.groupby("molecule_name").count().max()[0]) <= MAX_ATOM_COUNT
    assert int(bonds.groupby("molecule_name").count().max()[0]) <= MAX_BOND_COUNT
    assert int(triplets.groupby("molecule_name").count().max()[0]) <= MAX_TRIPLET_COUNT
    assert int(quadruplets.groupby("molecule_name").count().max()[0]) <= MAX_QUAD_COUNT

    print('Adding embedding and scaling...')
    add_embedding(atoms,bonds,triplets,quadruplets, embeddings=embeddings)
    bonds = add_scaling(bonds,means,stds)

    print('Creating test dataset...')
    D_sub = create_dataset(atoms, bonds, triplets, quadruplets, labeled=False)

    print('Saving file...')
    with gzip.open("torch_proc_submission.pkl.gz", "wb") as f:
        pickle.dump(D_sub, f, protocol=4)

    return


if __name__=='__main__':
    # There is a segmentation fault if stage1 is run while torch is loaded. So we have to run them separately.
    if '1' in sys.argv:
        auto_preproc_stage1()
    elif '2' in sys.argv:
        auto_preproc_stage2()
    else:
        print('Please identify either stage 1 or stage 2.')
