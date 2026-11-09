#!/usr/bin/env python3

## Copyright (c) 2017 Robert Bosch GmbH
## All rights reserved.
##
## This source code is licensed under the MIT license found in the
## LICENSE file in the root directory of this source tree.

import bz2
import gzip
import importlib
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader


root = '../'  # This should be the root of the archive
with open(os.path.join(root,'SETTINGS.json')) as f:
    settings = json.load(f)
with open(os.path.join(root,settings['CONFIG_DIR'],'models.json')) as f:
    models = json.load(f)
# Determine the number of chosen models and medians to be averaged for each type
model_count_for_median = [9,9,9,9,9,9,9,9]
median_mean_counts = [5,5,5,5,5,5,5,5]


def load_model(name):
    model_folder = os.path.join(root,settings['MODEL_DIR'],models[name+'_dir'])
    if not os.path.isdir(model_folder):
        sys.stderr.write("Error reading model from {}\n".format(model_folder))
        return None
    if not os.path.isfile(os.path.join(model_folder,'model.ckpt')):
        sys.stderr.write("Error reading model from {}/model.ckpt\n".format(model_folder))
        return None
    sys.path = [model_folder] + sys.path
    import graph_transformer
    importlib.reload(graph_transformer)
    print("Loading {} from {}".format(name,graph_transformer.__file__))
    with open(os.path.join(model_folder,'config')) as f:
        # JSON standard is double quotes, but some config files use single quotes.
        config_str = f.read().replace("'",'"')
        config = json.loads(config_str)
    # Clean it up if necessary
    to_del = ['name','optim','lr','mom','scheduler','warmup_step','decay_rate',
            'lr_min','clip','max_epoch','batch_size','seed','cuda','debug',
            'patience','champs_loss','multi_gpu','fp16','max_bond_count',
            'log_interval','batch_chunk','work_dir','restart','restart_dir',
            'load','mode','eta_min','gpu0_bsz','n_all_param','max_step','d_embed',
            'cutout']
    for new,old in {'dim':'d_model', 'n_layers':'n_layer', 'fdim':'feature_dim',
            'dist_embedding':'dist_embed_type', 'atom_angle_embedding':'angle_embed_type',
            'trip_angle_embedding':'quad_angle_embed_type',
            'quad_angle_embedding':'quad_angle_embed_type',
            }.items():
        if old in config and new not in config:
            config[new] = config[old]
            to_del.append(old)
    for old in to_del:
        if old in config:
            del config[old]
    # It would be nice to read the atom types from loaders, but it takes too long.
    config.update({k:models[k] for k in models if k.startswith('num') and k.endswith('types')})
    model = graph_transformer.GraphTransformer(**config)
    to_load_st = torch.load(os.path.join(model_folder,'model.ckpt')).state_dict()
    model.load_state_dict(to_load_st)
    sys.path.remove(model_folder)
    return model


def single_model_predict(loader, model, modelname):
    MAX_BOND_COUNT = 406
    out_str = "id,scalar_coupling_constant\n"
    dev = "cuda"
    #dev = "cpu"
    model = model.to(dev)
    model.eval()
    with torch.no_grad():
        for arr in tqdm(loader):
            x_idx, x_atom, x_atom_pos, x_bond, x_bond_dist, x_triplet, x_triplet_angle, x_quad, x_quad_angle, y = arr
            x_atom, x_atom_pos, x_bond, x_bond_dist, x_triplet, x_triplet_angle, x_quad, x_quad_angle, y = \
                x_atom.to(dev), x_atom_pos.to(dev), x_bond.to(dev), x_bond_dist.to(dev), \
                x_triplet.to(dev), x_triplet_angle.to(dev), x_quad.to(dev), x_quad_angle.to(dev), y.to(dev)

            x_bond, x_bond_dist, y = x_bond[:, :MAX_BOND_COUNT], x_bond_dist[:, :MAX_BOND_COUNT], y[:,:MAX_BOND_COUNT]
            y_pred, _ = model(x_atom, x_atom_pos, x_bond, x_bond_dist, x_triplet, x_triplet_angle, x_quad, x_quad_angle)
            y_pred_pad = torch.cat([torch.zeros(y_pred.shape[0], 1, y_pred.shape[2], device=y_pred.device), y_pred], dim=1)
            y_pred_scaled = y_pred_pad.gather(1,x_bond[:,:,1][:,None,:])[:,0,:] * y[:,:,2] + y[:,:,1]

            y_selected = y_pred_scaled.masked_select((x_bond[:,:,0] > 0) & (y[:,:,3] > 0)).cpu().numpy()
            ids_selected = y[:,:,0].masked_select((x_bond[:,:,0] > 0) & (y[:,:,3] > 0))

            if dev=='cuda':
                ids_selected = ids_selected.cpu()
            ids_selected = ids_selected.numpy()

            for id_, pred in zip(ids_selected, y_selected):
                out_str += "{0:d},{1:f}\n".format(int(id_), pred)
    with open(os.path.join(root,settings['SUBMISSION_DIR'],modelname+'.csv.bz2'), "wb") as f:
        f.write(bz2.compress(out_str.encode('utf-8')))
    return


def load_submission(modelname):
    data = pd.read_csv(os.path.join(root,settings['SUBMISSION_DIR'],modelname+'.csv.bz2'))
    sort_idx = np.argsort(data['id'])
    out = np.vstack((data['id'],data['scalar_coupling_constant']))[:,sort_idx].T
    return out


def select_models(n_model,chosen_models,hardcoded=True):
    if not hardcoded:
        model_mask = np.zeros((n_model,8), dtype=np.bool)
        for i in range(8):
            best_models = np.flip( np.argsort( chosen_models[:,i] ) )[0:model_count_for_median[i]]
            for j in best_models:
                model_mask[j,i] = True
    else:
        #assert models['names'] == ["gt15_3047fine3069", "gt16_3044fine3068", "gt18_3020", "gtA_174932_3015fine3042", "gtB_124323_2997", "gtC_091310_3010fine3025", "gtD_092424_3018fine3038", "gtE_114919_2823", "gtF_115725_2940", "gtG_120830_3001", "gtH_125215_3020", "gt15_J_3049", "gt14_K_3074"]
        model_mask = [[ True,  True,  True,  True,  True,  True,  True,  True],  # gt15_3047fine3069
                      [ True,  True,  True,  True,  True,  True,  True,  True],  # gt16_3044fine3068
                      [ True, False,  True, False, False,  True,  True, False],  # gt18_3020
                      [ True,  True,  True,  True,  True,  True,  True,  True],  # gtA_174932_3015fine3042
                      [False,  True, False,  True,  True, False,  True,  True],  # gtB_124323_2997
                      [ True,  True,  True,  True,  True, False, False,  True],  # gtC_091310_3010fine3025
                      [ True,  True,  True,  True,  True,  True,  True,  True],  # gtD_092424_3018fine3038
                      [False, False, False, False,  True, False, False, False],  # gtE_114919_2823
                      [False, False, False,  True,  True, False, False,  True],  # gtF_115725_2940
                      [False, False, False, False, False,  True, False, False],  # gtG_120830_3001
                      [ True,  True,  True, False, False,  True,  True, False],  # gtH_125215_3020
                      [ True,  True,  True,  True, False,  True,  True,  True],  # gt15_J_3049
                      [ True,  True,  True,  True,  True,  True,  True,  True]]  # gt14_K_3074
        model_mask = np.array(model_mask, dtype=np.bool)
    return model_mask


def ensemble(modelnames):
    xs = [load_submission(name) for name in modelnames]
    n_model = len(xs)
    idx = xs[0][:,0]
    x_all = np.vstack(( [xs[i][:,1] for i in range(n_model)] ))

    with open(os.path.join(root,settings['RAW_DATA_DIR'],'test.csv'),'r') as f:
        lines = f.read().strip().split('\n')[1:]
        types = ['1JHC','1JHN','2JHC','2JHH','2JHN','3JHC','3JHH','3JHN']
        line_types = np.array([types.index(line.split(',')[4]) for line in lines])
        line_type_indices = [(line_types == i) for i in range(8)]

    median_index = (n_model-1)/2
    even_models = False
    if n_model % 2 == 0:
        even_models = True
        print('WARNING: even number of models supplied')
    median_index = int(median_index)
    indices = np.argsort(x_all, axis=0)
    chosen_models = np.zeros((n_model,8), dtype=np.int)
    for i in range(n_model):
        for j in range(8):
            chosen_models[i,j] = int((indices[median_index, line_type_indices[j]] == i).sum())
            if even_models:
                chosen_models[i,j] += int((indices[median_index+1, line_type_indices[j]] == i).sum())
    if even_models:
        chosen_models = chosen_models / 2.0
    print('Count of a model & type being chosen in a raw median procedure:')
    for i in range(n_model):
        print(str(chosen_models[i,:])+' '+modelnames[i])
    print('')

    model_mask = select_models(n_model,chosen_models)
    print('Model mask based on model count for median (now hardcoded):')
    for i in range(n_model):
        print(str(model_mask[i,:])+' '+modelnames[i])
    print('')

    x_out = np.zeros(x_all.shape[1])
    answer_count = 0
    for i in range(8):
        sorted_x_part = np.sort(x_all[ model_mask[:,i],: ][ :,line_type_indices[i] ], axis=0)
        model_count = model_mask[:,i].sum()
        start_idx = int((model_count - median_mean_counts[i])/2)
        end_idx = start_idx + median_mean_counts[i]
        mean_x_median = np.mean(sorted_x_part[ start_idx:end_idx,: ], axis=0)
        x_out[ line_type_indices[i] ] = mean_x_median
        print('Type '+str(i)+': '+str(model_count)+' models average '+str(median_mean_counts[i])+' median')
        print('Output shape after mean: ',str(mean_x_median.shape))
        print('Sorted indices for median mean: ',str([j for j in range(start_idx,end_idx)]))
        answer_count += mean_x_median.shape[0]
    assert answer_count == 2505542
    return idx,x_out


def write_final(idx,x_out):
    with open(os.path.join(root,settings['SUBMISSION_DIR'],models['output_file']),'w') as f:
        f.write('id,scalar_coupling_constant\n')
        for i in range(idx.shape[0]):
            f.write(str(int(idx[i]))+','+str(x_out[i])+'\n')


if __name__=='__main__':
    batch_size = 64
    if 'fast' not in sys.argv:
        print("Loading submission loaders...")
        with gzip.open(os.path.join(root,settings['PROCESSED_DATA_DIR'],'torch_proc_submission.pkl.gz'),'rb') as f:
            sub_dataset = TensorDataset(*pickle.load(f))
        loader = DataLoader(sub_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        for modelname in models['names']:
            model = load_model(modelname)
            if model is None:
                continue
            print('Predicting {}...'.format(modelname))
            single_model_predict(loader, model, modelname)
            # Free up memory
            del model
    idx, x_out = ensemble(models['names'])
    write_final(idx,x_out)
