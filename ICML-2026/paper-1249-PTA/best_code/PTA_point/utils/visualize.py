import os
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
from collections import OrderedDict

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import openshape
from utils.utils import load_config
from utils.utils import build_test_data_loader
from utils.utils import get_arguments


# 1. model
@torch.no_grad()
def get_model(args):
    config = load_config("models/openshape/config.yaml", cli_args = vars(args))
    lm3d_model = openshape.create_openshape(config)
    lm3d_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(lm3d_model)

    checkpoint = torch.load(f"weights/openshape/openshape-pointbert-vitg14-rgb/model.pt")
    model_dict = OrderedDict()

    pattern = re.compile('module.')
    for k,v in checkpoint['state_dict'].items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v
    lm3d_model.load_state_dict(model_dict)

    print('len(model_dict):', len(model_dict))
    
    lm3d_model.half().to(args.device)
    lm3d_model.eval()
    
    return lm3d_model

# 2. dataset
def get_dataset(args):
    print('='*20)
    print(args)
    print('='*20)
    dataset_name = args.dataset
    test_loader, _, _ = build_test_data_loader(args, dataset_name, args.data_root, None)
    print('len(test_loader):', len(test_loader))
    
    return test_loader

# 3. forward pass
@torch.no_grad()
def get_feats_labels(args, lm3d_model, test_loader):
    cache_type = args.cache_type    # hierarchical

    global_feats, all_local_feats, local_feats = [], [], []
    global_labels, all_local_labels, local_labels = [], [], []
    for i, (pc, target, _, rgb) in enumerate(test_loader):
        if i == 5:
            break
        
        # pc: (1, n, 3)     rgb: (1, n, 3)
        pc, rgb = pc.half().cuda(), rgb.half().cuda()
        feature = torch.cat([pc, rgb], dim=-1).half().cuda()

        # NOTE hierarchical caches
        if cache_type == 'hierarchical': 
            # store these features in `pth` file
            pc_feats, all_patches, patch_centers = lm3d_model(pc, feature)
            pc_feats = pc_feats / pc_feats.norm(dim=-1, keepdim=True)
            all_patches = all_patches / all_patches.norm(dim=-1, keepdim=True)
            patch_centers = patch_centers / patch_centers.norm(dim=-1, keepdim=True)
        else:
            raise ValueError('Now, only `hierarchical` cache can be visualized!')
        
        print('all_patches.size():', all_patches.size())
        
        global_feats.append(pc_feats)
        all_local_feats.append(all_patches)
        local_feats.append(patch_centers)
        global_labels.append(target)
        all_local_labels.append(target.repeat(all_patches.shape[1]))
        local_labels.append(target.repeat(patch_centers.shape[0]))
        
    global_feats = torch.cat(global_feats, dim=0)
    all_local_feats = torch.cat(all_local_feats, dim=0)
    local_feats = torch.cat(local_feats, dim=0)
    global_labels = torch.cat(global_labels, dim=0)
    all_local_labels = torch.cat(all_local_labels, dim=0)
    local_labels = torch.cat(local_labels, dim=0)
    
    print('global_feats.shape:', global_feats.shape)
    print('all_local_feats.shape:', all_local_feats.shape)
    print('local_feats.shape:', local_feats.shape)
    print('global_labels.shape:', global_labels.shape)
    print('all_local_labels.shape:', all_local_labels.shape)
    print('local_labels.shape:', local_labels.shape)

    feats_labels_dict = {
        'global_feats': global_feats,
        'all_local_feats': all_local_feats,
        'local_feats': local_feats,
        'global_labels': global_labels,
        'all_local_labels': all_local_labels,
        'local_labels': local_labels
    }
    torch.save(feats_labels_dict, 'feats_labels.pth')
    

# 4. visualization
def make_plot(obj_id):
    state_dict = torch.load('feats_labels.pth', map_location=torch.device('cpu'))
    
    global_feats = state_dict['global_feats'][obj_id:(obj_id+1)]
    global_labels = state_dict['global_labels'][obj_id:(obj_id+1)]
    all_local_feats = state_dict['all_local_feats'][obj_id:(obj_id+1)].squeeze()
    all_local_labels = state_dict['all_local_labels'][obj_id:(obj_id+1)].squeeze()
    local_feats = state_dict['local_feats'][obj_id*5:(obj_id+1)*5]
    local_labels = state_dict['local_labels'][obj_id*5:(obj_id+1)*5]
    
    print('global_feats.shape:', global_feats.shape)
    print('global_labels.shape', global_labels.shape)
    print('all_local_feats.shape:', all_local_feats.shape)
    print('all_local_labels.shape:', all_local_labels.shape)
    print('local_feats.shape:', local_feats.shape)
    print('local_labels.shape:', local_labels.shape)
    
    all_local_and_clustered_feats = torch.cat([all_local_feats, local_feats], dim=0)
 
    # Step 1: Initialize t-SNE model
    tsne = TSNE(n_components=2, learning_rate='auto', random_state=42, init='random')

    # Step 2: Apply t-SNE to reduce dimensions
    combined_data_2d = tsne.fit_transform(all_local_and_clustered_feats)
    
    features_2d = combined_data_2d[:384]      # First 384 rows are the original features
    centers_2d = combined_data_2d[384:]       # Last 5 rows are the cluster centers
    
    plt.figure(figsize=(5, 5))

    # Plot the original features
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c='blue', label='Part Features', alpha=0.6)

    # Plot the cluster centers
    plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='X', s=200, label='Cluster Centers')

    plt.title("t-SNE Visualization of Parts and Cluster Centers")
    plt.xlabel("")
    plt.ylabel("")
    plt.legend()
    plt.grid(True)

    plt.savefig(f"notebook/images/mn40_obj_{obj_id}.pdf", format="pdf", dpi=400.0)


if '__main__' == __name__:
    args = get_arguments()
    lm3d_model = get_model(args)
    test_loader = get_dataset(args)
    get_feats_labels(args, lm3d_model, test_loader)
    obj_id = args.obj_id
    make_plot(obj_id)