import torch
import torch.nn as nn
from functools import partial
from .CSBrain import *

class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()

        # Brain region encoding: Frontal (0) | Parietal (1) | Temporal (2) | Occipital (3) | Central (4)
        brain_regions = [
            2, # 'FT7'
            2, # 'FT8'
            2, # 'T7'
            2, # 'T8'
            2, # 'TP7'
            2, # 'TP8'
            4, # 'CP1'
            4, # 'CP2'
            1, # 'P1'
            1, # 'PZ'
            1, # 'P2'
            1, # 'PO3'
            1, # 'POZ'
            1, # 'PO4'
            3, # 'O1'
            3, # 'OZ'
            3  # 'O2'
        ]

        electrode_labels = [
            'FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8',
            'CP1', 'CP2', 'P1', 'PZ', 'P2', 'PO3',
            'POZ', 'PO4', 'O1', 'OZ', 'O2'
        ]

        topology = {
            0: [],
            1: ['P1', 'PZ', 'P2', 'PO3', 'POZ', 'PO4'],
            2: ['FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8'],
            3: ['O1', 'OZ', 'O2'],
            4: ['CP1', 'CP2']
        }

        # Group electrode indices by brain region
        region_groups = {}
        for i, region in enumerate(brain_regions):
            if region not in region_groups:
                region_groups[region] = []
            region_groups[region].append((i, electrode_labels[i]))

        # Sort based on topology
        sorted_indices = []
        for region in sorted(region_groups.keys()):
            region_electrodes = region_groups[region]
            sorted_electrodes = sorted(region_electrodes, key=lambda x: topology[region].index(x[1]))
            sorted_indices.extend([e[0] for e in sorted_electrodes])

        print("Sorted Indices:", sorted_indices)

        if param.model == 'CSBrain':
            self.backbone = CSBrain(
                in_dim=200, out_dim=200, d_model=200,
                dim_feedforward=800, seq_len=30,
                n_layer=param.n_layer, nhead=8,
                brain_regions=brain_regions,
                sorted_indices=sorted_indices
            )
        else:
            return 0

        if param.use_pretrained_weights:
            map_location = torch.device(f'cuda:{param.cuda}')
            state_dict = torch.load(param.foundation_dir, map_location=map_location)
            # Remove "module." prefix
            new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
            
            model_state_dict = self.backbone.state_dict()

            # Filter matching weights by shape
            matching_dict = {k: v for k, v in new_state_dict.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}

            model_state_dict.update(matching_dict)
            self.backbone.load_state_dict(model_state_dict)

        self.backbone.proj_out = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(17 * 8 * 200, 8 * 200),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(8 * 200, 200),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(200, 1)
        )

    def forward(self, x):
        bz, ch_num, seq_len, patch_size = x.shape
        feats = self.backbone(x)
        feats = feats.contiguous().view(bz, ch_num * seq_len * patch_size)
        out = self.classifier(feats)
        out = out[:, 0]
        return out