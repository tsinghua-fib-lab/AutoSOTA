import torch
import torch.nn as nn
from functools import partial
from .CSBrain import *


class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()

        # Brain region encoding: Frontal (0) | Parietal (1) | Temporal (2) | Occipital (3) | Central (4)
        brain_regions = [
            0,  # 'Fp1'
            0,  # 'F3'
            4,  # 'C3'
            1,  # 'P3'
            3,  # 'O1'
            0,  # 'F7'
            2,  # 'T3'
            2,  # 'T5'
            4,  # 'Fc1'
            4,  # 'Fc5'
            4,  # 'Cp1'
            4,  # 'Cp5'
            0,  # 'F9'
            0,  # 'Fz'
            4,  # 'Cz'
            1,  # 'Pz'
            0,  # 'Fp2'
            0,  # 'F4'
            4,  # 'C4'
            1,  # 'P4'
            3,  # 'O2'
            0,  # 'F8'
            2,  # 'T4'
            2,  # 'T6'
            4,  # 'Fc2'
            4,  # 'Fc6'
            4,  # 'Cp2'
            4,  # 'Cp6'
            0,  # 'F10'
        ]

        electrode_labels = [
            'Fp1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T3', 'T5', 'Fc1', 'Fc5',
            'Cp1', 'Cp5', 'F9', 'Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'C4', 'P4',
            'O2', 'F8', 'T4', 'T6', 'Fc2', 'Fc6', 'Cp2', 'Cp6', 'F10'
        ]

        topology = {
            0: ['Fp1', 'F9', 'F7', 'F3', 'Fz', 'F4', 'F8', 'F10', 'Fp2'],
            1: ['P3', 'Pz', 'P4'],
            2: ['T3', 'T5', 'T4', 'T6'],
            3: ['O1', 'O2'],
            4: ['Cp5', 'Fc5', 'Fc1', 'C3', 'Cp1', 'Cz', 'Cp2', 'C4', 'Fc2', 'Fc6', 'Cp6']
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
                n_layer=12, nhead=8,
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

        self.feed_forward = nn.Sequential(
            nn.Linear(29 * 10 * 200, 10 * 200),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(10 * 200, 200),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(200, 1),
        )

    def forward(self, x):
        bz, ch_num, seq_len, patch_size = x.shape
        feats = self.backbone(x)
        feats = feats.contiguous().view(bz, ch_num * seq_len * 200)
        out = self.feed_forward(feats)
        out = out.contiguous().view(bz)
        return out