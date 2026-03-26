import torch
import torch.nn as nn
from functools import partial
from .CSBrain import *


class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()

        selected_channels = [
            'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 
            'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6',
            'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
            'FP1', 'FPZ', 'FP2', 'AF7', 'AF3', 'AFZ', 'AF4', 'AF8',
            'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
            'FT7', 'FT8',
            'T7', 'T8', 'T9', 'T10', 'TP7', 'TP8',
            'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
            'PO7', 'PO3', 'POZ', 'PO4', 'PO8',
            'O1', 'OZ', 'O2', 'IZ'
        ]

        # Brain region encoding: Frontal (0) | Parietal (1) | Temporal (2) | Occipital (3) | Central (4)
        brain_regions = [
            0, 0, 0, 0, 0, 0, 0,  # FC series
            4, 4, 4, 4, 4, 4, 4,  # C series
            4, 4, 4, 4, 4, 4, 4,  # CP series
            0, 0, 0, 0, 0, 0, 0, 0,  # FP and AF series
            0, 0, 0, 0, 0, 0, 0, 0, 0,  # F series
            2, 2,  # FT series
            2, 2, 2, 2, 2, 2,  # T and TP series
            1, 1, 1, 1, 1, 1, 1, 1, 1,  # P series
            3, 3, 3, 3, 3,  # PO series
            3, 3, 3, 3  # O and IZ series
        ]

        # Topological structure
        topology = {
            0: ['AF7', 'AF3', 'AFZ', 'AF4', 'AF8',
                'FP1', 'FPZ', 'FP2',
                'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
                'FT7', 'FT8',
                'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6'],
            4: ['C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6',
                'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6'],
            2: ['T7', 'T8', 'T9', 'T10', 'TP7', 'TP8'],
            1: ['P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8'],
            3: ['PO7', 'PO3', 'POZ', 'PO4', 'PO8',
                'O1', 'OZ', 'O2', 'IZ']
        }

        # Group electrode indices by brain region
        region_groups = {}
        for i, region in enumerate(brain_regions):
            if region not in region_groups:
                region_groups[region] = []
            region_groups[region].append((i, selected_channels[i]))

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
            nn.Linear(64 * 4 * 200, 4 * 200),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(4 * 200, 200),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(200, param.num_of_classes)
        )

    def forward(self, x):
        bz, ch_num, seq_len, patch_size = x.shape
        feats = self.backbone(x)
        out = feats.contiguous().view(bz, ch_num * seq_len * 200)
        out = self.classifier(out)
        return out