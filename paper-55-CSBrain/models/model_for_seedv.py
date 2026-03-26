import torch
import torch.nn as nn
from functools import partial
from .CSBrain import *

class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()
        # Brain region encoding: Frontal (0) | Parietal (1) | Temporal (2) | Occipital (3) | Central (4)
        SEED5_brain_regions = [
            0, # 'FP1'
            0, # 'FPZ'
            0, # 'FP2'
            0, # 'AF3'
            0, # 'AF4'
            0, # 'F7'
            0, # 'F5'
            0, # 'F3'
            0, # 'F1'
            0, # 'FZ'
            0, # 'F2'
            0, # 'F4'
            0, # 'F6'
            0, # 'F8'
            2, # 'FT7'
            4, # 'FC5'
            4, # 'FC3'
            4, # 'FC1'
            4, # 'FCZ'
            4, # 'FC2'
            4, # 'FC4'
            4, # 'FC6'
            2, # 'FT8'
            2, # 'T7'
            4, # 'C5'
            4, # 'C3'
            4, # 'C1'
            4, # 'CZ'
            4, # 'C2'
            4, # 'C4'
            4, # 'C6'
            2, # 'T8'
            2, # 'TP7'
            4, # 'CP5'
            4, # 'CP3'
            4, # 'CP1'
            4, # 'CPZ'
            4, # 'CP2'
            4, # 'CP4'
            4, # 'CP6'
            2, # 'TP8'
            1, # 'P7'
            1, # 'P5'
            1, # 'P3'
            1, # 'P1'
            1, # 'PZ'
            1, # 'P2'
            1, # 'P4'
            1, # 'P6'
            1, # 'P8'
            1, # 'PO7'
            1, # 'PO5'
            1, # 'PO3'
            1, # 'POZ'
            1, # 'PO4'
            1, # 'PO6'
            1, # 'PO8'
            3, # 'CB1'
            3, # 'O1'
            3, # 'OZ'
            3, # 'O2'
            3, # 'CB2'
        ]

        SEED5_electrode_labels = [
            "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
            "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
            "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
            "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2"
        ]

        # Define local topological relationships within brain regions
        SEED5_topology = {
            0: ["FP1", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "FP2", "FPZ", "AF3", "AF4"],
            4: ["FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "C5", "C3", "C1", "CZ", "C2", "C4", "C6",
                "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6"],
            1: ["P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8"],
            2: ["FT7", "FT8", "T7", "T8", "TP7", "TP8"],
            3: ["CB1", "O1", "OZ", "O2", "CB2"],
        }

        # Group electrode indices by brain region
        SEED5_region_groups = {}
        for i, region in enumerate(SEED5_brain_regions):
            if region not in SEED5_region_groups:
                SEED5_region_groups[region] = []
            SEED5_region_groups[region].append((i, SEED5_electrode_labels[i]))

        # Sort based on topological relationships
        SEED5_sorted_indices = []
        for region in sorted(SEED5_region_groups.keys()):
            region_electrodes = SEED5_region_groups[region]
            sorted_electrodes = sorted(region_electrodes, key=lambda x: SEED5_topology[region].index(x[1]))
            SEED5_sorted_indices.extend([e[0] for e in sorted_electrodes])

        print("Sorted Indices:", SEED5_sorted_indices)

        if param.model == 'CSBrain':
            self.backbone = CSBrain(
                in_dim=200, out_dim=200, d_model=200,
                dim_feedforward=800, seq_len=30,
                n_layer=12, nhead=8,
                brain_regions=SEED5_brain_regions,
                sorted_indices=SEED5_sorted_indices
            )
        else:
            return 0
        
        if param.use_pretrained_weights:
            map_location = torch.device(f'cuda:{param.cuda}')
            state_dict = torch.load(param.foundation_dir, map_location=map_location)
            # Remove "module." and "backbone." prefixes
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace("module.", "").replace("backbone.", "")
                new_state_dict[new_key] = value
            
            model_state_dict = self.backbone.state_dict()

            # Filter matching weights by shape
            matching_dict = {k: v for k, v in new_state_dict.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}

            model_state_dict.update(matching_dict)
            self.backbone.load_state_dict(model_state_dict)

        self.backbone.proj_out = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(62 * 1 * 200, 4 * 200),
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
        feats = feats.contiguous().view(bz, ch_num * seq_len * patch_size)
        out = self.classifier(feats)
        return out