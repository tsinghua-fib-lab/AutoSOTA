import torch
import torch.nn as nn
from functools import partial
from .CSBrain import *

class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()

        # Brain region encoding: Frontal (0) | Parietal (1) | Temporal (2) | Occipital (3) | Central (4)
        brain_regions = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # Fp1, Fp2, Fz, F3, F4, F7, F8, FC1, FC2, FC5, FC6
            4, 4, 4,                          # Cz, C3, C4
            2, 2,                             # T3, T4
            4, 4, 4, 4,                       # CP1, CP2, CP5, CP6
            1, 1, 1,                          # Pz, P3, P4
            2, 2,                             # T5, T6
            3, 3, 3, 3, 3                     # PO3, PO4, Oz, O1, O2
        ]

        electrode_labels = ['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 
                            'FC6', 'Cz', 'C3', 'C4', 'T3', 'T4', 
                            'CP1', 'CP2', 'CP5', 'CP6', 'Pz', 'P3', 'P4', 'T5', 'T6', 'PO3', 'PO4', 'Oz', 'O1', 'O2']

        topology = {
            0: ["Fp1", "F7", "F3", "FC5", "FC1", "Fz", "FC2", "FC6", "F4", "F8", "Fp2"],
            
            4: ["C3", "CP5", "CP1", "Cz", "CP2", "CP6", "C4"],
            
            1: ["P3", "Pz", "P4"],
            
            2: ["T3", "T5", "T4", "T6"],
            
            3: ["PO3", "O1", "Oz", "O2", "PO4"]
        }

        # Group signal electrodes by brain region
        SHU_region_groups = {}
        for i, region in enumerate(brain_regions):
            if region not in SHU_region_groups:
                SHU_region_groups[region] = []
            SHU_region_groups[region].append((i, electrode_labels[i]))

        # Sort by topological relationship
        SHU_sorted_indices = []
        for region in sorted(SHU_region_groups.keys()):
            if region == -1: # Skip ignored regions
                continue
            region_electrodes = SHU_region_groups[region]
            sorted_electrodes = sorted(region_electrodes, key=lambda x: topology[region].index(x[1]))
            SHU_sorted_indices.extend([e[0] for e in sorted_electrodes])

        print("Sorted Indices:", SHU_sorted_indices)

        if param.model == 'CSBrain':
            self.backbone = CSBrain(
                in_dim=200, out_dim=200, d_model=200,
                dim_feedforward=800, seq_len=30,
                n_layer=12, nhead=8,
                brain_regions=brain_regions,
                sorted_indices=SHU_sorted_indices
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

        self.backbone.proj_out = nn.Sequential()

        self.classifier = nn.Sequential(
            nn.Linear(30 * 4 * 200, 4 * 200),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(4 * 200, 200),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(200, 1)
        )

    def forward(self, x):
        bz, ch_num, seq_len, patch_size = x.shape
        x = torch.cat((x[:, 0:16, :, :], x[:, 18:, :, :]), dim=1)  # Exclude channels at indices 16 and 17 (A1, A2)
        ch_num = ch_num - 2
        feats = self.backbone(x)
        feats = feats.contiguous().view(bz, ch_num * seq_len * patch_size)
        out = self.classifier(feats)
        out = out[:, 0]
        return out