import torch
import torch.nn as nn
from .CSBrain import *

class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()
        FACED_brain_regions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 2, 2, 4, 4, 4, 4, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3]

        FACED_electrode_labels = [
            "Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5", "FC6",
            "Cz", "C3", "C4",
            "T7", "T8",
            "CP1", "CP2", "CP5", "CP6",
            "Pz", "P3", "P4", "P7", "P8",
            "PO3", "PO4", "Oz", "O1", "O2"
        ]

        FACED_topology = {
            0: ["Fp1", "FC5", "FC1", "F7", "F3", "Fz", "F4", "F8", "FC2", "FC6", "Fp2"],
            4: ["CP5", "CP1", "C3", "Cz", "C4", "CP2", "CP6"],
            1: ["P7", "P3", "Pz", "P4", "P8"],
            2: ["T7", "T8"],
            3: ["PO3", "O1", "Oz", "O2", "PO4"]
        }

        # Group electrode indices by brain region
        FACED_region_groups = {}
        for i, region in enumerate(FACED_brain_regions):
            if region not in FACED_region_groups:
                FACED_region_groups[region] = []
            FACED_region_groups[region].append((i, FACED_electrode_labels[i]))

        # Sort based on topology
        FACED_sorted_indices = []
        for region in sorted(FACED_region_groups.keys()):
            region_electrodes = FACED_region_groups[region]
            sorted_electrodes = sorted(region_electrodes, key=lambda x: FACED_topology[region].index(x[1]))
            FACED_sorted_indices.extend([e[0] for e in sorted_electrodes])

        print("FACED Sorted Indices:", FACED_sorted_indices)

        if param.model == 'CSBrain':
            self.backbone = CSBrain(
                in_dim=200, out_dim=200, d_model=200,
                dim_feedforward=800, seq_len=30,
                n_layer=param.n_layer, nhead=8,
                brain_regions=FACED_brain_regions,
                sorted_indices=FACED_sorted_indices
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
            nn.Linear(30 * 10 * 200, 10 * 200),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(10 * 200, 200),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(200, param.num_of_classes)
        )

    def forward(self, x):
        bz, ch_num, seq_len, patch_size = x.shape
        x = x[:, :30, :, :]
        feats = self.backbone(x)
        out = feats.contiguous().view(bz, (ch_num - 2) * seq_len * 200)
        out = self.classifier(out)
        return out