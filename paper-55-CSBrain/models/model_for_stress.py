import torch
import torch.nn as nn
from functools import partial
from .CSBrain import *


class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()

        # Brain region encoding: Frontal (0) | Parietal (1) | Temporal (2) | Occipital (3) | Central (4)
        brain_regions = [
            0, 0, 0, 0, 0, 0,  # Fp1, Fp2, F3, F4, F7, F8
            2, 2,             # T3, T4
            4, 4,             # C3, C4
            2, 2,             # T5, T6
            1, 1,             # P3, P4
            3, 3,             # O1, O2
            0, 4, 1           # Fz, Cz, Pz
        ]

        electrode_labels = [
            "FP1", "FP2", "F3", "F4", "F7", "F8",
            "T3", "T4", "C3", "C4", "T5", "T6",
            "P3", "P4", "O1", "O2", "FZ", "CZ", "PZ"
        ]

        # Topological sorting within brain regions (following International 10-20 system)
        topology = {
            0: ["FP1", "F7", "F3", "FZ", "F4", "F8", "FP2"],  # Frontal (Anterior-Posterior, Midline-Lateral)
            4: ["C3", "CZ", "C4"],                            # Central (Left-Midline-Right)
            1: ["P3", "PZ", "P4"],                            # Parietal (Left-Midline-Right)
            2: ["T5", "T3", "T4", "T6"],                      # Temporal (Anterior-Posterior, Left-Right)
            3: ["O1", "O2"]                                   # Occipital (Left-Right)
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
        self.classifier = nn.Sequential(
            nn.Linear(19 * 5 * 200, 5 * 200),
            nn.GELU(),
            nn.Dropout(param.dropout),
            nn.Linear(5 * 200, 200),
            nn.GELU(),
            nn.Dropout(param.dropout),
            nn.Linear(200, 1)
        )

    def forward(self, x):
        bz, ch_num, seq_len, patch_size = x.shape
        ch_num = ch_num - 1 # Exclude the A1 electrode if present
        feats = self.backbone(x[:, :-1, :, :]) # Pass all channels except the last one
        feats = feats.contiguous().view(bz, ch_num * seq_len * patch_size)
        out = self.classifier(feats)
        out = out[:, 0]
        return out