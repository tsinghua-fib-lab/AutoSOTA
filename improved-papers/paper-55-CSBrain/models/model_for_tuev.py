import torch
import torch.nn as nn
from .CSBrain import *

class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()
        
        # Brain region encoding: Frontal (0) | Parietal (1) | Temporal (2) | Occipital (3) | Central (4)
        TUEV_brain_regions = [
            0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 4, 1, 0, 0, 4, 1
        ]

        # Signal electrodes (reference removed)
        TUEV_signal_electrodes = [
            "FP1", "F7", "T3", "T5", 
            "FP2", "F8", "T4", "T6",
            "FP1", "F3", "C3", "P3", 
            "FP2", "F4", "C4", "P4"
        ]

        # Topological sorting within brain regions
        TUEV_topology = {
            0: ["FP1", "F3", "F7", "FZ", "F4", "F8", "FP2"],  # Frontal
            4: ["C3", "CZ", "C4"],  # Central
            1: ["P3", "PZ", "P4"],  # Parietal
            2: ["T3", "T5", "T6", "T4"],  # Temporal
            3: ["O1", "O2"],  # Occipital
        }

        # Group signal electrodes by brain region
        TUEV_region_groups = {}
        for i, region in enumerate(TUEV_brain_regions):
            if region not in TUEV_region_groups:
                TUEV_region_groups[region] = []
            TUEV_region_groups[region].append((i, TUEV_signal_electrodes[i]))

        # Sort by topological relationship
        TUEV_sorted_indices = []
        for region in sorted(TUEV_region_groups.keys()):
            region_electrodes = TUEV_region_groups[region]
            sorted_electrodes = sorted(region_electrodes, key=lambda x: TUEV_topology[region].index(x[1]))
            TUEV_sorted_indices.extend([e[0] for e in sorted_electrodes])

        print("Sorted Indices:", TUEV_sorted_indices)

        if param.model == 'CSBrain':
            self.backbone = CSBrain(
                in_dim=200, out_dim=200, d_model=200,
                dim_feedforward=800, seq_len=30,
                n_layer=param.n_layer, nhead=8,
                brain_regions=TUEV_brain_regions,
                sorted_indices=TUEV_sorted_indices
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
            nn.Linear(16 * 5 * 200, 5 * 200),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(5 * 200, 200),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(200, param.num_of_classes)
        )

        if param.use_finetune_weights:
            self._load_weights(param.foundation_dir, param)

    def forward(self, x):
        bz, ch_num, seq_len, patch_size = x.shape
        feats = self.backbone(x)
        feats = feats.contiguous().view(bz, ch_num * seq_len * patch_size)
        out = self.classifier(feats)
        return out
    
    def _load_weights(self, path, param):
        map_location = torch.device(f'cuda:{param.cuda}')
        print(f"Loading pretrained weights from: {path}")
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict, strict=True)