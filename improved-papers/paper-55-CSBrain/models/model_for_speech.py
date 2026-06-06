import torch
import torch.nn as nn
from functools import partial
from .CSBrain import *


class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()

        # Brain region encoding: Frontal (0) | Parietal (1) | Temporal (2) | Occipital (3) | Central (4)
        brain_regions = [
            0,  # Fp1
            0,  # Fp2
            0,  # F7
            0,  # F3
            0,  # Fz
            0,  # F4
            0,  # F8
            0,  # FC5
            0,  # FC1
            0,  # FC2
            0,  # FC6
            2,  # T7
            4,  # C3
            4,  # Cz
            4,  # C4
            2,  # T8
            2,  # TP9
            4,  # CP5
            4,  # CP1
            4,  # CP2
            4,  # CP6
            2,  # TP10
            1,  # P7
            1,  # P3
            1,  # Pz
            1,  # P4
            1,  # P8
            3,  # PO9
            3,  # O1
            3,  # Oz
            3,  # O2
            3,  # PO10
            0,  # AF7
            0,  # AF3
            0,  # AF4
            0,  # AF8
            0,  # F5
            0,  # F1
            0,  # F2
            0,  # F6
            0,  # FT9
            0,  # FT7
            0,  # FC3
            0,  # FC4
            0,  # FT8
            0,  # FT10
            4,  # C5
            4,  # C1
            4,  # C2
            4,  # C6
            2,  # TP7
            4,  # CP3
            4,  # CPz
            4,  # CP4
            2,  # TP8
            1,  # P5
            1,  # P1
            1,  # P2
            1,  # P6
            3,  # PO7
            3,  # PO3
            3,  # POz
            3,  # PO4
            3   # PO8
        ]

        electrode_labels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4',
                            'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz',
                            'O2', 'PO10', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FT9', 'FT7', 'FC3', 'FC4', 'FT8',
                            'FT10', 'C5', 'C1', 'C2', 'C6', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'P5', 'P1', 'P2', 'P6', 'PO7',
                            'PO3', 'POz', 'PO4', 'PO8']

        topology = {
            0: ["FT9", "FT7", "Fp1", "AF7", "AF3", "F7", "F5", "F3", "FC5", "FC3", "FC1", 'F1',
                "Fz",
                'F2', "FC2", "FC4", "FC6", "F4", "F6", "F8", "AF4", "AF8", "Fp2", "FT8", "FT10"],
            4: ["C5", "C3", "C1", "CP5", "CP3", "CP1",
                "Cz", "CPz",
                "CP2", "CP4", "CP6", "C2", "C4", "C6"],
            1: ["P7", "P5", "P3", "P1",
                "Pz",
                "P2", "P4", "P6", "P8"],
            2: ["TP9", "TP7", "T7",
                "T8",
                "TP8", "TP10"],
            3: ["PO9", "PO7", "PO3",
                "O1", "Oz", "POz",
                "O2", "PO4", "PO8", "PO10"]
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
            nn.Linear(64 * 3 * 200, 3 * 200),
            nn.ELU(),
            nn.Linear(3 * 200, 200),
            nn.ELU(),
            nn.Linear(200, param.num_of_classes)
        )

    def forward(self, x):
        bz, ch_num, seq_len, patch_size = x.shape
        feats = self.backbone(x)
        feats = feats.contiguous().view(bz, ch_num * seq_len * 200)
        out = self.classifier(feats)
        return out