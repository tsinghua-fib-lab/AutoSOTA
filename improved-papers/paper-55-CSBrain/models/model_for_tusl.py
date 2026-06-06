import torch
import torch.nn as nn

from .CSBrain import *


class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()
        # Brain region encoding: Frontal (0) | Parietal (1) | Temporal (2) | Occipital (3) | Central (4)
        brain_regions = [
            0,  # FP1-REF
            0,  # FP2-REF
            0,  # F3-REF
            0,  # F4-REF
            4,  # C3-REF
            4,  # C4-REF
            1,  # P3-REF
            1,  # P4-REF
            3,  # O1-REF
            3,  # O2-REF
            0,  # F7-REF
            0,  # F8-REF
            2,  # T3-REF
            2,  # T4-REF
            2,  # T5-REF
            2,  # T6-REF
            0,  # FZ-REF
            4,  # CZ-REF
            1   # PZ-REF
        ]
        
        electrode_labels = [
            "FP1-REF", "FP2-REF", "F3-REF", "F4-REF",
            "C3-REF", "C4-REF",
            "P3-REF", "P4-REF",
            "O1-REF", "O2-REF",
            "F7-REF", "F8-REF",
            "T3-REF", "T4-REF", "T5-REF", "T6-REF",
            "FZ-REF", "CZ-REF", "PZ-REF"
        ]
        
        # Define local topological relationships within brain regions
        topology = {
            0: ["FP1-REF", "F7-REF", "F3-REF", "FZ-REF", "F4-REF", "F8-REF", "FP2-REF"], 
            4: ["C3-REF", "CZ-REF", "C4-REF"],
            1: ["P3-REF", "PZ-REF", "P4-REF"],
            3: ["O1-REF", "O2-REF"],
            2: ["T3-REF", "T5-REF", "T6-REF", "T4-REF"],
            -1: ["A1-REF"] # Reference electrode placed separately
        }

        # Group signal electrodes by brain region
        TUEV_region_groups = {}
        for i, region in enumerate(brain_regions):
            if region not in TUEV_region_groups:
                TUEV_region_groups[region] = []
            TUEV_region_groups[region].append((i, electrode_labels[i]))

        # Sort by topological relationship
        sorted_indices = []
        for region in sorted(TUEV_region_groups.keys()):
            region_electrodes = TUEV_region_groups[region]
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

        self.backbone.proj_out = nn.Sequential()
        self.classifier = nn.Sequential(
            nn.Linear(19 * 10 * 200, 10 * 200),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(10 * 200, 200),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(200, param.num_of_classes)
        )

    def forward(self, x):
        bz, ch_num, seq_len, patch_size = x.shape
        # Select specific channels: 0 to 15, and 18 to 20. This excludes channels 16 and 17.
        x = torch.cat((x[:, 0:16, :, :], x[:, 18:21, :, :]), dim=1)
        ch_num = ch_num - 4 # Adjust channel count based on exclusion of 4 channels (T1-REF, T2-REF, and two more based on the previous context provided in the code)
        feats = self.backbone(x)
        feats = feats.contiguous().view(bz, ch_num * seq_len * patch_size)
        out = self.classifier(feats)
        return out