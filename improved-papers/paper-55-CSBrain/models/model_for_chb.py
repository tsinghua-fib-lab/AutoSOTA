import torch
import torch.nn as nn
from .CSBrain import *

class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()
        chbmit_brain_regions = [
            0,
            0,
            2,
            1,
            0,
            0,
            2,
            1,
            0,
            0,
            4,
            1,
            0,
            0,
            4,
            1
        ]

        chbmit_electrode_labels = [
            "FP1-F7", "F7-T7","T7-P7",
            "P7-O1","FP2-F8","F8-T8",
            "T8-P8","P8-O2","FP1-F3",
            "F3-C3","C3-P3","P3-O1","FP2-F4",
            "F4-C4","C4-P4","P4-O2"
        ]

        # Topological structure: Left - Midline - Right symmetry
        chbmit_topology = {
            0: ["F7-T7", "FP1-F7", "FP1-F3", "F3-C3", "F4-C4", "FP2-F4","FP2-F8" ,"F8-T8"],
            4: ["C3-P3", "C4-P4"],
            1: ["P7-O1", "P3-O1", "P4-O2", "P8-O2"],
            2: ["T7-P7", "T8-P8"],
        }

        # Group electrode indices by brain region
        chbmit_region_groups = {}
        for i, region in enumerate(chbmit_brain_regions):
            if region not in chbmit_region_groups:
                chbmit_region_groups[region] = []
            chbmit_region_groups[region].append((i, chbmit_electrode_labels[i]))

        # Sort by topological relationship
        chbmit_sorted_indices = []
        for region in sorted(chbmit_region_groups.keys()):
            region_electrodes = chbmit_region_groups[region]
            sorted_electrodes = sorted(
                region_electrodes,
                key=lambda x: chbmit_topology[region].index(x[1])
            )
            chbmit_sorted_indices.extend([e[0] for e in sorted_electrodes])

        print("CHB-MIT Sorted Indices:", chbmit_sorted_indices)

        if param.model == 'CSBrain':
            self.backbone = CSBrain(
                in_dim=200, out_dim=200, d_model=200,
                dim_feedforward=800, seq_len=30,
                n_layer=param.n_layer, nhead=8,
                brain_regions=chbmit_brain_regions,
                sorted_indices=chbmit_sorted_indices
            )
        else:
            return 0

        if param.use_pretrained_weights:
            map_location = torch.device(f'cuda:{param.cuda}')
            state_dict = torch.load(param.foundation_dir, map_location=map_location)
            # Remove "module." prefix from keys
            new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
            
            model_state_dict = self.backbone.state_dict()

            # Filter matching weights by shape
            matching_dict = {k: v for k, v in new_state_dict.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}

            # Update and load
            model_state_dict.update(matching_dict)
            self.backbone.load_state_dict(model_state_dict)

        self.backbone.proj_out = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(16 * 10 * 200, 10 * 200),
            nn.ELU(),
            nn.Linear(10 * 200, 200),
            nn.ELU(),
            nn.Linear(200, 1)
        )

    def forward(self, x):
        bz, ch_num, seq_len, patch_size = x.shape
        feats = self.backbone(x)
        feats = feats.contiguous().view(bz, ch_num * seq_len * 200)
        out = self.classifier(feats)
        out = out.contiguous().view(bz)
        return out