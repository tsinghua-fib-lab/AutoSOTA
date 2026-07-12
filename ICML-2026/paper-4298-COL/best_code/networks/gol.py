import torch
import torch.nn as nn

from networks.base import BaseModel


class ConOrd(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.backbone == 'vitB16':
            # hdim = 768
            hdim = 512
        else:
            raise ValueError('no backbone was found.')
        
        if cfg.ref_mode == 'flex':
            if cfg.ref_point_num == 2:
                self.ref_points = torch.randn([hdim])
                self.ref_points = torch.stack([self.ref_points, -self.ref_points])
                self.ref_points = nn.functional.normalize(self.ref_points, dim=-1)

            elif cfg.ref_point_num == 3:
                self.ref_points = nn.functional.normalize(torch.randn([hdim]), dim=0)
                noise = (1e-3) * torch.randn(hdim)
                max_point = nn.functional.normalize(-self.ref_points + noise, dim=0)
                mid_point = nn.functional.normalize(self.ref_points + max_point, dim=0)
                self.ref_points = torch.stack([self.ref_points, mid_point, -self.ref_points])

            elif cfg.ref_point_num == 5:
                self.ref_points = nn.functional.normalize(torch.randn([hdim]), dim=0)
                noise = (1e-4) * torch.randn(hdim)
                max_point = nn.functional.normalize(-self.ref_points + noise, dim=0)
                r2_point = nn.functional.normalize(self.ref_points + max_point, dim=0)
                r1_point = nn.functional.normalize(self.ref_points + r2_point, dim=0)
                r3_point = nn.functional.normalize(r2_point - self.ref_points, dim=0)
                self.ref_points = torch.stack([self.ref_points, r1_point, r2_point, r3_point, -self.ref_points])
                print(torch.sum(self.ref_points[0] * self.ref_points[1]))
                print(torch.sum(self.ref_points[1] * self.ref_points[2]))
                print(torch.sum(self.ref_points[2] * self.ref_points[3]))
                print(torch.sum(self.ref_points[3] * self.ref_points[4]))

            else:
                self.ref_points = torch.randn([cfg.ref_point_num, hdim])
                if cfg.start_norm:
                    self.ref_points = nn.functional.normalize(self.ref_points, dim=-1)

            self.ref_points = nn.parameter.Parameter(self.ref_points)

        else:
            pass

    def _forward(self, base_embs, ref_embs=None):
        return base_embs
