import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# from model import LayerNorm


class LA(nn.Module):
    def __init__(self, in_dim, bias=True):
        super(LA, self).__init__()
        self.chanel_in = in_dim
        self.chanel = in_dim // 16

        self.temperature = nn.Parameter(torch.ones(1))

        self.qkv = nn.Conv2d(self.chanel_in, self.chanel * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(self.chanel * 3, self.chanel * 3, kernel_size=3, stride=1, padding=1,
                                    groups=self.chanel * 3, bias=bias)
        self.project_out = nn.Conv2d(self.chanel, self.chanel_in, kernel_size=1, bias=bias)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, N, C, height, width = x.size()

        x_input = x.view(m_batchsize, N * C, height, width)
        qkv = self.qkv_dwconv(self.qkv(x_input))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(m_batchsize, N, -1)
        k = k.view(m_batchsize, N, -1)
        v = v.view(m_batchsize, N, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out_1 = (attn @ v)
        out_1 = out_1.view(m_batchsize, -1, height, width)

        out_1 = self.project_out(out_1)
        out_1 = out_1.view(m_batchsize, N, C, height, width)

        out = out_1 + x
        out = out.view(m_batchsize, -1, height, width)
        return out


class DEM(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone  # torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg').cuda().eval()
        self.transform = torchvision.transforms.Normalize(
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
        )
        backbone_dim = 768
        num_layer = 4
        fea_dim = 256
        self.adapter = nn.Sequential(
            LA(backbone_dim * num_layer),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(backbone_dim * num_layer, fea_dim),
            nn.LayerNorm(fea_dim),
            nn.SiLU(),
            nn.Linear(fea_dim, fea_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        with torch.no_grad():
            _, _, h, w = x.shape
            dino_inp = F.interpolate(x, size=(int(h / 14.) * 14, int(w / 14.) * 14), mode='bilinear')
            dino_inp = torch.clip(dino_inp * 0.5 + 0.5, 0, 1) * 255
            dino_inp = self.transform(dino_inp)
            f_list = [3 - 1, 6 - 1, 9 - 1, 12 - 1] #
            fea = self.backbone.get_intermediate_layers(dino_inp, f_list, reshape=True, return_class_token=False, norm=True)
            fea = torch.stack(fea, dim=1)

        fea = self.adapter(fea)

        return fea
    
    def save_model(self, outf):
        sd = {}
        sd["state_dict_adapter"] = {k: v for k, v in self.adapter.state_dict().items()}
        torch.save(sd, outf)
        
    def load_model(self, f):
        state_dict = torch.load(f)["state_dict_adapter"]
        self.adapter.load_state_dict(state_dict)
        