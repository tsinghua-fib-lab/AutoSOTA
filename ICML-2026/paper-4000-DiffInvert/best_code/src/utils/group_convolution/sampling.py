import torch
import numpy as np
from src import groups


class generate_haar_PI_parallel():
    def __init__(self, group, num_pooling_samples, num_path_samples, num_conv_layers, scales, device):
        if group == 'affine':
            self.G = groups.ImageAffine(device)
        elif group == 'homography':
            self.G = groups.ImageHomography(device)
        else:
            raise NotImplementedError
        self.num_pooling_samples = num_pooling_samples
        self.num_path_samples = num_path_samples
        self.num_conv_layers = num_conv_layers
        self.scales = scales
        self.device = device

    def generate(self):
        v = []
        dim = self.G.basis.shape[0]
        uvec = torch.mul((1/(self.num_conv_layers+1))*torch.tensor(self.scales), -torch.randn(self.num_pooling_samples, dim).to(self.device)).to(self.device) # pooling samples
        umtx = self.G.exp(uvec)
        x = torch.mul(0.5*(1/(self.num_conv_layers+1))*torch.tensor(self.scales), torch.randn(self.num_conv_layers, self.num_path_samples, dim).to(self.device)).to(self.device)
        rejects = torch.ones((self.num_conv_layers, self.num_path_samples), dtype=bool).to(self.device)
        while torch.sum(rejects) != 0:
            xnew =  torch.mul(0.5*(1/(self.num_conv_layers+1))*torch.tensor(self.scales), torch.randn(self.num_conv_layers, self.num_path_samples, dim).to(self.device)).to(self.device) + x
            alphas = self.G.dvol(-xnew) / self.G.dvol(-x)
            uniforms = torch.rand(self.num_conv_layers, self.num_path_samples).to(self.device)
            x[rejects] = xnew[rejects]
            rejects = torch.mul(rejects, (alphas < uniforms))
            rejects = rejects.reshape(self.num_conv_layers, self.num_path_samples)
        for i in range(self.num_conv_layers):
            v.append(x[i])
        prod = umtx.unsqueeze(1).expand(self.num_pooling_samples, self.num_path_samples, 3, 3)
        for i in range(len(v)-1, -1, -1):
            vmtx = self.G.exp(v[i])
            prod = torch.matmul(vmtx, prod)
        v.append(prod)
        return v

# class generate_haar_imgs():
#     def __init__(self, group, num_samples, scales, device):
#         self.G = lie.LieGroup(group, device)
#         self.num_samples = num_samples
#         self.scales = scales
#         self.device = device

#     def generate(self):
#         x = torch.mul(0.5*torch.tensor(self.scales), torch.randn(self.num_samples, self.G.dim)).to(self.device)
#         rejects = torch.ones(self.num_samples, dtype=bool).to(self.device)
#         while torch.sum(rejects) != 0:
#             xnew = torch.mul(0.5*torch.tensor(self.scales), torch.randn(self.num_samples, self.G.dim)).to(self.device) + x
#             alphas = self.G.dvol(-xnew) / self.G.dvol(-x)
#             uniforms = torch.rand(self.num_samples).to(self.device)
#             x[rejects] = xnew[rejects]
#             rejects = torch.mul(rejects, (alphas < uniforms))
#             rejects = rejects.reshape(self.num_samples)
#         xexp = self.G.exp(x)
#         return xexp

