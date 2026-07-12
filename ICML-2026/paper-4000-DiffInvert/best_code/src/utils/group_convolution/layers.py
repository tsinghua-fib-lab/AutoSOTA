import torch
import torch.nn as nn
import torch.nn.functional as F
from src import groups

def rbf_embedding(x, output_dim, sig, device):
    input_dim = x.shape[-1]
    mu = torch.linspace(0,1,output_dim).to(device)
    rbfcoords = []
    for i in range(input_dim):
        rbfcoords.append((-0.5*(x[...,i].unsqueeze(-1)-mu)**2/(sig**2)).exp())
    rbfemb = torch.cat(rbfcoords, -1)
    rbfemb = rbfemb/(rbfemb.norm(dim=-1).max())
    return rbfemb

def heavi(x, a, device):
    vals = torch.ones(1).to(device)
    out = []
    for i in range(x.shape[-1]):
        out.append(torch.heaviside(x[...,i]+a[i],vals) * torch.heaviside(a[i]-x[...,i], vals))
    output = out[0]
    for i in range(1, len(out)):
        output = output * out[i]
    return output

class FilterMLP(nn.Module):
    # creates an rbf embedding into input_dim * embedding_dim-space, an MLP that maps from here to some other space, together with a final linear layer.
    def __init__(self, embedding_dim, MLP_dims, output_dims, device):
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.output_dims = output_dims
        self.num_layers = len(MLP_dims) - 1
        self.layers = []
        self.activations = []
        for i in range(self.num_layers):
            self.layers.append(nn.Linear(MLP_dims[i], MLP_dims[i+1]).to(device))
            self.activations.append(nn.SiLU())
            for p in list(self.layers[i].named_parameters()):
                self.register_parameter(p[0] + ' {}'.format(i), p[1])
        self.linear_layer = nn.Linear(MLP_dims[self.num_layers], output_dims[0]*output_dims[1]).to(device)
        
    def forward(self, x):
        x = rbf_embedding(x, self.embedding_dim, 0.07, self.device)
        for i in range(self.num_layers):
            x = self.activations[i](self.layers[i](x))
        x = self.linear_layer(x).reshape(torch.tensor(x.shape).tolist()[:-1] + self.output_dims)
        return x

class FirstLayer(nn.Module):
    # feature dims [C, W, H]
    # filter dims [D_{0}, ..., D_{N-1}, Cin, Cout]
    def __init__(self, group, feature_dims, embedding_dim, MLP_dims, output_dims, scales, device):
        super().__init__()
        if group == 'affine':
            self.G = groups.ImageAffine(device)
        elif group == 'homography':
            self.G = groups.ImageHomography(device)
        else:
            raise NotImplementedError
        self.device = device
        self.output_dims = output_dims
        self.filter = FilterMLP(embedding_dim, MLP_dims, output_dims, device)
        self.scales = scales
        self.feature_size = feature_dims[1:]
        # set up Euclidean coordinates
        coord_vectors = []
        for i in range(2):
            coord_vectors.append(torch.linspace(-1 * self.scales[i], 1 * self.scales[i], self.feature_size[i]))
        self.euc_coords = torch.stack(torch.meshgrid(coord_vectors), 2).to(device)
        # set up homogeneous coordinates
        self.hom_coords = torch.cat((self.euc_coords, torch.ones(self.feature_size + [1]).to(device)), dim = 2).to(device)
        # set up the scaling factor for approximating Euclidean volume element in R2 by finite sum
        vol_scaling = []
        for i in range(2):
            vol_scaling.append( 2 / self.feature_size[i] )
        self.eucl_scaling = torch.prod(torch.tensor(vol_scaling))

    def forward(self,data,v):
        # v is a batch of matrices, computed from a batch of paths
        # assume input tensor is of the form NxCxWxH, and corresponds to a square image
        # coordinates are assigned to image by assuming it to be contained in the square [-1, 1]\times [-1, 1]

        # warp_grid, jac = self.G.act(v, self.hom_coords) #compute deformed coordinates, drop the 1s coming from the homogeneous coordinates

        # modified version
        data = data.float()
        num_pool, num_path, _, _ = v.shape
        B = num_pool * num_path
        g_batch = v.reshape(B, 3, 3)  # (B, 3, 3)
        H, W, d = self.hom_coords.shape # (H, W, 3)
        x_batch = self.hom_coords.unsqueeze(0).expand(B, H, W, d)
        x_batch = x_batch.permute(0, 3, 1, 2)

        warp_grid, _ = self.G.act_warp(g_batch, x_batch)
        warp_grid = warp_grid[:, :2]  # drop the 1s coming from the homogeneous coordinates
        warp_grid = warp_grid.permute(2, 3, 0, 1).reshape(H, W, num_pool, num_path, -1)
        jac = torch.exp(self.G.log_det_jacobian(g_batch, x_batch)).permute(1, 2, 0).reshape(H, W, num_pool, num_path)
        
        warp_grid_shape = torch.tensor(warp_grid.shape).tolist()
        integral = torch.zeros([data.shape[0]] + warp_grid_shape[2:-1] + [self.output_dims[1]]).to(self.device)
        num = 2 # make this number higher if GPU memory is running short
        num2 = int(self.feature_size[0] / num)
        for i in range(num):
            filtertensor = torch.mul(jac[i*num2:(i+1)*num2].unsqueeze(-1).unsqueeze(-1).to(data.dtype),
                                     self.filter(warp_grid[i*num2:(i+1)*num2]).to(data.dtype)) # shape W x H x N_pool x N1 x N2 x Cin x Cout
            filtertensor = torch.mul(heavi(warp_grid[i*num2:(i+1)*num2], [0.1, 0.1], self.device).unsqueeze(-1).unsqueeze(-1).to(data.dtype),
                                     filtertensor)
            integral = integral + torch.tensordot(data[...,i*num2:(i+1)*num2,:], filtertensor, dims = ([1, -2, -1], [-2, 0, 1])) # data shape B x cin x W x H
        return integral

class PoolingLayer(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super().__init__()
        self.Cin = input_dim
        self.Cout = output_dim
        self.linear_classifier = nn.Linear(self.Cin, self.Cout).to(device)

    def forward(self, feature):
        outputs = self.linear_classifier(feature) # shape b x n x Cin
        return torch.amax(outputs, dim=1)
