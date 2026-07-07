import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperbolicUtils:
    def __init__(self, eps=1e-6): 
        self.eps = eps
        
    def exp_map(self, v, c=1.0):
        v_norm = torch.norm(v, p=2, dim=-1, keepdim=True).clamp(min=self.eps)

        scale = torch.tanh(v_norm) * (1 / (v_norm * (c ** 0.5)))
        z = scale * v
        
        z_norm = torch.norm(z, p=2, dim=-1, keepdim=True)
        max_norm = (1 - self.eps) / (c ** 0.5)
        
        cond = z_norm > max_norm
        projected = z / (z_norm + 1e-8) * max_norm 
        return torch.where(cond, projected, z)

    def dist(self, x, y, c=1.0):
        sqdist = torch.sum((x - y) ** 2, dim=-1)
        
        x_sqnorm = torch.sum(x ** 2, dim=-1).clamp(max=1 - self.eps)
        y_sqnorm = torch.sum(y ** 2, dim=-1).clamp(max=1 - self.eps)
        
        num = 2 * sqdist
        den = (1 - x_sqnorm) * (1 - y_sqnorm) + 1e-8 

        arg = 1 + num / den

        return torch.log(arg + torch.sqrt(torch.clamp(arg**2 - 1, min=self.eps)))
    
class ViewEncoder(nn.Module):
    def __init__(self, input_dim, depth=4, width=1000, latent_dim=128):
        super(ViewEncoder, self).__init__()
        layers = []
        
        layers.append(nn.Linear(input_dim, width))
        layers.append(nn.ReLU())
        
        for _ in range(depth - 2):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
            
        layers.append(nn.Linear(width, latent_dim))
        
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)

class ViewDecoder(nn.Module):
    def __init__(self, input_dim, depth=4, width=1000, latent_dim=128):
        super(ViewDecoder, self).__init__()
        layers = []
        
        layers.append(nn.Linear(latent_dim, width))
        layers.append(nn.ReLU())
        
        for _ in range(depth - 2):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
            
        layers.append(nn.Linear(width, input_dim))
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, z):
        return self.decoder(z)
    
class HAMC_Model(nn.Module):

    def __init__(self, input_dims, latent_dim=128, n_clusters=10, net_depth=4, net_width=1000):
        super(HAMC_Model, self).__init__()
        self.encoders = nn.ModuleList([
            ViewEncoder(dim, depth=net_depth, width=net_width, latent_dim=latent_dim) 
            for dim in input_dims
        ])
        self.decoders = nn.ModuleList([
            ViewDecoder(dim, depth=net_depth, width=net_width, latent_dim=latent_dim) 
            for dim in input_dims
        ])
        self.hyp = HyperbolicUtils()
        self.register_buffer('prototypes', torch.randn(n_clusters, latent_dim))
        self.prototypes = F.normalize(self.prototypes, dim=1)

    def forward(self, xs):
        vs = [enc(x) for enc, x in zip(self.encoders, xs)]
        zs = [self.hyp.exp_map(v) for v in vs]
        xs_rec = [dec(v) for dec, v in zip(self.decoders, vs)]
        return vs, zs, xs_rec

    def get_hyp_prototypes(self):
        return self.hyp.exp_map(self.prototypes)
    
def mse_loss(input, target):
    return torch.mean((target - input) ** 2)

@torch.no_grad()
def sinkhorn_knopp(logits, epsilon=0.05, iterations=3):

    max_logits = torch.max(logits, dim=1, keepdim=True)[0]
    logits = logits - max_logits.detach()
    
    Q = torch.exp(logits / epsilon).t() # [K, B]
    B = Q.shape[1]
    K = Q.shape[0]
    
    Q /= (torch.sum(Q, dim=0, keepdim=True) + 1e-8) 
    
    for _ in range(iterations):
        # Normalize Rows
        row_sum = torch.sum(Q, dim=1, keepdim=True) + 1e-8
        Q /= row_sum
        Q /= K 
        
        # Normalize Cols
        col_sum = torch.sum(Q, dim=0, keepdim=True) + 1e-8
        Q /= col_sum
        Q /= B
        
    Q *= B
    return Q.t()

def contrastive_loss_euclidean(z_i, z_j, temperature=0.5):
    batch_size = z_i.shape[0]
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)

    sim_i2j = torch.matmul(z_i, z_j.T) / temperature
    labels = torch.arange(batch_size).to(z_i.device)
    loss_i2j = F.cross_entropy(sim_i2j, labels)

    sim_j2i = torch.matmul(z_j, z_i.T) / temperature
    loss_j2i = F.cross_entropy(sim_j2i, labels)

    return 0.5 * (loss_i2j + loss_j2i)