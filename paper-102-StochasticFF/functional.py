import torch

def effective_dimensionality(cov, mode='fast'):
    if mode == 'fast':
        trace = torch.sum(torch.diagonal(cov, dim1=-2, dim2=-1), dim=-1)
        f_norm = torch.sum(cov ** 2, dim=(-2, -1)) + 1e-6
        ed = trace ** 2 / f_norm
    elif mode == 'exact':
        eigenvalues= torch.linalg.eigvalsh(cov)
        ed = torch.sum(eigenvalues, dim=-1, keepdim=True) ** 2 / (torch.sum(eigenvalues ** 2, dim=-1, keepdim=True) + 1e-6)
    else:
        raise ValueError('mode should be fast or exact')
    return ed 

def consistency_loss(self, post: torch.Tensor, label=None):
    b, c, _, _ = post.size()
    
    if self.sampling_len > 1:
        b = b // self.sampling_len
        post = post.reshape(self.sampling_len, b, c, -1) # (n, b, c, h * w)
    else:
        post = post.reshape(1, b, c, -1)
        
    if 'supervised' == self.consistency_mode and label is not None:
            post = torch.permute(post, (2, 1, 0, 3)).flatten(2) # ( c, b, n * h * w)
            group, weight =[], []
            for i in range(self.num_classes):
                idx = label == i
                if idx.sum() > 0:
                    feature = post[:, idx].reshape(c, -1)
                    feature = torch.einsum('ik,jk->ij', feature, feature) / (feature.size(1) - 1)
                    group.append(feature.unsqueeze(0))
                    weight.append(idx.sum() / b)
            weight = torch.tensor(weight, device=post.device)
            cov = torch.cat(group, dim=0) # (b, c, c)
    else:
        post = torch.permute(post, (1, 2, 0, 3)).flatten(2) # ( b, c, n * h * w)
        cov = torch.einsum('ijk,ilk->ijl', post, post) / (post.size(2) - 1)
        weight = torch.ones(b, device=post.device) / b
    consistency= effective_dimensionality(cov).squeeze()
    return torch.sum(consistency * weight)

def diversity_loss(self, post: torch.Tensor):
    b, c, h, w = post.size()
    if self.sampling_len > 1:
        post = post.reshape(self.sampling_len, -1, c, h*w) # (n, b, c, h * w)
    else:
        post = post.reshape(1, b, c, h * w)
    post = torch.mean(post, dim=0) # (b, c, h * w) 
    post = torch.permute(post, (1, 0, 2)).flatten(1) # (c, b * n * h * w)
    cov = torch.einsum('ik,jk->ij', post, post) / (post.size(-1) - 1)
    diversity_loss = effective_dimensionality(cov)
    return -diversity_loss


def local_loss(self, post, label=None):  
    if self.consistency_mode is not None:
        consistency = consistency_loss(self, post, label=label) * self.consistency_factor
    else:
        consistency = 0
    if self.diversity_mode is not None:
        diversity = diversity_loss(self, post) * self.diversity_factor
    else:   
        diversity = 0
    loss = consistency + diversity
    return loss