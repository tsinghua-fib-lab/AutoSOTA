import torch
import torch.nn.functional as F
import torch.nn as nn



def get_zero_shot_logits(query_features, query_labels, clip_prototypes):

    clip_logits = 100 * query_features @ clip_prototypes

    return clip_logits.squeeze()


def build_affinity_matrix(query_features, n_neighbors):
    
    device = query_features.device
    num_samples = query_features.size(0)
    affinity = query_features.matmul(query_features.T).cpu()
    num_rows = num_samples
    num_cols = num_samples
        
    knn_index = affinity.topk(n_neighbors + 1, -1, largest=True).indices[:, 1:]
    row_indices = torch.arange(num_rows).unsqueeze(1).repeat(1, n_neighbors).flatten()
    col_indices = knn_index.flatten()
    values = affinity[row_indices, col_indices].to(device)
    W = torch.sparse_coo_tensor(torch.stack([row_indices, col_indices]).to(device), values, size=(num_rows, num_cols),
                                device=device)
    return W




class Gaussian(nn.Module):
    def __init__(self, mu, cov):
        super().__init__()
        self.mu = mu.clone()
        self.cov = cov.clone()

    def forward(self, x, no_exp=False):
        chunk_size = 2500
        N = x.shape[0]
        M, D = self.mu.shape[0], self.cov.shape[0]

        likelihoods = torch.empty((N, M), dtype=x.dtype, device=x.device)
        
        for start_idx in range(0, N, chunk_size):
            end_idx = min(start_idx + chunk_size, N)
            likelihoods[start_idx:end_idx] = -0.5 * (
                    ((x[start_idx:end_idx][:, None, :] - self.mu[None, :, 0, :]) ** 2) *
                    (1 / self.cov[None, :, :])
                ).sum(dim=2)


        if not no_exp:
            likelihoods = torch.exp(likelihoods)
        
        return likelihoods

    def set_cov(self, cov):
        self.cov = cov
        
    def set_mu(self, mu):
        self.mu = mu



def update_z(likelihoods, y_hat, z, W, lambda_y_hat, lambda_laplacian, n_neighbors, sigma, max_iter=5):
    for it in range(max_iter):
        intermediate = likelihoods.clone()
        intermediate += lambda_laplacian*(50 / (n_neighbors * 2)) * (
                W.T @ z + (W @ z))
        
        sigma_log_sum = sigma.log().sum(dim=1)
        intermediate -= 0.5 * sigma_log_sum.unsqueeze(0)
        
        # For numerical stability
        intermediate -= torch.max(intermediate, dim=1, keepdim=True)[0]
        intermediate = (y_hat ** lambda_y_hat) * torch.exp(1 / 50 * intermediate)
        z = intermediate / torch.sum(intermediate, dim=1, keepdim=True)
    return z


def update_mu(adapter, query_features, z, beta, init_prototypes):

    mu = torch.einsum('ij,ik->jk', z, query_features) 
    mu /= torch.sum(z, dim=0).unsqueeze(-1)
    mu = mu.unsqueeze(1)
    mu /= mu.norm(dim=-1, keepdim=True)
    mu = beta.unsqueeze(-1).unsqueeze(-1) * mu + (1-beta).unsqueeze(-1).unsqueeze(-1) * init_prototypes
    mu /= mu.norm(dim=-1, keepdim=True)
    return mu


def update_cov(adapter, query_features, z, beta, init_prototypes, init_covariance):
    n_query = z.size(0)
    num_classes = z.size(1)  # Assuming z is a one-hot encoded tensor or soft labels of shape [n_query, num_classes]
    chunk_size = 2500  # Iterate over query_features in chunks to avoid large memory consumption

    for start_idx in range(0, n_query, chunk_size):
        end_idx = min(start_idx + chunk_size, n_query)
        query_features_chunk = query_features[start_idx:end_idx]  # Shape: [chunk_size, num_dim]
        
        # Compute diff, squared_diff, and weighted_sum in one line
        weighted_sum = ((query_features_chunk[:, None, :] - adapter.mu[None, :, 0, :]) ** 2 * z[start_idx:end_idx, :, None]).sum(dim=0)  # Shape: [num_classes, num_dim]

        # Accumulate the covariance
        if start_idx == 0:
            cov = weighted_sum  # Initialize the covariance matrix for all classes
        else:
            cov += weighted_sum  # Accumulate per-class contributions

    cov /= z.sum(dim=0)[:, None]  
    
    # Compute delta_mu
    delta_mu = (init_prototypes - adapter.mu).squeeze()  # Shape: [num_classes, num_features]
    result = torch.bmm(delta_mu.unsqueeze(2), delta_mu.unsqueeze(1))  # Shape: [num_classes, num_features, num_features]
    diagonal_result = torch.diagonal(result, dim1=1, dim2=2)  # Extract diagonal, Shape: [num_classes, num_features]
    
    # Final update to std
    cov = beta.unsqueeze(-1) * cov + (1 - beta).unsqueeze(-1) * (init_covariance + diagonal_result)
    
    return cov



def init_cov(clip_prototypes, query_features, z):
    
    n_query = z.size(0)
    chunk_size = 2500  # Iterate over query_features in chunks to avoid large memory consumption
    
    for start_idx in range(0, n_query, chunk_size):
        end_idx = min(start_idx + chunk_size, n_query)
        query_features_chunk = query_features[start_idx:end_idx]
        
        chunk_result = torch.einsum(
            'ij,ijk->k',
            z[start_idx:end_idx, :],
            (query_features_chunk[:, None, :] - clip_prototypes[None, :,
                                               0, :]) ** 2)
        # If this is the first chunk, initialize cov; otherwise, accumulate
        if start_idx == 0:
            cov = chunk_result
        else:
            cov += chunk_result
        cov /= n_query 
    return cov

def update_beta(z, alpha, soft=False):

    if soft:
        sum_z = torch.sum(z, dim=0)  # [num_classes]
        beta = sum_z / (alpha + sum_z)
    else:
        predicted_classes = torch.argmax(z, dim=1) 
        sum_z = torch.bincount(predicted_classes, minlength=z.size(1))  
        beta = sum_z / (alpha + sum_z + 1e-12)
    return beta

    





def StatA_solver(query_features, query_labels, clip_prototypes, alpha=1, soft_beta=False, lambda_y_hat=1, lambda_laplacian=1, n_neighbors=3, max_iter=10):
    
    query_labels = query_labels.cuda().float()
    clip_prototypes = clip_prototypes.cuda().float()
    query_features = query_features.cuda().float()
    
    ##########
    # Z init #
    ##########

    zs_logits = get_zero_shot_logits(query_features, query_labels, clip_prototypes)
    y_hat = F.softmax(zs_logits, dim=1)
    z = y_hat.clone()
    
    ###########
    # MU init #
    ###########

    mu = clip_prototypes.permute(2,0,1) 

    ############
    # COV init #
    ############
     
    cov = init_cov(clip_prototypes.permute(2, 0, 1), query_features, z)
    cov = cov.unsqueeze(0).repeat(y_hat.size(-1), 1) 
    init_covariance = cov

    adapter = Gaussian(mu=mu, cov=cov).cuda()
    
    ###################
    # Affinity matrix #
    ###################
    
    W = build_affinity_matrix(query_features.float(), n_neighbors)
    
    for k in range(max_iter + 1):
        
        likelihoods = adapter(query_features, no_exp=True)
        
        ############
        # Z update #
        ############

        z = update_z(likelihoods, y_hat, z, W, lambda_y_hat, lambda_laplacian, n_neighbors, adapter.cov)
        
        if k == max_iter:  # STOP
            break
        
        ###############
        # BETA update #
        ###############
        
        beta = update_beta(z, alpha, soft=soft_beta)
        
        #############
        # MU update #
        #############

        mu = update_mu(adapter, query_features, z, beta, clip_prototypes.permute(2,0,1))
        adapter.set_mu(mu)

        ################
        # SIGMA update #
        ################
        
        cov = update_cov(adapter, query_features, z, beta, clip_prototypes.permute(2,0,1), init_covariance)
        adapter.set_cov(cov)

    return y_hat.cpu(), z.cpu()

