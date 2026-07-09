import torch
import torch.nn.functional as F
import torch.nn as nn

# ==========================================
# Helper functions for vMF distribution
# ==========================================

def log_Cd_approx(kappa, d):
    '''
    Approximate the normalization constant log C_d(kappa) for the vMF distribution.
    Ref: Clustering on the unit hypersphere using von Mises-Fisher distributions (JMLR 2005)
    Formula: log C_d(k) = (d/2 - 1) * log(k) - (d/2) * log(2*pi) - log I_{d/2-1}(k)
    Approximation: log I_v(k) approx k - 0.5 * log(2 * pi * k) under high dimension / large kappa.
    '''
    # Prevent numerical instability when kappa is too small.
    kappa = torch.clamp(kappa, min=1e-6)
    
    # Approximation under large kappa and high dimension.
    # log C_d(kappa) approx (d-1)/2 * log(kappa) - kappa - (d/2)*log(2*pi)
    # The constant term (d/2)*log(2*pi) cancels in softmax when d is fixed.
    return (d-1)/2 * torch.log(kappa) - kappa


def Ad_inverse_approx(r_bar, d):
    '''
    Approximate the inverse of A_d(k) = r_bar with the Banerjee formula.
    Ref: Sra, S. (2012). A short note on parameter approximation for von Mises-Fisher distributions.
    kappa approx (r * d - r^3) / (1 - r^2)
    '''
    r = torch.clamp(r_bar, min=0.0, max=0.999) # Avoid division by zero.
    return (r * d - r ** 3) / (1 - r ** 2)

# ==========================================
# Core code
# ==========================================

# Get zero-shot logits
def get_zero_shot_logits(query_features, query_labels, clip_prototypes, zs_temperature=100.0):
    
    clip_logits = zs_temperature * query_features @ clip_prototypes
    
    return clip_logits.squeeze()

# Build affinity matrix
def build_affinity_matrix(query_features, n_neighbors):
    '''
        Adjacency matrix W for the Laplacian term.
        Output: sparse tensor, (N, N)
    '''
    device = query_features.device
    num_samples = query_features.size(0)
    affinity = query_features.matmul(query_features.T).cpu()
    num_rows = num_samples
    num_cols = num_samples
        
    if n_neighbors <= 0:
        empty_indices = torch.empty((2, 0), dtype=torch.long, device=device)
        empty_values = torch.empty((0,), dtype=query_features.dtype, device=device)
        return torch.sparse_coo_tensor(empty_indices, empty_values, size=(num_rows, num_cols), device=device)

    knn_index = affinity.topk(n_neighbors + 1, -1, largest=True).indices[:, 1:]
    row_indices = torch.arange(num_rows).unsqueeze(1).repeat(1, n_neighbors).flatten()
    col_indices = knn_index.flatten()
    values = affinity[row_indices, col_indices].to(device)
    W = torch.sparse_coo_tensor(torch.stack([row_indices, col_indices]).to(device), values, size=(num_rows, num_cols),
                                device=device)
    return W


class vMF(nn.Module):
    '''
    vMF mixture component adapter.
    '''
    def __init__(self, mu, kappa):
        super().__init__()
        self.mu = mu.clone()       # (K, 1, d)
        self.kappa = kappa.clone() # (K, 1), concentration parameter

    def forward(self, x, no_exp=False):
        '''
        Compute the exponential part of vMF log-likelihood: kappa * mu^T * x.
        Input: x: (N, d)
        Output: (N, K)
        '''
        chunk_size = 2500
        N = x.shape[0]
        K = self.mu.shape[0]
        likelihoods = torch.empty((N, K), dtype=x.dtype, device=x.device)
         
        # x: (N, d), mu: (K, 1, d)
        for start_idx in range(0, N, chunk_size):
            end_idx = min(start_idx + chunk_size, N)
            # Calculate cosine similarity (since x and mu are normalized)
            # x: (N, d), mu: (K, 1, d). 
            # (N, d) @ (K, d)^T -> (N, K)
            cosine_sim = x[start_idx:end_idx] @ self.mu.squeeze(1).T
            
            # Log-likelihood = kappa * cosine_sim
            # kappa: (K, 1) -> transpose to (1, K)
            likelihoods[start_idx:end_idx] = self.kappa.T * cosine_sim 

        if not no_exp:
            likelihoods = torch.exp(likelihoods)
        
        return likelihoods

    def set_kappa(self, kappa):
        self.kappa = kappa
        
    def set_mu(self, mu):
        self.mu = mu


def update_z(likelihoods, y_hat, z, W, lambda_y_hat, lambda_laplacian, n_neighbors, kappa, d, e_step_temperature=50.0, max_iter=1):
    '''
    E-step: update assignment matrix z
    Likelihoods contains kappa * mu^T * x.
    '''
    prior = y_hat ** lambda_y_hat
    
    # Normalization constant log C_d(kappa)
    # kappa: (K, 1).
    norm_const = log_Cd_approx(kappa, d).T # (1, K)

    for it in range(max_iter):
        intermediate = likelihoods.clone()
        
        # vMF Log-Prob = (kappa * mu^T x) + log C_d(kappa)
        intermediate += norm_const
        
        # Laplacian term with linear approximation
        if n_neighbors > 0 and lambda_laplacian != 0:
            intermediate += lambda_laplacian * (e_step_temperature / (n_neighbors * 2)) * (
                W.T @ z + (W @ z)) 
        
        # Numerical stability
        intermediate -= torch.max(intermediate, dim=1, keepdim=True)[0]
        
        # Softmax step
        intermediate = prior * torch.exp(1.0 / e_step_temperature * intermediate) 
        z = intermediate / torch.sum(intermediate, dim=1, keepdim=True)
        
    return z


def update_mu(adapter, query_features, z, beta, init_prototypes, sample_weights=None):
    '''
    M-step: update mean direction mu.
    Formula: mu_new = Normalize(beta * v_k + (1-beta) * mu_init)
    This is equivalent to Normalize(Sum(z_i * f_i) + alpha * mu_init).
    '''
    if sample_weights is not None:
        z = z * sample_weights.unsqueeze(1) # Apply sample weights
        
    # 1. Compute weighted sample mean vector v_k (without normalization)
    weighted_sum = torch.einsum('ij,ik->jk', z, query_features) 
    sum_z = torch.sum(z, dim=0).unsqueeze(-1) + 1e-10 # (K, 1)
    v_k = weighted_sum / sum_z
    
    # 2. Combine with text anchors (shrinkage form in StatA)
    b = beta.unsqueeze(-1)
    mu_p = init_prototypes.squeeze(1)
    
    # Convex combination in mean-parameter space
    new_mu = b * v_k + (1 - b) * mu_p
    
    # 3. Project back to the hyper-sphere
    new_mu = F.normalize(new_mu, p=2, dim=-1)
    
    return new_mu.unsqueeze(1) # Reshape back to (K, 1, d)


def update_kappa(adapter, query_features, z, beta, init_prototypes, d, sample_weights=None):
    '''
    M-step: update concentration kappa with chunked computation for v_k.
    r_bar_k = || beta * v_k + (1-beta) * mu_init ||
    '''
    chunk_size = 2500
    N = query_features.size(0)
    K = z.size(1)
    device = query_features.device
    dtype = query_features.dtype
    num = torch.zeros((K, query_features.size(1)), device=device, dtype=dtype)  # (K, d)
    denom = torch.zeros((K, 1), device=device, dtype=dtype)                     # (K, 1)
    
    if sample_weights is not None:
        z = z * sample_weights.unsqueeze(1) # Apply sample weights
    
    for start_idx in range(0, N, chunk_size):
        end_idx = min(start_idx + chunk_size, N)
        # 1. Calculate the mean resultant length r_bar for each class k
        # r_bar = || Sum(z * f) + alpha * mu_init || / (Sum(z) + alpha)
        # Using the definition of beta: beta = Sum(z) / (Sum(z) + alpha), we can rearrange to get:
        # r_bar_k = || beta * v_k + (1-beta) * mu_init ||
        # (c,K) and (c,d) -> (K,d): sum_i z_ik * f_i
        num += torch.einsum('ik, id -> kd', z[start_idx:end_idx], query_features[start_idx:end_idx])   # (K, d)
        denom += z[start_idx:end_idx].sum(dim=0, keepdim=True).T       # (K, 1)
    
    v_k = num / (denom + 1e-10)
    
    b = beta.unsqueeze(-1)
    mu_p = init_prototypes.squeeze(1)
    
    resultant_vector = b * v_k + (1 - b) * mu_p
    r_bar = resultant_vector.norm(dim=-1, keepdim=True) # (K, 1)
    
    # 2. Update kappa using the Banerjee approximation
    # A_d(k) = r_bar
    new_kappa = Ad_inverse_approx(r_bar, d)
    # Upper bound for stability
    new_kappa = torch.clamp(new_kappa, max=500.0)
    
    return new_kappa


def init_kappa(clip_prototypes, query_features, z, d):
    '''
    Initialize kappa.
    Uses the relationship between variance and kappa:
    MeanSqDist approx 2*d / (2*kappa + d), or E[x^T mu] = A_d(k) approx 1 - d/(2k).
    '''
    # clip_prototypes: (1, d, K) -> (K, d)
    mu_init = clip_prototypes.permute(2, 0, 1).squeeze(1)  # (K, d)
    
    chunk_size = 2500
    N = query_features.size(0)
    K = mu_init.size(0)
    device = query_features.device
    dtype = query_features.dtype
    weighted_sum = torch.zeros(K, device=device, dtype=dtype)
    denom = torch.zeros(K, device=device, dtype=dtype)
    
    for start_idx in range(0, N, chunk_size):
        end_idx = min(start_idx + chunk_size, N)
        diff = query_features[start_idx:end_idx].unsqueeze(1) - mu_init.unsqueeze(0)  # (N, K, d)
        dist_sq = (diff ** 2).sum(dim=-1) # (N, K)
        weighted_sum += (z[start_idx:end_idx] * dist_sq).sum(dim=0)            # (K,)
        denom += z[start_idx:end_idx].sum(dim=0)  
    
    # Weighted average of squared distances for each class k
    # Then use Banerjee approximation to initialize kappa
    weighted_dist = weighted_sum / (denom + 1e-10) # (K,)
    Ad_init_approx = 1 - weighted_dist / 2
    kappa_init = Ad_inverse_approx(Ad_init_approx, d)
    return kappa_init.unsqueeze(1), Ad_init_approx # (K, 1)


def update_beta(z, alpha, soft=False, class_weights=None, sample_weights=None):
    '''
    Update shrinkage strength beta.
    Same form as StatA, but with class-level and sample-level dynamic weights.
    '''
    if sample_weights is not None:
        z = z * sample_weights.unsqueeze(1) # Apply sample weights
        
    if class_weights is not None:
        # Use class-specific alpha_k based on class_weights
        alpha_k = alpha / (class_weights + 1e-12)
    else:
        alpha_k = alpha
        
    if soft:
        sum_z = torch.sum(z, dim=0)  # [num_classes]
        beta = sum_z / (alpha_k + sum_z)
    else:
        predicted_classes = torch.argmax(z, dim=1) 
        sum_z = torch.bincount(predicted_classes, minlength=z.size(1))  
        beta = sum_z / (alpha_k + sum_z + 1e-12)
    return beta


def MOON_solver(query_features, query_labels, clip_prototypes, alpha=1, soft_beta=False, lambda_y_hat=1, lambda_laplacian=1, n_neighbors=3, max_iter=10, e_step_temperature=50.0, zs_temperature=100.0):
    
    query_labels = query_labels.cuda().float() 
    clip_prototypes = clip_prototypes.cuda().float()  # Text features, (1, d, K)
    query_features = query_features.cuda().float() # Image features, (N, d)
    
    # Feature normalization on the hypersphere
    query_features = F.normalize(query_features, p=2, dim=-1)
    clip_prototypes = F.normalize(clip_prototypes, p=2, dim=1)
    
    d = query_features.shape[1] # Feature dimension
    
    ##########
    # Z init #
    ##########
    zs_logits = get_zero_shot_logits(query_features, query_labels, clip_prototypes, zs_temperature)
    y_hat = F.softmax(zs_logits, dim=1)
    z = y_hat.clone() 
    
    ###########
    # MU init #
    ###########
    # (num_classes, 1, d)
    mu = clip_prototypes.permute(2,0,1) 

    ##############
    # KAPPA init #
    ##############
    kappa, _ = init_kappa(clip_prototypes, query_features, z, d)
    
    # Initialize vMF adapter
    adapter = vMF(mu=mu, kappa=kappa).cuda()
    
    ###################
    # Affinity matrix #
    ###################
    W = build_affinity_matrix(query_features.float(), n_neighbors)
    
    # Calculate class-wise weights based on zero-shot confidence
    # Geometric mean of average and max confidence
    class_weights = y_hat.mean(dim=0)
    class_weights_max, _ = torch.max(y_hat, dim=0)
    class_weights = torch.sqrt(class_weights * class_weights_max)
    class_weights = (class_weights / torch.max(class_weights)).cuda().view(-1)
    
    
    # EM iterations
    for k in range(max_iter + 1):
        
        likelihoods = adapter(query_features, no_exp=True)
        
        ''' E-step '''
        ############
        # Z update #
        ############
        z = update_z(likelihoods, y_hat, z, W, lambda_y_hat, lambda_laplacian, n_neighbors, adapter.kappa, d, e_step_temperature)
        
        if k == max_iter: 
            break
        
        ''' M-step ''' 
        # Calculate entropy-based sample weights
        entropies = -torch.sum(z * torch.log(z + 1e-12), dim=1)
        sample_weights = 1 - (entropies / (torch.log(torch.tensor(z.size(1), dtype=entropies.dtype)) + 1e-12))
    
        ###############
        # BETA update #
        ###############
        beta = update_beta(z, alpha, soft=soft_beta, class_weights=class_weights, sample_weights=sample_weights)
        
        #############
        # MU update #
        #############
        mu = update_mu(adapter, query_features, z, beta, clip_prototypes.permute(2,0,1), sample_weights=sample_weights)
        adapter.set_mu(mu)

        ################
        # KAPPA update #
        ################
        kappa = update_kappa(adapter, query_features, z, beta, clip_prototypes.permute(2,0,1), d, sample_weights=sample_weights)
        adapter.set_kappa(kappa)

    return y_hat.cpu(), z.cpu()
