import torch

class GaussAdapt(torch.nn.Module):
    def __init__(self, clip_prototypes, shot_capacity = 8, sig_type = 'RidgeMoorePenrose'):
        '''shot_capacity: maximum number of stored samples per class.
        sig_type: type of estimator for teh covariance. One of 'Ridge', 'MoorePenrose' or the recommended 'RidgeMoorePenrose'. 
        The latter transitions from empirical Bayes Ridge (see https://doi.org/10.1016/j.jmva.2008.01.016) to inverse when more than 4d sampels are available. '''
        super(GaussAdapt, self).__init__()
        assert sig_type in ['RidgeMoorePenrose', 'Ridge', 'MoorePenrose']
        K,d = clip_prototypes.shape
        self.shot_capacity = shot_capacity
        self.K = K
        self.clip_prototypes = clip_prototypes #should be (K,d)
        self.mus = clip_prototypes.clone().type(torch.float32)
        self.temp = 100
        self.d = clip_prototypes.shape[-1]
        self.count = torch.nn.Parameter(torch.zeros(self.K), requires_grad = False)
        self.sig_type = sig_type
        self.Sig = torch.nn.Parameter(1/d * torch.eye(d, dtype = torch.float32), requires_grad = False)
        self.inv_Sig = torch.nn.Parameter(d * torch.eye(d, dtype = torch.float32), requires_grad = False)
        self.memory_state = torch.nn.Parameter(torch.zeros((K,shot_capacity), dtype = torch.bool), requires_grad = False)
        
        self.memory = torch.nn.Parameter(torch.zeros((K, shot_capacity, d), dtype = torch.float16), 
                                         requires_grad = False)
        self.memory_soft_labels = torch.nn.Parameter(torch.zeros((K, shot_capacity), dtype = torch.float16), 
                                         requires_grad = False)
        self.__init_entropy(prop_max = 1)

        return None
    
    def __init_entropy(self, prop_max = 1):
        max_entropy = -torch.log(torch.tensor(1/self.K))
        init_val = prop_max * max_entropy
        self.memory_entropy = torch.nn.Parameter(init_val * torch.ones((self.K,self.shot_capacity), dtype = torch.float16, device = self.memory.device),
                                                 requires_grad = False)
        return init_val
    
        
    def get_entropy(self, probs):
        sh_entropy = - torch.sum(torch.log(probs+1e-6)*probs, dim = -1)
        return sh_entropy
    
    def __update_memory_entropy(self, x, text_prob, entropy, pseudo_label, gauss_prob = None):
        updated = False
        if torch.any(entropy<self.memory_entropy[pseudo_label,:]):
            idx_max = torch.argmax(self.memory_entropy[pseudo_label,:])
            self.memory[pseudo_label, idx_max] = x[...]
            self.memory_entropy[pseudo_label, idx_max] = entropy
            self.memory_state[pseudo_label, idx_max] = True
            self.memory_soft_labels[pseudo_label,idx_max] = text_prob[pseudo_label]
            updated = True
    
        return updated
    

            
    def update_memory(self,
                      features,
                      text_logits, 
                      zs_probs, 
                      zs_entropy,
                      zs_labels,
                      tau = 0.03,
                      normalize_mu = False):
        '''This method updates the memory as well as the means and covariance if necessary. '''
        selected_samples = []
        updated = False
        
        # update labels
        for ji in range(zs_labels.shape[0]):
            up = self.__update_memory_entropy(features[ji,:], zs_probs[ji,:], zs_entropy[ji], zs_labels[ji])
            if up:
                selected_samples.append(ji)
                updated = True

        if updated:
            self.__update_mu(normalize_mu = normalize_mu)
            self.__update_sigma()
            
        return updated, selected_samples #, zs_entropy, upd_entropy
    
    def __update_mu(self, normalize_mu = False):
        means = torch.mean(self.memory_state[...,None]*self.memory, dim = 1).float()
        mask = torch.sum(self.memory_state,dim=1)>=2 # was >2
        self.mus[mask,:] = means[mask,:].type(torch.float32)
        if normalize_mu:
            self.mus[mask,:] = self.mus[mask,:] / torch.linalg.norm(self.mus[mask,:], dim = -1, keepdims = True)
        return None
    
    
    def __update_sigma(self, use_soft_labels = False):      
        if 'Ridge' == self.sig_type:
            d = self.mus.shape[-1]
            x = self.memory.view((self.K*self.shot_capacity, d))
            x_mem_state = self.memory_state.view((self.K*self.shot_capacity))
            if torch.any(torch.sum(self.memory_state, dim = -1)>2):
                x_labels = torch.tensor([k for k in range(self.K) for _ in range(self.shot_capacity)], device = x.device)
                center_vecs = torch.cat([x[torch.logical_and(x_mem_state, x_labels == k)] - self.mus[k:k+1,:] for k in range(self.K)])
                M = center_vecs.T.cov()
                trace = torch.sum(M[range(d), range(d)])
                # shape 1 = d / shape 0 = n
                n,d = center_vecs.shape
                cov_inv = d * torch.linalg.pinv((n - 1) * M + trace * torch.eye(d, device = center_vecs.device))    
                self.Sig[...] = M
                self.inv_Sig[...] = cov_inv
        elif 'RidgeMoorePenrose' == self.sig_type:
            d = self.mus.shape[-1]
            n = torch.sum(self.memory_state)
            
            if torch.any(torch.sum(self.memory_state, dim = -1)>2):
                x = self.memory.view((self.K*self.shot_capacity, d))
                x_labels = torch.tensor([k for k in range(self.K) for _ in range(self.shot_capacity)], device = x.device)
                x_mem_state = self.memory_state.view((self.K*self.shot_capacity))
                
                class_probs = self.memory_soft_labels[self.memory_state]
                center_vecs = torch.cat([x[torch.logical_and(x_mem_state, x_labels == k)] - self.mus[k:k+1,:] for k in range(self.K)])
                center_vec_mean = center_vecs.mean(dim=0)
                if use_soft_labels:
                    c_center_vecs = (center_vecs - center_vec_mean[None,:]) * class_probs[:,None]
                    M = c_center_vecs.T @ c_center_vecs / torch.sum(class_probs)
                else:
                    c_center_vecs = (center_vecs - center_vec_mean[None,:])
                    M = c_center_vecs.T @ c_center_vecs / (n-1)
                
                if n<=4*d:
                    # use shrinkage
                    
                    trace = torch.sum(M[range(d), range(d)])
                    # shape 1 = d / shape 0 = n
                    cov_inv = d * torch.linalg.pinv((n - 1) * M + trace * torch.eye(d, device = center_vecs.device))    
                    self.Sig[...] = M
                    self.inv_Sig[...] = cov_inv
                else:
                    # Use pinv
                    self.Sig[...] = M
                    self.inv_Sig[...] = torch.linalg.pinv(M.type(torch.float32))
        elif 'MoorePenrose' == self.sig_type:
            d = self.mus.shape[-1]
            x = self.memory.view((self.K*self.shot_capacity, d))
            x_mem_state = self.memory_state.view((self.K*self.shot_capacity))
            if torch.any(torch.sum(self.memory_state, dim = -1)>2):
                x_labels = torch.tensor([k for k in range(self.K) for _ in range(self.shot_capacity)], device = x.device)
                center_vecs = torch.cat([x[torch.logical_and(x_mem_state, x_labels == k)] - self.mus[k:k+1,:] for k in range(self.K)])
                M = center_vecs.T.cov()
                self.Sig[...] = M
                self.inv_Sig[...] = torch.linalg.pinv(M.type(torch.float32)) 
                
        return None
    
    def get_log_probs(self,x):
        W = torch.einsum('nd, dc -> cn', self.mus, self.inv_Sig)
        b =  - torch.einsum('nd, dc, nc -> n', self.mus, self.inv_Sig, self.mus) / 2
        Q =  - torch.einsum('nd, dc, nc -> n', x.float(), self.inv_Sig, x.float()) / 2
        log_probs = (x.float() @ W + b)
        log_probs += Q[:,None]
        return log_probs
        
    
    def get_MAP(self, y_hat, memory_logits, tau = 0.01, simplex_p = False):
        '''y_hat: zero shot soft labels. memory_logits: log probabilities obtained from the cached samples. '''
        lambd = 1.0
        assert type(tau) is float or type(lambd) is float
        # Compute gaussian probs
        if type(tau) is float:
            if not simplex_p:
                p_ = torch.exp(tau * memory_logits)
            else:
                p_ = (tau*memory_logits).softmax(-1)
        else:
            if not simplex_p:
                p_ = torch.exp(tau[None,None,:] * memory_logits[...,None])
            else:
                p_ = (tau[None,None,:] * memory_logits[...,None]).softmax(-1)
            
        # Compute MAP (only if y_hat is not None)       
        if y_hat is None:
            z = None
        else:
            if type(lambd) is float:
                if len(p_.shape) == 2:
                    z = (y_hat**lambd) * p_
                    z = z/torch.sum(z, dim = 1, keepdims = True)
                elif len(p_.shape) == 3:
                    z = (y_hat**lambd)[...,None] * p_
                    z = z/torch.sum(z, dim = 1, keepdims = True)
                else:
                    raise RuntimeError(f'Incompatible p_ shape {p_.shape}')
                    
            else:
                if len(p_.shape) == 2:
                    z = (y_hat[:,:,None]**lambd[None,None,:]) * p_[:,:,None]
                elif len(p_.shape) == 3:
                    z = (y_hat[:,:,None]**lambd[None,None,:])[...,None] * p_[...,None]
                else:
                    raise RuntimeError(f'Incompatible p_ shape {p_.shape}')
        return z, p_
    


class OGA_solver(torch.nn.Module):
    def __init__(self,
                 shot_capacity: int = 8,
                 sig_type: str = 'RidgeMoorePenrose',
                 tau: float = 0.01,
                 normalize_mu: bool = False,
                 simplex_p: bool = True,
                 device: str = 'cuda'):
        super().__init__()
        self.shot_capacity = shot_capacity
        self.sig_type = sig_type
        self.tau = float(tau)
        self.normalize_mu = normalize_mu
        self.simplex_p = simplex_p
        self.device = torch.device(device)
        self.oga_model: GaussAdapt | None = None
        self.K: int | None = None
        self.d: int | None = None

    def reset(self):
        """Clears the online memory to start a new data stream task."""
        if self.oga_model is None:
            return
        K, C = self.oga_model.K, self.oga_model.shot_capacity
        d = self.oga_model.d
        # Reset memory and statistics
        self.oga_model.memory_state[...] = False
        self.oga_model.memory[...] = torch.zeros((K, C, d), dtype=torch.float16, device=self.device)
        self.oga_model.memory_soft_labels[...] = torch.zeros((K, C), dtype=torch.float16, device=self.device)
        # Reset means to initial CLIP prototypes
        _ = self.oga_model.__init_entropy(prop_max=1)

    def _ensure_model(self, clip_prototypes: torch.Tensor, feat_dim: int):
        """Initialize the underlying OGA model on first call."""
        # clip_prototypes is expected as [D, K] or [D, 1, K], then converted to [K, D].
        proto = clip_prototypes.squeeze()
        if proto.dim() != 2:
            raise RuntimeError(f'clip_prototypes has invalid shape {clip_prototypes.shape}')
        d, K = proto.shape
        if self.oga_model is None or self.K != K or self.d != d:
            self.K, self.d = K, d
            self.oga_model = GaussAdapt(proto.T, shot_capacity=self.shot_capacity, sig_type=self.sig_type).to(self.device)

    @torch.no_grad()
    def forward(self,
                query_features: torch.Tensor,
                query_labels: torch.Tensor,
                clip_prototypes: torch.Tensor,
                device: str = None):
        dev = torch.device(device) if device is not None else self.device

        x = query_features.to(dev).float()
        clip_w = clip_prototypes.to(dev).float().squeeze()

        self._ensure_model(clip_w, x.shape[-1])

        zs_logits = 100.0 * (x @ clip_w)                # [N, K]
        y_hat = zs_logits.softmax(-1)                   # [N, K]
        zs_entropy = -torch.sum(torch.log(y_hat + 1e-9) * y_hat, dim=-1)  # [N]
        zs_labels = torch.argmax(zs_logits, dim=-1)     # [N]

        # Online memory update
        _updated, _sel = self.oga_model.update_memory(
            x,
            zs_logits,
            y_hat,
            zs_entropy,
            zs_labels,
            tau=self.tau,
            normalize_mu=self.normalize_mu
        )

        # Calculate adapted predictions
        log_probs = self.oga_model.get_log_probs(x)                        # [N, K]
        z, _p = self.oga_model.get_MAP(y_hat, log_probs, tau=self.tau, simplex_p=self.simplex_p)

        return y_hat.cpu(), z.cpu()
