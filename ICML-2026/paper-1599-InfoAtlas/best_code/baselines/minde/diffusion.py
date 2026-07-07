import torch
import numpy as np

from estimators.minde.importance import *
from estimators.minde.diff_utils import *

class VP_SDE():
    def __init__(self,
                 beta_min=0.1,
                 beta_max=20,
                 N=1000,
                 importance_sampling=True,
                 type="j",
                 weight_s_functions=True,
                 var_sizes=[1, 1],
                 device="cuda",
                 ):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.var_sizes = var_sizes
        self.rand_batch = False
        self.N = N
        self.T = 1
        self.importance_sampling = importance_sampling
        self.nb_var = len(self.var_sizes)
        self.weight_s_functions = weight_s_functions
        self.device = device
        self.type = type
        self.masks = self.get_masks_training()
       

    def set_device(self, device):
        self.device = device
        self.masks = self.masks.to(device)

    def beta_t(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def sde(self, t):
        # Returns the drift and diffusion coefficient of the SDE ( f(t), g(t)) respectively.
        return -0.5*self.beta_t(t), torch.sqrt(self.beta_t(t))

    def marg_prob(self, t, x):
        
        ## Returns mean and std of the marginal distribution P_t(x_t) at time t.
        log_mean_coeff = -0.25 * t ** 2 * \
            (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        mean = torch.exp(log_mean_coeff)
        std = torch.sqrt(1 - torch.exp(2 * log_mean_coeff))
        return mean.view(-1, 1) * torch.ones_like(x, device=self.device), std.view(-1, 1) * torch.ones_like(x, device=self.device)

    def sample(self, x_0, t):
        ## Forward SDE
        # Sample from P(x_t | x_0) at time t. Returns A noisy version of x_0.
        mean, std = self.marg_prob(t, t)
        z = torch.randn_like(x_0, device=self.device)
        x_t = x_0 * mean + std * z
        return x_t, z, mean, std

    def train_step(self, data, score_net, eps=1e-5):
        """
        Perform a single training step for the SDE model.

        Args:
            data : The input data for the training step.
            score_net : The score network used for computing the score.
            eps: A small value used for numerical stability when importance sampling is Off. Defaults to 1e-5.
        Returns:
            Tensor: The loss value computed during the training step.
        """

        x_0 = torch.cat(data, dim=1)
        bs = x_0.size(0)

        if self.importance_sampling:
            t = (self.sample_importance_sampling_t(
                shape=(x_0.shape[0], 1))).to(self.device)
        else:
            t = ((self.T - eps) *
                 torch.rand((x_0.shape[0], 1)) + eps).to(self.device)
            
        # randomly sample an index to choose a masks
        if self.rand_batch:
            i = torch.randint(low=1, high=len(self.masks)+1, size=(bs,)) - 1
        else:
            i = (torch.randint(low=1, high=len(self.masks)+1, size=(1,)) - 1 ).expand(bs)

        # Select the mask randomly from the list of masks to learn the denoising score functions.
        mask = self.masks[i.long(), :]
        mask_data = expand_mask(mask, self.var_sizes)
        # Varaibles that are not marginal
        mask_data_marg = (mask_data < 0).float()
        # Varaibles that will be diffused
        mask_data_diffused = mask_data.clamp(0, 1)         # clip(0, 1)

        x_t, Z, _, _ = self.sample(x_0=x_0, t=t)
        x_t = mask_data_diffused * x_t + (1 - mask_data_diffused) * x_0
        x_t = x_t * (1 - mask_data_marg)+  torch.zeros_like(x_0, device=self.device) *mask_data_marg

        score = score_net(x_t, t=t, mask=mask, std=None) * mask_data_diffused
        Z = Z * mask_data_diffused

        #Score matching of diffused data reweithed proportionnaly to the size of the diffused data.
        total_size = score.size(1)
        n_diff=torch.sum(mask_data_diffused, dim=1)
        weight = (((total_size - n_diff) / total_size) + 1).view(bs, 1)
        loss = (weight * (torch.pow(score - Z, 2))).sum(1, keepdim=False)/n_diff

        return loss
    
    def get_masks_training(self):
        """
        Returns a list of masks each corresponds to a score function needed to compute MI.
        
        
        """
        if self.type=="c":
            masks= np.array([[1,-1],[1,0]]) 
        elif self.type=="j":
            masks= np.array([[1,1],[1,0],[0,1]])  
        
        if self.weight_s_functions:
            return torch.tensor(self.weight_masks(masks), device=self.device)
        else:
            return  torch.tensor(masks, device=self.device)


    def weight_masks(self, masks):
        """ Weighting the mask list so the more complex score functions are picked more often durring the training step. 
        This is done by duplicating the mask with the list of masks.
        """
        masks_w = []
  
        #print("Weighting the scores to learn ")
        for s in masks:
                nb_var_inset = np.sum(s == 1)
                for i in range(nb_var_inset):
                    masks_w.append(s)
        np.random.shuffle(masks_w)
        return np.array(masks_w)

    def sample_importance_sampling_t(self, shape):
        """
        Non-uniform sampling of t to importance_sampling. See [1,2] for more details.
        [1] https://arxiv.org/abs/2106.02808
        [2] https://github.com/CW-Huang/sdeflow-light
        """
        return sample_vp_truncated_q(shape, self.beta_min, self.beta_max, T=self.T)