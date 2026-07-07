import torch
import numpy as np
import torch.nn as nn

import optimizer
from estimators.minde.diffusion import VP_SDE
from estimators.layers.UnetMLP import UnetMLP_simple

from estimators.minde.diff_utils import concat_vect, deconcat, marginalize_data, cond_x_data, EMA
from estimators.minde.info_measures import mi_cond, mi_cond_sigma, mi_joint, mi_joint_sigma

class MINDE(nn.Module):
    """ 
        Mutual information neural estimate
    """
    def __init__(self, architecture_encoder_x, architecture_encoder_y, architecture_critic, hyperparams, var_list=None):
        super().__init__()

        if var_list ==None:
            var_list = {"x" + str(i): hyperparams.dim for i in range(2)}
        self.var_list = list(var_list.keys())
        self.sizes = list(var_list.values())
        
        # hyperparameters
        self.bs = 250 if not hasattr(hyperparams, 'bs') else hyperparams.bs 
        self.lr = 1e-3                                                         # <-- following original MINDE paper, overriding lr
        self.wd = 0e-5                                                         # <-- following original MINDE paper, overriding wd
        self.max_iteration = 1500 if not hasattr(hyperparams, 'max_iteration') else hyperparams.max_iteration
        self.early_stop = True if not hasattr(hyperparams, 'early_stop') else hyperparams.early_stop
        self.t_patience = 500 if not hasattr(hyperparams, 't_patience') else hyperparams.t_patience
        self.device = 'cuda:0' if not hasattr(hyperparams, 'device') else hyperparams.device

        self.type = 'c' if not hasattr(hyperparams, 'type') else hyperparams.type
        self.arch = 'mlp' if not hasattr(hyperparams, 'arch') else hyperparams.arch
        self.sigma = 1.0 if not hasattr(hyperparams, 'sigma') else hyperparams.sigma
        self.mc_iter = 100 if not hasattr(hyperparams, 'mc_iter') else hyperparams.mc_iter
        self.importance_sampling = True if not hasattr(hyperparams, 'importance_sampling') else hyperparams.importance_sampling
        self.use_ema = True if not hasattr(hyperparams, 'use_ema') else hyperparams.use_ema
        

        if hasattr(hyperparams, 'hidden_dim')==False or hyperparams.hidden_dim == None:
            hidden_dim = self.calculate_hidden_dim()
        else:
            hidden_dim = hyperparams.hidden_dim

        self.score = UnetMLP_simple(dim=np.sum(self.sizes), init_dim=hidden_dim, dim_mults=[],
                                        time_dim=hidden_dim, nb_var=len(var_list.keys()))
        
        self.sde = VP_SDE(
            importance_sampling=self.importance_sampling,
            var_sizes=self.sizes,
            device=self.device,
            type =self.type
        )
        self.model_ema = EMA(self.score, decay=0.999) if self.use_ema else None
        print('use ema:', self.use_ema, 'bs:', self.bs)

    def objective_func(self, x, y):
        batch = [x, y]
        loss = self.sde.train_step(batch, self.score_forward).mean()
        if self.model_ema and self.training:
            self.model_ema.update(self.score)
        return -loss    # objective function is negative loss

    def learn(self, x, y):
        return optimizer.NNOptimizer.learn(self, x, y)
    
    def MI(self, x, y, eps=1e-5):
        self.eval()

        data = {'x0': x, 'x1': y}
        var_list = list(data.keys())
        data_0 = {x_i: data[x_i].to(self.device) for x_i in var_list}
        z_0 = concat_vect(data_0)

        N = len(self.sizes)
        M = z_0.shape[0]

        mi = []
        mi_sigma = []
        marg_masks, cond_mask = self.get_masks(var_list)

        for i in range(self.mc_iter):
            # Sample t
            if self.importance_sampling:
                t = (self.sde.sample_importance_sampling_t(
                    shape=(M, 1))).to(self.device)
            else:
                t = ((self.sde.T - eps) * torch.rand((M, 1)) + eps).to(self.device)
            _, g = self.sde.sde(t)
            # Sample from the SDE (pertrbe the data with noise at time)
            z_t, _, mean, std = self.sde.sample(z_0, t=t)
            
            std_w = None if self.importance_sampling else std 
            z_t = deconcat(z_t, self.var_list, self.sizes)
            
            if self.type =="c":
                s_marg, s_cond = self.infer_scores(z_t,t, data_0, std_w, marg_masks, cond_mask)
                
                mi.append(
                    mi_cond(s_marg=s_marg,s_cond=s_cond,g=g,importance_sampling=self.importance_sampling)
                )
                mi_sigma.append(
                     mi_cond_sigma(s_marg=s_marg,s_cond=s_cond,
                                   g=g,mean=mean,std=std,x_t= z_t[self.var_list[0]],sigma=self.sigma,
                                   importance_sampling=self.importance_sampling)
                )
                
            elif self.type=="j":
                s_joint, s_cond_x,s_cond_y = self.infer_scores(z_t,t, data_0, std_w, marg_masks, cond_mask)
                mi.append(
                    mi_joint(s_joint=s_joint,
                                    s_cond_x=s_cond_x,
                                    s_cond_y=s_cond_y,g=g,importance_sampling=self.importance_sampling)
                )
                mi_sigma.append(
                     mi_joint_sigma(s_joint=s_joint,
                                    s_cond_x=s_cond_x,
                                    s_cond_y=s_cond_y,
                                    x_t= z_t[self.var_list[0]],
                                    y_t=z_t[self.var_list[1]] ,
                                    g=g,mean=mean,std=std,
                                    sigma=self.sigma,
                                    importance_sampling=self.importance_sampling)
                )
            
        ret = np.mean(mi), np.mean(mi_sigma)
        return ret[0]
    
    def infer_scores(self,z_t,t, data_0, std_w,marg_masks,cond_mask):
        

        with torch.no_grad():
            if self.type=="c":
                
                marg_x = concat_vect(marginalize_data(z_t, self.var_list[0],fill_zeros=True))
                cond_x = concat_vect(cond_x_data(z_t, data_0, self.var_list[0]))
                
                s_marg = - self.score_inference(marg_x, t=t, mask=marg_masks[self.var_list[0]], std=std_w).detach()
                s_cond = - self.score_inference(cond_x, t=t, mask=cond_mask[self.var_list[0]], std=std_w).detach()
                return deconcat(s_marg,self.var_list,self.sizes)[self.var_list[0]] , deconcat(s_cond,self.var_list,self.sizes)[self.var_list[0]]
                
            elif self.type=="j":
                
                s_joint = - self.score_inference( concat_vect(z_t), t=t, std=std_w, mask=torch.ones_like(marg_masks[self.var_list[0]])).detach()
                
                cond_x = concat_vect(cond_x_data(z_t, data_0, self.var_list[0]))
                cond_y = concat_vect(cond_x_data(z_t, data_0, self.var_list[1]))
                
                s_cond_x = - self.score_inference( cond_x, t=t, mask=cond_mask[self.var_list[0]], std=std_w).detach() ##S(X|Y)
                s_cond_y = - self.score_inference( cond_y, t=t, mask=cond_mask[self.var_list[1]], std=std_w).detach() ##S(Y|X)
                
                return s_joint,deconcat(s_cond_x,self.var_list,self.sizes)[self.var_list[0]], deconcat(s_cond_y,self.var_list,self.sizes)[self.var_list[1]]
            
    def score_inference(self, x, t=None, mask=None, std=None):
        """
        Perform score inference on the input data.

        Args:
            x (torch.Tensor): Concatenated variables.
            t (torch.Tensor, optional): The time t. 
            mask (torch.Tensor, optional): The mask data.
            std (torch.Tensor, optional): The standard deviation to rescale the network output.

        Returns:
            torch.Tensor: The output score function (noise/std) if std !=None , else return noise .
        """
        # Get the model to use for inference, use the ema model if use_ema is set to True

        score = self.model_ema.module if self.use_ema else self.score
        with torch.no_grad():
            score.eval()
            
            if self.arch == "mlp":
                t = t.expand(t.shape[0],mask.size(-1)) 
          
                marg = (- mask).clamp(0, 1) ## max <0 
                cond = 1 - (mask.clamp(0, 1)) - marg  ##mask ==0
             
                t = t * (1- cond)  + 0.0 * cond
                t = t* (1-marg) + 1 * marg

                return score(x, t=t, std=std)

    def score_forward(self, x, t=None, mask=None, std=None):
        """
        Perform score inference on the input data.

        Args:
            x (torch.Tensor): Concatenated variables.
            t (torch.Tensor, optional): The time t. 
            mask (torch.Tensor, optional): The mask data.
            std (torch.Tensor, optional): The standard deviation to rescale the network output.

        Returns:
            torch.Tensor: The output score function (noise/std) if std !=None , else return noise .
        """

        if self.arch == "mlp":
          
            # MLP network requires the multitime vector
            #t = t.expand(mask.size()) * mask.clamp(0, 1)
            t = t.expand(t.shape[0],mask.size(-1)) 
          
            marg = (- mask).clamp(0, 1) ## max <0 
            cond = 1 - (mask.clamp(0, 1)) - marg  ##mask ==0
             
            t = t * (1- cond)  + 0.0 * cond
            t = t* (1-marg) + 1 * marg

            return self.score(x, t=t, std=std)

    def calculate_hidden_dim(self):
        # return dimensions for the hidden layers
        if self.arch == "mlp":
            dim = np.sum(self.sizes)
            if dim <= 10:
                hidden_dim = 64
            elif dim <= 50:
                hidden_dim = 128
            else:
                hidden_dim = 256
            return hidden_dim

    def get_masks(self, var_list):
        """_summary_
        Returns:
            dict , dict :  marginal masks, conditional masks 
        """
        return {self.var_list[0]: torch.tensor([1,-1]).to(self.device),
                self.var_list[1]: torch.tensor([-1,1]).to(self.device),
                },{self.var_list[0]: torch.tensor([1,0]).to(self.device),
                self.var_list[1]: torch.tensor([0,1]).to(self.device),
                }

    
