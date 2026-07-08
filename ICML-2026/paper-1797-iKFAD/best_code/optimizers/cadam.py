import torch
import numpy as np
from torch.optim import Optimizer

class Cadam(Optimizer):
    def __init__(self, params, h=0.001, gamma=1, c=1, alpha=1, eps=1e-8, device='cpu'):
        if h < 0: raise ValueError(f"Invalid h: {h}")
        
        defaults = dict(h=torch.tensor([h], device=device), 
                        gamma=torch.tensor([gamma], device=device), 
                        c=torch.tensor([c], device=device), 
                        alpha=torch.tensor([alpha], device=device), 
                        eps=torch.tensor([eps], device=device))   
        super(Cadam, self).__init__(params, defaults)  
        self.t = 0
        self.initialized = False
        self.device = device

    @torch.no_grad()
    def initialize_state(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                state = self.state[p]
                state['momentum_buffer'] = torch.zeros_like(p.data, device=self.device)
                state['exp_avg_sq']      = torch.zeros_like(p.data, device=self.device)
        self.initialized = True

    @torch.no_grad()
    def A_step(self):
        for group in self.param_groups:
            h = group["h"].item() 
            eps = group["eps"]
            for q in group["params"]:
                if q.grad is None: continue
                param_state = self.state[q]
                p_buf = param_state["momentum_buffer"]
                zeta_buf = param_state["exp_avg_sq"]
                denom = zeta_buf.sqrt().add_(eps)
                q.addcdiv_(p_buf, denom, value=h)

    @torch.no_grad()
    def B_step(self):
        for group in self.param_groups:
            h = group["h"].item()
            for q in group["params"]:
                if q.grad is None: continue
                param_state = self.state[q]
                p_buf = param_state["momentum_buffer"]
                d_q = q.grad
                p_buf.add_(d_q, alpha=-h)  

    @torch.no_grad()
    def C_step(self):
        for group in self.param_groups:
            h = group["h"].item()                      
            alpha = group["alpha"].item()
            for q in group["params"]:
                if q.grad is None: continue
                param_state = self.state[q]
                zeta_buf = param_state["exp_avg_sq"]
                F = -torch.clone(q.grad).detach()
                expnt = np.exp(-alpha * h)
                zeta_buf.mul_(expnt).addcmul_(F, F, value=(1-expnt)/alpha)
                
    @torch.no_grad()
    def D_step(self):
        for group in self.param_groups:
            h = group["h"]                   
            c = group["c"]
            for q in group["params"]:
                if q.grad is None: continue
                param_state = self.state[q]
                p_buf = param_state["momentum_buffer"]
                aux = 2*c*h*p_buf.pow(2) + 1
                denom = aux.sqrt()
                p_buf.div_(denom)
                
    @torch.no_grad()
    def E_step(self):
        for group in self.param_groups:
            h = group["h"]          
            gamma = group["gamma"]
            for q in group["params"]:
                if q.grad is None: continue
                param_state = self.state[q]
                p_buf = param_state["momentum_buffer"]
                k = torch.exp(-gamma * h)                     
                p_buf.mul_(k)
       
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if not self.initialized:
            self.initialize_state()
            
        # Internal calls instead of arguments
        self.B_step()        
        self.C_step()
        self.A_step()
        self.D_step()
        self.E_step()
        
        return loss
    
