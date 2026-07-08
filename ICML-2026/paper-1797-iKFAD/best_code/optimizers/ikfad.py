import torch
from torch.optim import Optimizer

class iKFAD(Optimizer):
    def __init__(self, params, h=0.001, alpha=1, mu=1, gamma=1, device='cpu'):
        if h < 0: raise ValueError(f"Invalid h: {h}")

        defaults = dict(h=torch.tensor([h], device=device), 
                        alph=torch.tensor([alpha], device=device), 
                        mu=torch.tensor([mu], device=device),
                        gamma=torch.tensor([gamma], device=device))

        super(iKFAD, self).__init__(params, defaults)  
        self.initialized = False
        self.device = device

    @torch.no_grad()
    def initialize_state(self):
        for group in self.param_groups:        
            for q in group["params"]:
                if q.grad is None: continue                 
                param_state = self.state[q]   
                if "momentum_buffer" not in param_state:
                    param_state["momentum_buffer"] = torch.zeros_like(q)
                    param_state["ksi_buffer"] = torch.full_like(q, fill_value=1e-6)
        self.initialized = True

    @torch.no_grad()
    def A_step(self):
        for group in self.param_groups:
            h = group["h"].item()
            for q in group["params"]:
                if q.grad is None: continue
                p_buf = self.state[q]["momentum_buffer"]
                q.add_(p_buf, alpha=h)

    @torch.no_grad()
    def B_step(self):
        for group in self.param_groups:
            h = group["h"].item()
            for q in group["params"]:
                if q.grad is None: continue
                p_buf = self.state[q]["momentum_buffer"]
                d_q = q.grad
                p_buf.add_(d_q, alpha=-h)

    @torch.no_grad()
    def C_step(self):
        for group in self.param_groups:
            h = group["h"]
            alph = group["alph"]
            mu = group["mu"]
            for q in group["params"]:
                if q.grad is None: continue
                param_state = self.state[q]
                p_buf = param_state["momentum_buffer"]
                ksi_buf = param_state["ksi_buffer"]
                
                p_buf.mul_(torch.exp(-h * ksi_buf / 2))
                ksi_buf.mul_(torch.exp(-alph*h)).add_( (1-torch.exp(-alph * h)) * p_buf.pow(2) /(mu*alph) ) 
                p_buf.mul_(torch.exp(-h * ksi_buf / 2))

    @torch.no_grad()
    def D_step(self):
        for group in self.param_groups:
            h = group["h"]
            gamma = group["gamma"]
            for q in group["params"]:
                if q.grad is None: continue
                p_buf = self.state[q]["momentum_buffer"]
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
            
        self.B_step()
        self.A_step()
        self.C_step()
        self.D_step()
        
        return loss
        
        





























