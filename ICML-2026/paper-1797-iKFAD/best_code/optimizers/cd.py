import torch
from torch.optim import Optimizer

class cubic_damping_opt(Optimizer):
    def __init__(self, params, h=0.001, gamma=1, c=1, device='cpu'):
        if h < 0: raise ValueError(f"Invalid h: {h}")

        defaults = dict(h=torch.tensor([h], device=device), 
                        gamma=torch.tensor([gamma], device=device), 
                        c=torch.tensor([c], device=device)) 

        super(cubic_damping_opt, self).__init__(params, defaults)  
        self.initialized = False
        self.device = device
        
    @torch.no_grad()
    def initialize_state(self):
        for group in self.param_groups:       
            for q in group["params"]:
                if q.grad is None: continue                   
                d_q = q.grad
                param_state = self.state[q]   
                if "momentum_buffer" not in param_state:
                    param_state["momentum_buffer"] = torch.clone(d_q).detach()
                    param_state["ksi_buffer"] = torch.tensor([0.0], device=self.device)
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
            c = group["c"]
            for q in group["params"]:
                if q.grad is None: continue
                p_buf = self.state[q]["momentum_buffer"]
                aux = 2*c*h*p_buf.pow(2) + 1
                denom = aux.sqrt()
                p_buf.div_(denom)

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