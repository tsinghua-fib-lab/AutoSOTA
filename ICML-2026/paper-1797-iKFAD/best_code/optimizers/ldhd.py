import torch
from torch.optim import Optimizer

class LDHD(Optimizer):
    """
    LDHD (Linearly Dissipative Hamiltonian Dynamics)
    
    1. Hyperparameters stored as 1-element Tensors on 'device'.
    2. Initialization mimics PyTorch SGD (p = grad at step 0).
    3. Splitting scheme: B-D-A (Matches PyTorch mSGD).
    """

    def __init__(self, params, h=0.001, gamma=0.1, device='cuda'):
        if h < 0:
            raise ValueError(f"Invalid h: {h}")
        if gamma < 0:
            raise ValueError(f"Invalid gamma: {gamma}")

        # FORMAT MATCH: Store params as tensors on device, same as CD
        defaults = dict(h=torch.tensor([h], device=device), 
                        gamma=torch.tensor([gamma], device=device))
        
        super(LDHD, self).__init__(params, defaults)
        self.initialized = False
        self.device = device

    @torch.no_grad()
    def initialize_state(self):
        """Matches CD initialization: p_buf = clone(grad)."""
        for group in self.param_groups:
            for q in group['params']:
                if q.grad is None:
                    continue
                d_q = q.grad
                state = self.state[q]
                if 'momentum_buffer' not in state:
                    # Initialize momentum to gradient (standard PyTorch behavior)
                    state['momentum_buffer'] = torch.clone(d_q).detach()
        self.initialized = True

    # ------------------------------------------------------------
    # Splitting steps (BDA Scheme)
    # ------------------------------------------------------------

    @torch.no_grad()
    def B_step(self):
        """B: momentum update p <- p - h * grad f(x)"""
        for group in self.param_groups:
            h = group['h'].item() # Access as item for calculation
            for q in group['params']:
                if q.grad is None:
                    continue
                state = self.state[q]
                p_buf = state['momentum_buffer']
                g = q.grad
                p_buf.add_(g, alpha=-h)

    @torch.no_grad()
    def D_step(self):
        """D: linear damping p <- exp(-gamma * h) * p"""
        for group in self.param_groups:
            h = group['h']      # Tensor access to match format
            gamma = group['gamma']
            
            # Calculate decay factor
            decay = torch.exp(-gamma * h)
            
            for q in group['params']:
                if q.grad is None:
                    continue
                state = self.state[q]
                p_buf = state['momentum_buffer']
                p_buf.mul_(decay)

    @torch.no_grad()
    def A_step(self):
        """A: position update x <- x + h * p"""
        for group in self.param_groups:
            h = group['h'].item()
            for q in group['params']:
                if q.grad is None:
                    continue
                state = self.state[q]
                p_buf = state['momentum_buffer']
                q.add_(p_buf, alpha=h)

    # ------------------------------------------------------------
    # Integration step
    # ------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single B->D->A step to match mSGD correspondence.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if not self.initialized:
            self.initialize_state()

        # Execute BDA splitting [cite: 39]
        self.B_step()
        self.D_step() # Swapped order (was A->D, now D->A)
        self.A_step() 

        return loss
