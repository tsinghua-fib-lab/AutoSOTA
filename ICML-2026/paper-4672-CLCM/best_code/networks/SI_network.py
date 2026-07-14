import torch

from networks.network_interface import Network
from networks.layers import BP_layer
from networks.activation_function import ReLU, Softplus, Linear


class SI_network(Network):
    """
    Synaptic Intelligence (SI) network.
    
    Based on Zenke et al. (2017) "Continual Learning through Synaptic Intelligence"
    (ICML 2017).
    
    Key idea: Track the contribution of each parameter to the loss decrease along 
    the entire training trajectory (online), then use this as importance weights.
    
    The importance measure is:
        ω_k = Σ_t g_k(t) * Δθ_k(t)  (path integral of gradient × parameter change)
        Ω_k = ω_k / ((Δθ_k)² + ξ)   (normalized by total displacement)
    
    The regularization loss is:
        L_SI = (c/2) * Σ_k Ω_k * (θ_k - θ*_k)²
    
    where c is the importance hyperparameter and ξ is a damping term.
    """
    
    def __init__(self, config, name="SI_network"):
        super().__init__(BP_layer, ReLU, Linear, config, name)
        
        # SI hyperparameters
        self.importance = config.importance_ewc  # c: regularization strength (reuse ewc param)
        self.damping = getattr(config, 'si_damping', 0.1)  # ξ: damping term
        
        # Storage for SI
        self._omega = {}  # Accumulated importance: Ω (summed across tasks)
        self._theta_star = {}  # Reference parameters after each task: θ*
        self._first_task = True
        
        # Online tracking during training
        self._w = {}  # Running importance for current task: ω
        self._theta_task_start = {}  # Parameters at start of current task
        self._prev_params = {}  # Parameters from previous step (for computing Δθ)
        self._prev_grads = {}  # Gradients from previous step
        self._initialized = False  # Whether tracking has been initialized
        
        self._init_tracking()

    def to(self, *args, **kwargs):
        """Override to() to also move tracking tensors to the correct device."""
        self = super().to(*args, **kwargs)

        # Move all tracking dictionaries to the new device
        device = next(self.parameters()).device
        for dict_attr in [self._omega, self._theta_star, self._w,
                          self._theta_task_start, self._prev_params, self._prev_grads]:
            for key in dict_attr:
                dict_attr[key] = dict_attr[key].to(device)

        return self

    def _init_tracking(self):
        """Initialize tracking variables for all parameters."""
        for n, p in self.named_parameters():
            if p.requires_grad:
                self._w[n] = torch.zeros_like(p)
                self._theta_task_start[n] = p.data.clone()  # Also init task start params
                self._prev_params[n] = p.data.clone()
                self._prev_grads[n] = torch.zeros_like(p)
                
    def _init_task(self):
        """Initialize tracking for a new task."""
        for n, p in self.named_parameters():
            if p.requires_grad:
                # Reset running importance for new task
                self._w[n] = torch.zeros_like(p)
                # Store parameters at task start (for normalization)
                self._theta_task_start[n] = p.data.clone()
                # Reset previous params/grads
                self._prev_params[n] = p.data.clone()
                self._prev_grads[n] = torch.zeros_like(p)

    def backward(self, y):
        """
        Compute loss with SI regularization and track importance online.
        
        The key trick: We accumulate ω at the START of backward() using the 
        gradients and parameter changes from the PREVIOUS step. This way no 
        extra method calls are needed in the training loop.
        """
        # Step 1: Accumulate importance from PREVIOUS step
        # ω += -g_prev * Δθ_prev (negative because we descend the gradient)
        if self._initialized:
            for n, p in self.named_parameters():
                if n in self._prev_grads:
                    delta = p.data - self._prev_params[n]
                    # Accumulate: gradient × parameter change
                    # Using negative gradient since we're doing gradient descent
                    self._w[n] += (-self._prev_grads[n] * delta)
        
        # Step 2: Compute loss with SI regularization
        loss = self.loss_fn(self.y_hat, y)
        if not self._first_task:
            loss += self.si_loss()
        
        # Step 3: Backpropagate
        loss.backward()
        
        # Step 4: Store current parameters and gradients for next iteration
        for n, p in self.named_parameters():
            if p.requires_grad:
                self._prev_params[n] = p.data.clone()
                if p.grad is not None:
                    self._prev_grads[n] = p.grad.clone()
                    
        self._initialized = True

    def si_loss(self):
        """
        Compute SI regularization loss.
        
        Loss = (c/2) * Σ_k Ω_k * (θ_k - θ*_k)²
        
        Where Ω is the accumulated importance and θ* is the reference.
        """
        loss = 0.0
        for n, p in self.named_parameters():
            if n in self._theta_star and n in self._omega:
                loss += torch.sum(self._omega[n] * (p - self._theta_star[n]) ** 2)
        return self.importance * loss

    def complete_task(self, dataloader=None):
        """
        Finalize importance computation after task completion.
        
        Computes normalized importance:
            Ω_k = ω_k / ((θ_final - θ_init)² + ξ)
        
        And accumulates into total importance across tasks.
        
        Args:
            dataloader: Not used for SI (kept for interface compatibility)
        """
        # Store current parameters as reference for next task
        self._theta_star = {
            n: p.data.clone() for n, p in self.named_parameters() if p.requires_grad
        }
        
        if self._first_task:
            # First task: initialize omega with normalized importance
            for n, p in self.named_parameters():
                if p.requires_grad and n in self._w and n in self._theta_task_start:
                    # Compute total parameter displacement during task
                    delta_squared = (p.data - self._theta_task_start[n]) ** 2
                    # Normalize importance by displacement (with damping)
                    self._omega[n] = self._w[n] / (delta_squared + self.damping)
            self._first_task = False
        else:
            # Subsequent tasks: accumulate importance
            for n, p in self.named_parameters():
                if p.requires_grad and n in self._w and n in self._theta_task_start:
                    delta_squared = (p.data - self._theta_task_start[n]) ** 2
                    # Add normalized importance from this task
                    self._omega[n] += self._w[n] / (delta_squared + self.damping)
        
        # Re-initialize tracking for next task
        self._init_task()
        self._initialized = False

    def start_task(self):
        """
        Call this at the start of each task to initialize tracking.
        
        Note: This is called automatically in complete_task() for subsequent tasks,
        but should be called manually before training on the first task if you want
        proper tracking from the very beginning.
        """
        self._init_task()
        self._initialized = False

    def get_importance_stats(self):
        """Return importance statistics for monitoring."""
        if not self._omega:
            return {}
        
        all_omega = torch.cat([o.flatten() for o in self._omega.values()])
        stats = {
            'omega_mean': all_omega.mean().item(),
            'omega_max': all_omega.max().item(),
            'omega_min': all_omega.min().item(),
            'omega_std': all_omega.std().item(),
        }
        
        # Also report current running importance if tracking
        if self._w:
            all_w = torch.cat([w.flatten() for w in self._w.values()])
            stats['w_mean'] = all_w.mean().item()
            stats['w_max'] = all_w.max().item()
            
        return stats