import time 
import numpy as np
import torch
import os
from opt_einsum import contract,  contract_path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def swap(model_str: str, param_idx: int) -> str:
    lhs, rhs = model_str.split('->')
    params = lhs.split(',')
    if param_idx < 0 or param_idx >= len(params):
        raise ValueError(f"Parameter index {param_idx} is out of bounds for the model string '{lhs}'.")
    other_params = [p for i, p in enumerate(params) if i != param_idx]
    new_lhs = ",".join(other_params + [rhs])
    new_rhs = params[param_idx]
    return f"{new_lhs}->{new_rhs}"

class NNEinFact:
    """
    A class for Nonnegative Einsum Factorization usingmultiplicative updates.
    The model is defined by an einsum-style string.
    """
    def __init__(self, model_str: str, shape_dict: dict, device=None, alpha = 1.0, beta=1.0):
        """
        Initializes the factorization model.
        Args:
            model_str (str): An einsum-style string defining the model 
                             (e.g., 'iq,jq,aq,tq,q->ijat' for CP decomposition).
            shape_dict (dict): A dictionary mapping dimension characters (case-sensitive) to their sizes.
            device (str or torch.device, optional): The device to run computations on.
            alpha (float, optional): The alpha parameter for the alpha-beta divergence.
            beta (float, optional): The beta parameter for the alpha-beta divergence.
        """
        self.model_str = model_str 
        self.shape_dict = shape_dict
        self.data_str = model_str.split('->')[1]
        self.param_strs = model_str.split('->')[0].split(',')
        self._validate_inputs()
        self.P_params = []   
        self._initialize_params()
        self.history = None
        self.device = device
        self.contract_paths = [None]*len(self.P_params)
        self.alpha = alpha
        self.beta = beta
        self.P_params = [
            torch.from_numpy(param).to(device=self.device, dtype=torch.float32) 
            if isinstance(param, np.ndarray) 
            else torch.tensor(param, device=self.device, dtype=torch.float32)
            for param in self.P_params
        ]
        self.y_path = contract_path(self.model_str, *self.P_params)[0]
        self.Y_hat = contract(self.model_str, *self.P_params, optimize=self.y_path).clamp(min=1e-10)
        self.einsum_str = [swap(self.model_str, i) for i in range(len(self.P_params))]
        for i in range(len(self.P_params)):
            path = contract_path(self.einsum_str[i], *([self.P_params[j] for j in range(len(self.P_params)) if j != i] + [self.Y_hat]), memory_limit = 1e10)[0]
            self.contract_paths[i] = path

    def _validate_inputs(self):
        assert all(c in self.shape_dict.keys() for c in set(self.data_str)), \
            "All characters in the model string must be in the shape_dict."
        assert all(c in self.shape_dict.keys() for c in set(''.join(self.param_strs))), \
            "All characters in the model string must be in the shape_dict."

    def _initialize_params(self):
        """Initializes model factors using uniform initialization."""
        for param_str in self.param_strs:
            param_shape = tuple(self.shape_dict[c] for c in param_str)
            self.P_params.append(np.random.uniform(0.0, 1.0, size=param_shape))
    
    def _calculate_ab_divergence(self, Y: torch.Tensor, Y_hat: torch.Tensor) -> float:
        """Calculates the beta-divergence, averaged over elements of Y."""
        Y_hat = Y_hat.clamp(min=1e-10)
        if self.alpha == 1.0 and self.beta == 1.0: #mse
            loss = torch.mean((Y - Y_hat) ** 2)
        elif self.alpha != 0 and self.beta == 0.0:
            loss = torch.sum(Y[Y > 0]**self.alpha * torch.log(Y[Y > 0]**self.alpha / Y_hat[Y > 0]**self.alpha))/Y.numel() + torch.mean(Y_hat**self.alpha - Y**self.alpha)
        elif self.alpha == 1.0 and self.beta == -1.0: #itakura-saito
            mask = Y > 0
            loss = torch.mean(Y[mask] / Y_hat[mask] - torch.log(Y[mask] / Y_hat[mask]) - 1) #don't include terms where Y == 0
        else:
            term1 = (Y ** self.alpha * Y_hat ** self.beta) / (self.alpha * self.beta)
            term2 = (Y ** (self.alpha + self.beta)) / (self.beta * (self.alpha + self.beta))
            term3 = (Y_hat ** (self.alpha + self.beta)) / (self.alpha * (self.alpha + self.beta))
            loss = torch.mean(term2 + term3 - term1)
        return loss

    def prepare_masks(self, Y, mask=None):
        assert isinstance(Y, np.ndarray), "Y must be a numpy.ndarray."
        if mask is not None:
            initial_mask_bool = torch.as_tensor(mask, device=self.device, dtype=torch.bool)
        else:
            initial_mask_bool = torch.ones(Y.shape, device=self.device, dtype=torch.bool)
        
        assert initial_mask_bool.shape == Y.shape, "Mask shape must match the shape of Y."

        # Generate random validation set to evaluate early stopping
        val_selector = torch.rand(initial_mask_bool.shape, device=self.device) < 0.05

        # Create mutually exclusive boolean masks
        self.val_mask_bool = initial_mask_bool & val_selector
        self.mask_bool = initial_mask_bool & ~val_selector
        self.heldout_mask_bool = ~initial_mask_bool

        # Create float versions for computation
        self.mask = self.mask_bool.to(torch.float32)
        self.val_mask = self.val_mask_bool.to(torch.float32)
    
    def fit(self, Y,
              max_iter: int = 200,
              verbose: bool = True, 
              mask=None,
              early_stopping=False):
        '''Trains the tensor factorization model using multiplicative updates.
        Args:
            Y (np.ndarray or sparse.COO): The observed data tensor to be factorized.
            max_iter (int, optional): Maximum number of iterations for training.
            verbose (bool, optional): If True, prints progress during training.
            mask (np.ndarray, optional): A boolean mask (the same size of Y) indicating observed entries in Y.
            early_stopping (bool, optional): If True, enables early stopping based on validation loss.'''
        self.history = {'loss': [], 'heldout_loss': [], 'time': [], 'validation_loss': []}

        self.prepare_masks(Y, mask)
        Y = torch.as_tensor(Y, device=self.device, dtype=torch.float32)

        self.history['loss'].append(self._calculate_ab_divergence(Y[self.mask_bool], self.Y_hat[self.mask_bool]).detach().cpu().numpy())
        self.history['validation_loss'].append(self._calculate_ab_divergence(Y[self.val_mask_bool], self.Y_hat[self.val_mask_bool]).detach().cpu().numpy())
        self.history['heldout_loss'].append(self._calculate_ab_divergence(Y[self.heldout_mask_bool], self.Y_hat[self.heldout_mask_bool]).detach().cpu().numpy())
        self.history['time'].append(0.0)
        start_time = time.time()
        
        if self.alpha == 0 and self.beta == 1:
            gamma_ab = 1.0
        elif self.alpha == 0: #don't set alpha = 0
            raise NotImplementedError("Alpha = 0 not implemented.")
        elif np.abs(self.alpha + self.beta - 1) < 1e-4:
            gamma_ab = 1/self.alpha
        elif 1/self.alpha - 1 - self.beta/self.alpha > -1e-4:
            gamma_ab = 1/(1 - self.beta)
        elif self.beta/self.alpha > 1/self.alpha:
            gamma_ab = 1/(self.alpha + self.beta - 1)
        else:
            gamma_ab = 1/self.alpha

        if self.alpha != 1.0:
            Y_alpha = Y**self.alpha*self.mask
        else:
            Y_alpha = Y*self.mask

        consecutive_increases = 0

        with torch.no_grad():
            for i in range(max_iter):
                for param_idx, param in enumerate(self.P_params):
                    self.Y_hat = contract(self.model_str, *self.P_params, optimize=self.y_path).clamp(1e-10)
                    
                    A = Y_alpha*self.Y_hat**(self.beta-1)
                    B = self.Y_hat**(self.alpha + self.beta - 1)*self.mask
                    
                    if self.alpha == 1.0 and self.beta == -1.0:
                        B *= (Y_alpha > 0).float()
                    others = self.P_params[:param_idx] + self.P_params[param_idx+1:]
                    
                    num = contract(self.einsum_str[param_idx], *others, A, optimize=self.contract_paths[param_idx])
                    den = contract(self.einsum_str[param_idx], *others, B, optimize=self.contract_paths[param_idx]).clamp(1e-10)

                    ratio = (num/den).pow_(gamma_ab).clamp(min=1e-10, max=1e10)
                    param.mul_(ratio)

                #self.Y_hat = contract(self.model_str, *self.P_params, optimize=self.y_path).clamp(1e-10)

                if self.device is not None and 'cuda' in str(self.device):
                    torch.cuda.synchronize()
                current_time = time.time() - start_time
                
                self.history['time'].append(current_time)
                self.history['loss'].append(self._calculate_ab_divergence(Y[self.mask_bool], self.Y_hat[self.mask_bool]).detach().cpu().numpy())
                self.history['validation_loss'].append(self._calculate_ab_divergence(Y[self.val_mask_bool], self.Y_hat[self.val_mask_bool]).detach().cpu().numpy())
                self.history['heldout_loss'].append(self._calculate_ab_divergence(Y[self.heldout_mask_bool], self.Y_hat[self.heldout_mask_bool]).detach().cpu().numpy())
                if self.history['loss'][-1] > self.history['loss'][-2] + 1e-2:
                    print(f"WARNING: Training loss increased by {self.history['loss'][-1] - self.history['loss'][-2]:.6f}", flush=True)

                if verbose:
                    print(f"\n══════════ Iteration {i:03d} ══════════")
                    print(f"Elapsed Time     : {current_time:8.3f} s")
                    print(f"Training Loss     : {self.history['loss'][-1]:.6e}")
                    if early_stopping:
                        print(f"Validation Loss   : {self.history['validation_loss'][-1]:.6e}")
                    if mask is not None:
                        print(f"Held-out Loss     : {self.history['heldout_loss'][-1]:.6e}")
                    print("═══════════════════════════════════════")

                if len(self.history['loss']) > 100 and \
                np.abs(self.history['loss'][-1] - self.history['loss'][-2]) < 1e-6:
                    print("Convergence reached.")
                    break

                if early_stopping:
                    if len(self.history['validation_loss']) >= 2:
                        if self.history['validation_loss'][-1] > self.history['validation_loss'][-2]:
                            consecutive_increases += 1
                        else:
                            consecutive_increases = 0
                        if consecutive_increases >= 5:
                            print("Early stopping.")
                            break

        return self.history
    
    def get_params(self):
        """Returns the fitted parameters as a list of numpy arrays."""
        return [param.detach().cpu().numpy() for param in self.P_params]
