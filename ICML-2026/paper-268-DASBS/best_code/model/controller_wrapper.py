import torch
from torch import nn


class ControllerWrapper(nn.Module):
    def __init__(self, log_discrete_score: callable, inner_controller: nn.Module, corrector: nn.Module,
                 time_interp: callable = lambda t: t):
        r'''
        Wrapper for the reparameterized controller:

        log varPhi = time_interp(t) * log (discrete score / hat_varPhi) + (1 - time_interp(t)) * inner_controller 

        Args:
            log_discrete_score: function to compute the log discrete score of the target distribution
            inner_controller: inner neural network model tilde_varPhi
            corrector: neural network model for hat_varPhi
            time_interp: time interpolation function. By default, t \mapsto t.
        
        Remarks:
            The following identities should hold:
            ```
            id(controller_wrapper.inner_controller) == id(controller)
            id(controller_wrapper.corrector) == id(corrector)
            id(controller_wrapper.parameters()) == id(controller.parameters())
            ``` 
        '''
        super(ControllerWrapper, self).__init__()
        self.log_discrete_score = log_discrete_score
        self.inner_controller = inner_controller
        self.corrector = corrector
        # self.corrector.eval()  # corrector is fixed during controller training
        self.time_interp = time_interp

    # def parameters(self, recurse: bool = True):
    #     """Return parameters of the inner controller"""
    #     return self.inner_controller.parameters(recurse=recurse)

    # def named_parameters(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True):
    #     return self.inner_controller.named_parameters(prefix, recurse, remove_duplicate)

    # @property
    # def is_training(self):
    #     """
    #     Return training status of the inner controller.

    #     Don't use self.training as it would always be True.

    #     TODO: If naively replaces the name of this property by 'training', then an error would occur:
    #     `AttributeError: property 'training' of 'ControllerWrapper' object has no setter`
    #     """
    #     return self.inner_controller.training

    # def train(self, mode: bool = True):
    #     self.training = mode
    #     self.inner_controller.train(mode)
    #     self.corrector.eval()
    #     return self

    # def eval(self):
    #     self.inner_controller.eval()
    #     self.corrector.eval()
    #     return self

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor = None):
        r'''
        Args:
            x: state, shape (B, D), values in range(N)
            t: time, shape (B, 1)
            cond: optional conditioning information
       
        Returns:
            log varPhi(x, t), shape (B, D, N)
        '''
        with torch.no_grad():
            log_target_discrete_score = self.log_discrete_score(x, cond=cond)  # (B, D, N)
            log_hat_varPhi = self.corrector(x, cond=cond)  # (B, D, N)
        sigma_t = self.time_interp(t)[..., None]  # (B, 1, 1)
        tilde_varPhi = self.inner_controller(x, t, cond=cond)  # (B, D, N)
        return sigma_t * (log_target_discrete_score - log_hat_varPhi) + (1 - sigma_t) * tilde_varPhi
