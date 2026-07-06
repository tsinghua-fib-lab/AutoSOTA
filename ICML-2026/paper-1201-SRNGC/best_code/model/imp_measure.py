import torch

class Shapley_Value():
    def __init__(self, individual_effect_only=False, device='cpu'):
        super().__init__()
        # individual_effect_only = True: Jacobian as importance measure
        
        self.device = device
        self.individual_effect_only = individual_effect_only

    def _full_jacobian(self, y, x):
        batch_size, output_dim = y.shape
        input_dim = x.shape[1]
        jac = torch.zeros(batch_size, output_dim, input_dim, device=self.device)
        for i in range(output_dim):
            grad_y = torch.zeros_like(y)
            grad_y[:, i] = 1.0
            grad_x, = torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=grad_y,
                retain_graph=True
            )
            jac[:, i, :] = grad_x
        return jac

    def cal_shapley_value(self, y, x):
        full_jacobian = self._full_jacobian(y, x) # [batch, output_dim, input_dim]

        if self.individual_effect_only:
            return full_jacobian.abs().mean(dim=0)
        else:
            # F-norm on Jacobian
            individual_effect = full_jacobian.pow(2).mean(dim=0)
            jacobian_norm = full_jacobian.abs()
            interaction_effect = torch.einsum('boi,bok->boi', jacobian_norm, jacobian_norm).mean(dim=0)
            return 0.5 * (individual_effect + interaction_effect)

class Layer_Weight:
    def __init__(self, model_type: str):
        super().__init__()
        assert model_type in ["cLSTM", "cMLP"], "model_type must be 'cLSTM' or 'cMLP'"
        self.model_type = model_type

    # --------- cLSTM GC (from weight_ih_l0) ---------
    def _gc_cLSTM(self, model) -> torch.Tensor:
        GC_list = []
        for block in model.blocks:
            # W: [4*hidden, p]
            W = block.lstm.weight_ih_l0
            g = torch.norm(W, dim=0)  # [p]
            GC_list.append(g)

        GC = torch.stack(GC_list, dim=0)  # [p, p]
        
        return GC

    # --------- cMLP GC (from first conv/linear layer) ---------
    def _gc_cMLP(self, model) -> torch.Tensor:
        GC_list = []

        for block in model.blocks:
            W = block.net[0].weight
            g = torch.norm(W, dim=0)
            GC_list.append(g)

        GC = torch.stack(GC_list, dim=0)

        return GC

    # --------- main entry point (like cal_shapley_value) ---------
    def cal_gc(self, model) -> torch.Tensor:
        if self.model_type == "cLSTM":
            return self._gc_cLSTM(model)
        elif self.model_type == "cMLP":
            return self._gc_cMLP(model)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
