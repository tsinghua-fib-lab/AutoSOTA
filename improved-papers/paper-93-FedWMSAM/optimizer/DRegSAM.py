import torch
import torch.nn.functional as F

class DRegSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid perturbation rate, should be non-negative: {rho}"
        self.max_norm = 10

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(DRegSAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = rho
            group["adaptive"] = adaptive
        self.paras = None
        

    @torch.no_grad()
    def first_step(self, mu, global_mu):
        ### update g with dynamic dual
        for group in self.param_groups:
            for p, _mu_, _global_mu_ in zip(group["params"], mu, global_mu):
                p.requires_grad = True 
                if p.grad is None: 
                    continue
                p.grad.sub_(_mu_ + _global_mu_)

        ### gradient ascent
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-7)
            for p in group["params"]:
                p.requires_grad = True 
                if p.grad is None: 
                    continue
                # original SAM 
                # e_w = p.grad * scale.to(p)
                # ASAM 
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                # climb to the local maximum "w + e(w)"
                p.add_(e_w * 1)  
                self.state[p]["e_w"] = e_w
        
        ### update local dynamic dual
        for group in self.param_groups:
            for p, _mu_, _global_mu_ in zip(group["params"], mu, global_mu):
                p.requires_grad = True 
                if p.grad is None: 
                    continue
                _mu_ += self.state[p]["e_w"] - _global_mu_
        
        return mu
                

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not self.state[p]:
                    continue
                # go back to "w" from "w + e(w)"
                p.sub_(self.state[p]["e_w"])  
                self.state[p]["e_w"] = 0


    def step(self, mu):
        # model.require_backward_grad_sync = False
        # model.require_forward_param_sync = True
        inputs, labels, loss_func, model, global_dual_correction = self.paras
        
        predictions = model(inputs)
        loss = loss_func(predictions, labels)
        self.zero_grad()
        loss.backward()

        mu = self.first_step(mu, global_dual_correction)
        # model.require_backward_grad_sync = True
        # model.require_forward_param_sync = False

        predictions = model(inputs)
        loss = loss_func(predictions, labels)
        self.zero_grad()
        loss.backward()
        
        self.second_step()

        return mu
        
        
    def _grad_norm(self):
        norm = torch.norm(torch.stack([
                        # original SAM
                        # p.grad.norm(p=2).to(shared_device)
                        # ASAM 
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None]), p=2)
        return norm