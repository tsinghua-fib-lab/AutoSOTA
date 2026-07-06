import torch

class load_sampler:
    def __init__(self, algorithm_name, order, NFE, model, use_ldm, device, rho=7.0, sigma_min=0.002, sigma_max=80.0):
        self.algorithm_name = algorithm_name
        self.order = order
        self.NFE = NFE
        self.model = model
        self.use_ldm = use_ldm
        self.device = device
        self.rho = rho
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.gamma = 0.0  # bidirectional correction weight (0 = disabled)
        self.get_schedule(NFE+1, device, use_ldm, model)
    
    def get_sampler(self):
        if self.algorithm_name == "F-DPMSolver":
            self.sampler = self.Forward_DPMSolver
        elif self.algorithm_name == "DDIM":
            self.order = 1
            self.sampler = self.ODESolver
        elif self.algorithm_name == "DPMSolver":
            self.sampler = self.ODESolver
        elif self.algorithm_name == "UniPC":
            self.sampler = self.UniPC
        else:
            raise ValueError(f"No existing algorithm {self.algorithm_name}!")
        return self.sampler
        
    def Forward_DPMSolver(self, xt, class_labels):
        device = xt.device

        # load model and evaluate
        with torch.no_grad():
            img_size = xt.shape[2]
            n_channel = xt.shape[1]
            x = xt.reshape(xt.shape[0],-1).clone()
            eps_vec = torch.stack([x.clone() for _ in range(self.order)], dim=0)

            for t in range(self.NFE):
                t_start = max(t-self.order+1, 0)
                at_vec = self.alphas_bar[t_start:t+1].flip(0)
                at_next = self.alphas_bar[t+1]
                at = self.alphas_bar[t]
                
                xhat = self.ODESampler_onestep(x, min(t+1, self.order), at_next, at_vec, eps_vec[0:t-t_start+1])
                
                coeff_x = ((1-at_next) / (1-at))**(1/2)
                coeff_x0 = ((1-at_next) / (1-at) * at)**(1/2) - (at_next)**(1/2)
                
                t_vec = self.sigma[t+1] * torch.ones(x.shape[0], device=device, dtype=torch.long)
                eps = self.model.apply_model(xhat.reshape(xhat.shape[0], n_channel, img_size, img_size), t_vec, class_labels, at_next)
                
                # Bidirectional correction: blend forward-value epsilon with current-state epsilon
                # PFDiff-style: gamma * eps_current + (1-gamma) * eps_forward
                if hasattr(self, "gamma") and self.gamma > 0:
                    t_vec_curr = self.sigma[t] * torch.ones(x.shape[0], device=device, dtype=torch.long)
                    eps_current = self.model.apply_model(
                        x.reshape(x.shape[0], n_channel, img_size, img_size),
                        t_vec_curr, class_labels, self.alphas_bar[t]
                    )
                    eps = (1 - self.gamma) * eps + self.gamma * eps_current
                
                x0_pred = (xhat - (1-at_next)**(1/2)*eps)/(at_next)**(1/2)
                
                x = coeff_x * x - coeff_x0 * x0_pred
                for idx in range(self.order-1, 0, -1):
                    eps_vec[idx] = eps_vec[idx-1]
                eps_vec[0] = eps.clone()

            return x.reshape(x.shape[0], n_channel, img_size, img_size)
        
    def ODESolver(self, xt, class_labels):
        device = xt.device

        # load model and evaluate
        with torch.no_grad():
            img_size = xt.shape[2]
            n_channel = xt.shape[1]
            x = xt.reshape(xt.shape[0], -1).clone()
            
            eps_vec = torch.stack([x.clone() for _ in range(self.order)], dim=0)
            xhat = x.clone()
            
            for t in range(self.NFE):
                t_start = max(t-self.order+1, 0)
                at_vec = self.alphas_bar[t_start:t+1].flip(0)
                at_next = self.alphas_bar[t+1]
                
                t_vec = self.sigma[t] * torch.ones(x.shape[0], device=device, dtype=torch.long)
                eps_vec[0] = self.model.apply_model(x.reshape(x.shape[0], n_channel, img_size, img_size), t_vec, class_labels, self.alphas_bar[t])
                
                xhat = self.ODESampler_onestep(x, p=min(t+1, self.order), at_next=at_next, at_vec=at_vec, eps_vec=eps_vec)

                x = xhat.clone()
                for idx in range(self.order-1, 0, -1):
                    eps_vec[idx] = eps_vec[idx-1]
            return x.reshape(x.shape[0], n_channel, img_size, img_size)
        
    def UniPC(self, xt, class_labels):
        device = torch.device(xt.device)

        with torch.no_grad():
            img_size = xt.shape[2]
            n_channel = xt.shape[1]
            x = xt.reshape(xt.shape[0],-1).clone()
            
            eps_vec = torch.stack([torch.zeros_like(x.clone()) for _ in range(self.order)], dim=0)
            eps_vec[0] = x.clone()
            xc = x.clone()
            
            for t in range(self.NFE):
                t_start = max(t-self.order+1, 0)
                at_vec = self.alphas_bar[t_start:t+1].flip(0)
                at_next = self.alphas_bar[t+1]
                
                t_vec = self.sigma[t] * torch.ones(x.shape[0], device=device, dtype=torch.long)
                eps_vec[0] = self.model.apply_model(x.reshape(x.shape[0], n_channel, img_size, img_size), t_vec, class_labels, self.alphas_bar[t])
                
                # UniPC 3rd corrector
                if t > 0:
                    xc = self.ODESampler_onestep(xc, p=min(t+1, self.order), at_next=self.alphas_bar[t], at_vec=at_vec[[1, 0] + list(range(2, at_vec.shape[0]))], eps_vec=eps_vec[[1, 0] + list(range(2, eps_vec.shape[0]))])
                    
                x = self.ODESampler_onestep(xc, p=min(t+1, self.order-1), at_next=at_next, at_vec=at_vec[:self.order-1], eps_vec=eps_vec[:self.order-1])
                
                for idx in range(self.order-1, 0, -1):
                    eps_vec[idx] = eps_vec[idx-1]
                    
            return x.reshape(x.shape[0], n_channel, img_size, img_size)
    
    def get_schedule(self, T, device, use_ldm, model=None):
        if use_ldm:
            self.alphas_bar = model.alphas_cumprod
            step_num = T-1
            t_idx = torch.round(torch.linspace(0, len(self.alphas_bar)-1, step_num+1)).to(device).to(torch.int)
            t_idx = torch.unique(t_idx)
            self.alphas_bar = self.alphas_bar[t_idx].flip(0)
            self.sigma = t_idx.flip(0)

        else:
            t_vec = torch.linspace(0, T-1, T)
            self.sigma = ((self.sigma_max**(1/self.rho) + t_vec/(T-1) * (self.sigma_min**(1/self.rho) - self.sigma_max**(1/self.rho)))**(self.rho)).to(device)
            self.sigma = self.sigma.to(torch.float32)
            self.alphas_bar = 1 / (self.sigma**2+1)
            self.alphas_bar = self.alphas_bar.clamp(min=1e-8, max=1-1e-8)
            self.alphas_bar = self.alphas_bar.to(torch.float32)

    def ODESampler_onestep(self, x, p, at_next=None, at_vec=[], eps_vec=[]):
        device = x.device
        at_vec = at_vec[:p]
        eps_vec = eps_vec[:p]
        lambd = (1/2) * torch.log(at_vec / (1 - at_vec))
        diff = lambd - lambd[0]

        # Order-1 closed-form shortcut: skip Vandermonde solve
        if p == 1:
            delta = (1/2) * torch.log(at_next / (1 - at_next)) - lambd[0]
            delta = delta.clamp(min=-50, max=50)
            phi0 = 1 - torch.exp(-delta)
            Delta = (phi0 * eps_vec[0]).reshape(eps_vec.shape[1], eps_vec.shape[2])
        else:
            exponents = torch.arange(len(at_vec), device=device)
            A = (diff.unsqueeze(1) ** exponents.unsqueeze(0))
            # Tikhonov regularization for numerical stability at order >= 2
            A = A + 1e-8 * torch.eye(p, device=device)
            B = eps_vec.reshape(A.shape[0], -1)
            AinvB = torch.linalg.solve(A, B)

            delta = (1/2) * torch.log(at_next / (1 - at_next)) - lambd[0]
            delta = delta.clamp(min=-50, max=50)
            phi = torch.zeros(AinvB.shape[0], device=device)
            phi[0] = 1 - torch.exp(-delta)
            for i in range(1, AinvB.shape[0]):
                phi[i] = i * phi[i-1] - (delta**i) * torch.exp(-delta)
            Delta = (phi @ AinvB).reshape(eps_vec.shape[1], eps_vec.shape[2])

        x_coef = torch.sqrt(at_next / at_vec[0])
        eps_coef = torch.sqrt(at_next)
        x = x_coef * x - eps_coef * torch.exp(-lambd[0]) * Delta
        return x
