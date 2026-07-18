import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.SiLU()

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.activation(out)
        out = self.fc3(out)
        return out + residual
    

class CondResBlock(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_dim):
        super(CondResBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim+cond_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.SiLU()

    def forward(self, x, cond):
        residual = x
        input = torch.cat([x, cond], dim=-1)  # Concatenate condition
        out = self.fc1(input)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.activation(out)
        out = self.fc3(out)
        return out + residual

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings
    

class ToyModel(nn.Module):
    def __init__(self,
                 betas,
                 alphas,
                 alpha_bars,
                 pred_type='epsilon',  # 'epsilon' or 'v' or 'flow'
                 data_dim=2, 
                 n_layers=5, 
                 n_resblocks=0,
                 hidden_dim=128, 
                 pos_emb_dim=32,
                 t_emb_dim=8,
                 cond_emb_dim=0,
                 cond_drop_prob=0.2,
                 num_classes=0,
                 device=None):
        super().__init__()
        self.device = device
        assert n_layers >= 2, "n_layers must be at least 2"
        assert t_emb_dim % 2 == 0, "t_emb_dim must be even for sinusoidal embeddings"

        self.pred_type = pred_type
        self.betas = betas.to(self.device)
        self.alphas = alphas.to(self.device)
        self.alpha_bars = alpha_bars.to(self.device)

        self.n_layers = n_layers
        self.n_resblocks= n_resblocks
        self.data_dim = data_dim
        self.t_emb_dim = t_emb_dim
        self.cond_emb_dim = cond_emb_dim
        self.all_cond_dim = t_emb_dim + cond_emb_dim
        self.pos_emb_dim = pos_emb_dim
        self.input_dim = data_dim * pos_emb_dim + self.all_cond_dim

        self.hidden_dim = hidden_dim
        self.output_dim = data_dim
        
        self.use_cond = cond_emb_dim > 0
        if self.use_cond:
            self.cond_embedder = LabelEmbedder(num_classes, cond_emb_dim, cond_drop_prob).to(self.device)

        self.layers = []

        if self.n_resblocks == 0:
            # self.layers.append(nn.BatchNorm1d(self.input_dim).to(self.device))
            self.layers.append(nn.Linear(self.input_dim, hidden_dim).to(self.device))
            # self.layers.append(nn.BatchNorm1d(hidden_dim).to(self.device))
            self.layers.append(nn.ReLU().to(self.device))

            for i in range(n_layers-2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim).to(self.device))
                # self.layers.append(nn.BatchNorm1d(hidden_dim).to(self.device))
                self.layers.append(nn.ReLU().to(self.device))

            self.layers.append(nn.Linear(hidden_dim, self.output_dim).to(self.device))
            # self.layers.append(nn.BatchNorm1d(self.output_dim).to(self.device))
        else:
            self.input_layer = nn.Linear(data_dim * pos_emb_dim, hidden_dim).to(self.device)
            # self.layers.append(nn.ReLU().to(self.device))

            for i in range(n_resblocks):
                    self.layers.append(CondResBlock(hidden_dim, self.all_cond_dim, hidden_dim).to(self.device))

            self.output_layer = nn.Linear(hidden_dim, self.output_dim).to(self.device)
            
        self.mlp = nn.Sequential(*self.layers).to(self.device)

    def get_sinusoidal_embedding(self, x, dim):
        """
        Computes sinusoidal time embeddings.
        t: Tensor of shape (batch,), (batch, 1), scalar tensor, or float/int
        Returns: (batch, tembed_dim)
        """
        half_dim = dim // 2
        exponent = torch.arange(half_dim, dtype=torch.float32, device=self.device) / half_dim
        exponent = 10000 ** (-exponent)
        emb = x * exponent  # broadcasting
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # shape: (batch, tembed_dim)

        return emb  # (batch, tembed_dim)
    
    def get_position_embedding(self, x_t):
        x_t_emb = []
        for d in range(self.data_dim):
            x_t_d = x_t[:, d].unsqueeze(-1)  # (batch, 1)
            x_t_d_emb = self.get_sinusoidal_embedding(x_t_d, self.pos_emb_dim)  # (batch, pos_emb_dim)
            x_t_emb.append(x_t_d_emb)
            
        x_t_emb = torch.cat(x_t_emb, dim=-1)
        return x_t_emb

    def forward(self, x_t, t, cond=None, k=None):
        """
        Predict score (i.e., estimated noise) given noisy input x_t and timestep t.
        x_t: (batch, data_dim)
        t:   (batch,) or (batch, 1) or scalar
        Returns: (batch, data_dim)
        """

        if isinstance(t, (int, float)):
            t = torch.full((x_t.shape[0], 1), t, dtype=torch.float32, device=self.device)
        elif isinstance(t, torch.Tensor):
            if t.dim() <= 0:
                t = torch.full((x_t.shape[0], 1), t.item(), dtype=torch.float32, device=self.device)
            elif t.dim() == 1:
                t = t.unsqueeze(1)
            elif t.dim() == 2 and t.shape[1] != 1:
                raise ValueError("Time tensor must have shape (batch,), (batch,1), or be a scalar.")
            t = t.to(self.device).to(torch.float32)
        else:
            raise TypeError("t must be int, float, or torch.Tensor")

        t_emb = self.get_sinusoidal_embedding(t, self.t_emb_dim)  # (batch, t_emb_dim)
        all_cond = t_emb

        if self.use_cond:
            cond_emb = self.cond_embedder(cond, train=self.training)  # (batch, cond_emb_dim)
            # x_in = torch.cat([x_in, cond_emb], dim=-1)  # (batch, data_dim + tembed_dim + cond_emb_dim)
            all_cond = torch.cat([all_cond, cond_emb], dim=-1)  # (batch, tembed_dim + cond_emb_dim)

        # if self.k_emb_dim > 0:
        #     if isinstance(k, (int, float)):
        #         k = torch.full((x_t.shape[0], 1), k, dtype=torch.float32, device=self.device)
        #     elif isinstance(k, torch.Tensor):
        #         if k.dim() <= 0:
        #             k = torch.full((x_t.shape[0], 1), k.item(), dtype=torch.float32, device=self.device)
        #         elif k.dim() == 1:
        #             k = k.unsqueeze(1)
        #         elif k.dim() == 2 and k.shape[1] != 1:
        #             raise ValueError("k tensor must have shape (batch,), (batch,1), or be a scalar.")
        #     else:
        #         raise TypeError("k must be int, float, or torch.Tensor")

        #     k_emb = self.get_sinusoidal_embedding(k, self.kembed_dim)  # (batch, kembed_dim)
        #     # x_in = torch.cat([x_in, k_emb], dim=-1)  # (batch, data_dim + tembed_dim + kembed_dim)
        #     all_cond = torch.cat([all_cond, k_emb], dim=-1)  # (batch, tembed_dim + kembed_dim + cond_emb_dim)

        x_t_emb = self.get_position_embedding(x_t)

        if self.n_resblocks == 0:
            x_in = torch.cat([x_t_emb, all_cond], dim=-1)
            return self.mlp(x_in)
        else:
            intermediate = self.input_layer(x_t_emb)  # (batch, hidden_dim)
            # cond_zeros = torch.zeros_like(all_cond)
            # hidden_zeros = torch.zeros_like(intermediate)
            # intermediate = torch.cat([intermediate, cond_zeros], dim=-1)  # (batch, hidden_dim + all_cond_dim)
            # all_cond_padded = torch.cat([hidden_zeros, all_cond], dim=-1)  # (batch, hidden_dim + all_cond_dim)

            for layer in self.layers:
                # x_in = torch.cat([intermediate, all_cond], dim=-1)
                # x_in = intermediate + all_cond_padded  # (batch, hidden_dim + all_cond_dim)
                intermediate = layer(intermediate, all_cond)
            output = self.output_layer(intermediate)  # (batch, data_dim)

            return output


    def add_noise(self, x_0, t):
        # t = t.long()
        if self.pred_type == 'flow':
            x_1 = torch.randn_like(x_0)
            x_t = (1 - t) * x_0 + t * x_1
            velocity = x_1 - x_0
            return x_t, velocity, x_1
        else:
            sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t]).unsqueeze(-1)
            sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bars[t]).unsqueeze(-1)
            epsilon = torch.randn_like(x_0)
            x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * epsilon
            v = self.alpha_bars[t] ** 0.5 * epsilon - (1 - self.alpha_bars[t]) ** 0.5 * x_0
            if self.pred_type == 'v':
                return x_t, v, epsilon
            elif self.pred_type == 'epsilon':
                return x_t, epsilon, epsilon
            else:
                raise ValueError(f"Unknown prediction type: {self.pred_type}")
    

    def energy_fn(self, x_t, t, k=None):
        model_output = self.forward(x_t, t, k=k)
        energy = torch.sum(model_output**2, dim=-1)
        return energy
    
    def score_fn(self, x_t, t, k=None):
        if self.pred_type == 'epsilon':
            epsilon = self.forward(x_t, t, k=k)
            return - epsilon / ((1 - self.alpha_bars[t]) ** 0.5)
        elif self.pred_type == 'v':
            # Implement the 'v' prediction type
            v = self.forward(x_t, t, k=k)
            epsilon = self.alpha_bars[t]**0.5 * v + (1 - self.alpha_bars[t])**0.5 * x_t
            return - epsilon / ((1 - self.alpha_bars[t]) ** 0.5)
        elif self.pred_type == 'energy':
            # only for inference
            x_t = x_t.requires_grad_()
            energy = self.energy_fn(x_t, t, k=k)
            epsilon = torch.autograd.grad(energy, x_t, create_graph=False)[0]
            return - epsilon / ((1 - self.alpha_bars[t]) ** 0.5)
        elif self.pred_type == 'flow':
            # Implement the 'flow' prediction type
            pass

    