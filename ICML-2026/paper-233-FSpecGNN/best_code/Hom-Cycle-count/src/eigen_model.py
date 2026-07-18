from . import *
import math

# --------------------------------- EMBEDDING -------------------------------- #

class NodeEmbedding(nn.Module):
    def __init__(self, dim: int, max_dis: int, enc: Optional[nn.Module]):
        super().__init__()
        self.max_dis = max_dis
        self.embed_v = enc and enc(dim)
        self.embed_d = nn.Embedding(max_dis + 1, dim)

    def forward(self, batch):
        x = self.embed_v(batch.x) if self.embed_v else 0
        d = self.embed_d(torch.clamp(batch.d, 0, max=self.max_dis))
        batch.x = x + d
        del batch.d
        return batch


class EdgeEmbedding(nn.Module):
    def __init__(self, dim: int, enc: Optional[nn.Module]):
        super().__init__()
        self.embed = enc and enc(dim)

    def forward(self, message, attrs=None):
        if not self.embed: return F.relu(message)
        return F.relu(message + self.embed(attrs))


class MLP(nn.Module):
    def __init__(self, idim: int, odim: int, hdim: int = None):
        super().__init__()
        hdim = hdim or idim

        self.fc1 = nn.Linear(idim, hdim, bias=True)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hdim, odim, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Sine(nn.Module):
    def forward(self, input):
        return torch.sin(input)

class Cosine(nn.Module):
    def forward(self, input):
        return torch.cos(input)


class BSGL(nn.Module):
    # Bivariate Spectral-Gated Layer
    
    def __init__(self, args, idim, odim, enc=None):
        super().__init__()

        task, _ = args.task
        if task in ["chordal5_31", "chordal5_13", "chordal5_24"]:
            self.enc_E = nn.Sequential(
                nn.Linear(2, idim),
                Sine(),
                nn.Linear(idim, idim),
                Sine(),
                nn.Linear(idim, idim),
            )
        elif task in ["cycle5", "cycle6"]:
            self.freq_layer = nn.Linear(2, idim//2)
            self.enc_E = nn.Sequential(
                nn.Linear(idim, idim),
                nn.GELU(),
                nn.Linear(idim, idim),
            )
        else:
            self.enc_E = nn.Sequential(
                nn.Linear(2, idim),
                nn.GELU(),
                nn.Linear(idim, idim),
                nn.GELU(),
                nn.Linear(idim, idim),
            )


        self.enc = EdgeEmbedding(idim, enc)
        self.eps = nn.Parameter(torch.zeros(1))
        self.linear = nn.Linear(idim, idim)
        self.mlp = MLP(idim, odim)
        
    def forward(self, args, E, U, S, M2, A=None):
        
        # E = E.unsqueeze(-1)                          # [B, N, 1]
        # feat_EE = self.enc_E(E)
        # feat_EE = self.enc_ExE(feat_E)

        Ei = E.unsqueeze(2).expand(-1, -1, E.size(1))  # [B, N, N]
        Ej = E.unsqueeze(1).expand(-1, E.size(1), -1)  # [B, N, N]

        EE = torch.stack([Ei, Ej], dim=-1)             # [B, N, N, 2]

        task, _ = args.task
        if task in ["cycle5", "cycle6"]:
            freq_EE = self.freq_layer(EE)
            feat_EE = torch.cat([torch.sin(freq_EE), torch.cos(freq_EE)], dim=-1)
            feat_EE = self.enc_E(feat_EE)
        else:
            feat_EE = self.enc_E(EE)

        S_linear = self.linear(S)
        S_attr = self.enc(S_linear, A)

        S_attr = S_attr * M2.unsqueeze(-1)
        
        S_in = torch.einsum(
            'bij,bjkd,bkl->bild',
            U.transpose(-1, -2),  # U^T
            S_attr, U
        )
        
        # S_filter = S_in * feat_LE[:, :, None, :] * feat_RE[:, None, :, :]
        S_filter = S_in * feat_EE

        S_filter = S_filter * M2.unsqueeze(-1)

        S_out = torch.einsum(
            'bij,bjkd,bkl->bild',
            U, S_filter,
            U.transpose(-1, -2),  # U^T
        )
 
        S_out = self.mlp(S * (1 + self.eps) + S_out)

        return F.relu(S_out)

class Pool(nn.Module):

    def __init__(self, idim, odim, task, bn, gin=True):
        super().__init__()
        self.task = task
        self.pooling = task != "e" and MLP(idim, idim)
        self.predict = MLP(idim, odim, hdim=2*idim)
        self.eps = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm1d(idim) if bn else nn.Identity()
        self.gin = gin

    def forward(self, S):
        # S [B, N, N, d]
        assert S.ndim == 4
        B, N, _, d = S.size()
        
        if self.task == "e":
            x = S.reshape(B, N*N, d)
        else:
            x = torch.mean(S, dim=2)    # [B, N, d]
            
            if self.gin:
                diag = S.diagonal(dim1=1, dim2=2)
                x = x + (1. + self.eps) * diag  
            x = F.relu(self.bn(self.pooling(x)))
            
            if self.task == "g":
                x = torch.mean(x, dim=1)  # [B, d]

        return self.predict(x)


class Eigen_GNN(nn.Sequential):

    def __init__(self, args, task, odim, bn=True):
        super().__init__()
        
        idim = args.dim_embed
        self.num_layer = args.num_layer
        
        self.spe_encoding = nn.Embedding(6, idim)      # max_dis + 1
        
        self.layers = nn.ModuleList()
        for _ in range(args.num_layer):
            self.layers.append(BSGL(args, idim, idim))

        self.pool = Pool(idim, odim, task, bn=False, gin=False)

    def forward(self, args, X, E, U, S, A, M2):

        # X:  [B, N, d]
        # E:  [B, N]
        # U:  [B, N, N]
        # S:  [B, N, N, d]
        # A:  [B, N, N, d]
        # M2: [B, N, N]

        S = self.spe_encoding(S)
        S = S * M2.unsqueeze(-1)

        for i in range(self.num_layer):
            S = self.layers[i](args, E, U, S, M2, A=None)
            S = S * M2.unsqueeze(-1)

        out = self.pool(S)
        
        return out