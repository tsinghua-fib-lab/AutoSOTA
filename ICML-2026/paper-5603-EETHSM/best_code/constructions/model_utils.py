import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSSM(nn.Module):
    """
    Minimal discrete-time linear SSM:
        x_{t+1} = A x_t + B u_t
        y_t = C x_t
    """

    def __init__(self, d_model, d_state):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state

        # Learnable parameters
        self.A = nn.Parameter(torch.randn(d_model, d_state, d_state) * 0.01)
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.Cb = nn.Parameter(torch.zeros(d_model, d_state))
        self.Delta = nn.Parameter(torch.randn(d_model) * 0.01)

        self.Wo = nn.Linear(d_model, d_model, bias=False)

        self.h0 = nn.Parameter(torch.zeros(d_state, d_model))

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        returns: (batch, seq_len, d_model)
        """

        B, T, _ = x.shape
        h = self.h0.unsqueeze(0).expand(B, -1, -1)  # (batch, d_state, d_model)

        outputs = []

        for t in range(T):
            xt = x[:, t]                         # (batch, d_model)
            Delta = xt @ self.Delta
            # edA = torch.exp(-torch.einsum('b,aij->baij', Delta, self.A))
            edA = torch.einsum('b,aij->baij', 1-Delta, torch.eye(self.d_state, device=x.device).unsqueeze(0)) \
                + torch.einsum('b,aij->baij', Delta, self.A)  # (batch, d_model, d_state, d_state)
            B = xt @ self.B
            # B = torch.einsum('ba,ak->bk', xt, self.B)  # (batch, d_model, d_state)
            C = xt @ self.C

            # print(edA.shape, h.shape, xt.shape, Delta.shape, B.shape)

            h = torch.einsum('bja,bakj->bka', h, edA) + torch.einsum('b,ba,bk->bka', Delta, xt, B) # state update
            # h = h @ self.A.T + xt @ self.B.T     # state update
            # print(h.shape, self.Cb.shape)
            yt = torch.einsum('bka,bk->ba', h, C) + torch.einsum('bka,ak->ba', h, self.Cb)   # output
            # yt = h @ self.C.T                    # output
            outputs.append(yt)
            # if Delta == 1:
            #     print("Next is number!")
                # print(torch.max(torch.abs(xt - h)))
            # print(h[0,0])
            # print(torch.max(edA), torch.max(torch.einsum('baj,akj->bak', h, edA)), torch.max(torch.einsum('b,bk,ba->bak', Delta, xt, B)), torch.max(yt), torch.max(torch.abs(h)))

            # print(yt[0, 25:30])
        # print(torch.stack(outputs, dim=1)[0, -1, 25:30])
        return self.Wo(torch.stack(outputs, dim=1))
    
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        B, T, D = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape for heads
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)

        # causal mask
        mask = torch.tril(torch.ones(T, T, device=x.device))
        # mask = torch.tril(torch.ones(T, T, device=x.device)).T
        scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out), attn
    
class SSMTransformerBlock(nn.Module):
    def __init__(self, d_model, d_state, n_heads, d_ff, layer_type):
        super().__init__()

        self.layer_type = layer_type

        if layer_type == 'SSM':
            self.layer = SimpleSSM(d_model, d_state)
        elif layer_type == 'TF':
            self.layer = CausalSelfAttention(d_model, n_heads)

    def forward(self, x):
        out = self.layer(x)
        if self.layer_type == 'SSM':
            x = x + out
        elif self.layer_type == 'TF':
            x = x + out[0]

        return x

class SSMTransformer(nn.Module):
    def __init__(self, num_vocab, d_model, d_state, n_heads, d_ff, layers):
        super().__init__()

        self.embedding = nn.Embedding(num_vocab, d_model)

        self.pos_emb = nn.Parameter(torch.randn(1, 100, d_model))

        self.layers = nn.ModuleList([
            SSMTransformerBlock(d_model, d_state, n_heads, d_ff, layer)
            for layer in layers
        ])

        self.lm_head = nn.Linear(d_model, num_vocab)

    def forward(self, x):
        x = self.embedding(x.long()) + self.pos_emb

        for layer in self.layers:
            x = layer(x)
        
        return self.lm_head(x)
