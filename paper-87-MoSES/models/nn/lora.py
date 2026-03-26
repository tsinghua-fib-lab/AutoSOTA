import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.nn.transformer import RMSNorm


class LoRALayer(nn.Module):
    def __init__(self, input_dim, output_dim, rank, alpha=1.0, use_linear=False):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.alpha = alpha
        self.use_linear = use_linear
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())

        if not use_linear:
            self.A = torch.nn.Parameter(torch.randn(input_dim, rank) * std_dev)
            self.B = torch.nn.Parameter(torch.zeros(rank, output_dim))
        else:
            self.weight = torch.nn.Parameter(torch.zeros(input_dim, output_dim))
        
    def forward(self, x, *args, **kwargs):
        if not self.use_linear:
            output = self.alpha * (x @ self.A @ self.B)
        else:
            output = self.alpha * (x @ self.weight)
        return output




class LoRANorm(nn.Module):
    def __init__(self, dim: int, alpha: float = 1.0, eps: float = 1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.alpha = alpha
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return self.alpha * output * self.weight





class GateLayer(nn.Module):
    def __init__(self, input_dim, ouptut_dim=1, act_func=None):
        super(GateLayer, self).__init__()
        self.gate_weight = torch.nn.Parameter(torch.zeros(input_dim, ouptut_dim))

        if act_func == 'sigmoid':
            self.act_func = torch.nn.Sigmoid()
        elif act_func == None:
            self.act_func = torch.nn.Identity()
        else:
            raise NotImplementedError

    def forward(self, x):
        output = self.act_func(x @ self.gate_weight)
        return output



class LinearWithLoRA(nn.Module):
    def __init__(self, linear_layer, rank=64, alpha=1.0, use_gate=True, act_func='sigmoid', use_linear=False):
        super(LinearWithLoRA, self).__init__()
        self.linear_layer = linear_layer
        self.use_gate = use_gate

        if isinstance(linear_layer, nn.Linear):
            input_dim = linear_layer.in_features
            output_dim = linear_layer.out_features
            self.lora_layer = LoRALayer(input_dim, output_dim, rank, alpha, use_linear=use_linear)

        elif isinstance(linear_layer, RMSNorm):
            assert linear_layer.weight.dim() == 1
            input_dim = linear_layer.weight.size(0)
            output_dim = input_dim
            self.lora_layer = LoRANorm(input_dim, alpha)

        if use_gate:
            self.gate_layer = GateLayer(input_dim, act_func=act_func)
        else:
            self.gate_layer = None


    def forward(self, x, *args, **kwargs):
        if self.use_gate:
            output = self.gate_layer(x) * self.linear_layer(x, *args, **kwargs) + self.lora_layer(x)
        else:
            output = self.linear_layer(x, *args, **kwargs) + self.lora_layer(x)
        return output



class GatedMultiLoRALayer(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 rank,
                 alpha,
                 act_func,
                 n_experts,
                 top_k,
                 temperature,
                 linear_layer,
                 use_trainable_layer: bool = False,
                 use_dynamic_topK: bool = False,
                 use_basis_variants: bool = False,
                 use_linear: bool=False,
                 use_basis_variants_as_input: bool=False,
                 ):
        super(GatedMultiLoRALayer, self).__init__()

        # 1. n_experts: the number of fixed lora layers
        # 2. top_k: the number of selected lora layers from fixed lora layers
        # 3. use_trainable_layer: add an additional trainable layer
        # 4. use_dynamic_topK: when use_dynamic_topK=True, some basis variants are selected and the number equals to the number of basis variants present in the given variant
        # 5. use_basis_variants: when use_dynamic_topK=True and use_basis_variants=True, basis variants present in the given variant are selected.

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act_func = act_func
        self.n_experts = n_experts
        self.top_k = top_k
        self.temperature = temperature
        self.linear_layer = linear_layer
        assert self.top_k <= self.n_experts

        self.use_trainable_layer = use_trainable_layer
        self.use_dynamic_topK = use_dynamic_topK
        self.use_basis_variants = use_basis_variants
        self.use_basis_variants_as_input = use_basis_variants_as_input
        self.basis_variant_binary_vector = None

        if isinstance(linear_layer, nn.Linear):
            self.lora_layers = nn.ModuleList(
                [
                    LoRALayer(input_dim, output_dim, rank[_], alpha, use_linear=use_linear)
                    for _ in range(n_experts if not use_trainable_layer else n_experts + 1)
                ]
            )

        elif isinstance(linear_layer, RMSNorm):
            self.lora_layers = nn.ModuleList(
                [
                    LoRANorm(input_dim, alpha=alpha)
                    for _ in range(n_experts if not use_trainable_layer else n_experts + 1)
                ]
            )
        
        if not use_basis_variants_as_input:
            self.gate_layers = nn.ModuleList([GateLayer(input_dim) for _ in range(n_experts + 1)])
        else:
            self.gate_layers = nn.ModuleList([GateLayer(input_dim + n_experts) for _ in range(n_experts + 1)])


    def normalize_logits(self, logits, dim):
        if self.act_func == 'softmax':
            output = F.softmax(logits / self.temperature, dim=dim)
        elif self.act_func == 'softplus':
            output = F.softplus(logits / self.temperature)
            output = output / output.sum(dim=dim, keepdim=True)
        elif self.act_func == 'sigmoid':
            output = F.sigmoid(logits)
        else:
            raise NotImplementedError
        return output

    def wrap_basis_variant_binary_vector(self, x, basis_variant_binary_vector):
        assert basis_variant_binary_vector.size() == (x.size(0), self.n_experts)

        if len(basis_variant_binary_vector.size()) != len(x.size()):
            assert len(basis_variant_binary_vector.size()) < len(x.size())

            basis_variant_binary_vector = basis_variant_binary_vector[:, None, :].expand(
                x.size(0), x.size(1), self.n_experts
            )
            # (n_nodes, n_experts)
            basis_variant_binary_vector = basis_variant_binary_vector.reshape(-1, self.n_experts)
        return basis_variant_binary_vector

    def reset_basis_variant_binary_vector(self):
        self.basis_variant_binary_vector = None

    def forward(self, x, *args, **kwargs):
        # x: (batch, graph, input_dim)

        # (batch, graph, output_dim)
        base_output = self.linear_layer(x, *args, **kwargs)

        basis_variant_binary_vector = self.basis_variant_binary_vector
        basis_variant_binary_vector = self.wrap_basis_variant_binary_vector(x, basis_variant_binary_vector)

        output_shape = list(x.size()[:-1]) + [self.output_dim]
        n_nodes = np.prod(x.size()[:-1])

        # (n_nodes, embed_dim)
        x = x.reshape(-1, self.input_dim) if x.dim() != 2 else x
        # (n_nodes, output_dim)
        base_output = base_output.reshape(-1, self.output_dim) if base_output.dim != 2 else base_output

        # (n_nodes, n_experts, output_dim)
        outputs = torch.stack([lora_layer(x) for lora_layer in self.lora_layers[:self.n_experts]]).permute(1, 0, 2)
        # (n_nodes, 1+n_experts, output_dim)
        outputs = torch.cat((base_output[:, None, :], outputs), dim=1)

        # (n_nodes, 1+n_experts)
        if self.use_dynamic_topK:
            gates = self.dynamic_top_k_gating(x, basis_variant_binary_vector)
        else:
            gates = self.static_top_k_gating(x, top_k=self.top_k, basis_variant_binary_vector=basis_variant_binary_vector)

        # (n_nodes, output_dim)
        if self.use_trainable_layer:
            outputs = (outputs * gates[..., None]).sum(1) + self.lora_layers[self.n_experts](x)
        else:
            outputs = (outputs * gates[..., None]).sum(1)

        return outputs.reshape(output_shape)


    def dynamic_top_k_gating(self, x, basis_variant_binary_vector):
        # x: (n_nodes, input_dim)
        # basis_variant_binary_vector: (n_nodes, n_experts)

        if self.use_basis_variants_as_input:
            x = torch.cat((x, basis_variant_binary_vector), dim=-1)

        # (n_nodes, 1+n_experts)
        logits = torch.cat([gate_layer(x) for gate_layer in self.gate_layers], dim=1)

        # (n_nodes, 1): [0, 4]
        top_k = basis_variant_binary_vector.sum(-1, keepdim=True).to(torch.int64)
        assert basis_variant_binary_vector.size() == (x.size(0), self.n_experts)
        assert (logits.size(-1) - 1 >= top_k).all() and logits.size(-1) == self.n_experts + 1

        # (n_nodes, 1), (n_nodes, n_experts)
        base_logits, lora_logits = logits[:, :1], logits[:, 1:self.n_experts + 1]

        if self.use_basis_variants:
            # (n_nodes, n_experts)
            lora_logits[~basis_variant_binary_vector.bool()] = float("-inf")
        else:
            # (n_nodes, n_experts)
            sorted_lora_logits = lora_logits.sort(dim=-1, descending=True)[0]
            # (n_nodes, 1)
            threshold_index = torch.clamp(top_k - 1, 0, self.n_experts - 1)
            threshold = sorted_lora_logits.gather(index=threshold_index, dim=-1)
            # (n_nodes, n_experts)
            lora_logits[lora_logits < threshold] = float("-inf")
            lora_logits[(top_k == 0).expand_as(lora_logits)] = float("-inf")

        logits = torch.cat((base_logits, lora_logits), dim=-1)
        gates = self.normalize_logits(logits, dim=-1)
        return gates



    def static_top_k_gating(self, x, top_k, basis_variant_binary_vector):
        # x: (n_nodes, input_dim)

        if self.use_basis_variants_as_input:
            x = torch.cat((x, basis_variant_binary_vector), dim=-1)

        # (n_nodes, 1+n_experts)
        logits = torch.cat([gate_layer(x) for gate_layer in self.gate_layers], dim=1)
        assert logits.size(-1) - 1 >= top_k and logits.size(-1) == self.n_experts + 1

        # (n_nodes, 1), (n_nodes, n_experts)
        base_logits, lora_logits = logits[:, :1], logits[:, 1:self.n_experts + 1]

        # (n_nodes, top_k)
        top_logits, top_indices = lora_logits.topk(top_k, dim=-1)

        zeros = torch.zeros_like(lora_logits, requires_grad=True)
        lora_logits = zeros.scatter(dim=-1, index=top_indices, src=top_logits)

        logits = torch.cat((base_logits, lora_logits), dim=-1)
        gates = self.normalize_logits(logits, dim=-1)
        return gates








class LinearWithGatedMultiLoRA(nn.Module):
    def __init__(self,
                 linear_layer,
                 rank=64,
                 alpha=1.0,
                 act_func='softmax',
                 n_experts=4,
                 top_k=4,
                 temperature=1.0,
                 use_trainable_layer=False,
                 use_dynamic_topK=False,
                 use_basis_variants=False,
                 use_linear=False,
                 use_basis_variants_as_input=False,
                 ):

        super(LinearWithGatedMultiLoRA, self).__init__()
        self.linear_layer = linear_layer

        if isinstance(linear_layer, nn.Linear):
            input_dim = linear_layer.in_features
            output_dim = linear_layer.out_features
        elif isinstance(linear_layer, RMSNorm):
            assert linear_layer.weight.dim() == 1
            input_dim = linear_layer.weight.size(0)
            output_dim = input_dim

        self.lora_layer = GatedMultiLoRALayer(
            input_dim=input_dim,
            output_dim=output_dim,
            rank=rank,
            alpha=alpha,
            act_func=act_func,
            n_experts=n_experts,
            top_k=top_k,
            temperature=temperature,
            use_trainable_layer=use_trainable_layer,
            use_dynamic_topK=use_dynamic_topK,
            use_basis_variants=use_basis_variants,
            use_linear=use_linear,
            use_basis_variants_as_input=use_basis_variants_as_input,
            linear_layer=linear_layer,
        )

    def forward(self, x, *args, **kwargs):
        output = self.lora_layer(x, *args, **kwargs)
        return output




