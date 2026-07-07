import math
import einops
import torch
import torch.nn.functional as F
from torch import nn

from .target_networks import PolicyTargetNet
from .modules import MLP, ModuleEncoder, ModuleDecoder, ParamLN
from .opt_blocks import get_opt_layer
from .weight_init import InitWeightHypernet


class HyperTr(nn.Module):
    def __init__(
        self,
        ftask_dim,
        dim2,
        dim3,
        dim4,
        num_layers=None,
        use_norm=False,
        weight_dim=None,
        enc_dec_dim=None,
        opt_block_dim=None,
        opt_mid_dim=None,
        num_opt_mlp_layer=None,
        num_enc_dec_layer=None,
        tanh=False,
        target_layers_num=3,
        **kwargs,
    ):
        super().__init__()
        mid_act = F.relu
        self.target_layers_num = target_layers_num
        self.ablation = kwargs.get("ablation", None)
        print(f'============== Ablation: {self.ablation}')

        target_net = PolicyTargetNet(
            dim2, dim3, dim4, nn.LeakyReLU(), mid_act=mid_act, target_layers_num=target_layers_num
        )

        self.target_net = target_net
        self.ftask_dim = ftask_dim
        self.weight_dim = weight_dim
        # Pop to avoid passing duplicate positional and keyword arguments downstream
        self.weight_split_dim = kwargs.pop("weight_split_dim", 512)
        self.num_layers = num_layers

        # Adapter to map task embedding dim (ftask_dim) -> latent weight dim for tokens
        self.ftask_adapter = (
            nn.Linear(self.ftask_dim, self.weight_dim) if self.ftask_dim != self.weight_dim else nn.Identity()
        )

        lr_scheme_method = kwargs.get("lr_scheme_method", "cosine")
        assert lr_scheme_method in ["karras", "cosine", "constant"]
        if lr_scheme_method == "karras":
            eta_max = 1e-2
            eta_min = 1e-4
            rho = 7
            lr_scheme = eta_max * (eta_min / eta_max) ** (torch.linspace(0.0, 1.0, num_layers) ** rho)
        elif lr_scheme_method == "cosine":
            eta_max = 1e-2
            eta_min = 1e-4
            lr_scheme = eta_min + 0.5 * (eta_max - eta_min) * (1 + torch.cos(torch.linspace(0.0, math.pi, num_layers)))
        else:
            lr_scheme = torch.ones(num_layers) * 1e-2
        self.register_buffer("lr_scheme", lr_scheme)

        self.encoders = nn.ModuleList(
            [
                ModuleEncoder(target_module, weight_dim, self.weight_split_dim, enc_dec_dim, num_enc_dec_layer, **kwargs)
                for target_module in target_net.get_submodules()
            ]
        )
        self.decoders = nn.ModuleList(
            [
                ModuleDecoder(target_module, weight_dim, self.weight_split_dim, enc_dec_dim, num_enc_dec_layer, **kwargs)
                for target_module in target_net.get_submodules()
            ]
        )
        OptLayerClass = get_opt_layer(use_compile=kwargs.get("use_compile", False))
        self.replicate_blocks = kwargs.get("replicate_blocks", True)
        if self.replicate_blocks:
            self.opt_layer = OptLayerClass(
                target_net=target_net,
                ftask_dim=ftask_dim,
                weight_dim=weight_dim,
                deriv_hidden_dim=opt_block_dim,
                driv_num_layers=num_opt_mlp_layer,
                weight_split_dim=self.weight_split_dim,
                **kwargs,
            )
        else:
            self.opt_layers = nn.ModuleList([OptLayerClass(
                target_net=target_net,
                ftask_dim=ftask_dim,
                weight_dim=weight_dim,
                deriv_hidden_dim=opt_block_dim,
                driv_num_layers=num_opt_mlp_layer,
                weight_split_dim=self.weight_split_dim,
                **kwargs,
            ) for _ in range(num_layers)])           

        self.layer_norms = nn.ModuleList([ParamLN(weight_dim) for _ in self.target_net.get_submodules()])
        self.init_weight_hypernet = InitWeightHypernet(
            self.target_net,
            self.weight_dim,
            self.weight_split_dim,
            hidden_dim=128,
            num_layers=4,
        )

    def forward_block(self, ftask, weight_dicts, opt_block):
        weight_upd_dicts = opt_block(ftask, weight_dicts)
        weight_upd_dicts = [ln(submodule) for ln, submodule in zip(self.layer_norms, weight_upd_dicts)]
        lrs = opt_block.get_lrs()
        return weight_upd_dicts, lrs

    def forward_blocks(self, ftask, warmup_lr=1.0):
        base_weight_dict = {k: v for k, v in self.target_net.named_parameters()}

        # Project task embeddings to weight token dimension if needed
        ftask_proj = self.ftask_adapter(ftask)

        # Get initial weights using init_weight_hypernet (expects (B, L, weight_dim))
        weight_embs = self.init_weight_hypernet(ftask_proj, ablation=self.ablation)

        # Decode initial weights to real space
        weight_dicts = [decoder(w) for (decoder, w) in zip(self.decoders, weight_embs)]

        # For init_only ablation, return early with just the initial weights
        if self.ablation == "init_only":
            weight_dict = self.target_net.merge_submodule_weights(weight_dicts)
            weight_dict = {
                k: einops.einsum(v, base_weight_dict[k], "b ..., ... -> b ...") for k, v in weight_dict.items()
            }
            return [weight_dict]

        final_weight_dicts = []
        self.lrs = []
        for i in range(self.num_layers):
            if self.replicate_blocks:
                weight_upd_embs, lrs = self.forward_block(ftask_proj, weight_embs, self.opt_layer)
            else:   
                weight_upd_embs, lrs = self.forward_block(ftask_proj, weight_embs, self.opt_layers[i])
            weight_upd_dicts = [decoder(w) for (decoder, w) in zip(self.decoders, weight_upd_embs)]
            weight_dicts = [
                {k: weights[k] + einops.einsum(upds[k], lr, "b ..., b -> b ...") * self.lr_scheme[i] for k in weights}
                for weights, upds, lr in zip(weight_dicts, weight_upd_dicts, lrs)
            ]
            self.lrs.append(torch.stack(lrs) * self.lr_scheme[i])
            weight_embs = [encoder(weight_dict) for encoder, weight_dict in zip(self.encoders, weight_dicts)]

            weight_dict = self.target_net.merge_submodule_weights(weight_dicts)
            weight_dict = {
                k: einops.einsum(v, base_weight_dict[k], "b ..., ... -> b ...") for k, v in weight_dict.items()
            }
            final_weight_dicts.append(weight_dict)
        
        self.lrs = torch.stack(self.lrs, dim=0)
        return final_weight_dicts

    def generate_weights(self, ftask, warmup_lr=1.0):
        final_weight_dicts = self.forward_blocks(ftask, warmup_lr)
        self.generated_weights = final_weight_dicts
        return final_weight_dicts

    def forward_with_weights(self, z, base_v, early_sup=False):
        if early_sup:
            return torch.stack([torch.vmap(torch.func.functional_call, in_dims=(None, 0, 0), randomness="different")(
                self.target_net, weights, base_v
            ) for weights in self.generated_weights], dim=0)
        else:
            return torch.vmap(torch.func.functional_call, in_dims=(None, 0, 0), randomness="different")(
                self.target_net, self.generated_weights[-1], base_v
            )

    def forward(self, z, base_v, early_sup=False):
        self.generate_weights(z)
        return self.forward_with_weights(z, base_v, early_sup)
