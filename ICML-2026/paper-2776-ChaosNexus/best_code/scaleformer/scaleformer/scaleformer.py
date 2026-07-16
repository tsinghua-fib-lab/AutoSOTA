MODEL_NAME="Nexus"

if MODEL_NAME == "Nexus":

    try:
        from flash_attn import flash_attn_func
        _flash_attn_available = True
        print("Flash Attention is available and will be used.")
    except ImportError:
        _flash_attn_available = False
        print("Flash Attention is not available, falling back to standard attention.")

    from dataclasses import dataclass
    from typing import Optional, Tuple, Union

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import PatchTSTConfig, PatchTSTPreTrainedModel
    from transformers.models.patchtst.modeling_patchtst import (
        ACT2CLS,
        BaseModelOutput,
        NegativeBinomialOutput,
        NormalOutput,
        PatchTSTForPredictionOutput,
        PatchTSTForPretrainingOutput,
        PatchTSTMasking,
        PatchTSTModelOutput,
        PatchTSTScaler,
        SamplePatchTSTOutput,
        StudentTOutput,
        nll,
        weighted_average,
    )
    from transformers.utils import ModelOutput

    from .modules import (
        DyT,
        PatchTSTKernelEmbedding,
        PatchTSTPatchify,
        PatchTSTRMSNorm,
        apply_p_rope_to_qk,
    )

    import kymatio.torch as kymatio

    import numpy as np

    def calculate_activated_params(model: nn.Module, top_k: int, num_experts: int):





        total_params = 0
        shared_params = 0
        expert_params = 0


        param_details = {
            "shared": {"count": 0, "size": []},
            "expert": {"count": 0, "size": []},
            "gate": {"count": 0, "size": []}
        }

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            num_params = param.numel()
            total_params += num_params


            if 'experts' in name:
                expert_params += num_params
                param_details["expert"]["count"] += num_params
                param_details["expert"]["size"].append((name, num_params))

            elif 'gate' in name:
                shared_params += num_params
                param_details["gate"]["count"] += num_params
                param_details["gate"]["size"].append((name, num_params))
            else:
                shared_params += num_params
                param_details["shared"]["count"] += num_params
                param_details["shared"]["size"].append((name, num_params))


        if num_experts > 0 and expert_params > 0:

            params_per_expert = expert_params / num_experts


            activated_expert_params = params_per_expert * top_k


            total_activated_params = shared_params + activated_expert_params
        else:

            params_per_expert = 0
            activated_expert_params = 0
            total_activated_params = total_params


        def format_M(num):
            return f"{num / 1e6:.2f}M"

        print("="*60)
        print("Model Parameter Analysis (Scaleformer with MoE)")
        print("="*60)
        print(f"Total Parameters: {format_M(total_params)}")
        print(f"Activated Parameters: {format_M(total_activated_params)}")
        print("-"*60)
        print("Breakdown:")
        print(f"  - Shared Parameters: {format_M(shared_params)}")
        print(f"    (Includes embeddings, attention, norms, head, MoE gate, etc.)")
        print(f"  - Total Expert Parameters: {format_M(expert_params)} (across all {num_experts} experts)")
        print(f"  - Parameters per Expert: {format_M(params_per_expert)}")
        print(f"  - Activated Expert Params: {format_M(activated_expert_params)} (top_k={top_k})")
        print("="*60)


        details = {
            "total_params": total_params,
            "activated_params": total_activated_params,
            "shared_params": shared_params,
            "total_expert_params": expert_params,
            "params_per_expert": params_per_expert,
            "activated_expert_params": activated_expert_params,
        }

        return total_activated_params, total_params, details

    class CnnExtractorWithLayerNorm(nn.Module):
        def __init__(self, n_coeffs):
            super().__init__()
            self.conv1 = nn.Conv1d(n_coeffs, 64, kernel_size=7, padding=3)
            self.ln1 = nn.LayerNorm(64) 
            self.gelu1 = nn.GELU()

            self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
            self.ln2 = nn.LayerNorm(128)
            self.gelu2 = nn.GELU()

            self.pool = nn.AdaptiveAvgPool1d(1)
            self.flatten = nn.Flatten()

        def forward(self, x):
            x = self.conv1(x)

            x = x.permute(0, 2, 1)
            x = self.ln1(x)

            x = x.permute(0, 2, 1)
            x = self.gelu1(x)

            x = self.conv2(x)

            x = x.permute(0, 2, 1)
            x = self.ln2(x)

            x = x.permute(0, 2, 1)
            x = self.gelu2(x)

            x = self.pool(x)
            x = self.flatten(x)

            return x
    class WaveletAnalyzer(nn.Module):
        def __init__(self, input_timesteps, feature_dim, J=8, Q=8):
            super().__init__()
            self.scattering = kymatio.Scattering1D(J=J, shape=(input_timesteps,), Q=Q)
            with torch.no_grad():
                dummy_input = torch.randn(1, input_timesteps)
                n_coeffs = self.scattering(dummy_input).shape[1]

            self.cnn_extractor = CnnExtractorWithLayerNorm(n_coeffs)
            self.final_mlp = nn.Linear(128, feature_dim)

        def forward(self, x):
            B, V, T = x.shape

            x_reshaped = x.reshape(B * V, T)
            scattering_coeffs = self.scattering(x_reshaped.contiguous())
            cnn_features = self.cnn_extractor(scattering_coeffs)
            features = self.final_mlp(cnn_features)
            features_reshaped = features.view(B, V, -1)
            final_embedding = features_reshaped.mean(dim=1)
            stabilized_embedding = torch.sign(final_embedding) * torch.log(torch.abs(final_embedding) + 1)

            return stabilized_embedding

    class STFTAnalyzer(nn.Module):





        def __init__(self, input_timesteps, feature_dim, n_fft=256, hop_length=64):
            super().__init__()
            self.n_fft = n_fft
            self.hop_length = hop_length



            n_freq = self.n_fft // 2 + 1




            self.cnn_extractor = CnnExtractorWithLayerNorm(n_coeffs=n_freq)
            self.final_mlp = nn.Linear(128, feature_dim)

            print(f"STFTAnalyzer initialized: n_fft={n_fft}, hop_length={hop_length}, n_freq={n_freq}")

        def forward(self, x):
            B, V, T = x.shape

            x_reshaped = x.reshape(B * V, T)



            stft_coeffs = torch.stft(
                x_reshaped, 
                n_fft=self.n_fft, 
                hop_length=self.hop_length, 
                return_complex=True,
                center=True
            )


            stft_mag = torch.abs(stft_coeffs) + 1e-6






            cnn_features = self.cnn_extractor(stft_mag)


            features = self.final_mlp(cnn_features)


            features_reshaped = features.view(B, V, -1)
            final_embedding = features_reshaped.mean(dim=1)


            stabilized_embedding = torch.sign(final_embedding) * torch.log(torch.abs(final_embedding) + 1)

            return stabilized_embedding

    class LearnableEncoderAnalyzer(nn.Module):




        def __init__(self, input_timesteps, feature_dim):
            super().__init__()



            self.cnn_extractor = CnnExtractorWithLayerNorm(n_coeffs=1)
            self.final_mlp = nn.Linear(128, feature_dim)

            print(f"LearnableEncoderAnalyzer initialized: Input Channels=1")

        def forward(self, x):
            B, V, T = x.shape

            x_reshaped = x.reshape(B * V, T)



            x_with_channel = x_reshaped.unsqueeze(1)



            cnn_features = self.cnn_extractor(x_with_channel)


            features = self.final_mlp(cnn_features)


            features_reshaped = features.view(B, V, -1)
            final_embedding = features_reshaped.mean(dim=1)


            stabilized_embedding = torch.sign(final_embedding) * torch.log(torch.abs(final_embedding) + 1)

            return stabilized_embedding

    def _get_kernel_params(kernel_type: str) -> dict:



        if kernel_type == 'gaussian_combo':

            sigma_list = np.logspace(-3, 3, num=7, base=10.0).tolist()

            weights = [1.0 / len(sigma_list)] * len(sigma_list)
            return {'sigma_list': sigma_list, 'weights': weights}

        elif kernel_type == 'polynomial':

            return {'gamma': 1.0, 'c': 1.0, 'd': 2.0}

        elif kernel_type == 'linear':
            return {}

        elif kernel_type == 'rational_quadratic':

            return {'sigma_list': [0.2, 0.5, 0.9, 1.3]}

        else:
            raise ValueError(f"未知的 kernel_type: {kernel_type}")

    def _compute_squared_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:



        if x.dim() == 3:
            x = x.squeeze(1)
        if y.dim() == 3:
            y = y.squeeze(1)
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        return torch.sum((x - y) ** 2, dim=-1)

    def gaussian_kernel_combination(x, y, sigma_list, weights, **kwargs):


        squared_dist = _compute_squared_dist(x, y)


        sigma_list = torch.tensor(sigma_list, device=x.device).view(-1, 1, 1)
        weights = torch.tensor(weights, device=x.device).view(-1, 1, 1)


        kernel_val_per_sigma = torch.exp(-squared_dist.unsqueeze(0) / (2 * sigma_list**2))


        weighted_kernels = kernel_val_per_sigma * weights


        return weighted_kernels.sum(dim=0)

    def linear_kernel(x, y, **kwargs):
        if x.dim() == 3: x = x.squeeze(1)
        if y.dim() == 3: y = y.squeeze(1)

        return torch.matmul(x, y.t())

    def polynomial_kernel(x, y, gamma, c, d, **kwargs):

        if x.dim() == 3: x = x.squeeze(1)
        if y.dim() == 3: y = y.squeeze(1)

        linear_term = torch.matmul(x, y.t())
        return (gamma * linear_term + c) ** d

    def rational_quadratic_kernel(x, y, sigma_list, **kwargs):

        squared_dist = _compute_squared_dist(x, y)

        sigma = torch.tensor(sigma_list, device=x.device).view(-1, 1, 1)
        sigma_squared = sigma ** 2


        kernel_val = sigma_squared / (sigma_squared + squared_dist.unsqueeze(0)) 
        return kernel_val.sum(dim=0)

    KERNEL_FUNCTIONS = {
        'gaussian_combo': gaussian_kernel_combination,
        'linear': linear_kernel,
        'polynomial': polynomial_kernel,
        'rational_quadratic': rational_quadratic_kernel,
    }

    def compute_mmd(x, y, mean_value, variance_value, kernel_type: str, kernel_params: dict):
        if x.dim() == 3:
            x = x.squeeze(1)
        if y.dim() == 3:
            y = y.squeeze(1)
        if mean_value.dim() == 1:
            mean_value = mean_value.unsqueeze(0)
        if variance_value.dim() == 1:
            variance_value = variance_value.unsqueeze(0)


        x = (x - mean_value) / torch.sqrt(variance_value + 1e-6)
        y = (y - mean_value) / torch.sqrt(variance_value + 1e-6)



        if kernel_type not in KERNEL_FUNCTIONS:
            raise ValueError(f"未知的 kernel_type: {kernel_type}")
        kernel_func = KERNEL_FUNCTIONS[kernel_type]


        xx = kernel_func(x, x, **kernel_params)
        yy = kernel_func(y, y, **kernel_params)
        xy = kernel_func(x, y, **kernel_params)


        B = x.size(0)


        if B > 1:
            term1 = (xx.sum() - xx.diag().sum()) / (B * (B - 1))
            term2 = (yy.sum() - yy.diag().sum()) / (B * (B - 1))
            term3 = xy.sum() / (B * B)
        else:
            term1, term2, term3 = 0, 0, 0

        return (term1 + term2 - 2 * term3).clamp(min=0)

    def conditional_mmd_multi_step(input_traj, true_traj, pred_traj, mean, variance, kernel_type: str, kernel_params: dict, steps=None):

        H = pred_traj.shape[1]


        if steps is None:
            steps = range(H)

        mmd_sum = 0.0
        for t in steps:
            true_evolved = true_traj[:, t, :]
            model_evolved = pred_traj[:, t, :]


            mmd_sum += compute_mmd(
                true_evolved, 
                model_evolved, 
                mean, 
                variance, 
                kernel_type=kernel_type, 
                kernel_params=kernel_params
            )


        return mmd_sum / len(steps) if len(steps) > 0 else 0.0

    @dataclass
    class CompletionsPatchTSTOutput(ModelOutput):
        completions: torch.FloatTensor
        patched_past_values: Optional[torch.FloatTensor] = None
        mask: Optional[torch.FloatTensor] = None
        loc: Optional[torch.FloatTensor] = None
        scale: Optional[torch.FloatTensor] = None

    class Expert(nn.Module):




        def __init__(self, d_model: int, ffn_dim: int, config: PatchTSTConfig):
            super().__init__()
            self.ff = nn.Sequential(
                nn.Linear(d_model, ffn_dim, bias=config.bias),
                ACT2CLS[config.activation_function](),
                nn.Dropout(config.ff_dropout) if config.ff_dropout > 0 else nn.Identity(),
                nn.Linear(ffn_dim, d_model, bias=config.bias),
            )

        def forward(self, x):
            return self.ff(x)

    class NaiveMoE(nn.Module):





        def __init__(self, d_model: int, ffn_dim: int, num_experts: int, top_k: int, config: PatchTSTConfig):
            super().__init__()
            self.d_model = d_model
            self.num_experts = num_experts
            self.top_k = top_k


            self.gate = nn.Linear(self.d_model, self.num_experts, bias=False)

            self.experts = nn.ModuleList([Expert(d_model, ffn_dim, config) for _ in range(self.num_experts)])

        def forward(self, x: torch.Tensor):

            original_shape = x.shape

            x_reshaped = x.reshape(-1, self.d_model)
            num_tokens, _ = x_reshaped.shape


            gate_logits = self.gate(x_reshaped)


            router_probs = F.softmax(gate_logits, dim=-1)
            tokens_per_expert_prob = router_probs.mean(dim=0)

            load_balance_loss = self.num_experts * torch.sum(tokens_per_expert_prob * tokens_per_expert_prob)


            top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)


            top_k_weights = top_k_weights / torch.sum(top_k_weights, dim=-1, keepdim=True)



            flat_top_k_indices = top_k_indices.flatten()



            perm = torch.argsort(flat_top_k_indices)


            perm_flat_top_k_indices = flat_top_k_indices[perm]



            counts = torch.bincount(perm_flat_top_k_indices, minlength=self.num_experts)

            starts = torch.cat((torch.tensor([0], device=x.device), counts.cumsum(0)[:-1]))



            token_indices = torch.arange(num_tokens, device=x.device).repeat_interleave(self.top_k)
            perm_token_indices = token_indices[perm]
            perm_inputs = x_reshaped[perm_token_indices]




            perm_outputs = torch.zeros_like(perm_inputs)


            for i in range(self.num_experts):
                start, end = starts[i], starts[i] + counts[i]
                if start < end:
                    expert_input = perm_inputs[start:end]
                    expert_output = self.experts[i](expert_input)
                    perm_outputs[start:end] = expert_output



            inv_perm = torch.argsort(perm)


            unperm_outputs = perm_outputs[inv_perm]


            unperm_outputs = unperm_outputs * top_k_weights.flatten().unsqueeze(-1)


            final_output_reshaped = torch.zeros_like(x_reshaped).index_add_(0, token_indices, unperm_outputs)


            return final_output_reshaped.reshape(original_shape), load_balance_loss

    class ConvNeXtBlock1D(nn.Module):




        def __init__(self, dim, path_dropout=0.0, layer_scale_init_value=1e-6, norm_eps=1e-6):
            super().__init__()
            self.dwconv = nn.Conv1d(
                dim, dim, kernel_size=7, padding=3, groups=dim
            )
            self.norm = nn.LayerNorm(dim, eps=norm_eps)
            self.pwconv1 = nn.Linear(dim, 4 * dim)
            self.act = nn.GELU()
            self.pwconv2 = nn.Linear(4 * dim, dim)
            self.weight = (
                nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
                if layer_scale_init_value > 0
                else None
            )
            self.drop_path = nn.Dropout(path_dropout) if path_dropout > 0.0 else nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:

            input = x
            x = x.permute(0, 2, 1)
            x = self.dwconv(x)
            x = x.permute(0, 2, 1)
            x = self.norm(x)
            x = self.pwconv1(x)
            x = self.act(x)
            x = self.pwconv2(x)
            if self.weight is not None:
                x = self.weight * x

            x = input + self.drop_path(x)
            return x

    class PatchMerging(nn.Module):



        def __init__(self, dim: int, norm_eps=1e-6):
            super().__init__()
            self.dim = dim

            self.reduction = nn.Linear(2 * dim, 2 * dim, bias=False)
            self.norm = nn.LayerNorm(2 * dim, eps=norm_eps)

        def forward(self, x: torch.Tensor) -> torch.Tensor:

            batch_size, num_patches, dim = x.shape


            if num_patches % 2 != 0:

                x = nn.functional.pad(x, (0, 0, 0, 1))
                num_patches += 1

            x = x.reshape(batch_size, num_patches // 2, 2, dim)
            x = x.flatten(2)

            x = self.reduction(x)
            x = self.norm(x)

            return x

    class PatchExpansion(nn.Module):



        def __init__(self, dim: int):
            super().__init__()
            self.dim = dim



            self.expand = nn.Linear(dim, dim, bias=False)



        def forward(self, x: torch.Tensor) -> torch.Tensor:

            batch_size, num_patches, dim = x.shape
            x = self.expand(x)
            x = x.view(batch_size, num_patches, 2, dim // 2)
            x = x.view(batch_size, -1, dim // 2)
            return x

    class PatchTSTEmbedding(nn.Module):
        def __init__(self, config: PatchTSTConfig):
            super().__init__()
            self.input_embedding = nn.Linear(config.patch_length, config.d_model)

        def forward(self, patch_input: torch.Tensor):




            embeddings = self.input_embedding(patch_input)
            return embeddings


    class PatchTSTRopeAttention(nn.Module):




        def __init__(
            self,
            d_model: int,
            num_heads: int,
            dropout: float = 0.0,
            is_decoder: bool = False,
            bias: bool = True,
            is_causal: bool = False,
            use_rope: bool = True,
            max_wavelength: int = 10000,
            rope_percent: float = 0.5,
            config: Optional[PatchTSTConfig] = None,


            use_flash_attention: bool = True, 
        ):
            super().__init__()
            self.embed_dim = d_model
            self.num_heads = num_heads
            self.dropout = dropout
            self.head_dim = d_model // num_heads
            self.max_wavelength = max_wavelength
            self.rope_percent = rope_percent
            self.use_rope = use_rope
            self.config = config



            self.use_flash_attention = use_flash_attention and _flash_attn_available

            if (self.head_dim * num_heads) != self.embed_dim:
                raise ValueError(
                    f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                    f" and `num_heads`: {num_heads})."
                )
            self.scaling = self.head_dim**-0.5
            self.is_decoder = is_decoder
            self.is_causal = is_causal

            self.k_proj = nn.Linear(d_model, d_model, bias=bias)
            self.v_proj = nn.Linear(d_model, d_model, bias=bias)
            self.q_proj = nn.Linear(d_model, d_model, bias=bias)
            self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
            return (
                tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
            )

        def get_seq_pos(self, seq_len, device, dtype, offset=0):
            return torch.arange(seq_len, device=device, dtype=dtype) + offset

        def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            linear_attn: bool = False,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
            """Input shape: Batch x Time x Channel"""
            is_cross_attention = key_value_states is not None
            bsz, tgt_len, _ = hidden_states.size()


            query_states = self.q_proj(hidden_states)

            if is_cross_attention and past_key_value is not None:
                key_states = past_key_value[0]
                value_states = past_key_value[1]
            elif is_cross_attention:
                key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
                value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
            elif past_key_value is not None:
                key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
                value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
            else:
                key_states = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)

            if self.is_decoder:
                past_key_value = (key_states, value_states)


            query_states = self._shape(query_states, tgt_len, bsz)
            key_states = self._shape(key_states, -1, bsz)
            value_states = self._shape(value_states, -1, bsz)

            src_len = key_states.size(2)


            if self.use_rope:



                q_for_rope = query_states.reshape(-1, tgt_len, self.head_dim)
                k_for_rope = key_states.reshape(-1, src_len, self.head_dim)

                position_ids = self.get_seq_pos(
                    src_len, key_states.device, key_states.dtype
                )
                k_for_rope, q_for_rope = apply_p_rope_to_qk(
                    k_for_rope,
                    q_for_rope,
                    position_ids,
                    self.head_dim,
                    self.max_wavelength,
                    self.rope_percent,
                )

                query_states = q_for_rope.view(bsz, self.num_heads, tgt_len, self.head_dim)
                key_states = k_for_rope.view(bsz, self.num_heads, src_len, self.head_dim)




            can_use_flash_attn = (
                self.use_flash_attention
                and hidden_states.is_cuda
                and attention_mask is None
                and hidden_states.dtype in [torch.float16, torch.bfloat16]
            )

            if can_use_flash_attn:


                query_states = query_states.transpose(1, 2)
                key_states = key_states.transpose(1, 2)
                value_states = value_states.transpose(1, 2)


                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout_p=self.dropout if self.training else 0.0,
                    causal=self.is_causal,
                )


                attn_weights_reshaped = None


                attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

            else:
                query_states = query_states * self.scaling
                attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1))

                if attn_weights.size() != (bsz, self.num_heads, tgt_len, src_len):
                    raise ValueError(
                        f"Attention weights should be of size {(bsz, self.num_heads, tgt_len, src_len)}, but is"
                        f" {attn_weights.size()}"
                    )

                if attention_mask is not None:
                    if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                        raise ValueError(
                            f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                        )
                    attn_weights = attn_weights + attention_mask.to(attn_weights.device)

                if not linear_attn:
                    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

                if layer_head_mask is not None:
                    if layer_head_mask.size() != (self.num_heads,):
                        raise ValueError(
                            f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                            f" {layer_head_mask.size()}"
                        )
                    attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights

                if output_attentions:
                    attn_weights_reshaped = attn_weights
                else:
                    attn_weights_reshaped = None

                attn_probs = nn.functional.dropout(
                    attn_weights, p=self.dropout, training=self.training
                )

                attn_output = torch.matmul(attn_probs, value_states)

                if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
                    raise ValueError(
                        f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                        f" {attn_output.size()}"
                    )

                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)


            attn_output = self.out_proj(attn_output)

            return attn_output, attn_weights_reshaped, past_key_value


    class PatchTSTEncoderLayerWithRope(nn.Module):




        def __init__(self, config: PatchTSTConfig, d_model: int, num_heads: int):
            super().__init__()


            self.use_moe = True
            self.num_experts = 8
            self.top_k = 2



            self.channel_attention = config.channel_attention

            self.temporal_self_attn = PatchTSTRopeAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=config.attention_dropout,
                use_rope=True,
                max_wavelength=config.max_wavelength,
                rope_percent=config.rope_percent,
            )










            if self.channel_attention:
                self.channel_self_attn = PatchTSTRopeAttention(
                    d_model=d_model,
                    num_heads=num_heads,
                    dropout=config.attention_dropout,
                    use_rope=config.channel_rope,
                    max_wavelength=config.max_wavelength,
                    rope_percent=config.rope_percent,
                )


            self.dropout_path1 = (
                nn.Dropout(config.path_dropout)
                if config.path_dropout > 0
                else nn.Identity()
            )
            if config.norm_type == "rmsnorm":
                self.norm_sublayer1 = PatchTSTRMSNorm(d_model, config.norm_eps)
            elif config.norm_type == "layernorm":
                self.norm_sublayer1 = nn.LayerNorm(d_model, eps=config.norm_eps)
            elif config.norm_type == "dyt":
                self.norm_sublayer1 = DyT(d_model)
            else:
                raise ValueError(f"{config.norm_type} is not a supported norm layer type.")


            if self.channel_attention:
                self.dropout_path2 = (
                    nn.Dropout(config.path_dropout)
                    if config.path_dropout > 0
                    else nn.Identity()
                )
                if config.norm_type == "rmsnorm":
                    self.norm_sublayer2 = PatchTSTRMSNorm(d_model, config.norm_eps)
                elif config.norm_type == "layernorm":
                    self.norm_sublayer2 = nn.LayerNorm(d_model, eps=config.norm_eps)
                elif config.norm_type == "dyt":
                    self.norm_sublayer2 = DyT(d_model)
                else:
                    raise ValueError(
                        f"{config.norm_type} is not a supported norm layer type."
                    )

            ffn_dim = d_model * 4
            if self.use_moe:
                self.ff = NaiveMoE(
                    d_model=d_model, 
                    ffn_dim=ffn_dim, 
                    num_experts=self.num_experts, 
                    top_k=self.top_k, 
                    config=config
                )
            else:
                self.ff = nn.Sequential(
                    nn.Linear(d_model, ffn_dim, bias=config.bias),
                    ACT2CLS[config.activation_function](),
                    nn.Dropout(config.ff_dropout) if config.ff_dropout > 0 else nn.Identity(),
                    nn.Linear(ffn_dim, d_model, bias=config.bias),
                )



            self.dropout_path3 = (
                nn.Dropout(config.path_dropout)
                if config.path_dropout > 0
                else nn.Identity()
            )
            if config.norm_type == "rmsnorm":
                self.norm_sublayer3 = PatchTSTRMSNorm(d_model, config.norm_eps)
            elif config.norm_type == "layernorm":
                self.norm_sublayer3 = nn.LayerNorm(d_model, eps=config.norm_eps)
            elif config.norm_type == "dyt":
                self.norm_sublayer3 = DyT(d_model)
            else:
                raise ValueError(f"{config.norm_type} is not a supported norm layer type.")

            self.pre_norm = config.pre_norm

        def forward(
            self,
            hidden_state: torch.Tensor,
            output_attentions: Optional[bool] = None,
            channel_attention_mask: Optional[torch.Tensor] = None,
            linear_attn: bool = False,
        ):









            batch_size, num_input_channels, sequence_length, d_model = hidden_state.shape



            hidden_state = hidden_state.view(
                batch_size * num_input_channels, sequence_length, d_model
            )

            if self.pre_norm:

                attn_output, attn_weights, _ = self.temporal_self_attn(
                    hidden_states=self.norm_sublayer1(hidden_state),
                    output_attentions=output_attentions,
                )

                hidden_state = hidden_state + self.dropout_path1(attn_output)




            else:

                attn_output, attn_weights, _ = self.temporal_self_attn(
                    hidden_states=hidden_state,
                    output_attentions=output_attentions,
                    linear_attn=linear_attn,
                )

                hidden_state = self.norm_sublayer1(
                    hidden_state + self.dropout_path1(attn_output)
                )








            hidden_state = hidden_state.reshape(
                batch_size, num_input_channels, sequence_length, d_model
            )


            if self.channel_attention:

                hidden_state = hidden_state.transpose(2, 1).contiguous()

                hidden_state = hidden_state.view(
                    batch_size * sequence_length, num_input_channels, d_model
                )
                if self.pre_norm:

                    attn_output, channel_attn_weights, _ = self.channel_self_attn(
                        hidden_states=self.norm_sublayer2(hidden_state),
                        output_attentions=output_attentions,
                        attention_mask=channel_attention_mask,
                    )

                    hidden_state = hidden_state + self.dropout_path2(attn_output)
                else:

                    attn_output, channel_attn_weights, _ = self.channel_self_attn(
                        hidden_states=hidden_state,
                        output_attentions=output_attentions,
                        attention_mask=channel_attention_mask,
                        linear_attn=linear_attn,
                    )

                    hidden_state = self.norm_sublayer2(
                        hidden_state + self.dropout_path2(attn_output)
                    )



                hidden_state = hidden_state.reshape(
                    batch_size, sequence_length, num_input_channels, d_model
                )

                hidden_state = hidden_state.transpose(1, 2).contiguous()



            hidden_state = hidden_state.view(
                batch_size * num_input_channels, sequence_length, d_model
            )

            moe_loss = torch.tensor(0.0, device=hidden_state.device)
            if self.pre_norm:
                normalized_hidden_state = self.norm_sublayer3(hidden_state)
                if self.use_moe:
                    ff_output, moe_loss = self.ff(normalized_hidden_state)
                else:
                    ff_output = self.ff(normalized_hidden_state)
                hidden_state = hidden_state + self.dropout_path3(ff_output)
            else:
                if self.use_moe:
                    ff_output, moe_loss = self.ff(hidden_state)
                else:
                    ff_output = self.ff(hidden_state)
                hidden_state = self.norm_sublayer3(
                    hidden_state + self.dropout_path3(ff_output)
                )



            hidden_state = hidden_state.reshape(
                batch_size, num_input_channels, sequence_length, d_model
            )


            outputs = (hidden_state,)
            if output_attentions:
                outputs += (
                    (attn_weights, channel_attn_weights)
                    if self.channel_attention
                    else (attn_weights,)
                )
            outputs += (moe_loss,)


            return outputs

    class PatchTSTUNetEncoder(nn.Module):
        def __init__(self, config: PatchTSTConfig, depths: list, num_heads_list: list):
            super().__init__()
            self.config = config
            self.stages = nn.ModuleList()
            current_dim = config.d_model
            for i, depth in enumerate(depths):
                stage_layers = nn.ModuleList(
                    [PatchTSTEncoderLayerWithRope(config, d_model=current_dim, num_heads=num_heads_list[i]) for _ in range(depth)]
                )

                downsample = PatchMerging(dim=current_dim, norm_eps=config.norm_eps) if i < len(depths) - 1 else None

                self.stages.append(nn.ModuleDict({
                    "layers": stage_layers,
                    "downsample": downsample
                }))

                if downsample:
                    current_dim *= 2

        def forward(self, hidden_state, output_attentions=None, channel_attention_mask=None, linear_attn=False):
            skip_connections = []
            total_moe_loss = 0.0

            for stage in self.stages:

                skip_connections.append(hidden_state) 
                for layer in stage["layers"]:
                    layer_outputs = layer(
                        hidden_state,
                        output_attentions=output_attentions,
                        channel_attention_mask=channel_attention_mask,
                        linear_attn=linear_attn,
                    )
                    hidden_state = layer_outputs[0]
                    total_moe_loss += layer_outputs[-1]

                if stage["downsample"] is not None:
                    batch_size, num_channels, num_patches, d_model = hidden_state.shape
                    hidden_state_reshaped = hidden_state.view(batch_size * num_channels, num_patches, d_model)
                    hidden_state_downsampled = stage["downsample"](hidden_state_reshaped)

                    num_patches = hidden_state_downsampled.shape[1]
                    d_model = hidden_state_downsampled.shape[2]
                    hidden_state = hidden_state_downsampled.view(batch_size, num_channels, num_patches, d_model)

            return hidden_state, skip_connections, total_moe_loss

    class PatchTSTUNetDecoder(nn.Module):
        def __init__(self, config: PatchTSTConfig, depths: list, skip_connections_depths: list, num_heads_list: list):
            super().__init__()
            self.config = config
            self.stages = nn.ModuleList()

            reversed_depths = list(reversed(depths))
            reversed_num_heads = list(reversed(num_heads_list))


            encoder_bottleneck_dim = config.d_model


            current_decoder_dim = encoder_bottleneck_dim

            for i, depth in enumerate(reversed_depths):


                target_dim = encoder_bottleneck_dim // (2 ** i)
                current_num_heads = reversed_num_heads[i]



                upsample = PatchExpansion(dim=current_decoder_dim) if i > 0 else None


                skip_processor_dim = target_dim
                skip_connection_processor = nn.ModuleList(
                    [ConvNeXtBlock1D(skip_processor_dim, norm_eps=config.norm_eps) for _ in range(skip_connections_depths[len(depths)-1-i])]
                )


                stage_layers = nn.ModuleList(
                    [PatchTSTEncoderLayerWithRope(config, d_model=target_dim, num_heads=current_num_heads) for _ in range(depth)]
                )

                self.stages.append(nn.ModuleDict({
                    "upsample": upsample,
                    "skip_processor": skip_connection_processor,
                    "layers": stage_layers,
                }))


                current_decoder_dim = target_dim

        def forward(self, hidden_state, skip_connections, output_attentions=None, channel_attention_mask=None, linear_attn=False):
            reversed_skips = list(reversed(skip_connections))
            total_moe_loss = 0.0


            all_stage_outputs = []

            for i, stage in enumerate(self.stages):

                if stage["upsample"] is not None:

                    batch_size, num_channels, num_patches, d_model = hidden_state.shape
                    hidden_state_reshaped = hidden_state.view(batch_size * num_channels, num_patches, d_model)
                    hidden_state_upsampled = stage["upsample"](hidden_state_reshaped)


                    skip = reversed_skips[i]
                    bs_s, nc_s, np_s, nd_s = skip.shape
                    skip_reshaped = skip.view(bs_s * nc_s, np_s, nd_s)
                    for processor in stage["skip_processor"]:
                        skip_reshaped = processor(skip_reshaped)



                    if hidden_state_upsampled.shape[1] != skip_reshaped.shape[1]:
                        diff = hidden_state_upsampled.shape[1] - skip_reshaped.shape[1]
                        skip_reshaped = nn.functional.pad(skip_reshaped, (0, 0, 0, diff))

                    hidden_state = hidden_state_upsampled + skip_reshaped


                    num_patches, d_model = hidden_state.shape[1], hidden_state.shape[2]
                    hidden_state = hidden_state.view(batch_size, num_channels, num_patches, d_model)


                for layer in stage["layers"]:
                    layer_outputs = layer(
                        hidden_state,
                        output_attentions=output_attentions,
                        channel_attention_mask=channel_attention_mask,
                        linear_attn=linear_attn,
                    )
                    hidden_state = layer_outputs[0]
                    total_moe_loss += layer_outputs[-1]


                all_stage_outputs.append(hidden_state)


            return all_stage_outputs, total_moe_loss

    class PatchTSTEncoder(PatchTSTPreTrainedModel):




        def __init__(self, config: PatchTSTConfig):
            super().__init__(config)
            self.gradient_checkpointing = False
            if config.use_dynamics_embedding:

                self.embedder = PatchTSTKernelEmbedding(config)
            else:
                self.embedder = PatchTSTEmbedding(config)

            self.layers = nn.ModuleList(
                [
                    PatchTSTEncoderLayerWithRope(config)
                    for i in range(config.num_hidden_layers)
                ]
            )


            self.post_init()

        def forward(
            self,
            patch_input: torch.Tensor,
            channel_attention_mask: Optional[torch.Tensor] = None,
            output_hidden_states: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            linear_attn: bool = False,
        ) -> BaseModelOutput:










            output_attentions = (
                output_attentions
                if output_attentions is not None
                else self.config.output_attentions
            )
            output_hidden_states = (
                output_hidden_states
                if output_hidden_states is not None
                else self.config.output_hidden_states
            )


            patch_input = self.embedder(patch_input)
            hidden_state = patch_input

            encoder_states = () if output_hidden_states else None
            all_attentions = () if output_attentions else None

            for encoder_layer in self.layers:
                if output_hidden_states:
                    encoder_states = encoder_states + (hidden_state,)

                layer_outputs = encoder_layer(
                    hidden_state=hidden_state,
                    output_attentions=output_attentions,
                    channel_attention_mask=channel_attention_mask,
                    linear_attn=linear_attn,
                )

                hidden_state = layer_outputs[0]

                if output_attentions:
                    all_attentions = all_attentions + layer_outputs[1:]

            return BaseModelOutput(
                last_hidden_state=hidden_state,
                hidden_states=encoder_states,
                attentions=all_attentions,
            )


    class PatchTSTModel(PatchTSTPreTrainedModel):
        def __init__(self, config: PatchTSTConfig):
            super().__init__(config)

            self.scaler = PatchTSTScaler(config)
            self.patchifier = PatchTSTPatchify(config)

            self.do_mask_input = config.do_mask_input

            if self.do_mask_input:
                self.masking = PatchTSTMasking(config)
            else:
                self.masking = nn.Identity()
            self.encoder = PatchTSTEncoder(config)


            self.post_init()

        def forward(
            self,
            past_values: torch.Tensor,
            past_observed_mask: Optional[torch.Tensor] = None,
            future_values: Optional[torch.Tensor] = None,
            output_hidden_states: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            channel_attention_mask: Optional[torch.Tensor] = None,
            linear_attn: bool = False,
            return_dict: Optional[bool] = None,
        ) -> Union[Tuple, PatchTSTModelOutput]:
            r"""




















            """

            return_dict = (
                return_dict if return_dict is not None else self.config.use_return_dict
            )
            output_attentions = (
                output_attentions
                if output_attentions is not None
                else self.config.output_attentions
            )
            output_hidden_states = (
                output_hidden_states
                if output_hidden_states is not None
                else self.config.output_hidden_states
            )

            if past_observed_mask is None:
                past_observed_mask = torch.ones_like(past_values)

            scaled_past_values, loc, scale = self.scaler(past_values, past_observed_mask)
            patched_values = self.patchifier(scaled_past_values)

            if self.do_mask_input:
                masked_values, mask = self.masking(patched_values)
            else:
                masked_values, mask = self.masking(patched_values), None

            encoder_output = self.encoder(
                patch_input=masked_values,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                channel_attention_mask=channel_attention_mask,
                linear_attn=linear_attn,
            )

            if not return_dict:
                outputs = (
                    encoder_output.last_hidden_state,
                    encoder_output.hidden_states,
                    encoder_output.attentions,
                )
                outputs = outputs + (mask, loc, scale, patched_values)
                return tuple(v for v in outputs if v is not None)

            return PatchTSTModelOutput(
                last_hidden_state=encoder_output.last_hidden_state,
                hidden_states=encoder_output.hidden_states,
                attentions=encoder_output.attentions,
                mask=mask,
                loc=loc,
                scale=scale,
                patch_input=patched_values,
            )


    class PatchTSTMaskPretrainHead(nn.Module):




        def __init__(
            self,
            d_model: int,
            patch_length: int,
            head_dropout: float = 0.0,
            use_cls_token: bool = False,
        ):
            super().__init__()
            self.dropout = nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity()
            self.linear = nn.Linear(d_model, patch_length)
            self.use_cls_token = use_cls_token

        def forward(self, embedding: torch.Tensor) -> torch.Tensor:









            embedding = self.linear(
                self.dropout(embedding)
            )
            if self.use_cls_token:
                embedding = embedding[:, :, 1:, :]
            return embedding


    class ChannelIndependentMasking(nn.Module):




        def __init__(self, config: PatchTSTConfig):
            super().__init__()
            self.mask_ratio = 0.5

            self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, config.d_model))

            torch.nn.init.normal_(self.mask_token, std=0.02)

        def forward(self, x: torch.Tensor):








            B, V, N, D = x.shape


            noise = torch.rand(B, V, N, device=x.device)


            num_masked = int(N * self.mask_ratio)
            if num_masked == 0:

                return x, torch.zeros(B, V, N, device=x.device)



            _, masked_indices = torch.topk(noise, num_masked, dim=2, largest=False)


            mask = torch.zeros(B, V, N, device=x.device, dtype=torch.bool)
            mask.scatter_(2, masked_indices, True)



            bool_mask = mask.unsqueeze(-1)


            masked_x = torch.where(bool_mask, self.mask_token, x)


            return masked_x, mask.float()

    class PatchTSTForPretraining(PatchTSTPreTrainedModel):
        def __init__(self, config: PatchTSTConfig):
            super().__init__(config)
            self.config = config


            self.depths = [2, 2, 2, 2]
            self.skip_connections_depths = [2, 2, 2, 0]
            self.num_heads_list = [3, 6, 12, 24]


            self.load_balance_coeff = 0.5




            self.scaler = PatchTSTScaler(config)
            self.patchifier = PatchTSTPatchify(config)


            self.masking = ChannelIndependentMasking(config)


            if config.use_dynamics_embedding:
                self.encoder_embedding = PatchTSTKernelEmbedding(config)
            else:
                self.encoder_embedding = PatchTSTEmbedding(config)



            original_d_model = config.d_model
            self.encoder = PatchTSTUNetEncoder(config, depths=self.depths, num_heads_list=self.num_heads_list)

            config.d_model = original_d_model * (2 ** (len(self.depths) - 1))
            self.decoder = PatchTSTUNetDecoder(config, depths=self.depths, skip_connections_depths=self.skip_connections_depths, num_heads_list=self.num_heads_list)

            config.d_model = original_d_model


            self.head = PatchTSTMaskPretrainHead(
                d_model=config.d_model,
                patch_length=config.patch_length,
                head_dropout=config.head_dropout,
                use_cls_token=config.use_cls_token,
            )


            if config.loss == "mse":
                self.loss = nn.MSELoss(reduction="none")
            elif config.loss == "huber":
                self.loss = nn.HuberLoss(reduction="none", delta=config.huber_delta)
            else:
                raise ValueError(f"Unknown loss {config.loss}")

            self.post_init()

        def forward(
            self,
            past_values: torch.Tensor,
            past_observed_mask: Optional[torch.Tensor] = None,
            output_hidden_states: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            channel_attention_mask: Optional[torch.Tensor] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[Tuple, PatchTSTForPretrainingOutput]:

            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if past_observed_mask is None:
                past_observed_mask = torch.ones_like(past_values)


            scaled_past_values, loc, scale = self.scaler(past_values, past_observed_mask)

            target_patches = self.patchifier(scaled_past_values)
            embedded_values = self.encoder_embedding(target_patches)




            masked_values, mask = self.masking(embedded_values)


            encoder_output, skip_connections, encoder_moe_loss = self.encoder(
                masked_values,
                output_attentions=output_attentions,
                channel_attention_mask=channel_attention_mask,
            )


            decoder_outputs_list, decoder_moe_loss = self.decoder(
                encoder_output,
                skip_connections,
                output_attentions=output_attentions,
                channel_attention_mask=channel_attention_mask,
            )

            final_decoder_output = decoder_outputs_list[-1]


            reconstructed_patches = self.head(final_decoder_output)



            loss_per_patch = self.loss(reconstructed_patches, target_patches)



            reconstruction_loss = (loss_per_patch.mean(dim=-1) * mask).sum() / (mask.sum() + 1e-10)


            total_moe_loss = encoder_moe_loss + decoder_moe_loss
            total_loss = reconstruction_loss + self.load_balance_coeff * total_moe_loss

            if not return_dict:
                output = (reconstructed_patches,)
                return (total_loss,) + output

            return PatchTSTForPretrainingOutput(
                loss=total_loss,
                prediction_output=reconstructed_patches,
                hidden_states=None,
                attentions=None,
            )


        @torch.no_grad()
        def generate_completions(
            self,
            past_values: torch.Tensor,
            past_observed_mask: Optional[torch.Tensor] = None,
            channel_attention_mask: Optional[torch.Tensor] = None,
        ) -> CompletionsPatchTSTOutput:



            if past_observed_mask is None:
                past_observed_mask = torch.ones_like(past_values)


            scaled_past_values, loc, scale = self.scaler(past_values, past_observed_mask)
            target_patches = self.patchifier(scaled_past_values)
            embedded_values = self.encoder_embedding(target_patches)



            masked_values, mask = self.masking(embedded_values)


            encoder_output, skip_connections, _ = self.encoder(
                masked_values,
                output_attentions=False,
                channel_attention_mask=channel_attention_mask,
            )


            decoder_outputs_list, _ = self.decoder(
                encoder_output,
                skip_connections,
                output_attentions=False,
                channel_attention_mask=channel_attention_mask,
            )
            final_decoder_output = decoder_outputs_list[-1]


            reconstructed_patches = self.head(final_decoder_output)


            return CompletionsPatchTSTOutput(
                completions=reconstructed_patches,
                patched_past_values=target_patches,
                loc=loc,
                scale=scale,
                mask=mask,
            )


    class PatchTSTPredictionHead(nn.Module):
        def __init__(
            self, config: PatchTSTConfig, num_patches: int = 1, distribution_output=None
        ):
            super().__init__()

            self.use_cls_token = config.use_cls_token
            self.pooling_type = config.pooling_type
            if self.pooling_type or self.use_cls_token:
                head_dim = config.d_model
            else:

                head_dim = config.d_model * num_patches


            self.flatten = nn.Flatten(start_dim=2)
            if distribution_output is None:

                self.projection = nn.Linear(head_dim, config.prediction_length, bias=False)
            else:

                self.projection = distribution_output.get_parameter_projection(head_dim)
            self.dropout = (
                nn.Dropout(config.head_dropout)
                if config.head_dropout > 0
                else nn.Identity()
            )

        def forward(self, embedding: torch.Tensor):









            if self.use_cls_token:

                pooled_embedding = embedding[:, :, 0, :]
            else:
                if self.pooling_type == "mean":

                    pooled_embedding = embedding.mean(dim=2)
                elif self.pooling_type == "max":

                    pooled_embedding = embedding.max(dim=2).values
                else:

                    pooled_embedding = embedding


            pooled_embedding = self.flatten(pooled_embedding)
            pooled_embedding = self.dropout(pooled_embedding)


            output = self.projection(pooled_embedding)

            if isinstance(output, tuple):

                output = tuple(z.transpose(2, 1) for z in output)
            else:
                output = output.transpose(2, 1)
            return output

    class MultiStagePredictionHead(nn.Module):


        def __init__(self, config: PatchTSTConfig, depths: list, distribution_output=None, wavelet_feature_dim: int = 0):
            super().__init__()
            self.config = config
            self.wavelet_feature_dim = wavelet_feature_dim


            if self.wavelet_feature_dim > 0:
                self.wavelet_mlp = nn.Sequential(
                    nn.Linear(self.wavelet_feature_dim, self.wavelet_feature_dim * 2),
                    nn.GELU(),
                    nn.Linear(self.wavelet_feature_dim * 2, self.wavelet_feature_dim)
                )


            encoder_bottleneck_dim = config.d_model * (2 ** (len(depths) - 1))
            decoder_dims = [encoder_bottleneck_dim // (2 ** i) for i in range(len(depths))]

            total_time_domain_dim = sum(decoder_dims)

            head_dim = total_time_domain_dim + self.wavelet_feature_dim

            self.flatten = nn.Flatten(start_dim=2)
            if distribution_output is None:
                self.projection = nn.Linear(head_dim, config.prediction_length, bias=False)
            else:
                self.projection = distribution_output.get_parameter_projection(head_dim)

            self.dropout = (
                nn.Dropout(config.head_dropout)
                if config.head_dropout > 0
                else nn.Identity()
            )

        def forward(self, decoder_outputs_list: list, wavelet_embedding: Optional[torch.Tensor] = None):
            B, V, _, _ = decoder_outputs_list[0].shape

            pooled_outputs = []
            for embedding in decoder_outputs_list:
                pooled_embedding = embedding.mean(dim=2)
                pooled_outputs.append(pooled_embedding)

            time_domain_embedding = torch.cat(pooled_outputs, dim=-1)

            if wavelet_embedding is not None and self.wavelet_feature_dim > 0:

                processed_wavelet_embedding = self.wavelet_mlp(wavelet_embedding)

                wavelet_embedding_expanded = processed_wavelet_embedding.unsqueeze(1).expand(-1, V, -1)
                final_embedding = torch.cat([time_domain_embedding, wavelet_embedding_expanded], dim=-1)
            else:
                final_embedding = time_domain_embedding

            flattened_embedding = self.flatten(final_embedding)
            dropped_embedding = self.dropout(flattened_embedding)

            output = self.projection(dropped_embedding)

            if isinstance(output, tuple):
                output = tuple(z.transpose(2, 1) for z in output)
            else:
                output = output.transpose(2, 1)

            return output

    class PatchTSTForPrediction(PatchTSTPreTrainedModel):
        def __init__(self, config: PatchTSTConfig):
            super().__init__(config)
            self.config = config




            self.training_truncate_lengths = [128, 256, 384, 512]  
            self.inference_truncate_length = 512


            if self.inference_truncate_length not in self.training_truncate_lengths:
                self.training_truncate_lengths.append(self.inference_truncate_length)


            self.training_truncate_lengths = [
                min(l, config.context_length) for l in self.training_truncate_lengths
            ]
            self.inference_truncate_length = min(self.inference_truncate_length, config.context_length)


            import random
            self.random = random




            self.depths = [2, 2, 2, 2]
            self.skip_connections_depths = [2, 2, 2, 0]
            self.num_heads_list = [3, 6, 12, 24]


            self.load_balance_coeff = 0.1


            self.mmd_loss_coeff = 0.5






            self.mmd_kernel_type = 'rational_quadratic' 


            self.mmd_kernel_params = _get_kernel_params(self.mmd_kernel_type)

            print("="*50)
            print(f"MMD Kernel Initialized for Experiment")
            print(f"  Type: {self.mmd_kernel_type}")
            print(f"  Params: {self.mmd_kernel_params}")
            print("="*50)



            self.wavelet_feature_dim = 48


            self.freq_analyzer_type = 'wst' 



            if self.freq_analyzer_type == 'wst':
                self.freq_analyzer = WaveletAnalyzer(
                    input_timesteps=config.context_length,
                    feature_dim=self.wavelet_feature_dim
                )
            elif self.freq_analyzer_type == 'stft':

                self.freq_analyzer = STFTAnalyzer(
                    input_timesteps=config.context_length,
                    feature_dim=self.wavelet_feature_dim,
                    n_fft=256, 
                    hop_length=64 
                )
            elif self.freq_analyzer_type == 'learnable_encoder':
                self.freq_analyzer = LearnableEncoderAnalyzer(
                    input_timesteps=config.context_length,
                    feature_dim=self.wavelet_feature_dim
                )
            else:
                raise ValueError(f"未知的 freq_analyzer_type: {self.freq_analyzer_type}")

            print(f"Frequency Analyzer Initialized: {self.freq_analyzer_type.upper()}")


            self.scaler = PatchTSTScaler(config)
            self.patchifier = PatchTSTPatchify(config)

            if config.use_dynamics_embedding:

                self.encoder_embedding = PatchTSTKernelEmbedding(config)
            else:
                self.encoder_embedding = PatchTSTEmbedding(config)

            original_d_model = config.d_model
            self.encoder = PatchTSTUNetEncoder(config, depths=self.depths, num_heads_list=self.num_heads_list)
            config.d_model = original_d_model * (2 ** (len(self.depths) - 1))
            self.decoder = PatchTSTUNetDecoder(config, depths=self.depths, skip_connections_depths=self.skip_connections_depths, num_heads_list=self.num_heads_list)
            config.d_model = original_d_model


            if config.loss == "mse" or config.loss == "huber":
                self.distribution_output = None
            else:
                if config.distribution_output == "student_t":
                    self.distribution_output = StudentTOutput(dim=config.prediction_length)
                elif config.distribution_output == "normal":
                    self.distribution_output = NormalOutput(dim=config.prediction_length)
                elif config.distribution_output == "negative_binomial":
                    self.distribution_output = NegativeBinomialOutput(dim=config.prediction_length)
                else:
                    raise ValueError(f"Unknown distribution output {config.distribution_output}")

            self.head = MultiStagePredictionHead(config, depths=self.depths, distribution_output=self.distribution_output, wavelet_feature_dim=self.wavelet_feature_dim)

            if config.loss == "mse":
                self.loss = nn.MSELoss(reduction="mean")
            elif config.loss == "huber":
                self.loss = nn.HuberLoss(reduction="mean", delta=config.huber_delta)




            self.post_init()

        def forward(
            self,
            past_values: torch.Tensor,
            past_observed_mask: Optional[torch.Tensor] = None,
            future_values: Optional[torch.Tensor] = None,
            output_hidden_states: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            channel_attention_mask: Optional[torch.Tensor] = None,
            return_dict: Optional[bool] = None,
            linear_attn: bool = False,
        ) -> Union[Tuple, PatchTSTForPredictionOutput]:
            return_dict = (
                return_dict if return_dict is not None else self.config.use_return_dict
            )

            if past_observed_mask is None:
                past_observed_mask = torch.ones_like(past_values)




            if self.training:

                target_len = self.random.choice(self.training_truncate_lengths)
            else:

                target_len = self.inference_truncate_length


            current_seq_len = past_values.shape[1]
            target_len = min(target_len, current_seq_len)


            past_values_truncated = past_values[:, :target_len, :]

            if past_observed_mask is not None:
                past_observed_mask_truncated = past_observed_mask[:, :target_len, :]
            else:
                past_observed_mask_truncated = torch.ones_like(past_values_truncated)



            wavelet_input_truncated = past_values[:, -target_len:, :].permute(0, 2, 1)



            padding_needed = self.config.context_length - wavelet_input_truncated.shape[2]


            if padding_needed > 0:

                wavelet_input_padded = F.pad(wavelet_input_truncated, (0, padding_needed), "reflect")
            else:
                wavelet_input_padded = wavelet_input_truncated






            wavelet_embedding = self.freq_analyzer(wavelet_input_padded)




            scaled_past_values, loc, scale = self.scaler(past_values_truncated, past_observed_mask_truncated)



            patched_values = self.patchifier(scaled_past_values)


            embedded_values = self.encoder_embedding(patched_values)


            encoder_output, skip_connections, encoder_moe_loss = self.encoder(
                embedded_values,
                output_attentions=output_attentions,
                channel_attention_mask=channel_attention_mask,
                linear_attn=linear_attn,
            )


            decoder_outputs_list, decoder_moe_loss = self.decoder(
                encoder_output,
                skip_connections,
                output_attentions=output_attentions,
                channel_attention_mask=channel_attention_mask,
                linear_attn=linear_attn,
            )


            y_hat = self.head(decoder_outputs_list, wavelet_embedding=wavelet_embedding)

            loss_val = None
            if future_values is not None:

                prediction_loss = torch.tensor(0.0, device=past_values.device)
                mmd_loss = torch.tensor(0.0, device=past_values.device)

                if self.distribution_output:

                    y_hat_out = y_hat
                    distribution = self.distribution_output.distribution(y_hat, loc=loc, scale=scale)
                    prediction_loss = nll(distribution, future_values)
                    prediction_loss = weighted_average(prediction_loss)
                else:

                    y_hat_out = y_hat * scale + loc
                    prediction_loss = self.loss(y_hat_out, future_values)


                    if self.mmd_loss_coeff > 0:
                        batch_mean = loc.mean(dim=0)
                        batch_variance = (scale**2).mean(dim=0)

                        mmd_loss = conditional_mmd_multi_step(
                            input_traj=None,
                            true_traj=future_values,
                            pred_traj=y_hat_out,
                            mean=batch_mean,
                            variance=batch_variance,
                            kernel_type=self.mmd_kernel_type,
                            kernel_params=self.mmd_kernel_params
                        )


                total_moe_loss = encoder_moe_loss + decoder_moe_loss










                loss_val = (
                    prediction_loss 
                    + self.mmd_loss_coeff * mmd_loss 
                    + self.load_balance_coeff * total_moe_loss
                )


            if self.distribution_output:
                y_hat_out = y_hat
            else:
                y_hat_out = y_hat * scale + loc

            if not return_dict:
                outputs = (y_hat_out, loc, scale)
                return (loss_val,) + outputs if loss_val is not None else outputs

            return PatchTSTForPredictionOutput(
                loss=loss_val,
                prediction_outputs=y_hat_out,
                hidden_states=None,
                attentions=None,
                loc=loc,
                scale=scale,
            )

        def generate(
            self,
            past_values: torch.Tensor,
            past_observed_mask: Optional[torch.Tensor] = None,
            channel_attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
        ) -> SamplePatchTSTOutput:



            num_parallel_samples = self.config.num_parallel_samples


            outputs = self(
                past_values=past_values,
                future_values=None,
                past_observed_mask=past_observed_mask,
                output_hidden_states=False,
                channel_attention_mask=channel_attention_mask,
                output_attentions=output_attentions,
            )

            if self.distribution_output:

                distribution = self.distribution_output.distribution(
                    outputs.prediction_outputs, loc=outputs.loc, scale=outputs.scale
                )

                samples = [distribution.sample() for _ in range(num_parallel_samples)]

                samples = torch.stack(samples, dim=1)
            else:
                samples = outputs.prediction_outputs.unsqueeze(1)

            return SamplePatchTSTOutput(sequences=samples)
        
elif MODEL_NAME == "Panda":
    from dataclasses import dataclass
    from typing import Optional, Tuple, Union

    import torch
    import torch.nn as nn
    # from mamba_ssm import Mamba2
    from transformers import PatchTSTConfig, PatchTSTPreTrainedModel
    from transformers.models.patchtst.modeling_patchtst import (
        ACT2CLS,
        BaseModelOutput,
        NegativeBinomialOutput,
        NormalOutput,
        PatchTSTForPredictionOutput,
        PatchTSTForPretrainingOutput,
        PatchTSTMasking,
        PatchTSTModelOutput,
        PatchTSTScaler,
        SamplePatchTSTOutput,
        StudentTOutput,
        nll,
        weighted_average,
    )
    from transformers.utils import ModelOutput

    from .modules import (
        DyT,
        PatchTSTKernelEmbedding,
        PatchTSTPatchify,
        PatchTSTRMSNorm,
        apply_p_rope_to_qk,
    )


    @dataclass
    class CompletionsPatchTSTOutput(ModelOutput):
        completions: torch.FloatTensor
        patched_past_values: Optional[torch.FloatTensor] = None
        mask: Optional[torch.FloatTensor] = None
        loc: Optional[torch.FloatTensor] = None
        scale: Optional[torch.FloatTensor] = None


    class PatchTSTEmbedding(nn.Module):
        def __init__(self, config: PatchTSTConfig):
            super().__init__()
            self.input_embedding = nn.Linear(config.patch_length, config.d_model)

        def forward(self, patch_input: torch.Tensor):
            """
            Parameters:
                patch_input (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`, *required*):
                    Patch input for embedding
            return:
                `torch.Tensor` of shape `(batch_size, num_channels, num_patches, d_model)`
            """
            embeddings = self.input_embedding(patch_input)
            return embeddings


    class PatchTSTRopeAttention(nn.Module):
        """
        Multi-headed attention from 'Attention Is All You Need' paper

        Implemented with p-rotary positional embeddings
        """

        def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            is_decoder: bool = False,
            bias: bool = True,
            is_causal: bool = False,
            use_rope: bool = True,
            max_wavelength: int = 10000,
            rope_percent: float = 0.5,
            config: Optional[PatchTSTConfig] = None,
        ):
            super().__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.dropout = dropout
            self.head_dim = embed_dim // num_heads
            self.max_wavelength = max_wavelength
            self.rope_percent = rope_percent
            self.use_rope = use_rope
            self.config = config

            if (self.head_dim * num_heads) != self.embed_dim:
                raise ValueError(
                    f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                    f" and `num_heads`: {num_heads})."
                )
            self.scaling = self.head_dim**-0.5
            self.is_decoder = is_decoder
            self.is_causal = is_causal

            self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
            return (
                tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
            )

        def get_seq_pos(self, seq_len, device, dtype, offset=0):
            return torch.arange(seq_len, device=device, dtype=dtype) + offset

        def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            linear_attn: bool = False,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
            """Input shape: Batch x Time x Channel"""

            # if key_value_states are provided this layer is used as a cross-attention layer
            # for the decoder
            is_cross_attention = key_value_states is not None

            bsz, tgt_len, _ = hidden_states.size()

            # get query proj
            query_states = self.q_proj(hidden_states) * self.scaling
            # get key, value proj
            # `past_key_value[0].shape[2] == key_value_states.shape[1]`
            # is checking that the `sequence_length` of the `past_key_value` is the same as
            # the provided `key_value_states` to support prefix tuning
            if (
                is_cross_attention
                and past_key_value is not None
                and past_key_value[0].shape[2] == key_value_states.shape[1]
            ):
                # reuse k,v, cross_attentions
                key_states = past_key_value[0]
                value_states = past_key_value[1]  # type: ignore
            elif is_cross_attention:
                # cross_attentions
                key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
                value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
            elif past_key_value is not None:
                # reuse k, v, self_attention
                key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
                value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
                key_states = torch.cat([past_key_value[0], key_states], dim=2)  # type: ignore
                value_states = torch.cat([past_key_value[1], value_states], dim=2)  # type: ignore
            else:
                # self_attention
                key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
                value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

            if self.is_decoder:
                # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
                # Further calls to cross_attention layer can then reuse all cross-attention
                # key/value_states (first "if" case)
                # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
                # all previous decoder key/value_states. Further calls to uni-directional self-attention
                # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
                # if encoder bi-directional self-attention `past_key_value` is always `None`
                past_key_value = (key_states, value_states)  # type: ignore

            proj_shape = (bsz * self.num_heads, -1, self.head_dim)
            query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
            key_states = key_states.reshape(*proj_shape)
            value_states = value_states.reshape(*proj_shape)
            src_len = key_states.size(1)

            # apply rotary positional embeddings
            if self.use_rope:
                position_ids = self.get_seq_pos(
                    src_len, key_states.device, key_states.dtype
                )
                key_states, query_states = apply_p_rope_to_qk(
                    key_states,
                    query_states,
                    position_ids,
                    self.head_dim,
                    self.max_wavelength,
                    self.rope_percent,
                )

            attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

            if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights.view(
                    bsz, self.num_heads, tgt_len, src_len
                ) + attention_mask.to(attn_weights.device)
                attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

            if not linear_attn:
                attn_weights = nn.functional.softmax(attn_weights, dim=-1)

            if layer_head_mask is not None:
                if layer_head_mask.size() != (self.num_heads,):
                    raise ValueError(
                        f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                        f" {layer_head_mask.size()}"
                    )
                attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(
                    bsz, self.num_heads, tgt_len, src_len
                )
                attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

            if output_attentions:
                # this operation is a bit awkward, but it's required to
                # make sure that attn_weights keeps its gradient.
                # In order to do so, attn_weights have to be reshaped
                # twice and have to be reused in the following
                attn_weights_reshaped = attn_weights.view(
                    bsz, self.num_heads, tgt_len, src_len
                )
                attn_weights = attn_weights_reshaped.view(
                    bsz * self.num_heads, tgt_len, src_len
                )
            else:
                attn_weights_reshaped = None

            attn_probs = nn.functional.dropout(
                attn_weights, p=self.dropout, training=self.training
            )

            attn_output = torch.bmm(attn_probs, value_states)

            if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
            attn_output = attn_output.transpose(1, 2)

            # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
            # partitioned across GPUs when using tensor-parallelism.
            attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

            attn_output = self.out_proj(attn_output)

            return attn_output, attn_weights_reshaped, past_key_value


    class PatchTSTEncoderLayerWithRope(nn.Module):
        """
        PatchTST encoder layer with rope positional embeddings
        """

        def __init__(self, config: PatchTSTConfig):
            super().__init__()

            self.channel_attention = config.channel_attention
            # Multi-Head attention
            self.temporal_self_attn = PatchTSTRopeAttention(
                embed_dim=config.d_model,
                num_heads=config.num_attention_heads,
                dropout=config.attention_dropout,
                use_rope=True,
                max_wavelength=config.max_wavelength,
                rope_percent=config.rope_percent,
            )
            # self.temporal_mamba = Mamba2(
            #     d_model=config.d_model,
            #     d_state=1024,
            #     d_conv=4,
            #     expand=2,
            #     headdim=64
            # )
            if self.channel_attention:
                self.channel_self_attn = PatchTSTRopeAttention(
                    embed_dim=config.d_model,
                    num_heads=config.num_attention_heads,
                    dropout=config.attention_dropout,
                    use_rope=config.channel_rope,  # channels are not positional
                    max_wavelength=config.max_wavelength,
                    rope_percent=config.rope_percent,
                )

            # Add & Norm of the sublayer 1
            self.dropout_path1 = (
                nn.Dropout(config.path_dropout)
                if config.path_dropout > 0
                else nn.Identity()
            )
            if config.norm_type == "rmsnorm":
                self.norm_sublayer1 = PatchTSTRMSNorm(config.d_model, config.norm_eps)
            elif config.norm_type == "layernorm":
                self.norm_sublayer1 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
            elif config.norm_type == "dyt":
                self.norm_sublayer1 = DyT(config.d_model)
            else:
                raise ValueError(f"{config.norm_type} is not a supported norm layer type.")

            # Add & Norm of the sublayer 2
            if self.channel_attention:
                self.dropout_path2 = (
                    nn.Dropout(config.path_dropout)
                    if config.path_dropout > 0
                    else nn.Identity()
                )
                if config.norm_type == "rmsnorm":
                    self.norm_sublayer2 = PatchTSTRMSNorm(config.d_model, config.norm_eps)
                elif config.norm_type == "layernorm":
                    self.norm_sublayer2 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
                elif config.norm_type == "dyt":
                    self.norm_sublayer2 = DyT(config.d_model)
                else:
                    raise ValueError(
                        f"{config.norm_type} is not a supported norm layer type."
                    )

            # Position-wise Feed-Forward
            self.ff = nn.Sequential(
                nn.Linear(config.d_model, config.ffn_dim, bias=config.bias),
                ACT2CLS[config.activation_function](),
                nn.Dropout(config.ff_dropout) if config.ff_dropout > 0 else nn.Identity(),
                nn.Linear(config.ffn_dim, config.d_model, bias=config.bias),
            )

            # Add & Norm of sublayer 3
            self.dropout_path3 = (
                nn.Dropout(config.path_dropout)
                if config.path_dropout > 0
                else nn.Identity()
            )
            if config.norm_type == "rmsnorm":
                self.norm_sublayer3 = PatchTSTRMSNorm(config.d_model, config.norm_eps)
            elif config.norm_type == "layernorm":
                self.norm_sublayer3 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
            elif config.norm_type == "dyt":
                self.norm_sublayer3 = DyT(config.d_model)
            else:
                raise ValueError(f"{config.norm_type} is not a supported norm layer type.")

            self.pre_norm = config.pre_norm

        def forward(
            self,
            hidden_state: torch.Tensor,
            output_attentions: Optional[bool] = None,
            channel_attention_mask: Optional[torch.Tensor] = None,
            linear_attn: bool = False,
        ):
            """
            Parameters:
                hidden_state (`torch.Tensor` of shape `(batch_size, num_channels, sequence_length, d_model)`, *required*):
                    Past values of the time series
                output_attentions (`bool`, *optional*):
                    Whether or not to return the output attention of all layers
            Return:
                `torch.Tensor` of shape `(batch_size, num_channels, sequence_length, d_model)`

            """
            batch_size, num_input_channels, sequence_length, d_model = hidden_state.shape

            # First sublayer: attention across time
            # hidden_states: [(bs*num_channels) x sequence_length x d_model]
            hidden_state = hidden_state.view(
                batch_size * num_input_channels, sequence_length, d_model
            )

            if self.pre_norm:
                ## Norm and Multi-Head attention and Add residual connection
                attn_output, attn_weights, _ = self.temporal_self_attn(
                    hidden_states=self.norm_sublayer1(hidden_state),
                    output_attentions=output_attentions,
                )
                # Add: residual connection with residual dropout
                hidden_state = hidden_state + self.dropout_path1(attn_output)
                # mamba_input = self.norm_sublayer1(hidden_state)
                # mamba_output = self.temporal_mamba(mamba_input)
                # hidden_state = hidden_state + self.dropout_path1(mamba_output)
            else:
                ## Multi-Head attention and Add residual connection and Norm - Standard Transformer from BERT
                attn_output, attn_weights, _ = self.temporal_self_attn(
                    hidden_states=hidden_state,
                    output_attentions=output_attentions,
                    linear_attn=linear_attn,
                )
                # hidden_states: [(bs*num_channels) x sequence_length x d_model]
                hidden_state = self.norm_sublayer1(
                    hidden_state + self.dropout_path1(attn_output)
                )
                # mamba_output = self.temporal_mamba(hidden_state)
                # hidden_state = self.norm_sublayer1(hidden_state + self.dropout_path1(mamba_output))

            # attn_weights = None

            # hidden_state: [bs x num_channels x sequence_length x d_model]
            hidden_state = hidden_state.reshape(
                batch_size, num_input_channels, sequence_length, d_model
            )

            # second sublayer: attention across variable at any given time
            if self.channel_attention:
                # hidden_state: [bs x sequence_length x num_channels x d_model]
                hidden_state = hidden_state.transpose(2, 1).contiguous()
                # hidden_state: [(bs*sequence_length) x num_channels x d_model]
                hidden_state = hidden_state.view(
                    batch_size * sequence_length, num_input_channels, d_model
                )
                if self.pre_norm:
                    ## Norm and Multi-Head attention and Add residual connection
                    attn_output, channel_attn_weights, _ = self.channel_self_attn(
                        hidden_states=self.norm_sublayer2(hidden_state),
                        output_attentions=output_attentions,
                        attention_mask=channel_attention_mask,
                    )
                    # Add: residual connection with residual dropout
                    hidden_state = hidden_state + self.dropout_path2(attn_output)
                else:
                    ## Multi-Head attention and Add residual connection and Norm
                    attn_output, channel_attn_weights, _ = self.channel_self_attn(
                        hidden_states=hidden_state,
                        output_attentions=output_attentions,
                        attention_mask=channel_attention_mask,
                        linear_attn=linear_attn,
                    )
                    # hidden_states: [(bs*sequence_length) x num_channels x d_model]
                    hidden_state = self.norm_sublayer2(
                        hidden_state + self.dropout_path2(attn_output)
                    )

                # Reshape hidden state
                # hidden_state: [bs x sequence_length x num_channels x d_model]
                hidden_state = hidden_state.reshape(
                    batch_size, sequence_length, num_input_channels, d_model
                )
                # hidden_state: [bs x num_channels x sequence_length x d_model]
                hidden_state = hidden_state.transpose(1, 2).contiguous()

            # Third sublayer: mixing across hidden
            # hidden_state: [(batch_size*num_channels) x sequence_length x d_model]
            hidden_state = hidden_state.view(
                batch_size * num_input_channels, sequence_length, d_model
            )
            if self.pre_norm:
                ## Norm and Position-wise Feed-Forward and Add residual connection
                # Add: residual connection with residual dropout
                hidden_state = hidden_state + self.dropout_path3(
                    self.ff(self.norm_sublayer3(hidden_state))
                )
            else:
                ## Position-wise Feed-Forward and Add residual connection and Norm
                # Add: residual connection with residual dropout
                hidden_state = self.norm_sublayer3(
                    hidden_state + self.dropout_path3(self.ff(hidden_state))
                )

            # [bs x num_channels x sequence_length x d_model]
            hidden_state = hidden_state.reshape(
                batch_size, num_input_channels, sequence_length, d_model
            )

            outputs = (hidden_state,)
            if output_attentions:
                outputs += (
                    (attn_weights, channel_attn_weights)
                    if self.channel_attention
                    else (attn_weights,)
                )

            return outputs


    class PatchTSTEncoder(PatchTSTPreTrainedModel):
        """
        PatchTST Encoder
        """

        def __init__(self, config: PatchTSTConfig):
            super().__init__(config)
            self.gradient_checkpointing = False
            if config.use_dynamics_embedding:
                # self.embedder = PatchTSTPolynomialEmbedding(config)
                self.embedder = PatchTSTKernelEmbedding(config)
            else:
                self.embedder = PatchTSTEmbedding(config)

            self.layers = nn.ModuleList(
                [
                    PatchTSTEncoderLayerWithRope(config)
                    for i in range(config.num_hidden_layers)
                ]
            )

            # Initialize weights and apply final processing
            self.post_init()

        def forward(
            self,
            patch_input: torch.Tensor,
            channel_attention_mask: Optional[torch.Tensor] = None,
            output_hidden_states: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            linear_attn: bool = False,
        ) -> BaseModelOutput:
            """
            Parameters:
                patch_input (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`, *required*):
                    Past values of the time series
                output_hidden_states (bool, optional): Indicates if hidden states should be outputted.
                output_attentions (bool, optional): Indicates if attentions should be outputted.

            return:
                `BaseModelOutput`
            """
            output_attentions = (
                output_attentions
                if output_attentions is not None
                else self.config.output_attentions
            )
            output_hidden_states = (
                output_hidden_states
                if output_hidden_states is not None
                else self.config.output_hidden_states
            )

            # Input embedding
            patch_input = self.embedder(patch_input)
            hidden_state = patch_input

            encoder_states = () if output_hidden_states else None
            all_attentions = () if output_attentions else None

            for encoder_layer in self.layers:
                if output_hidden_states:
                    encoder_states = encoder_states + (hidden_state,)  # type: ignore

                layer_outputs = encoder_layer(
                    hidden_state=hidden_state,
                    output_attentions=output_attentions,
                    channel_attention_mask=channel_attention_mask,
                    linear_attn=linear_attn,
                )
                # get hidden state. hidden_state shape is [bs x num_channels x num_patches x d_model]
                # or [bs x num_channels x (num_patches+1) x d_model] if use cls_token
                hidden_state = layer_outputs[0]
                # append attention matrix at each layer
                if output_attentions:
                    all_attentions = all_attentions + layer_outputs[1:]  # type: ignore
            # return past_values, hidden_states
            return BaseModelOutput(
                last_hidden_state=hidden_state,  # type: ignore
                hidden_states=encoder_states,  # type: ignore
                attentions=all_attentions,
            )


    class PatchTSTModel(PatchTSTPreTrainedModel):
        def __init__(self, config: PatchTSTConfig):
            super().__init__(config)

            self.scaler = PatchTSTScaler(config)
            self.patchifier = PatchTSTPatchify(config)

            self.do_mask_input = config.do_mask_input

            if self.do_mask_input:
                self.masking = PatchTSTMasking(config)
            else:
                self.masking = nn.Identity()
            self.encoder = PatchTSTEncoder(config)

            # Initialize weights and apply final processing
            self.post_init()

        def forward(
            self,
            past_values: torch.Tensor,
            past_observed_mask: Optional[torch.Tensor] = None,
            future_values: Optional[torch.Tensor] = None,
            output_hidden_states: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            channel_attention_mask: Optional[torch.Tensor] = None,
            linear_attn: bool = False,
            return_dict: Optional[bool] = None,
        ) -> Union[Tuple, PatchTSTModelOutput]:
            r"""
            Parameters:
                past_values (`torch.Tensor` of shape `(bs, sequence_length, num_input_channels)`, *required*):
                    Input sequence to the model
                past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
                    Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
                    in `[0, 1]`:

                    - 1 for values that are **observed**,
                    - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
                future_values (`torch.BoolTensor` of shape `(batch_size, prediction_length, num_input_channels)`, *optional*):
                    Future target values associated with the `past_values`
                output_hidden_states (`bool`, *optional*):
                    Whether or not to return the hidden states of all layers
                output_attentions (`bool`, *optional*):
                    Whether or not to return the output attention of all layers
                return_dict (`bool`, *optional*):
                    Whether or not to return a `ModelOutput` instead of a plain tuple.

            Returns:
                `PatchTSTModelOutput` or tuple of `torch.Tensor` (if `return_dict`=False or `config.return_dict`=False)

            """

            return_dict = (
                return_dict if return_dict is not None else self.config.use_return_dict
            )
            output_attentions = (
                output_attentions
                if output_attentions is not None
                else self.config.output_attentions
            )
            output_hidden_states = (
                output_hidden_states
                if output_hidden_states is not None
                else self.config.output_hidden_states
            )

            if past_observed_mask is None:
                past_observed_mask = torch.ones_like(past_values)

            scaled_past_values, loc, scale = self.scaler(past_values, past_observed_mask)
            patched_values = self.patchifier(scaled_past_values)

            if self.do_mask_input:
                masked_values, mask = self.masking(patched_values)
            else:
                masked_values, mask = self.masking(patched_values), None

            encoder_output = self.encoder(
                patch_input=masked_values,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                channel_attention_mask=channel_attention_mask,
                linear_attn=linear_attn,
            )

            if not return_dict:
                outputs = (
                    encoder_output.last_hidden_state,
                    encoder_output.hidden_states,
                    encoder_output.attentions,
                )
                outputs = outputs + (mask, loc, scale, patched_values)
                return tuple(v for v in outputs if v is not None)

            return PatchTSTModelOutput(
                last_hidden_state=encoder_output.last_hidden_state,
                hidden_states=encoder_output.hidden_states,
                attentions=encoder_output.attentions,
                mask=mask,  # type: ignore
                loc=loc,
                scale=scale,
                patch_input=patched_values,
            )


    class PatchTSTMaskPretrainHead(nn.Module):
        """
        Pretraining head for mask modelling
        """

        def __init__(
            self,
            d_model: int,
            patch_length: int,
            head_dropout: float = 0.0,
            use_cls_token: bool = False,
        ):
            super().__init__()
            self.dropout = nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity()
            self.linear = nn.Linear(d_model, patch_length)
            self.use_cls_token = use_cls_token

        def forward(self, embedding: torch.Tensor) -> torch.Tensor:
            """
            Parameters:
                embedding (`torch.Tensor` of shape `(bs, num_channels, num_patches, d_model)` or
                        `(bs, num_channels, num_patches+1, d_model)` if `cls_token` is set to True, *required*):
                    Embedding from the model
            Returns:
                `torch.Tensor` of shape `(bs, num_channels, num_patches, d_model)` or
                                `(bs, num_channels, num_patches+1, d_model)` if `cls_token` is set to True

            """
            embedding = self.linear(
                self.dropout(embedding)
            )  # [bs x num_channels x num_patches x patch_length]
            if self.use_cls_token:
                embedding = embedding[:, :, 1:, :]  # remove the first cls token
            return embedding


    class PatchTSTForPretraining(PatchTSTPreTrainedModel):
        def __init__(self, config: PatchTSTConfig):
            super().__init__(config)

            config.do_mask_input = True
            self.model = PatchTSTModel(config=config)
            self.head = PatchTSTMaskPretrainHead(
                d_model=config.d_model,
                patch_length=config.patch_length,
                head_dropout=config.head_dropout,
                use_cls_token=config.use_cls_token,
            )

            if config.loss == "mse":
                self.loss = nn.MSELoss(reduction="none")
            elif config.loss == "huber":
                self.loss = nn.HuberLoss(reduction="none", delta=config.huber_delta)
            else:
                raise ValueError(f"Unknown loss {config.loss}")
            # Initialize weights and apply final processing
            self.post_init()

        def forward(
            self,
            past_values: torch.Tensor,
            past_observed_mask: Optional[torch.Tensor] = None,
            output_hidden_states: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            channel_attention_mask: Optional[torch.Tensor] = None,
            return_dict: Optional[bool] = None,
            schedule_param: float = 0.0,
        ) -> Union[Tuple, PatchTSTForPretrainingOutput]:
            r"""
            Parameters:
                past_values (`torch.Tensor` of shape `(bs, sequence_length, num_input_channels)`, *required*):
                    Input sequence to the model
                past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
                    Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
                    in `[0, 1]`:

                    - 1 for values that are **observed**,
                    - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
                output_hidden_states (`bool`, *optional*):
                    Whether or not to return the hidden states of all layers
                output_attentions (`bool`, *optional*):
                    Whether or not to return the output attention of all layers
                return_dict (`bool`, *optional*): Whether or not to return a `ModelOutput` instead of a plain tuple.

            Returns:
                `PatchTSTForPretrainingOutput` or tuple of `torch.Tensor` (if `return_dict`=False or
                `config.return_dict`=False)

            """
            return_dict = (
                return_dict if return_dict is not None else self.config.use_return_dict
            )

            # past_values: [bs x num_channels x num_patches x d_model] or
            # [bs x num_channels x (num_patches+1) x d_model] if use cls_token
            model_output = self.model(
                past_values=past_values,
                past_observed_mask=past_observed_mask,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                channel_attention_mask=channel_attention_mask,
                return_dict=True,
            )

            # last_hidden_state: [bs x num_channels x num_patches x d_model] or
            x_hat = model_output.last_hidden_state

            # [bs x num_channels x (num_patches+1) x d_model] if use cls_token
            # x_hat: [bs x num_channels x num_patches x patch_length]
            x_hat = self.head(x_hat)

            # reduce over the patch length dim first, then compute the masked loss over the tokens
            loss_val = self.loss(x_hat, model_output.patch_input)
            masked_loss = (loss_val.mean(dim=-1) * model_output.mask).sum() / (
                model_output.mask.sum() + 1e-10
            )

            encoder_states = model_output.hidden_states
            if not return_dict:
                outputs = (x_hat,) + model_output[1:-4]
                outputs = (masked_loss,) + outputs if masked_loss is not None else outputs
                return outputs
            return PatchTSTForPretrainingOutput(
                loss=masked_loss,
                prediction_output=x_hat,
                hidden_states=encoder_states,
                attentions=model_output.attentions,
            )

        @torch.no_grad()
        def generate_completions(
            self,
            past_values: torch.Tensor,
            past_observed_mask: Optional[torch.Tensor] = None,
            channel_attention_mask: Optional[torch.Tensor] = None,
        ) -> CompletionsPatchTSTOutput:
            r"""
            Parameters:
                past_values (`torch.Tensor` of shape `(bs, sequence_length, num_input_channels)`, *required*):
                    Input sequence to the model
                past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
                    Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
                    in `[0, 1]`:

                    - 1 for values that are **observed**,
                    - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).

            Returns:
                `CompletionPatchTSTOutput`

            """

            # past_values: [bs x num_channels x num_patches x d_model] or
            # [bs x num_channels x (num_patches+1) x d_model] if use cls_token
            model_output = self.model(
                past_values=past_values,
                past_observed_mask=past_observed_mask,
                return_dict=True,
                channel_attention_mask=channel_attention_mask,
            )

            # last_hidden_state: [bs x num_channels x num_patches x d_model] or
            x_hat = model_output.last_hidden_state

            # [bs x num_channels x (num_patches+1) x d_model] if use cls_token
            # x_hat: [bs x num_channels x num_patches x patch_length]
            x_hat = self.head(x_hat)

            return CompletionsPatchTSTOutput(
                completions=x_hat,
                patched_past_values=model_output.patch_input,
                loc=model_output.loc,
                scale=model_output.scale,
                mask=model_output.mask,
            )


    class PatchTSTPredictionHead(nn.Module):
        def __init__(
            self, config: PatchTSTConfig, num_patches: int = 1, distribution_output=None
        ):
            super().__init__()

            self.use_cls_token = config.use_cls_token
            self.pooling_type = config.pooling_type
            if self.pooling_type or self.use_cls_token:  # this should always be true
                head_dim = config.d_model
            else:  # included for completeness
                # num_patches is set to a dummy value,
                head_dim = config.d_model * num_patches

            # all the channels share the same head
            self.flatten = nn.Flatten(start_dim=2)
            if distribution_output is None:
                # use linear head with custom weight initialization
                self.projection = nn.Linear(head_dim, config.prediction_length, bias=False)
            else:
                # use distribution head
                self.projection = distribution_output.get_parameter_projection(head_dim)
            self.dropout = (
                nn.Dropout(config.head_dropout)
                if config.head_dropout > 0
                else nn.Identity()
            )

        def forward(self, embedding: torch.Tensor):
            """
            Parameters:
                embedding (`torch.Tensor` of shape `(bs, num_channels, num_patches, d_model)` or
                         `(bs, num_channels, num_patches+1, d_model)` if `cls_token` is set to True, *required*):
                    Embedding from the model
            Returns:
                `torch.Tensor` of shape `(bs, forecast_len, num_channels)`

            """
            if self.use_cls_token:
                # pooled_embedding: [bs x num_channels x d_model]
                pooled_embedding = embedding[:, :, 0, :]
            else:
                if self.pooling_type == "mean":
                    # pooled_embedding: [bs x num_channels x d_model]
                    pooled_embedding = embedding.mean(dim=2)
                elif self.pooling_type == "max":
                    # pooled_embedding: [bs x num_channels x d_model]
                    pooled_embedding = embedding.max(dim=2).values
                else:
                    # pooled_embedding: [bs x num_channels x num_patches x d_model]
                    pooled_embedding = embedding

            # pooled_embedding: [bs x num_channels x (d_model * num_patches)] or [bs x num_channels x d_model)]
            pooled_embedding = self.flatten(pooled_embedding)
            pooled_embedding = self.dropout(pooled_embedding)

            # output: [bs x num_channels x forecast_len] or
            # tuple ([bs x num_channels x forecast_len], [bs x num_channels x forecast_len]) if using distribution head
            output = self.projection(pooled_embedding)

            if isinstance(output, tuple):
                # output: ([bs x forecast_len x num_channels], [bs x forecast_len x num_channels])
                output = tuple(z.transpose(2, 1) for z in output)
            else:
                output = output.transpose(2, 1)  # [bs x forecast_len x num_channels]
            return output


    class PatchTSTForPrediction(PatchTSTPreTrainedModel):
        def __init__(self, config: PatchTSTConfig):
            super().__init__(config)

            # Turn off masking
            config.do_mask_input = False

            self.model = PatchTSTModel(config)

            if config.loss == "mse" or config.loss == "huber":
                self.distribution_output = None
            else:
                if config.distribution_output == "student_t":
                    self.distribution_output = StudentTOutput(dim=config.prediction_length)
                elif config.distribution_output == "normal":
                    self.distribution_output = NormalOutput(dim=config.prediction_length)
                elif config.distribution_output == "negative_binomial":
                    self.distribution_output = NegativeBinomialOutput(
                        dim=config.prediction_length
                    )
                else:
                    raise ValueError(
                        f"Unknown distribution output {config.distribution_output}"
                    )

            self.head = PatchTSTPredictionHead(
                config, distribution_output=self.distribution_output
            )

            if config.loss == "mse":
                self.loss = nn.MSELoss(reduction="mean")
            elif config.loss == "huber":
                self.loss = nn.HuberLoss(reduction="mean", delta=config.huber_delta)
            else:
                raise ValueError(f"Unknown loss {config.loss}")
            # Initialize weights and apply final processing
            self.post_init()

        def forward(
            self,
            past_values: torch.Tensor,
            past_observed_mask: Optional[torch.Tensor] = None,
            future_values: Optional[torch.Tensor] = None,
            output_hidden_states: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            channel_attention_mask: Optional[torch.Tensor] = None,
            return_dict: Optional[bool] = None,
            linear_attn: bool = False,
        ) -> Union[Tuple, PatchTSTForPredictionOutput]:
            r"""
            Parameters:
                past_values (`torch.Tensor` of shape `(bs, sequence_length, num_input_channels)`, *required*):
                    Input sequence to the model
                past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
                    Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
                    in `[0, 1]`:

                    - 1 for values that are **observed**,
                    - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
                future_values (`torch.Tensor` of shape `(bs, forecast_len, num_input_channels)`, *optional*):
                    Future target values associated with the `past_values`
                output_hidden_states (`bool`, *optional*):
                    Whether or not to return the hidden states of all layers
                output_attentions (`bool`, *optional*):
                    Whether or not to return the output attention of all layers
                return_dict (`bool`, *optional*):
                    Whether or not to return a `ModelOutput` instead of a plain tuple.

            Returns:
                `PatchTSTForPredictionOutput` or tuple of `torch.Tensor` (if `return_dict`=False or
                `config.return_dict`=False)

            """

            return_dict = (
                return_dict if return_dict is not None else self.config.use_return_dict
            )

            # get model output
            model_output = self.model(
                past_values=past_values,
                past_observed_mask=past_observed_mask,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                channel_attention_mask=channel_attention_mask,
                return_dict=True,
                linear_attn=linear_attn,
            )
            y_hat = self.head(model_output.last_hidden_state)

            if self.distribution_output:
                y_hat_out = y_hat
            else:
                y_hat_out = y_hat * model_output.scale + model_output.loc

            loss_val = None
            if future_values is not None:
                if self.distribution_output:
                    distribution = self.distribution_output.distribution(
                        y_hat, loc=model_output.loc, scale=model_output.scale
                    )
                    loss_val = nll(distribution, future_values)
                    loss_val = weighted_average(loss_val)
                else:
                    loss_val = self.loss(y_hat_out, future_values)

            loc = model_output.loc
            scale = model_output.scale

            if not return_dict:
                outputs = (
                    future_values,
                    y_hat_out,
                    loc,
                    scale,
                ) + model_output[1:-1]
                outputs = (loss_val,) + outputs if loss_val is not None else outputs
                return outputs

            return PatchTSTForPredictionOutput(
                loss=loss_val,  # type: ignore
                prediction_outputs=y_hat_out,
                hidden_states=model_output.hidden_states,
                attentions=model_output.attentions,
                loc=loc,
                scale=scale,
            )

        def generate(
            self,
            past_values: torch.Tensor,
            past_observed_mask: Optional[torch.Tensor] = None,
            channel_attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
        ) -> SamplePatchTSTOutput:
            """
            Generate sequences of sample predictions from a model with a probability distribution head.

            Parameters:
                past_values (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                    Past values of the time series that serves as context in order to predict the future.
                past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
                    Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
                    in `[0, 1]`:

                    - 1 for values that are **observed**,
                    - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).

            Return:
                [`SamplePatchTSTOutput`] where the outputs `sequences` tensor will have shape `(batch_size, number of
                samples, prediction_length, 1)` or `(batch_size, number of samples, prediction_length, num_input_channels)`
                for multivariate predictions.
            """
            # get number of samples
            num_parallel_samples = self.config.num_parallel_samples

            # get model output
            outputs = self(
                past_values=past_values,
                future_values=None,
                past_observed_mask=past_observed_mask,
                output_hidden_states=False,
                channel_attention_mask=channel_attention_mask,
                output_attentions=output_attentions,
            )

            if self.distribution_output:
                # get distribution
                distribution = self.distribution_output.distribution(
                    outputs.prediction_outputs, loc=outputs.loc, scale=outputs.scale
                )
                # get samples: list of [bs x forecast_len x num_channels]
                samples = [distribution.sample() for _ in range(num_parallel_samples)]
                # samples: [bs x num_samples x forecast_len x num_channels]
                samples = torch.stack(samples, dim=1)
            else:
                samples = outputs.prediction_outputs.unsqueeze(1)

            return SamplePatchTSTOutput(sequences=samples)  # type: ignore

else:
    raise NotImplementedError("Unrecognized model.")