import torch
from torch import nn
import torch.nn.functional as F
import einops
from diffusers import DiffusionPipeline
from jaxtyping import Float, Int
from pydoc import locate
from typing import Literal
from .layers import FeedForwardBlock, FourierFeatures, Linear, MappingNetwork
from .min_sd15 import SD15UNetModel
from .min_sd21 import SD21UNetModel


class SD15UNetFeatureExtractor(SD15UNetModel):
    def __init__(self):
        super().__init__()

    def forward(self, sample, timesteps, encoder_hidden_states, added_cond_kwargs, **kwargs):
        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps).to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb)

        sample = self.conv_in(sample)

        # 3. down
        s0 = sample
        sample, [s1, s2, s3] = self.down_blocks[0](
            sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
        )

        sample, [s4, s5, s6] = self.down_blocks[1](
            sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
        )

        sample, [s7, s8, s9] = self.down_blocks[2](
            sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
        )

        sample, [s10, s11] = self.down_blocks[3](
            sample,
            temb=emb,
        )

        # 4. mid
        sample_mid = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)

        # 5. up
        _, [us1, us2, us3] = self.up_blocks[0](
            hidden_states=sample_mid,
            temb=emb,
            res_hidden_states_tuple=[s9, s10, s11],
        )

        _, [us4, us5, us6] = self.up_blocks[1](
            hidden_states=us3,
            temb=emb,
            res_hidden_states_tuple=[s6, s7, s8],
            encoder_hidden_states=encoder_hidden_states,
        )

        _, [us7, us8, us9] = self.up_blocks[2](
            hidden_states=us6,
            temb=emb,
            res_hidden_states_tuple=[s3, s4, s5],
            encoder_hidden_states=encoder_hidden_states,
        )

        _, [us10, us11, _] = self.up_blocks[3](
            hidden_states=us9,
            temb=emb,
            res_hidden_states_tuple=[s0, s1, s2],
            encoder_hidden_states=encoder_hidden_states,
        )

        return {
            "mid": sample_mid,
            "us1": us1,
            "us2": us2,
            "us3": us3,
            "us4": us4,
            "us5": us5,
            "us6": us6,
            "us7": us7,
            "us8": us8,
            "us9": us9,
            "us10": us10,
        }


class SD21UNetFeatureExtractor(SD21UNetModel):
    def __init__(self):
        super().__init__()

    def forward(self, sample, timesteps, encoder_hidden_states, added_cond_kwargs, **kwargs):
        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps).to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb)

        sample = self.conv_in(sample)

        # 3. down
        s0 = sample
        sample, [s1, s2, s3] = self.down_blocks[0](
            sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
        )

        sample, [s4, s5, s6] = self.down_blocks[1](
            sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
        )

        sample, [s7, s8, s9] = self.down_blocks[2](
            sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
        )

        sample, [s10, s11] = self.down_blocks[3](
            sample,
            temb=emb,
        )

        # 4. mid
        sample_mid = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)

        # 5. up
        _, [us1, us2, us3] = self.up_blocks[0](
            hidden_states=sample_mid,
            temb=emb,
            res_hidden_states_tuple=[s9, s10, s11],
        )

        _, [us4, us5, us6] = self.up_blocks[1](
            hidden_states=us3,
            temb=emb,
            res_hidden_states_tuple=[s6, s7, s8],
            encoder_hidden_states=encoder_hidden_states,
        )

        _, [us7, us8, us9] = self.up_blocks[2](
            hidden_states=us6,
            temb=emb,
            res_hidden_states_tuple=[s3, s4, s5],
            encoder_hidden_states=encoder_hidden_states,
        )

        _, [us10, us11, _] = self.up_blocks[3](
            hidden_states=us9,
            temb=emb,
            res_hidden_states_tuple=[s0, s1, s2],
            encoder_hidden_states=encoder_hidden_states,
        )

        return {
            "mid": sample_mid,
            "us1": us1,
            "us2": us2,
            "us3": us3,
            "us4": us4,
            "us5": us5,
            "us6": us6,
            "us7": us7,
            "us8": us8,
            "us9": us9,
            "us10": us10,
        }

class FeedForwardBlockCustom(FeedForwardBlock):
    def __init__(self, d_model: int, d_ff: int, d_cond_norm: int = None, norm_type: Literal['AdaRMS', 'FiLM'] = 'AdaRMS', use_gating: bool = True):
        super().__init__(d_model=d_model, d_ff=d_ff, d_cond_norm=d_cond_norm)
        if not use_gating:
            self.up_proj = LinearSwish(d_model, d_ff, bias=False)
        if norm_type == 'FiLM':
            self.norm = FiLMNorm(d_model, d_cond_norm)

class FFNStack(nn.Module):
    def __init__(self, dim: int, depth: int, ffn_expansion: float, dim_cond: int, 
                 norm_type: Literal['AdaRMS', 'FiLM'] = 'AdaRMS', use_gating: bool = True) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [FeedForwardBlockCustom(d_model=dim, d_ff=int(dim * ffn_expansion), d_cond_norm=dim_cond, norm_type=norm_type, use_gating=use_gating) 
             for _ in range(depth)])

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, cond_norm=cond)
        return x

class FiLMNorm(nn.Module):
    def __init__(self, features, cond_features):
        super().__init__()
        self.linear = Linear(cond_features, features * 2, bias=False)
        self.feature_dim = features

    def forward(self, x, cond):
        B, _, D = x.shape
        scale, shift = self.linear(cond).chunk(2, dim=-1)
        # broadcast scale and shift across all features
        scale = scale.view(B, 1, D)
        shift = scale.view(B, 1, D) 
        return scale * x + shift

class LinearSwish(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, x):
        return F.silu(super().forward(x))
    

class ArgSequential(nn.Module):  # Utility class to enable instantiating nn.Sequential instances with Hydra
    def __init__(self, *layers) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, *args, **kwargs):
        for layer in self.layers:
            x = layer(x, *args, **kwargs)
        return x

class StableFeatureAligner(nn.Module):
    def __init__(
        self,
        ae: nn.Module,
        mapping,
        adapter_layer_class: str,
        feature_dims: dict[str, int],
        feature_extractor_cls: str,
        sd_version: Literal["sd15", "sd21"],
        adapter_layer_params: dict = {},
        use_text_condition: bool = False,
        t_min: int = 1,
        t_max: int = 999,
        t_max_model: int = 999,
        num_t_stratification_bins: int = 3,
        alignment_loss: Literal["cossim", "mse", "l1"] = "cossim",
        train_unet: bool = True,
        train_adapter: bool = True,
        t_init: int = 261,
        learn_timestep: bool = False,
        val_dataset: torch.utils.data.Dataset | None = None,
        val_t: int = 261,
        val_feature_key: str = "us6",
        val_chunk_size: int = 10,
        use_adapters: bool = True
    ):
        super().__init__()
        self.ae = ae
        self.sd_version = sd_version
        self.val_t = val_t
        self.val_feature_key = val_feature_key
        self.val_dataset = val_dataset
        self.val_chunk_size = val_chunk_size
        self.use_adapters = use_adapters

        if sd_version == "sd15":
            self.repo = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        elif sd_version == "sd21":
            self.repo = "stabilityai/stable-diffusion-2-1"
        else:
            raise ValueError(f"Invalid SD version: {sd_version}")

        self.mapping = None
        if use_adapters:
            self.time_emb = FourierFeatures(1, mapping.width)
            self.time_in_proj = Linear(mapping.width, mapping.width, bias=False)
            self.mapping = MappingNetwork(mapping.depth, mapping.width, mapping.d_ff, dropout=mapping.dropout)
            self.mapping.compile()

        if use_adapters:
            self.adapters = nn.ModuleDict()
            for k, dim in feature_dims.items():
                self.adapters[k] = locate(adapter_layer_class)(dim=dim, **adapter_layer_params)
                self.adapters[k].requires_grad_(train_adapter)

        self.unet_feature_extractor_base = locate(feature_extractor_cls)().cuda()
        self.pipe = DiffusionPipeline.from_pretrained(
            self.repo,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        ).to("cuda")
        self.unet_feature_extractor_base.load_state_dict(self.pipe.unet.state_dict())
        self.unet_feature_extractor_base.eval()
        self.unet_feature_extractor_base.requires_grad_(False)
        self.unet_feature_extractor_base.compile()

        self.unet_feature_extractor_cleandift = locate(feature_extractor_cls)().cuda()
        self.unet_feature_extractor_cleandift.load_state_dict(
            {k: v.detach().clone() for k, v in self.unet_feature_extractor_base.state_dict().items()}
        )

        if train_unet or learn_timestep:
            self.unet_feature_extractor_cleandift.train()
        else:
            self.unet_feature_extractor_cleandift.eval()
        self.unet_feature_extractor_cleandift.requires_grad_(train_unet)
        self.unet_feature_extractor_cleandift.compile()

        self.use_text_condition = use_text_condition
        if self.use_text_condition:
            self.pipe.text_encoder.compile()
        else:
            with torch.no_grad():
                prompt_embeds_dict = self.get_prompt_embeds([""])
                self._empty_prompt_embeds = prompt_embeds_dict["prompt_embeds"]
                del self.pipe.text_encoder

        del self.pipe.unet, self.pipe.vae

        self.t_min = t_min
        self.t_max = t_max
        self.t_max_model = t_max_model
        self.num_t_stratification_bins = num_t_stratification_bins
        self.alignment_loss = alignment_loss
        self.timestep = nn.Parameter(
            torch.tensor(float(t_init), requires_grad=learn_timestep), requires_grad=learn_timestep
        )

    def get_prompt_embeds(self, prompt: list[str]) -> dict[str, torch.Tensor | None]:
        self.prompt_embeds, _ = self.pipe.encode_prompt(
            prompt=prompt,
            device=torch.device("cuda"),
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        return {"prompt_embeds": self.prompt_embeds}

    def _get_unet_conds(self, prompts: list[str], device, dtype, N_T) -> dict[str, torch.Tensor]:
        B = len(prompts)
        if self.use_text_condition:
            prompt_embeds_dict = self.get_prompt_embeds(prompts)
        else:
            prompt_embeds_dict = {"prompt_embeds": einops.repeat(self._empty_prompt_embeds, "b ... -> (B b) ...", B=B)}

        unet_conds = {
            "encoder_hidden_states": einops.repeat(
                prompt_embeds_dict["prompt_embeds"], "B ... -> (B N_T) ...", N_T=N_T
            ).to(dtype=dtype, device=device),
            "added_cond_kwargs": {},
        }

        return unet_conds

    def forward(
        self, x: Float[torch.Tensor, "b c h w"], caption: list[str], **kwargs
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        B, *_ = x.shape
        device = x.device
        t_range = self.t_max - self.t_min
        t_range_per_bin = t_range / self.num_t_stratification_bins
        t: Int[torch.Tensor, "B N_T"] = (
            self.t_min
            + torch.rand((B, self.num_t_stratification_bins), device=device) * t_range_per_bin
            + torch.arange(0, self.num_t_stratification_bins, device=device)[None, :] * t_range_per_bin
        ).long()
        B, N_T = t.shape

        with torch.no_grad():
            unet_conds = self._get_unet_conds(caption, device, x.dtype, N_T)
            x_0: Float[torch.Tensor, "(B N_T) ..."] = self.ae.encode(x)
            x_0 = einops.repeat(x_0, "B ... -> (B N_T) ...", N_T=N_T)
            _, *latent_shape = x_0.shape
            noise_sample = torch.randn((B * N_T, *latent_shape), device=device, dtype=x.dtype)
            
            x_t: Float[torch.Tensor, "(B N_T) ..."] = self.pipe.scheduler.add_noise(
                x_0,
                noise_sample,
                einops.rearrange(t, "B N_T -> (B N_T)"),
            )

            feats_base: dict[str, Float[torch.Tensor, "B N_T ..."]] = {
                k: einops.rearrange(v, "(B N_T) D H W -> B N_T (H W) D", B=B, N_T=N_T)
                for k, v in self.unet_feature_extractor_base(
                    x_t,
                    einops.rearrange(t, "B N_T -> (B N_T)"),
                    **unet_conds,
                ).items()
            }

        feats_cleandift: dict[str, Float[torch.Tensor, "B N_T ..."]] = {
            k: einops.rearrange(v, "(B N_T) D H W -> B N_T (H W) D", N_T=N_T)
            for k, v in self.unet_feature_extractor_cleandift(
                x_0,
                einops.rearrange(torch.ones_like(t) * self.timestep, "B N_T -> (B N_T)"),
                **unet_conds,
            ).items()
        }

        if self.use_adapters:
            # time conditioning for adapters
            if not self.mapping is None:
                map_cond: Float[torch.Tensor, "(B N_T) ..."] = self.mapping(
                    self.time_in_proj(
                        self.time_emb(
                            einops.rearrange(t, "B N_T -> (B N_T) 1").to(dtype=x.dtype, device=device) / self.t_max_model
                        )
                    )
                )
   
            feats_cleandift: dict[str, Float[torch.Tensor, "B N_T ..."]] = {
                k: einops.rearrange(
                    self.adapters[k](einops.rearrange(v, "B N_T ... -> (B N_T) ..."), cond=map_cond),
                    "(B N_T) ... -> B N_T ...",
                    B=B,
                    N_T=N_T,
                )
                for k, v in feats_cleandift.items()
            }

        if self.alignment_loss == "mse":
            return {f"mse_{k}": F.mse_loss(feats_cleandift[k], v.detach()) for k, v in feats_base.items()}
        elif self.alignment_loss == "l1":
            return {f"l1_{k}": F.l1_loss(feats_cleandift[k], v.detach()) for k, v in feats_base.items()}
        elif self.alignment_loss == "cossim":
            return {
                f"neg_cossim_{k}": -F.cosine_similarity(feats_cleandift[k], v.detach(), dim=-1).mean()
                for k, v in feats_base.items()
            }
        else:
            raise ValueError(f"Invalid alignment loss type: {self.alignment_loss}")

    @torch.no_grad()
    def get_features(
        self,
        x: Float[torch.Tensor, "b c h w"],
        caption: list[str] | None,
        t: Int[torch.Tensor, "b"] | None,
        feat_key: str,
        use_base_model: bool = False,
        input_pure_noise: bool = False,
        eps: torch.Tensor = None,
    ) -> Float[torch.Tensor, "b d h' w'"]:
        if use_base_model:
            assert not t is None
            B, *_ = x.shape

            if caption is None:
                caption = [""] * B

            unet_conds = self._get_unet_conds(caption, x.device, x.dtype, 1)
            x_0 = self.ae.encode(x)
            eps = torch.randn_like(x_0) if eps is None else eps
            if input_pure_noise:
                assert torch.allclose(
                    t, torch.full_like(t, 999)
                ), "Sanity check. Pure noise means that no x_t is given to the U-Net, just pure noise (eps)."
                x_t = eps
            else:
                x_t = self.pipe.scheduler.add_noise(x_0, eps, t)
            
            if feat_key is None:
                feats = self.unet_feature_extractor_base(x_t, t, **unet_conds)
            else:
                feats = self.unet_feature_extractor_base(x_t, t, **unet_conds)[feat_key]
            return feats
        else:
            (B, *_), device = x.shape, x.device

            if caption is None:
                caption = [""] * B

            unet_conds = self._get_unet_conds(caption, device, x.dtype, 1)
            x_0 = self.ae.encode(x)

            feats = self.unet_feature_extractor_cleandift(
                    x_0,
                    torch.ones((B,), device=device, dtype=self.timestep.dtype) * self.timestep,
                    **unet_conds,
                )
            
            if feat_key is not None:
                feats = feats[feat_key]

            feats = einops.rearrange(feats,"B D H W -> B H W D",)
            if t is None:
                return einops.rearrange(feats, "B H W D -> B D H W")
            else:
                assert self.use_adapters, "Adapters must be enabled to use t conditioning on cleandift model"
                map_cond: Float[torch.Tensor, "B ..."] = self.mapping(
                    self.time_in_proj(self.time_emb(t[:, None].to(dtype=x.dtype, device=device) / self.t_max_model))
                )
                if feat_key is not None:
                    return einops.rearrange(self.adapters[feat_key](feats, cond=map_cond), "B H W D -> B D H W")
                else:
                    return {key: einops.rearrange(self.adapters[key](feats[key], cond=map_cond), "B H W D -> B D H W") for key in feats.keys()}