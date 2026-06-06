import types
from typing import List, Optional
import torch
from torch import nn
import torch.distributed as dist

from utils.scheduler import SchedulerInterface, FlowMatchScheduler
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.model import WanModel, RegisterTokens, GanAttentionBlock
from wan.modules.vae import _video_vae
from wan.modules.t5 import umt5_xxl
from peft import LoraConfig, inject_adapter_in_model, get_peft_model
import torch.cuda.amp as amp
# from wan.modules.causal_model import CausalWanModel


class WanTextEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device=torch.device('cpu')
        ).eval().requires_grad_(False)
        import os

        self.text_encoder.load_state_dict(
            torch.load("Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth",
                        map_location='cpu', weights_only=False)
        )

        self.tokenizer = HuggingfaceTokenizer(
            name="Wan2.1-T2V-14B/google/umt5-xxl/", seq_len=512, clean='whitespace')


    @property
    def device(self):
        # Assume we are always on GPU
        return torch.cuda.current_device()

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0

        return {
            "prompt_embeds": context
        }


class WanVAEWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        # init model
        import os

        self.model = _video_vae(
            pretrained_path="Wan2.1-T2V-14B/Wan2.1_VAE.pth",
            z_dim=16,
        ).eval().requires_grad_(False)


    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [batch_size, num_channels, num_frames, height, width]
        device, dtype = pixel.device, pixel.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        print(pixel.dtype)
        print(next(self.model.parameters()).dtype)
        output = [
            self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
            for u in pixel
        ]
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        with amp.autocast(dtype=torch.float):
            output = []
            for u in zs:
                output.append(decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
            output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output


class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
            self,
            model_name="Wan2.1-T2V-1.3B",
            timestep_shift=8.0,
            is_causal=False,
            local_attn_size=-1,
            sink_size=0,
            use_lora=False,
            dit_path=None,
    ):
        super().__init__()

        print('model_name:', model_name)
        if is_causal:
            self.model = CausalWanModel.from_pretrained(
                f"wan_models/{model_name}/", local_attn_size=local_attn_size, sink_size=sink_size)
        else:
            import os
            if "1.3" in model_name:
                self.model = WanModel.from_pretrained(f"Wan2.1-T2V-1.3B/")
            else:
                self.model = WanModel.from_pretrained(f"Wan2.1-T2V-14B/")

        self.model.eval()

        if use_lora:  # use acc wan to initialize the generator
            from safetensors.torch import load_file
            lora_checkpoint_dir = 'Wan21_acc_T2V_14B_lora_rank32_fp16.safetensors'
            lora_state_dict = load_file(lora_checkpoint_dir)
            lora_state_dict = {
                key.replace("diffusion_model.", "") if key.startswith("diffusion_model.") else key: value
                for key, value in lora_state_dict.items()
            }
            target_modules = []
            for idx in range(0, 40):
                target_modules.append(f"blocks.{idx}.cross_attn.k")
                target_modules.append(f"blocks.{idx}.cross_attn.q")
                target_modules.append(f"blocks.{idx}.cross_attn.v")
                target_modules.append(f"blocks.{idx}.cross_attn.o")
                target_modules.append(f"blocks.{idx}.self_attn.k")
                target_modules.append(f"blocks.{idx}.self_attn.q")
                target_modules.append(f"blocks.{idx}.self_attn.v")
                target_modules.append(f"blocks.{idx}.self_attn.o")
                target_modules.append(f"blocks.{idx}.ffn.0")
                target_modules.append(f"blocks.{idx}.ffn.2")

            target_modules.append(f"head.head")
            target_modules.append(f"text_embedding.0")
            target_modules.append(f"text_embedding.2")
            target_modules.append(f"time_embedding.0")
            target_modules.append(f"time_embedding.2")

            base_state_dict = self.model.state_dict()

            lora_rank = 32
            lora_alpha = lora_rank
            lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
            self.model = inject_adapter_in_model(lora_config, self.model)

            copy_state_dict = self.model.state_dict()

            for idx in range(0, 40):
                # attn
                for layer in ['q', 'k', 'v', 'o']:
                    # cross
                    copy_state_dict[f"blocks.{idx}.cross_attn.{layer}.base_layer.weight"] = base_state_dict[
                        f"blocks.{idx}.cross_attn.{layer}.weight"]
                    copy_state_dict[f"blocks.{idx}.cross_attn.{layer}.base_layer.bias"] = (
                            base_state_dict[f"blocks.{idx}.cross_attn.{layer}.bias"] + lora_state_dict[
                        f"blocks.{idx}.cross_attn.{layer}.diff_b"]
                    )
                    copy_state_dict[f"blocks.{idx}.cross_attn.{layer}.lora_A.default.weight"] = lora_state_dict[
                        f"blocks.{idx}.cross_attn.{layer}.lora_down.weight"]
                    copy_state_dict[f"blocks.{idx}.cross_attn.{layer}.lora_B.default.weight"] = lora_state_dict[
                        f"blocks.{idx}.cross_attn.{layer}.lora_up.weight"]
                    if layer == 'q' or layer == 'k':
                        copy_state_dict[f"blocks.{idx}.cross_attn.norm_{layer}.weight"] = (
                                base_state_dict[f"blocks.{idx}.cross_attn.norm_{layer}.weight"] + lora_state_dict[
                            f"blocks.{idx}.cross_attn.norm_{layer}.diff"]
                        )

                    # self
                    copy_state_dict[f"blocks.{idx}.self_attn.{layer}.base_layer.weight"] = base_state_dict[
                        f"blocks.{idx}.self_attn.{layer}.weight"]
                    copy_state_dict[f"blocks.{idx}.self_attn.{layer}.base_layer.bias"] = (
                            base_state_dict[f"blocks.{idx}.self_attn.{layer}.bias"] + lora_state_dict[
                        f"blocks.{idx}.self_attn.{layer}.diff_b"]
                    )
                    copy_state_dict[f"blocks.{idx}.self_attn.{layer}.lora_A.default.weight"] = lora_state_dict[
                        f"blocks.{idx}.self_attn.{layer}.lora_down.weight"]
                    copy_state_dict[f"blocks.{idx}.self_attn.{layer}.lora_B.default.weight"] = lora_state_dict[
                        f"blocks.{idx}.self_attn.{layer}.lora_up.weight"]
                    if layer == 'q' or layer == 'k':
                        copy_state_dict[f"blocks.{idx}.self_attn.norm_{layer}.weight"] = (
                                base_state_dict[f"blocks.{idx}.self_attn.norm_{layer}.weight"] + lora_state_dict[
                            f"blocks.{idx}.self_attn.norm_{layer}.diff"]
                        )

                # ffn
                for idx_ffn in ['0', '2']:
                    copy_state_dict[f"blocks.{idx}.ffn.{idx_ffn}.base_layer.weight"] = base_state_dict[
                        f"blocks.{idx}.ffn.{idx_ffn}.weight"]
                    copy_state_dict[f"blocks.{idx}.ffn.{idx_ffn}.base_layer.bias"] = (
                            base_state_dict[f"blocks.{idx}.ffn.{idx_ffn}.bias"] + lora_state_dict[
                        f"blocks.{idx}.ffn.{idx_ffn}.diff_b"]
                    )
                    copy_state_dict[f"blocks.{idx}.ffn.{idx_ffn}.lora_A.default.weight"] = lora_state_dict[
                        f"blocks.{idx}.ffn.{idx_ffn}.lora_down.weight"]
                    copy_state_dict[f"blocks.{idx}.ffn.{idx_ffn}.lora_B.default.weight"] = lora_state_dict[
                        f"blocks.{idx}.ffn.{idx_ffn}.lora_up.weight"]

            # head.head
            copy_state_dict[f"head.head.base_layer.weight"] = base_state_dict[f"head.head.weight"]
            copy_state_dict[f"head.head.base_layer.bias"] = (
                    base_state_dict[f"head.head.bias"] + lora_state_dict[f"head.head.diff_b"]
            )
            copy_state_dict[f"head.head.lora_A.default.weight"] = lora_state_dict[f"head.head.lora_down.weight"]
            copy_state_dict[f"head.head.lora_B.default.weight"] = lora_state_dict[f"head.head.lora_up.weight"]

            copy_state_dict[f"patch_embedding.bias"] = (
                    base_state_dict[f"patch_embedding.bias"] + lora_state_dict[f"patch_embedding.diff_b"]
            )

            copy_state_dict[f"time_projection.1.bias"] = (
                    base_state_dict[f"time_projection.1.bias"] + lora_state_dict[f"time_projection.1.diff_b"]
            )

            # text_embedding
            for idx_ffn in ['0', '2']:
                copy_state_dict[f"text_embedding.{idx_ffn}.base_layer.weight"] = base_state_dict[
                    f"text_embedding.{idx_ffn}.weight"]
                copy_state_dict[f"text_embedding.{idx_ffn}.base_layer.bias"] = (
                        base_state_dict[f"text_embedding.{idx_ffn}.bias"] + lora_state_dict[
                    f"text_embedding.{idx_ffn}.diff_b"]
                )
                copy_state_dict[f"text_embedding.{idx_ffn}.lora_A.default.weight"] = lora_state_dict[
                    f"text_embedding.{idx_ffn}.lora_down.weight"]
                copy_state_dict[f"text_embedding.{idx_ffn}.lora_B.default.weight"] = lora_state_dict[
                    f"text_embedding.{idx_ffn}.lora_up.weight"]

            # time_embedding
            for idx_ffn in ['0', '2']:
                copy_state_dict[f"time_embedding.{idx_ffn}.base_layer.weight"] = base_state_dict[
                    f"time_embedding.{idx_ffn}.weight"]
                copy_state_dict[f"time_embedding.{idx_ffn}.base_layer.bias"] = (
                        base_state_dict[f"time_embedding.{idx_ffn}.bias"] + lora_state_dict[
                    f"time_embedding.{idx_ffn}.diff_b"]
                )
                copy_state_dict[f"time_embedding.{idx_ffn}.lora_A.default.weight"] = lora_state_dict[
                    f"time_embedding.{idx_ffn}.lora_down.weight"]
                copy_state_dict[f"time_embedding.{idx_ffn}.lora_B.default.weight"] = lora_state_dict[
                    f"time_embedding.{idx_ffn}.lora_up.weight"]

            self.model.load_state_dict(copy_state_dict, strict=True)
            self.model = self.model.to(torch.bfloat16)

        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = not is_causal

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        self.seq_len = 32760  # [1, 21, 16, 60, 104]
        self.post_init()

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def adding_cls_branch(self, atten_dim=1536, num_class=4, time_embed_dim=0) -> None:
        self._cls_pred_branch = nn.Sequential(
            # Input: [B, 384, 21, 60, 104]
            nn.LayerNorm(atten_dim * 3 + time_embed_dim),
            nn.Linear(atten_dim * 3 + time_embed_dim, atten_dim),
            nn.SiLU(),
            nn.Linear(atten_dim, num_class)
        )
        self._cls_pred_branch.requires_grad_(True)
        num_registers = 3
        self._register_tokens = RegisterTokens(num_registers=num_registers, dim=atten_dim)
        self._register_tokens.requires_grad_(True)

        head_number = 16 if atten_dim == 1536 else 20

        gan_ca_blocks = []
        for _ in range(num_registers):
            block = GanAttentionBlock(dim=atten_dim, num_heads=head_number)
            gan_ca_blocks.append(block)
        self._gan_ca_blocks = nn.ModuleList(gan_ca_blocks)
        self._gan_ca_blocks.requires_grad_(True)
        # self.has_cls_branch = True

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )

        # timestep_id = torch.argmin(
        #     (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        # sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        sigma_t = timestep.reshape(-1, 1, 1, 1) / 1000.0
        print('pred_to_x0:', timestep[0], sigma_t[0,0,0,0])
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt,
                                                      scheduler.sigmas,
                                                      scheduler.timesteps]
        )
        # timestep_id = torch.argmin(
        #     (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        # sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        sigma_t = timestep.reshape(-1, 1, 1, 1) / 1000.0
        print('x0_to_pred:', timestep[0], sigma_t[0, 0, 0, 0])
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        classify_mode: Optional[bool] = False,
        concat_time_embeddings: Optional[bool] = False,
        clean_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
        cache_start: Optional[int] = None
    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]

        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        logits = None
        # X0 prediction
        if kv_cache is not None:
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start
            ).permute(0, 2, 1, 3, 4)
        else:
            if clean_x is not None:
                # teacher forcing
                flow_pred = self.model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4),
                    t=input_timestep, context=prompt_embeds,
                    seq_len=self.seq_len,
                    clean_x=clean_x.permute(0, 2, 1, 3, 4),
                    aug_t=aug_t,
                ).permute(0, 2, 1, 3, 4)
            else:
                if classify_mode:
                    flow_pred, logits = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=self.seq_len,
                        classify_mode=True,
                        register_tokens=self._register_tokens,
                        cls_pred_branch=self._cls_pred_branch,
                        gan_ca_blocks=self._gan_ca_blocks,
                        concat_time_embeddings=concat_time_embeddings
                    )
                    flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                else:
                    flow_pred = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=self.seq_len
                    ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])

        if logits is not None:
            return flow_pred, pred_x0, logits

        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()
