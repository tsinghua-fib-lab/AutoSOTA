import torch
import numpy as np
import torch.nn.functional as F
from dfm_utils.laplacian_decomposer import LaplacianDecomposer2D


def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))

class SILoss:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            accelerator=None, 
            latents_scale=None, 
            latents_bias=None,
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t


    def __call__(self, model, images, model_kwargs=None, zs=None):
        if model_kwargs == None:
            model_kwargs = {}
        # sample timesteps
        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], 1, 1, 1))
        elif self.weighting == "lognormal":
            # sample timestep according to log-normal distribution of sigmas following EDM
            rnd_normal = torch.randn((images.shape[0], 1 ,1, 1))
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)
                
        time_input = time_input.to(device=images.device, dtype=images.dtype)
        
        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
            
        model_input = alpha_t * images + sigma_t * noises
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
        else:
            raise NotImplementedError() # TODO: add x or eps prediction
        model_output, zs_tilde  = model(model_input, time_input.flatten(), **model_kwargs)
        denoising_loss = mean_flat((model_output - model_target) ** 2)

        # REPA projection loss (only computed when both encoder features and
        # model projector outputs are available)
        proj_loss = 0.
        if zs is not None and zs_tilde is not None:
            bsz = zs[0].shape[0]
            for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
                for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)):
                    z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1) 
                    z_j = torch.nn.functional.normalize(z_j, dim=-1) 
                    proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
            proj_loss /= (len(zs) * bsz)

        return denoising_loss, proj_loss





class DFMSILoss(SILoss):
    def __init__(
            self,
            *args,
            num_stages=2,
            stage_weights=[0.9, 0.1],
            **kwargs):
        # Remove 'encoders' from kwargs if present (legacy compat)
        kwargs.pop('encoders', None)
        super().__init__(*args, **kwargs)
        self.num_stages = num_stages
        self.stage_weights = stage_weights
        self.laplacian_decomposer = LaplacianDecomposer2D(stages_count=num_stages)

    def __call__(self, model, images, model_kwargs=None, zs=None):
        if model_kwargs == None:
            model_kwargs = {}
        
        # conver the image into multiscale
        multiscale_images = self.laplacian_decomposer.decompose(images)
        
        # sample stage index
        sampled_stage_idx = np.random.choice(self.num_stages, p=self.stage_weights)
        # drop mask
        drop_mask = torch.zeros(self.num_stages)
        for i in range(sampled_stage_idx + 1):
            drop_mask[i] = 1.0
        
        # sample timesteps for num_stages
        time_inputs = []
        for stage_idx in range(sampled_stage_idx+1):
            if stage_idx < sampled_stage_idx:
                # Earlier stages: sample from logit-normal distribution with location -2.0 (shifted to lower noise timesteps to simulate the inference behaviour)
                rnd_normal = torch.randn((images.shape[0], 1, 1, 1))
                logit_t = rnd_normal - 2.0  # location = -2.0
                time_input = torch.sigmoid(logit_t)
            else:
                # Last active stage: use the original sampling method
                if self.weighting == "uniform":
                    time_input = torch.rand((images.shape[0], 1, 1, 1))
                elif self.weighting == "lognormal":
                    # sample timestep according to log-normal distribution of sigmas following EDM
                    rnd_normal = torch.randn((images.shape[0], 1 ,1, 1))
                    sigma = rnd_normal.exp()
                    if self.path_type == "linear":
                        time_input = sigma / (1 + sigma)
                    elif self.path_type == "cosine":
                        time_input = 2 / np.pi * torch.atan(sigma)
            
            time_input = time_input.to(device=images.device, dtype=images.dtype)
            time_inputs.append(time_input)
        
        # for the rest stages that are masked, just append dummy ones
        for stage_idx in range(sampled_stage_idx+1, self.num_stages):
            dummy_time_input = torch.ones((images.shape[0], 1, 1, 1), device=images.device, dtype=images.dtype)
            time_inputs.append(dummy_time_input)
            
        time_inputs = torch.stack(time_inputs, dim=-1)  # shape: [batch_size, 1, 1, 1, num_stages]
        
        # create noisy multiscale inputs for all stages
        noisy_multiscale_model_input = {}
        multiscale_model_target = {}
        for stage_idx in range(self.num_stages):
            time_input = time_inputs[:, :, :, :, stage_idx]
            current_stage_image = multiscale_images[stage_idx]
            noises = torch.randn_like(current_stage_image)
            alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
            
            model_input = alpha_t * current_stage_image + sigma_t * noises
            noisy_multiscale_model_input[stage_idx] = model_input
            
            model_target = d_alpha_t * current_stage_image + d_sigma_t * noises
            multiscale_model_target[stage_idx] = model_target
            
            # Debug: Check for NaN/Inf in inputs
            if torch.isnan(model_input).any() or torch.isinf(model_input).any():
                print(f"WARNING: NaN/Inf in model_input for stage {stage_idx}")
                print(f"  time_input: {time_input[0].item()}")
                print(f"  alpha_t: {alpha_t[0].item()}, sigma_t: {sigma_t[0].item()}")
                print(f"  image stats: min={current_stage_image.min().item()}, max={current_stage_image.max().item()}, mean={current_stage_image.mean().item()}")
                print(f"  noise stats: min={noises.min().item()}, max={noises.max().item()}, mean={noises.mean().item()}")
        
        time_inputs_2d = time_inputs.view(images.shape[0], -1)  # shape: [batch_size, num_stages]
        
        # Debug: Check inputs before model forward
        for stage_idx in range(self.num_stages):
            inp = noisy_multiscale_model_input[stage_idx]
            if torch.isnan(inp).any() or torch.isinf(inp).any():
                print(f"ERROR: NaN/Inf in stage {stage_idx} input before model forward!")
                print(f"  Shape: {inp.shape}, Range: [{inp.min():.4f}, {inp.max():.4f}]")
        
        multiscale_model_output, zs_tilde  = model(noisy_multiscale_model_input, time_inputs_2d, drop_mask=drop_mask, **model_kwargs)
        
        # Debug: Check outputs after model forward
        for stage_idx in range(self.num_stages):
            if stage_idx in multiscale_model_output:
                out = multiscale_model_output[stage_idx]
                if torch.isnan(out).any() or torch.isinf(out).any():
                    print(f"ERROR: NaN/Inf in stage {stage_idx} MODEL OUTPUT!")
                    print(f"  Shape: {out.shape}, contains NaN: {torch.isnan(out).sum().item()}/{out.numel()}")
                    print(f"  Range: [{out.min():.4f}, {out.max():.4f}]")
                elif drop_mask[stage_idx]:
                    # Only log stats for active stages occasionally
                    if np.random.rand() < 0.01:  # 1% of the time
                        print(f"Stage {stage_idx} output: mean={out.mean():.4f}, std={out.std():.4f}, range=[{out.min():.4f}, {out.max():.4f}]")
        
        total_denoising_loss = 0.
        total_proj_loss = 0.
        
        for stage_idx in range(self.num_stages):
            if self.prediction == 'v':
                model_target = multiscale_model_target[stage_idx]
            else:
                raise NotImplementedError() # TODO: add x or eps prediction
            
            model_output = multiscale_model_output[stage_idx]
            denoising_loss = mean_flat((model_output - model_target) ** 2)
            
            # Debug: Check loss values
            if torch.isnan(denoising_loss).any() or torch.isinf(denoising_loss).any():
                print(f"ERROR: NaN/Inf in denoising_loss for stage {stage_idx}!")
                print(f"  model_output range: [{model_output.min():.4f}, {model_output.max():.4f}]")
                print(f"  model_target range: [{model_target.min():.4f}, {model_target.max():.4f}]")
                print(f"  loss value: {denoising_loss}")
            
            # cancel the loss if the scale is dropped
            current_drop_mask = drop_mask[stage_idx]
            weighted_loss = denoising_loss * current_drop_mask
            
            # Debug: Check weighted loss and log magnitude per scale
            if torch.isnan(weighted_loss).any() or torch.isinf(weighted_loss).any():
                print(f"ERROR: NaN/Inf in weighted_loss for stage {stage_idx}!")
                print(f"  drop_mask: {current_drop_mask}")
                print(f"  denoising_loss: {denoising_loss}")
            elif current_drop_mask.item() > 0 and np.random.rand() < 0.01:  # Log 1% of active stages
                print(f"Stage {stage_idx} loss: mean={denoising_loss.mean():.6f}, is_active={current_drop_mask.item()}")
            
            total_denoising_loss += weighted_loss
            
        
        # REPA projection loss (only computed when both encoder features and
        # model projector outputs are available)
        proj_loss = 0.
        if zs is not None and zs_tilde is not None:
            bsz = zs[0].shape[0]
            for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
                for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)):
                    z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1) 
                    z_j = torch.nn.functional.normalize(z_j, dim=-1) 
                    proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
            proj_loss /= (len(zs) * bsz)
        total_proj_loss += proj_loss
        
        # Debug: Check final loss values
        if isinstance(total_denoising_loss, torch.Tensor):
            if torch.isnan(total_denoising_loss).any() or torch.isinf(total_denoising_loss).any():
                print(f"ERROR: NaN/Inf in TOTAL denoising_loss!")
                print(f"  total_denoising_loss: {total_denoising_loss}")
        if isinstance(total_proj_loss, torch.Tensor):
            if torch.isnan(total_proj_loss).any() or torch.isinf(total_proj_loss).any():
                print(f"ERROR: NaN/Inf in TOTAL proj_loss!")
                print(f"  total_proj_loss: {total_proj_loss}")

        return total_denoising_loss, total_proj_loss
