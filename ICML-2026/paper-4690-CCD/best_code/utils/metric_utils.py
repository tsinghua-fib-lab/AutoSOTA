import torch
import numpy as np
from utils.optim_utils import set_random_seed


# =========================================================================================
# Utility Functions
# =========================================================================================

def sigma_from_scheduler(pipe, t, dtype, device):
    """Calculate sigma from scheduler"""
    alphas_cumprod = pipe.scheduler.alphas_cumprod.to(device=device, dtype=dtype)
    alpha_bar = alphas_cumprod[t.long()]
    return torch.sqrt(1 - alpha_bar)

def rademacher_like(x):
    """Generate Rademacher random variable (±1)"""
    return torch.empty_like(x).bernoulli_(0.5).mul_(2).sub_(1)

def hutchinson_diag_from_output(eps, z_input, t, pipe, k=16, h_seed=None):
    """Estimate diagonal of Hessian using Hutchinson estimator"""
    #sigma = sigma_from_scheduler(pipe, t, z_input.dtype, eps.device)
    # For evaluation purposes, we don't divide by sigma to avoid numerical instability
    # The relative magnitude is preserved, which is sufficient for IoU/Accuracy metrics
    s = -eps  # Instead of: s = -eps / sigma
 
    diag = torch.zeros_like(z_input)
    
    if h_seed is not None:
        gen = torch.Generator(device=z_input.device).manual_seed(h_seed)
    else:
        gen = None

    for _ in range(k):
        if gen is not None:
            rnd = torch.empty_like(z_input).uniform_(0, 1, generator=gen)
            v = (rnd > 0.5).to(z_input.dtype).mul_(2).sub_(1)
        else:
            v = rademacher_like(z_input)
            
        dot = (s * v).sum()
        (jtv,) = torch.autograd.grad(dot, z_input, retain_graph=True, create_graph=False)
        diag += v * jtv
    
    diag /= float(k)
    return diag.detach()

# =========================================================================================
# Metric Computation Functions
# =========================================================================================

def compute_diag_h_diff_only(prompt, pipe, args, text_embeddings, seed):
    """Compute diag_h (Hessian diagonal) only"""
    set_random_seed(seed)
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    
    pipe.scheduler.set_timesteps(args.num_inference_steps, device=pipe.device)
    timesteps = pipe.scheduler.timesteps
    
    latents = torch.randn(
        (1, pipe.unet.config.in_channels, 64, 64),
        device=pipe.device,
        dtype=torch.float16,
        generator=generator
    )
    latents = latents * pipe.scheduler.init_noise_sigma
    
    diag_h_tensor = None
    
    for i, t in enumerate(timesteps):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        
        if i == args.target_timestep:
            latent_for_hess = latents.clone().detach().requires_grad_(True)
            latent_model_input_hess = torch.cat([latent_for_hess] * 2)
            latent_model_input_hess = pipe.scheduler.scale_model_input(latent_model_input_hess, t)
            
            with torch.enable_grad():
                noise_pred = pipe.unet(
                    latent_model_input_hess,
                    t,
                    encoder_hidden_states=text_embeddings,
                ).sample
            
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            
            # Using difference for Hessian calculation
            base_h_seed = getattr(args, 'hutchinson_seed', None)
            actual_h_seed = base_h_seed + seed * 10000 if base_h_seed is not None else None
            diag_h = hutchinson_diag_from_output(
                (noise_pred_text - noise_pred_uncond), 
                latent_for_hess,
                t,
                pipe,
                k=args.hutchinson_k,
                h_seed=actual_h_seed
            )
            
            diag_h_tensor = -diag_h.sum(dim=1)[0]
            diag_h_tensor = diag_h_tensor.relu()
            diag_h_tensor = diag_h_tensor.cpu().numpy()
            
            break
        
        # Advance scheduler
        with torch.no_grad():
             noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
            ).sample
        
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
    
    if diag_h_tensor is None:
        # Check safety like in generate_metric_maps.py
        # If target timestep wasn't reached for some reason (e.g. loops mismatched)
        pass 
        
    return diag_h_tensor

def compute_diag_h_cond_only(prompt, pipe, args, text_embeddings, seed):
    """Compute diag_h (Hessian diagonal) for Conditional Term ONLY"""
    set_random_seed(seed)
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    
    pipe.scheduler.set_timesteps(args.num_inference_steps, device=pipe.device)
    timesteps = pipe.scheduler.timesteps
    
    latents = torch.randn(
        (1, pipe.unet.config.in_channels, 64, 64),
        device=pipe.device,
        dtype=torch.float16,
        generator=generator
    )
    latents = latents * pipe.scheduler.init_noise_sigma
    
    diag_h_tensor = None
    
    for i, t in enumerate(timesteps):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        
        if i == args.target_timestep:
            # We only need ONE latent for gradient (Cond)
            # But pipe expects batch 2 (Uncond, Cond) usually.
            # We can run batch 2 but only backprop through Cond.
            
            latent_for_hess = latents.clone().detach().requires_grad_(True)
            latent_model_input_hess = torch.cat([latent_for_hess] * 2)
            latent_model_input_hess = pipe.scheduler.scale_model_input(latent_model_input_hess, t)
            
            with torch.enable_grad():
                noise_pred = pipe.unet(
                    latent_model_input_hess,
                    t,
                    encoder_hidden_states=text_embeddings,
                ).sample
            
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            
            # Using ONLY noise_pred_text (Conditional) for Hessian
            base_h_seed = getattr(args, 'hutchinson_seed', None)
            actual_h_seed = base_h_seed + seed * 10000 if base_h_seed is not None else None
            diag_h = hutchinson_diag_from_output(
                noise_pred_text, 
                latent_for_hess,
                t,
                pipe,
                k=args.hutchinson_k,
                h_seed=actual_h_seed
            )
            
            diag_h_tensor = -diag_h.sum(dim=1)[0]
            diag_h_tensor = diag_h_tensor.relu()
            diag_h_tensor = diag_h_tensor.cpu().numpy()
            
            break
        
        # Advance scheduler
        with torch.no_grad():
             noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
            ).sample
        
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
    
    return diag_h_tensor

def compute_score_diffs_only(prompt, pipe, args, text_embeddings, seed):
    """Compute score difference metrics"""
    set_random_seed(seed)
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    
    pipe.scheduler.set_timesteps(args.num_inference_steps, device=pipe.device)
    timesteps = pipe.scheduler.timesteps
    
    latents = torch.randn(
        (1, pipe.unet.config.in_channels, 64, 64),
        device=pipe.device,
        dtype=torch.float16,
        generator=generator
    )
    latents = latents * pipe.scheduler.init_noise_sigma
    
    target_diff_norm_sum = torch.zeros(latents.shape[2], latents.shape[3], device=pipe.device, dtype=latents.dtype)
    count = 0
    
    for i, t in enumerate(timesteps):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        
        with torch.no_grad():
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
            ).sample
        
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        
        if i >= args.score_diff_step_start and i <= args.score_diff_step_end:
            if args.is_score:
                alpha_bar = pipe.scheduler.alphas_cumprod[t]
                sigma = torch.sqrt(1 - alpha_bar)
                target_cond = -noise_pred_cond / sigma
                target_uncond = -noise_pred_uncond / sigma
            else:
                target_cond = noise_pred_cond
                target_uncond = noise_pred_uncond
            
            # 1. ||s_cond - s_uncond||^2
            if getattr(args, 'use_score_sq', False):
                target_diff = target_cond
            else:
                target_diff = target_cond - target_uncond
            diff_norm = target_diff ** 2
            diff_norm = diff_norm.sum(dim=1)[0]
            target_diff_norm_sum += diff_norm
            
            count += 1
        
        noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
    
    if count == 0:
        return np.zeros((64, 64))

    target_diff_norm = target_diff_norm_sum / count
      
    return target_diff_norm.cpu().numpy()


def compute_metrics_for_prompt(prompt, pipe, args, seed):
    """Compute all metrics for a prompt"""
    device = pipe.device
    
    # Text Embeddings
    text_input = pipe.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipe.text_encoder(text_input.input_ids.to(device))[0]
    uncond_input = pipe.tokenizer(
        [""],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    )
    uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    
    results = {}

    # 1. diag_h (cov)
    if "cov" in args.metrics:
        results["cov"] = compute_diag_h_diff_only(prompt, pipe, args, text_embeddings, seed)
    

    
    # 3. Score Diffs / Score Sq
    if "score_diff" in args.metrics:
        results["score_diff"] = compute_score_diffs_only(prompt, pipe, args, text_embeddings, seed)
    if "score_sq" in args.metrics:
        results["score_sq"] = compute_score_diffs_only(prompt, pipe, args, text_embeddings, seed)
    
    return results


# =========================================================================================
# Bad Model Metric Functions
# =========================================================================================

def compute_diag_h_diff_only_with_bad_model(prompt, pipe, bad_unet, args, text_embeddings, seed, bad_text_embeddings=None):
    """Compute diag_h (Hessian diagonal) using bad_model as baseline"""
    set_random_seed(seed)
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    
    pipe.scheduler.set_timesteps(args.num_inference_steps, device=pipe.device)
    timesteps = pipe.scheduler.timesteps
    
    # Latent init
    latents = torch.randn(
        (1, pipe.unet.config.in_channels, 64, 64),
        device=pipe.device,
        dtype=torch.float16,
        generator=generator
    )
    latents = latents * pipe.scheduler.init_noise_sigma
    
    diag_h_tensor = None
    
    for i, t in enumerate(timesteps):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        
        if i == args.target_timestep:
            latent_for_hess = latents.clone().detach().requires_grad_(True)
            
            latent_model_input_hess = torch.cat([latent_for_hess] * 2) 
            latent_model_input_hess = pipe.scheduler.scale_model_input(latent_model_input_hess, t)
            
            with torch.enable_grad():
                noise_pred = pipe.unet(
                    latent_model_input_hess,
                    t,
                    encoder_hidden_states=text_embeddings,
                ).sample
            
            _, noise_pred_text = noise_pred.chunk(2)
            
            # Bad model inference (Gradient must flow through inputs for Hessian)
            with torch.enable_grad(): 
                 bad_noise_pred = bad_unet(
                    latent_model_input_hess,
                    t,
                    encoder_hidden_states=bad_text_embeddings if bad_text_embeddings is not None else text_embeddings,
                 ).sample
            
            _, bad_noise_pred_text = bad_noise_pred.chunk(2)
            
            # Target for Hessian: (Current_Cond - Bad_Cond)
            target_diff = noise_pred_text - bad_noise_pred_text
            
            base_h_seed = getattr(args, 'hutchinson_seed', None)
            actual_h_seed = base_h_seed + seed * 10000 if base_h_seed is not None else None
            diag_h = hutchinson_diag_from_output(
                target_diff, 
                latent_for_hess,
                t,
                pipe,
                k=args.hutchinson_k,
                h_seed=actual_h_seed
            )
            
            # Process diag_h
            diag_h_tensor = -diag_h.sum(dim=1)[0]
            diag_h_tensor = diag_h_tensor.relu()
            diag_h_tensor = diag_h_tensor.cpu().numpy()
            
            break
        
        # Advance scheduler
        with torch.no_grad():
             noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
            ).sample
        
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
    
    if diag_h_tensor is None:
        pass
        
    return diag_h_tensor


def compute_score_diffs_only_with_bad_model(prompt, pipe, bad_unet, args, text_embeddings, seed, bad_text_embeddings=None):
    """Compute score difference metrics using bad_model as baseline"""
    set_random_seed(seed)
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    
    pipe.scheduler.set_timesteps(args.num_inference_steps, device=pipe.device)
    timesteps = pipe.scheduler.timesteps
    
    latents = torch.randn(
        (1, pipe.unet.config.in_channels, 64, 64),
        device=pipe.device,
        dtype=torch.float16,
        generator=generator
    )
    latents = latents * pipe.scheduler.init_noise_sigma
    
    target_diff_norm_sum = torch.zeros(latents.shape[2], latents.shape[3], device=pipe.device, dtype=latents.dtype)
    count = 0
    
    for i, t in enumerate(timesteps):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        
        # Current Model
        with torch.no_grad():
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
            ).sample
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        
        # Bad Model
        with torch.no_grad():
            bad_noise_pred = bad_unet(
                latent_model_input,
                t,
                encoder_hidden_states=bad_text_embeddings if bad_text_embeddings is not None else text_embeddings,
            ).sample
        _, bad_noise_pred_cond = bad_noise_pred.chunk(2)

        
        if i >= args.score_diff_step_start and i <= args.score_diff_step_end:
            if args.is_score:
                alpha_bar = pipe.scheduler.alphas_cumprod[t]
                sigma = torch.sqrt(1 - alpha_bar)
                target_cond = -noise_pred_cond / sigma
                target_bad_cond = -bad_noise_pred_cond / sigma
                
            else:
                target_cond = noise_pred_cond
                target_bad_cond = bad_noise_pred_cond
            
            # 1. ||s_cond - s_bad_cond||^2
            if getattr(args, 'use_score_sq', False):
                target_diff = target_cond
            else:
                target_diff = target_cond - target_bad_cond
            diff_norm = target_diff ** 2
            diff_norm = diff_norm.sum(dim=1)[0]
            target_diff_norm_sum += diff_norm
            
            count += 1
        
        # Standard Guidance Steps (Using Current Model logic for trajectory)
        noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
    
    if count == 0:
        return np.zeros((64, 64))

    target_diff_norm = target_diff_norm_sum / count
    return target_diff_norm.cpu().numpy()


def compute_metrics_for_prompt_with_bad_model(prompt, pipe, bad_unet, args, seed, bad_tokenizer=None, bad_text_encoder=None):
    """Compute all metrics for a prompt using bad_model"""
    device = pipe.device
    
    # Text Embeddings
    text_input = pipe.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipe.text_encoder(text_input.input_ids.to(device))[0]
    uncond_input = pipe.tokenizer(
        [""],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    )
    uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    
    results = {}

    if bad_tokenizer is not None and bad_text_encoder is not None:
        bad_text_input = bad_tokenizer([prompt], padding="max_length", max_length=bad_tokenizer.model_max_length, truncation=True, return_tensors="pt")
        bad_text_embeds = bad_text_encoder(bad_text_input.input_ids.to(device))[0]
        bad_uncond_input = bad_tokenizer([""], padding="max_length", max_length=bad_tokenizer.model_max_length, return_tensors="pt")
        bad_uncond_embeds = bad_text_encoder(bad_uncond_input.input_ids.to(device))[0]
        bad_text_embeddings = torch.cat([bad_uncond_embeds, bad_text_embeds])
    else:
        bad_text_embeddings = None

    # 1. diag_h (Covariance with Bad Model Baseline)
    if "cov" in args.metrics:
        results["cov_bad"] = compute_diag_h_diff_only_with_bad_model(prompt, pipe, bad_unet, args, text_embeddings, seed, bad_text_embeddings=bad_text_embeddings)
    
    # 2. Attention (Skip for bad model as it is already computed)
    # attn = compute_attention_only(prompt, pipe, args, text_embeddings, text_input, seed)
    
    # 3. Score Diffs (Current vs Bad) / Score Sq
    if "score_diff" in args.metrics:
        results["score_diff_bad"] = compute_score_diffs_only_with_bad_model(prompt, pipe, bad_unet, args, text_embeddings, seed, bad_text_embeddings=bad_text_embeddings)
    if "score_sq" in args.metrics:
        results["score_sq_bad"] = compute_score_diffs_only_with_bad_model(prompt, pipe, bad_unet, args, text_embeddings, seed, bad_text_embeddings=bad_text_embeddings)
    
    return results
