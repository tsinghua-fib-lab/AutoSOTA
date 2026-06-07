import torch

@torch.no_grad()
def rf_sampler(
    model, 
    latents, 
    y=None, 
    cfg_scale=1.0,
    num_steps=1, 
    reverse=False,
    **kwargs
):

    batch_size = latents.shape[0]
    device = latents.device

    do_cfg = y is not None and cfg_scale > 0.0
    if do_cfg:
        if hasattr(model, 'module'):
            num_classes = model.module.num_classes
        else:
            num_classes = model.num_classes
        null_y = torch.full_like(y, num_classes)
    
    if num_steps == 1:
        t = torch.ones(batch_size, device=device)
        
        if do_cfg:
            z_combined = torch.cat([latents, latents], dim=0)
            t_combined = torch.cat([t, t], dim=0)
            y_combined = torch.cat([y, null_y], dim=0)
            
            u_combined = model(z_combined, t_combined, y=y_combined)
            u_cond, u_uncond = u_combined.chunk(2, dim=0)
            
            u = u_uncond + cfg_scale * (u_cond - u_uncond)
        else:
            u = model(latents, t, y=y)
        x0 = latents - u
        
    else:
        z = latents
        
        if reverse:
            time_steps = torch.linspace(0, 1, num_steps + 1, device=device)
        else:
            time_steps = torch.linspace(1, 0, num_steps + 1, device=device)
        
        for i in range(num_steps):
            t_cur = time_steps[i]
            t_next = time_steps[i + 1]
            
            t = torch.full((batch_size,), t_cur, device=device)
            
            if do_cfg:
                z_combined = torch.cat([z, z], dim=0)
                t_combined = torch.cat([t, t], dim=0)
                y_combined = torch.cat([y, null_y], dim=0)
                
                u_combined = model(z_combined, t_combined, y=y_combined)
                u_cond, u_uncond = u_combined.chunk(2, dim=0)
                
                u = u_uncond + cfg_scale * (u_cond - u_uncond)
            else:
                u = model(z, t, y=y)
            
            dt = t_next - t_cur
            z = z + dt * u
        
        x0 = z
    
    return x0
