import torch
import numpy as np
from typing import Dict, List, Tuple


class MultiscaleScheduler:
    """scheduler for multiscale DFM sampling"""
    
    def __init__(self, num_steps_per_scale, scale_thresholds):
        self.num_steps_per_scale = num_steps_per_scale
        self.scales_count = len(self.num_steps_per_scale)
        self.scale_thresholds = scale_thresholds
        
        # assert that scale_thresholds has the correct length and it is decreasing. 
        assert len(self.scale_thresholds) == self.scales_count - 1, \
            f"scale_thresholds should have {self.scales_count - 1} elements for {self.scales_count} scales"
        assert all(x > y for x, y in zip(self.scale_thresholds, self.scale_thresholds[1:])), \
            "scale_thresholds should be strictly decreasing"
        
    def _compute_single_schedule(self, num_steps: int, device, start_value=1.0, end_value=0.0) -> torch.Tensor:
        """Computes a linear sampling schedule"""
        step_indices = torch.flip(
            torch.arange(num_steps, dtype=torch.float64, device=device), dims=[0]
        ) + 1
        delta_t = (start_value - end_value) / num_steps
        t_steps = step_indices * delta_t + end_value
        return t_steps
    
    def _split_threshold(self, timesteps: torch.Tensor, threshold: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Splits timesteps with floating point tolerance to avoid numerical errors"""
        threshold_tensor = torch.tensor(threshold, device=timesteps.device, dtype=timesteps.dtype)
        lower_equal_timesteps = timesteps[torch.isclose(timesteps, threshold_tensor) | (timesteps < threshold)]
        greater_timesteps = timesteps[~torch.isclose(timesteps, threshold_tensor) & (timesteps > threshold)]
        return lower_equal_timesteps, greater_timesteps
    
    def schedule(self, device) -> List[Dict]:
        """Creates deterministic multiscale sampling schedule using explicit thresholds"""
        
        stage_end_threshold = torch.cat([torch.tensor(self.scale_thresholds), torch.tensor([0.0])])
        
        total_steps = sum(self.num_steps_per_scale)
        all_stages_scheduler = torch.ones((total_steps, self.scales_count), device="cpu")
        
        current_step = 0
        for i in range(self.scales_count):
            current_chunk_scheduler = self._compute_single_schedule(
                self.num_steps_per_scale[i], "cpu", 
                start_value=1.0, 
                end_value=float(stage_end_threshold[i])
            )
            
            all_stages_scheduler[current_step:current_step+self.num_steps_per_scale[i], i] = current_chunk_scheduler
            
            # Propagate to earlier scales (scales with lower indices)
            for j in range(i):
                # Map the current chunk to the range for previous scale
                # Linearly interpolate between the thresholds
                scale_factor = (float(stage_end_threshold[j]) - float(stage_end_threshold[j+1]))
                offset = float(stage_end_threshold[j+1])
                previous_chunk_scheduler = current_chunk_scheduler * scale_factor + offset
                all_stages_scheduler[current_step:current_step+self.num_steps_per_scale[i], j] = previous_chunk_scheduler
            
            current_step += self.num_steps_per_scale[i]
        
        final_schedule = all_stages_scheduler
        
        drop_masks = torch.ones((total_steps, self.scales_count), dtype=torch.float32)
        
        # Build the results
        sampling_schedule = []
        final_schedule = final_schedule.to(device)
        drop_masks = drop_masks.to(device)
        
        for step_idx in range(total_steps):
            current_times = final_schedule[step_idx]
            current_drop_masks = drop_masks[step_idx].clone()
            
            if step_idx == total_steps - 1:
                next_times = torch.zeros_like(current_times)
            else:
                next_times = final_schedule[step_idx + 1]
                
            current_delta_t = next_times - current_times
            
            # Find the rightmost scale that's being updated
            rightmost_negative_delta_idx = -1
            for i in range(current_delta_t.shape[0]):
                if current_delta_t[i].item() < 0: 
                    rightmost_negative_delta_idx = i
            
            # Only update scales up to the rightmost active one
            current_drop_masks[rightmost_negative_delta_idx+1:] = False
            
            sampling_schedule.append({
                "times": current_times,
                "next_times": next_times,
                "delta_t": current_delta_t,
                "drop_mask": current_drop_masks,
            })
            
            # Check for positive deltas (but allow small numerical errors)
            if torch.any(current_delta_t > 0):
                raise ValueError(f"Step {step_idx} has positive delta_t: {current_delta_t}")
                # Don't raise error, just warn for now
                
        return sampling_schedule

def dfm_euler_sampler(
        model,
        latents_dict,  # Dict with keys like '0', '1', etc.
        y,
        heun=False,
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        path_type="linear",  # not used, just for compatibility
        
        # DFM
        num_steps_per_scale=None,
        stage_thresholds=None,
        
        ):
    """
    Multiscale DFM Euler sampler using simplified scheduler
    """
    
    # Heun's method second step
    if heun:
        assert False, "Heun's method not implemented for multiscale DFM sampler yet"
        
    scheduler = MultiscaleScheduler(
        num_steps_per_scale,
        stage_thresholds
    )
    scales_count = len(num_steps_per_scale)
    # Setup conditioning
    if cfg_scale > 1.0:
        y_null = torch.tensor([1000] * y.size(0), device=y.device)
    
    # Get device and dtype from first latent
    first_key = list(latents_dict.keys())[0]
    device = latents_dict[first_key].device
    _dtype = latents_dict[first_key].dtype
    
    # Convert latents to double precision for sampling
    x_next_dict = {}
    for scale_idx in range(scales_count):
        x_next_dict[scale_idx] = latents_dict[scale_idx].to(torch.float64)
    
    # Get sampling schedule
    schedule = scheduler.schedule(device)
    
    with torch.no_grad():
        for step_idx, schedule_step in enumerate(schedule):
            current_times = schedule_step["times"]
            next_times = schedule_step["next_times"] 
            delta_t = schedule_step["delta_t"]
            drop_mask = schedule_step["drop_mask"]
            
            # Prepare model inputs for each scale
            model_inputs = {}
            time_inputs = []
            
            # Check if ANY scale needs CFG guidance
            any_scale_needs_cfg = False
            for scale_idx in range(len(current_times)):
                t_cur = current_times[scale_idx]
                if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                    any_scale_needs_cfg = True
                    break
            
            # Apply CFG consistently to ALL scales if any scale needs it
            for scale_idx, (scale_key, x_cur) in enumerate(x_next_dict.items()):
                t_cur = current_times[scale_idx]
                
                if any_scale_needs_cfg:
                    model_inputs[scale_key] = torch.cat([x_cur] * 2, dim=0).to(dtype=_dtype)
                else:
                    model_inputs[scale_key] = x_cur.to(dtype=_dtype)
                    
                time_inputs.append(torch.ones(model_inputs[scale_key].size(0)).to(
                    device=device, dtype=torch.float64
                ) * t_cur)
            
            # Setup y_cur based on whether CFG is needed
            if any_scale_needs_cfg:
                y_cur = torch.cat([y, y_null], dim=0)
            else:
                y_cur = y
            
            time_inputs = torch.stack(time_inputs, dim=-1).to(dtype=_dtype)  # shape: [batch_size, num_scales]
            model_outputs, _ = model(model_inputs, time_inputs, y=y_cur, drop_mask=drop_mask)
            
            # Apply CFG and update each scale
            for scale_idx, (scale_key, x_cur) in enumerate(x_next_dict.items()):
                if drop_mask[scale_idx] and scale_key in model_outputs:
                    t_cur = current_times[scale_idx]
                    dt = delta_t[scale_idx]
                    
                    d_cur = model_outputs[scale_key].to(torch.float64)
                    
                    # Apply CFG if it was used for this step (consistent with input preparation)
                    if any_scale_needs_cfg:
                        d_cur_cond, d_cur_uncond = d_cur.chunk(2)
                        # Only apply CFG to scales within the guidance window
                        if t_cur <= guidance_high and t_cur >= guidance_low:
                            d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)
                        else:
                            # Outside guidance window, use conditional output
                            d_cur = d_cur_cond
                    
                    # Euler step
                    x_next_dict[scale_key] = x_cur + dt * d_cur
    
    return x_next_dict



def main():
    """Test function for MultiscaleScheduler, DeterministicMultiscaleScheduler and dfm_euler_sampler_multiscale"""
    print("Testing Multiscale Schedulers...")
    
    print("\n=== Test 1: Basic Scheduler ===")
    scheduler_config = {
        "num_steps": [10, 5],
        "scale_thresholds": [0.1]
    }
    
    scheduler = MultiscaleScheduler(
        scheduler_config["num_steps"],
        scheduler_config["scale_thresholds"]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    schedule = scheduler.schedule(device)
    print(f"Generated schedule with {len(schedule)} steps")
    print(f"Scales count: {scheduler.scales_count}")
    
    # Print first few steps
    print("\nAll steps:")
    for i, step in enumerate(schedule):
        print(f"Step {i}: times={step['times']}, next_times={step['next_times']}, drop_mask={step['drop_mask']}, delta_t={step['delta_t']}")
    
    
if __name__ == "__main__":
    main()
