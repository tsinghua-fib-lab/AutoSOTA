#!/usr/bin/env python3
"""Patch generate_poisson_f.py with gradient clipping (CODE-04)."""
import sys

PATCH_TARGET = "/repo/scripts/generate_poisson_f.py"

MARKER = "combined_grad = (zeta_pde * grad_x_cur_pde + zeta_obs_a * grad_x_cur_obs_a + zeta_obs_u * grad_x_cur_obs_u)"

CLIP_LINE = """        # CODE-04: Gradient clipping by global norm
        max_norm = 1.0
        grad_norm = torch.norm(combined_grad)
        if grad_norm > max_norm:
            combined_grad = combined_grad * (max_norm / grad_norm)"""

ADAM_LINE = "        x_next, adam_state_pde = apply_adam_frequency_aware(x_next, combined_grad, adam_state_pde, optimizer_step, freq_weight, lr_low=lr_low, lr_high=lr_high, beta1=beta1, beta2=beta2)"

def apply():
    with open(PATCH_TARGET, 'r') as f:
        content = f.read()
    if MARKER not in content:
        print("ERROR: Could not find combined_grad line")
        return False
    if "CODE-04" in content:
        print("Patch already applied")
        return False
    # Insert clip line AFTER the combined_grad line and BEFORE the adam line
    content = content.replace(
        MARKER + "\n" + ADAM_LINE,
        MARKER + "\n" + CLIP_LINE + "\n" + ADAM_LINE
    )
    with open(PATCH_TARGET, 'w') as f:
        f.write(content)
    print("Applied gradient clipping patch (CODE-04)")
    return True

def revert():
    with open(PATCH_TARGET, 'r') as f:
        content = f.read()
    if "CODE-04" in content:
        # Remove the clip block
        content = content.replace("\n" + CLIP_LINE, "")
        with open(PATCH_TARGET, 'w') as f:
            f.write(content)
        print("Reverted gradient clipping patch")
        return True
    else:
        print("Patch not found, nothing to revert")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "revert":
        revert()
    else:
        apply()
