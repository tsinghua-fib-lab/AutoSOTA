#!/usr/bin/env python3
"""Patch generate_poisson_f.py with gradient normalization (CODE-02)."""
import sys

PATCH_TARGET = "/repo/scripts/generate_poisson_f.py"

OLD_LINE = "        combined_grad = (zeta_pde * grad_x_cur_pde + zeta_obs_a * grad_x_cur_obs_a + zeta_obs_u * grad_x_cur_obs_u)"

NEW_LINE = """        # CODE-02: Normalize each gradient term before combining
        def safe_norm(g):
            return g / (torch.norm(g) + 1e-8)
        combined_grad = (zeta_pde * safe_norm(grad_x_cur_pde) +
                         zeta_obs_a * safe_norm(grad_x_cur_obs_a) +
                         zeta_obs_u * safe_norm(grad_x_cur_obs_u))"""

def apply():
    with open(PATCH_TARGET, 'r') as f:
        content = f.read()
    if OLD_LINE not in content:
        print("ERROR: Could not find target line. Already patched?")
        print("Looking for:", repr(OLD_LINE[:80]))
        return False
    content = content.replace(OLD_LINE, NEW_LINE)
    with open(PATCH_TARGET, 'w') as f:
        f.write(content)
    print("Applied gradient normalization patch (CODE-02)")
    return True

def revert():
    with open(PATCH_TARGET, 'r') as f:
        content = f.read()
    if NEW_LINE.strip().split('\n')[0] in content:
        content = content.replace(NEW_LINE, OLD_LINE)
        with open(PATCH_TARGET, 'w') as f:
            f.write(content)
        print("Reverted gradient normalization patch")
        return True
    else:
        print("Patch not found, nothing to revert")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "revert":
        revert()
    else:
        apply()
