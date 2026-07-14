#!/usr/bin/env python3
"""Patch generate_poisson_f.py with annealing PDE guidance (ALGO-01)."""
import sys

PATCH_TARGET = "/repo/scripts/generate_poisson_f.py"

MARKER = "        combined_grad = (zeta_pde * grad_x_cur_pde + zeta_obs_a * grad_x_cur_obs_a + zeta_obs_u * grad_x_cur_obs_u)"

ANNEAL_LINE = """        # ALGO-01: Anneal PDE guidance from 100% to 20% over sampling
        anneal_factor = 1.0 - 0.8 * i / num_steps
        zeta_pde_t = zeta_pde * anneal_factor
        combined_grad = (zeta_pde_t * grad_x_cur_pde + zeta_obs_a * grad_x_cur_obs_a + zeta_obs_u * grad_x_cur_obs_u)"""

def apply():
    with open(PATCH_TARGET, 'r') as f:
        content = f.read()
    if MARKER not in content:
        print("ERROR: Could not find target line")
        return False
    if "ALGO-01" in content:
        print("Anneal patch already applied")
        return False
    content = content.replace(MARKER, ANNEAL_LINE)
    with open(PATCH_TARGET, 'w') as f:
        f.write(content)
    print("Applied PDE annealing patch (ALGO-01)")
    return True

def revert():
    with open(PATCH_TARGET, 'r') as f:
        content = f.read()
    if "ALGO-01" in content:
        content = content.replace(ANNEAL_LINE, MARKER)
        with open(PATCH_TARGET, 'w') as f:
            f.write(content)
        print("Reverted PDE annealing patch")
        return True
    else:
        print("Patch not found, nothing to revert")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "revert":
        revert()
    else:
        apply()
