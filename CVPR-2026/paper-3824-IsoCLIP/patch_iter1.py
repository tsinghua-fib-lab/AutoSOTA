"""Patch: Soft sigmoid spectral thresholding in apply_iso()."""
import sys
log = open('/repo/patch_result.txt', 'w')

path = '/repo/src/retrieval.py'
with open(path) as f:
    content = f.read()

# Find the apply_iso function
func_start = content.find('def apply_iso(')
if func_start == -1:
    log.write("ERROR: apply_iso not found\n")
    log.close()
    sys.exit(1)

# Find the end of the function (next def or decorator)
rest = content[func_start:]
lines = rest.split('\n')
func_lines = []
in_func = False
for i, line in enumerate(lines):
    if line.startswith('def apply_iso('):
        in_func = True
        func_lines.append(line)
    elif in_func:
        if line.startswith('def ') or line.startswith('@torch'):
            break
        func_lines.append(line)

old_func = '\n'.join(func_lines)
log.write(f"Found apply_iso, {len(func_lines)} lines\n")

new_func = '''def apply_iso(W_text, W_image, iso_ktop=0, iso_kbottom=0, iso_tau=0.0):

        # inter-modal operator
        Psi = W_image.T @ W_text

        U, S, V = torch.linalg.svd(Psi, full_matrices=False)
        V = V.T

        r = S.shape[0]

        # Sanity check
        if iso_ktop + iso_kbottom >= r:
            raise ValueError(f"Cannot remove {iso_ktop} top and {iso_kbottom} bottom components from {r}")

        if iso_tau > 0:
            # Soft sigmoid-weighted spectral thresholding
            print("Soft filtering: k_top = {}, k_bottom = {}, tau = {}".format(iso_ktop, iso_kbottom, iso_tau))
            indices = torch.arange(r, device=S.device, dtype=S.dtype)
            w_top = torch.sigmoid((indices - iso_ktop) / iso_tau)
            w_bottom = torch.sigmoid((r - iso_kbottom - 1 - indices) / iso_tau)
            weights = w_top * w_bottom
            W_text_iso = W_text @ V @ (weights.unsqueeze(1) * V.T)
            W_image_iso = W_image @ U @ (weights.unsqueeze(1) * U.T)
        else:
            # Original hard binary selection
            print("Manual filtering: k_top = {}, k_bottom = {}".format(iso_ktop, iso_kbottom))
            start = iso_ktop
            end = r - iso_kbottom
            U_k = U[:, start:end]
            V_k = V[:, start:end]
            W_text_iso = W_text @ V_k @ V_k.T
            W_image_iso = W_image @ U_k @ U_k.T

        # Perform the transpose to match the original shape of the features
        W_text_iso = W_text_iso.T
        W_image_iso = W_image_iso.T

        return W_text_iso, W_image_iso'''

if old_func in content:
    content = content.replace(old_func, new_func)
    with open(path, 'w') as f:
        f.write(content)
    log.write("PATCH SUCCESS: apply_iso updated\n")
else:
    log.write("PATCH FAILED: old function not found in content\n")
    log.write(f"Old function preview:\n{old_func[:500]}\n")

log.close()
