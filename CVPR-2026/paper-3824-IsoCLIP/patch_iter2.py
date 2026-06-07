"""Patch: Add gap-guided adaptive band selection to apply_iso()."""
log = open('/repo/patch2_result.txt', 'w')

path = '/repo/src/retrieval.py'
with open(path) as f:
    content = f.read()

# Add gap-guided band computation function before apply_iso
old_marker = 'def apply_iso(W_text, W_image, iso_ktop=0, iso_kbottom=0, iso_tau=0.0):'

gap_func = '''
def compute_gap_guided_band(S, gap_threshold=1.5):
    """Compute ktop and kbottom from singular value gap statistic.
    Finds natural spectral boundaries by locating large gaps between consecutive singular values.
    """
    r = S.shape[0]
    gaps = (S[:-1] - S[1:]).cpu().numpy()
    mean_gap = gaps.mean()

    # Search for the largest gap in the first half (top boundary)
    half = r // 2
    ktop = int((gaps[:half] > gap_threshold * mean_gap).nonzero()[0].max()) + 1 if (gaps[:half] > gap_threshold * mean_gap).any() else 0

    # Search for the largest gap in the second half (bottom boundary)
    bottom_gaps = gaps[half:]
    if (bottom_gaps > gap_threshold * mean_gap).any():
        kbottom = r - (half + int((bottom_gaps > gap_threshold * mean_gap).nonzero()[0].min())) - 1
    else:
        kbottom = 0

    ktop = max(0, min(ktop, r // 2))
    kbottom = max(0, min(kbottom, r // 2))

    if ktop + kbottom >= r:
        ktop = min(ktop, r - kbottom - 1)

    return ktop, kbottom


'''

if old_marker in content:
    content = content.replace(old_marker, gap_func + old_marker)
    with open(path, 'w') as f:
        f.write(content)
    log.write("PATCH: gap_guided_band function added before apply_iso\n")
else:
    log.write("ERROR: marker not found\n")

# Now modify apply_iso to support auto mode
old_iso_start = '''def apply_iso(W_text, W_image, iso_ktop=0, iso_kbottom=0, iso_tau=0.0):

        # inter-modal operator
        Psi = W_image.T @ W_text

        U, S, V = torch.linalg.svd(Psi, full_matrices=False)
        V = V.T

        r = S.shape[0]

        # Sanity check
        if iso_ktop + iso_kbottom >= r:
            raise ValueError(f"Cannot remove {iso_ktop} top and {iso_kbottom} bottom components from {r}")

        if iso_tau > 0:'''

new_iso_start = '''def apply_iso(W_text, W_image, iso_ktop=0, iso_kbottom=0, iso_tau=0.0, iso_gap_guided=False):

        # inter-modal operator
        Psi = W_image.T @ W_text

        U, S, V = torch.linalg.svd(Psi, full_matrices=False)
        V = V.T

        r = S.shape[0]

        # Auto-determine band from singular value gaps if requested
        if iso_gap_guided:
            auto_ktop, auto_kbottom = compute_gap_guided_band(S)
            print("Gap-guided band: auto k_top = {}, auto k_bottom = {}".format(auto_ktop, auto_kbottom))
            iso_ktop = auto_ktop
            iso_kbottom = auto_kbottom

        # Sanity check
        if iso_ktop + iso_kbottom >= r:
            raise ValueError(f"Cannot remove {iso_ktop} top and {iso_kbottom} bottom components from {r}")

        if iso_tau > 0:'''

if new_iso_start in content:
    log.write("already patched\n")
elif old_iso_start in content:
    content = content.replace(old_iso_start, new_iso_start)
    with open(path, 'w') as f:
        f.write(content)
    log.write("PATCH: gap-guided mode added to apply_iso\n")
else:
    log.write("ERROR: iso_start not found\n")

log.close()
