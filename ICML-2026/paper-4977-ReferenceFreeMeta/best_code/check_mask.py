import h5py, numpy as np
import os

path = "/datasets/fastmri_dl/singlecoil_test_file1000055.h5"
with h5py.File(path, "r") as f:
    mask = f["mask"][:]
    kspace = f["kspace"][:]
    print("Mask shape:", mask.shape)
    print("Kspace shape:", kspace.shape)
    print("Mask type:", mask.dtype)
    print("Mask values (first 30):", mask[:30].astype(int))
    # Count sampled lines
    sampled = mask.sum()
    total = mask.shape[0]
    print("Sampled lines: %d / %d" % (sampled, total))
    print("Approx AF: %.1f" % (total / sampled))

    mid = kspace.shape[0] // 2
    slice_ksp = kspace[mid]
    print("Middle slice kspace shape:", slice_ksp.shape)

    # Apply mask and do IFFT to understand image size
    masked_ksp = slice_ksp.copy()
    masked_ksp[:, ~mask] = 0
    recon = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(masked_ksp)))
    print("Reconstructed image shape:", recon.shape)
    print("Image min/max:", np.abs(recon).min(), np.abs(recon).max())
