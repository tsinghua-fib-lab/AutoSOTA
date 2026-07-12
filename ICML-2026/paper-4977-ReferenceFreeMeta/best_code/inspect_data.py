import h5py, numpy as np, os

path = "/datasets/fastmri_dl/singlecoil_train_file1000001.h5"
print("Inspecting:", path)
with h5py.File(path, "r") as f:
    kspace = f["kspace"][:]
    mask = f["mask"][:]
    print("kspace shape:", kspace.shape, "dtype:", kspace.dtype)
    print("mask shape:", mask.shape, "dtype:", mask.dtype)
    print("mask sum:", mask.sum(), "/", mask.shape[0])
    af = mask.shape[0] / mask.sum()
    print("Acceleration factor approx:", af)

path2 = "/datasets/fastmri_dl/singlecoil_test_file1000055.h5"
print("\nInspecting:", path2)
with h5py.File(path2, "r") as f:
    kspace2 = f["kspace"][:]
    mask2 = f["mask"][:]
    print("kspace shape:", kspace2.shape)
    print("mask shape:", mask2.shape)
    print("mask sum:", mask2.sum(), "/", mask2.shape[0])
    af2 = mask2.shape[0] / mask2.sum()
    print("Acceleration factor approx:", af2)
print("Done!")
