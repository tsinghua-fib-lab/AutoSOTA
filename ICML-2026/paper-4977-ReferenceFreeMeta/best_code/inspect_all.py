import h5py, os

paths = [
    "/datasets/fastmri_dl/singlecoil_train_file1000001.h5",
    "/datasets/fastmri_dl/singlecoil_test_file1000055.h5",
]

for p in paths:
    print("=" * 60)
    print("File:", os.path.basename(p))
    with h5py.File(p, "r") as f:
        print("Keys:", list(f.keys()))
        for key in f.keys():
            obj = f[key]
            if hasattr(obj, "shape"):
                print("  %s: shape=%s, dtype=%s" % (key, obj.shape, obj.dtype))
            else:
                print("  %s: type=%s" % (key, type(obj)))
    print()
print("Done!")
