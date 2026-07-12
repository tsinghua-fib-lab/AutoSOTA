import h5py, os, glob

files = sorted(glob.glob('/datasets/fastmri_dl/*test*.h5'))
for f in files[:3]:
    with h5py.File(f, 'r') as h:
        print('%s: keys=%s' % (os.path.basename(f), list(h.keys())))
        for k in h.keys():
            if hasattr(h[k], 'shape'):
                print('  %s: %s' % (k, h[k].shape))
