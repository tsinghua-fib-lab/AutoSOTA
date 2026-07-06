import sys
import h5py

import numpy as np

cor_type = sys.argv[1]  # add_global_2
file = f'data/modelnet_c/{cor_type}.h5'

h5_f = h5py.File(file)
label = h5_f['label'][:].astype(np.int32)

with open(f'/home/hongyu/data/pc_views/modelnet_c_views/{cor_type}/labels.txt', 'w') as fout:
    for val in label:
        fout.write(f'{val.item()}\n')

print(f"Generate /home/hongyu/data/pc_views/modelnet_c_views/{cor_type}/labels.txt done!")