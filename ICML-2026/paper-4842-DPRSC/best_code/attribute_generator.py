import numpy as np

n = 379
d = 2
samples = np.random.randn(n, d)

with open(f'ca-netscience/ca-netscience_attribute_d={d}.txt', 'w') as f:
    for sample in samples:
        f.write(" ".join(map(str, sample)) + "\n")