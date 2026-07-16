from config import *
from ks_mpc import KSMPC2D
# let's just use N=32 to verify compilation is fast and not bugged
N = 32
L = 16.0
c = [[8, 8]]
print("Instantiating generic...")
mpc = KSMPC2D(N, L, 0.05, c, 1.2, 5)
print("Success!")
