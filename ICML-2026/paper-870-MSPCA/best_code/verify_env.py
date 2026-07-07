import os
for f in ["eval.py", "main.py", "rebuttal.py", "rebuttal2.py", "README.md", "pyproject.toml"]:
    status = "OK" if os.path.exists(f) else "MISSING"
    print(f"  {f}: {status}")
print()
from sklearn.decomposition import TruncatedSVD, PCA
from rpca import RobustPCA
from scipy.stats.mstats import winsorize
from scipy.linalg import sqrtm
import numpy as np; import numpy.linalg as LA
import pandas as pd
print("All imports verified OK")
print(f"numpy={np.__version__}")
import sklearn; print(f"scikit-learn={sklearn.__version__}")
import scipy; print(f"scipy={scipy.__version__}")
