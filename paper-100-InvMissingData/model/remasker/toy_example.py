import numpy as np
import pandas as pd
from utils import get_args_parser
from remasker_impute import ReMasker

X_raw = np.arange(50).reshape(10, 5) * 1.0
X = pd.DataFrame(X_raw, columns=['0', '1', '2', '3', '4'])
X.iat[3,0] = np.nan

imputer = ReMasker()

imputed = imputer.fit_transform(X)
print(imputed[3, 0])