import os, json
from utils import get_dataset
import torch
from plugin_mae import MAEPlugin
# from hyperimpute.utils.benchmarks import compare_models
from benchmarks import compare_models, simulate_scenarios
from hyperimpute.plugins.imputers import Imputers

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np
from configs import get_args_parser
from sklearn.preprocessing import MinMaxScaler

args = get_args_parser().parse_args()
X_raw, y = get_dataset(args.dataset, args.path)
X = X_raw.values
no = len(X)
dim = len(X[0, :])

size = min(256, no)

min_val = np.zeros(dim)
max_val = np.zeros(dim)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# for i in range(dim):
#     min_val[i] = np.nanmin(X[:, i])
#     max_val[i] = np.nanmax(X[:, i])
#     X[:, i] = (X[:, i] - min_val[i]) / (max_val[i] -  min_val[i] + 1e-8)      

x = torch.tensor(X[:size], dtype=torch.float32)
print(x)
torch.save({  
    'ox' : x
    },
    '/data/ting/remasker2/data/' + args.dataset + '.pt'
)


# imputation_scenarios = simulate_scenarios(X_raw)

# miss_pct = [0.1, 0.3, 0.5, 0.7]

# for missingness in miss_pct:
#     x, x_miss, mask = imputation_scenarios['MAR'][missingness]
#     x = torch.tensor(x.values[:size], dtype=torch.float32)
#     x_miss = torch.tensor(x_miss.values[:size], dtype=torch.float32)
#     torch.save({  
#         'ox' : x,
#         'mx' : x_miss
#         },
#         '/data/ting/remasker2/data/' + args.dataset + '-mar-' +  str(missingness) + '.pt'
#     )
