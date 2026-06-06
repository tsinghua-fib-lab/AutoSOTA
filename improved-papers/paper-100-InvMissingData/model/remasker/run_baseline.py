import os, json
from utils import get_dataset

# from hyperimpute.utils.benchmarks import compare_models
from benchmarks import compare_models
from hyperimpute.plugins.imputers import Imputers

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np
from configs import get_args_parser


args = get_args_parser().parse_args()
X, y = get_dataset(args.dataset, args.path)

if len(np.unique(y)) > 20:
    org_auroc_score = 0
else:
    clf = LogisticRegression(solver="liblinear", random_state=0).fit(np.asarray(X), np.asarray(y))
    if len(np.unique(np.asarray(y))) > 2:
        org_auroc_score = roc_auc_score(np.asarray(y), clf.predict_proba(np.asarray(X)), multi_class='ovr')
    else:
        org_auroc_score = roc_auc_score(np.asarray(y), clf.predict_proba(np.asarray(X))[:,1])

datasets = ['climate', 'compression', 'wine', 'yacht', 'spam', 'letter', 'credit', 'raisin', 'bike', 'obesity', 'california', 'diabetes']
methods = ['gain', 'ice', 'mice', 'missforest', 'sinkhorn', 'miwae', 'miracle', 'EM', 'mean', 'median', 'most_frequent', 'softimpute']

imputer = Imputers().get("hyperimpute")

# create directories if not exist
# for part in ['model', 'output']
# :
dirpath = os.path.join(args.path, 'output', args.exp_name)
if not os.path.exists(dirpath):
    os.makedirs(dirpath)

filepath = '-'.join([str(val) for val in [args.dataset, args.note]])
filepath = os.path.join(args.path, 'output', args.exp_name, filepath + '.json')

results = {}
# for dataset in datasets: 
    
#     results[dataset] = compare_models(
#         name=args.exp_name,
#         evaluated_model=imputer,
#         X_raw=X,
#         ref_methods=[],
#         scenarios=["MAR"],
#         miss_pct=[0.3],
#         n_iter=3,
#     )

#     with open(file_path, 'w') as f:
#         f.write(json.dumps(results, indent=4))
#         f.close()

results[args.dataset] = compare_models(
    name=args.exp_name,
    evaluated_model=imputer,
    X_raw=X,
    y=y,
    ref_methods=methods,
    scenarios=["MAR"],
    miss_pct=[0.3],
    n_iter=3,
    n_jobs=4,
)

results[args.dataset]['org_auroc_score'] = org_auroc_score
    
with open(filepath, 'w') as f:
    f.write(json.dumps(results, indent=4))
    f.close()