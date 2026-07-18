import argparse
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', **{'family': 'serif', 'serif': 'cm'})
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
import torch
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.utils import loadSCData, tpSplitInd, modelParams, splitBySpec
from model.models import *
from model.running import train, predict
from optim.evaluation import globalEvaluation
from sklearn.decomposition import PCA
# ======================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='wot', help='Dataset name: zebrafish, drosophila, wot')
    parser.add_argument('--split_type', type=str, default='three_forecasting', help='Split type: three_interpolation, two_forecasting, three_forecasting, remove_recovery')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--pca', action='store_true', default=False)
    args = parser.parse_args()
    return args

args = parse_args()
# Load data and pre-processing
print("=" * 70)
SEED = args.seed
# Specify the dataset: zebrafish, drosophila, wot
# Representing ZB, DR, SC, repectively
data_name= args.data_name
print("[ {} ]".format(data_name).center(60))
# Specify the type of prediction tasks: three_interpolation, two_forecasting, three_forecasting, remove_recovery
# The tasks feasible for each dataset:
#   zebrafish (ZB): three_interpolation, two_forecasting, remove_recovery
#   drosophila (DR): three_interpolation, three_forecasting, remove_recovery
#   wot (SC): three_interpolation, three_forecasting, remove_recovery
# They denote easy, medium, and hard tasks respectively.
split_type = args.split_type
print("Split type: {}".format(split_type))
ann_data, cell_tps, cell_types, n_genes, n_tps = loadSCData(data_name, split_type)

train_tps, test_tps = tpSplitInd(data_name, split_type)

if "remove_recovery" in split_type:
    val_tps = [3] + train_tps[-2:]
elif "interpolation" in split_type:
    val_tps = [3, 7, 9]
else:
    val_tps = train_tps[-2:]
data = ann_data.X

# Convert to torch project
traj_data = [torch.FloatTensor(data[np.where(cell_tps == t)[0], :]) for t in range(1, n_tps + 1)]
if cell_types is not None:
    traj_cell_types = [cell_types[np.where(cell_tps == t)[0]] for t in range(1, n_tps + 1)]

all_tps = list(range(n_tps))
train_data, test_data = splitBySpec(traj_data, train_tps, test_tps)
if args.pca:
    pca_model = PCA(n_components=50)
    n_genes = 50
    pca_model.fit(np.concatenate(train_data, axis=0))
    train_data = [torch.FloatTensor(pca_model.transform(each)) for each in train_data]
    traj_data = [torch.FloatTensor(pca_model.transform(each)) for each in traj_data]
    test_data = [torch.FloatTensor(pca_model.transform(each)) for each in test_data]
    
tps = torch.FloatTensor(all_tps)
train_tps = torch.FloatTensor(train_tps)
test_tps = torch.FloatTensor(test_tps)
val_tps = torch.FloatTensor(val_tps)
n_cells = [each.shape[0] for each in traj_data]
print("# tps={}, # genes={}".format(n_tps, n_genes))
print("# cells={}".format(n_cells))
print("Train tps={}".format(train_tps))
print("Test tps={}".format(test_tps))

# ======================================================
# Model training
latent_coeff = 1.0 # regularization coefficient: beta
epochs = 500 if split_type == "all" else 1250
batch_size = 256
lr = 1e-3
n_sim_cells = 2000

latent_dim, dec_latent_list, M = modelParams(data_name, split_type) # use tuned hyperparameters
y_num_dim = n_genes
x_num_dim = 2 # id, time
P = n_sim_cells
id_embed_dim = P
id_handler = "none"
C = []
id_covariate = 0
se_idx = [1]
ca_idx = []
bin_idx = []
poly_idx = []
interactions = []
basis_funcs = "hs"
scale = np.log(1.6) - 0.5
alpha = 1.0
alpha_fixed = False
scale_fixed = False
vy_init = 1.0
vy_fixed = True
p_drop = 0.1
dataset_type = "LGPOT-" + data_name + ("-PCA" if args.pca else "")
k = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)

# Deep generative model with approximate GP prior in the latent space
model = DGBFGP(y_num_dim, x_num_dim, latent_dim, 
                         P, id_embed_dim, id_handler, M, C,
                         id_covariate, se_idx, ca_idx, bin_idx, poly_idx, interactions, basis_funcs, 
                         scale, alpha, alpha_fixed, scale_fixed, vy_init, vy_fixed, p_drop, dec_latent_list, k=k,
                         device = device).to(device)

# Noise model for makeing the model heteroscedastic
noise_model = CovariateModule({
                    "index": 1,
                    "type": "SE",
                    "basis": BasisFunction(M, basis_funcs, "SE", 
                                                scale, alpha, 
                                                alpha_fixed, scale_fixed, dim = latent_dim, device=device),
                    "A": BayesianLinear(M, latent_dim, device=device)
                }).to(device)
model.noise_model = noise_model


X = np.empty((n_sim_cells * len(all_tps), x_num_dim))
X_train = np.empty((n_sim_cells * len(train_tps), x_num_dim))
X_val = np.empty((n_sim_cells * len(val_tps), x_num_dim))
X_test = np.empty((n_sim_cells * len(test_tps), x_num_dim))
i = 0
i_train = 0
i_val = 0
i_test = 0
for id in range(n_sim_cells):
    for t in all_tps:
        X[i, 0] = id
        X[i, 1] = t
        i += 1
    for t in train_tps:
        if t in val_tps:
            X_val[i_val, 0] = id
            X_val[i_val, 1] = t
            i_val += 1
        X_train[i_train, 0] = id
        X_train[i_train, 1] = t
        i_train += 1
    for t in test_tps:
        X_test[i_test, 0] = id
        X_test[i_test, 1] = t
        i_test += 1
X_train[:, 1] = (X_train[:, 1] - np.mean(all_tps)) / np.std(all_tps)
X_test[:, 1] = (X_test[:, 1] - np.mean(all_tps)) / np.std(all_tps)
X_val[:, 1] = (X_val[:, 1] - np.mean(all_tps)) / np.std(all_tps)
X[:, 1] = (X[:, 1] - np.mean(all_tps)) / np.std(all_tps)
min_t = np.min(X[:, 1])
max_t = np.max(X[:, 1])
print("min tp: {}, max tp: {}".format(np.min(X[:, 1]), np.max(X[:, 1])))
assert i_train == X_train.shape[0]
assert i_val == X_val.shape[0]
assert i_test == X_test.shape[0]
assert i == X.shape[0]

X_train = torch.Tensor(X_train).to(device)
X_val = torch.Tensor(X_val).to(device)
X_test = torch.Tensor(X_test).to(device)
X = torch.Tensor(X).to(device)
# train_data = [each.to(device) for each in train_data]
val_data = []
for i, each in enumerate(train_data):
    t = train_tps[i]
    if t in val_tps:
        N_t = each.shape[0]
        val_ind = np.random.choice(N_t, int(N_t * 0.05), replace=False)
        val_data.append(each[val_ind, :].to(device))
        train_data[i] = each[np.setdiff1d(np.arange(N_t), val_ind), :].to(device)
    else:
        train_data[i] = each.to(device)

save_dir = "./res/experimental/{}".format(data_name)
res_filename = "{}/{}-{}-{}-LGPOT{}-res.npy".format(save_dir, data_name, split_type, str(SEED), "-PCA" if args.pca else "")
state_filename = "{}/{}-{}-{}-LGPOT{}-state_dict.pt".format(save_dir, data_name, split_type, str(SEED), "-PCA" if args.pca else "")
model, loss_list, recon_obs, latent_seq = train(train_data, X_train, val_data, X_val, model, latent_coeff, epochs, batch_size, lr, P, device, train_tps, val_tps, state_filename)

all_recon_obs, latent_seq = predict(model, X, device, all_tps)  # (# cells, # tps, # genes)

true_data = [each.detach().numpy() for each in traj_data]
true_cell_tps = np.concatenate([np.repeat(t, each.shape[0]) for t, each in enumerate(true_data)])
pred_cell_tps = np.concatenate([np.repeat(t, all_recon_obs[:, t, :].shape[0]) for t in range(all_recon_obs.shape[1])])


# Compute evaluation metrics
print("Compute metrics...")
test_tps_list = [int(t) for t in test_tps]
global_metrics = []
for t in test_tps_list:
    print("-" * 70)
    print("t = {}".format(t))
    # -----
    n_pred_cells = all_recon_obs[:, t].shape[0]
    print("Number of predicted cells: {}".format(n_pred_cells))
    pred_global_metric = globalEvaluation(traj_data[t].detach().numpy(), all_recon_obs[:, t, :])
    global_metrics.append(pred_global_metric)
    print("Predicted:", pred_global_metric)

# # ======================================================
# Save results

print("Saving to {}".format(res_filename))
np.save(
    res_filename,
    {"true": [each.detach().numpy() for each in traj_data],
     "pred": [all_recon_obs[:, t, :] for t in range(all_recon_obs.shape[1])],
     "latent_seq": latent_seq,
     "tps": {"all": tps.detach().numpy(), "train": train_tps.detach().numpy(), "test": test_tps.detach().numpy()},
     "loss": loss_list,
     "cell_tps": {"true": true_cell_tps, "pred": pred_cell_tps},
     "pred_global_metric": global_metrics,
     },
    allow_pickle=True
)