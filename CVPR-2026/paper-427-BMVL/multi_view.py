import warnings
warnings.filterwarnings("ignore")
import argparse
import os
import sys
import random
import logging
import time
import h5py
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sklearn.preprocessing as skp
import scipy.io as sio
from torch.utils.data import DataLoader
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from contextlib import contextmanager


def set_seed(seed=None):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@contextmanager
def preserve_rng_states():
    torch_cpu_state = torch.get_rng_state()
    cuda_states = (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    np_state = np.random.get_state()
    py_state = random.getstate()
    try:
        yield
    finally:
        torch.set_rng_state(torch_cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        np.random.set_state(np_state)
        random.setstate(py_state)

def get_log(args):
    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)
    if not os.path.exists(args.log_path + args.dataset_name + '/'):
        os.mkdir(args.log_path + args.dataset_name + '/')

    timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    log_folder_path = os.path.join(args.log_path + args.dataset_name + '/' + timestamp)
    if not os.path.exists(log_folder_path):
        os.mkdir(log_folder_path)

    log_format = '%(asctime)s %(message)s'
    # logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    log_file_path = os.path.join(log_folder_path, 'train.log')
    fh = logging.FileHandler(log_file_path)
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    args.log_path = log_folder_path
    return log_folder_path

def log_args(args):
    args_dict = args.__dict__
    arg_lines = [f"{key}: {repr(value)}" for key, value in args_dict.items()]
    multi_line_message = "Command Line Arguments:\n" + '\n'.join(arg_lines)
    logging.info(multi_line_message)

def to_begin():
    args = get_arguments()
    log_path = get_log(args)
    args.log_path = log_path
    log_args(args)
    return args

def load_multiviewdata(args):
    data_list = []
    if args.dataset_name in ['YoutubeFace_sel']:
        with h5py.File(args.dataset_path + args.dataset_name + '.mat', 'r') as f:
            data_list.append(f[f['X'][0, 0]][()].T)
            data_list.append(f[f['X'][1, 0]][()].T)
            data_list.append(f[f['X'][2, 0]][()].T)
            data_list.append(f[f['X'][3, 0]][()].T)
            data_list.append(f[f['X'][4, 0]][()].T)                      
            labels = np.squeeze(f['Y'][()]).astype(np.int64)
    elif args.dataset_name == 'AwAfea':
        with h5py.File(args.dataset_path + args.dataset_name + '.mat', 'r') as f:                
            data_list.append(f[f['X'][0, 0]][()].T)
            data_list.append(f[f['X'][1, 0]][()].T)
            data_list.append(f[f['X'][2, 0]][()].T)
            data_list.append(f[f['X'][3, 0]][()].T)
            data_list.append(f[f['X'][4, 0]][()].T)        
            data_list.append(f[f['X'][5, 0]][()].T)              
            labels = np.squeeze(f['Y'][()]).astype(np.int64)
    else:
        mat = sio.loadmat(args.dataset_path + args.dataset_name + '.mat')
        if args.dataset_name == '100Leaves':
            data_list.append(mat['X'][0][0])
            data_list.append(mat['X'][0][1])
            data_list.append(mat['X'][0][2])
            labels = np.squeeze(mat['Y'].astype(np.int64))
        elif args.dataset_name == 'handwritten':
            data_list.append(mat['X'][0][0])
            data_list.append(mat['X'][0][1])
            data_list.append(mat['X'][0][2])
            data_list.append(mat['X'][0][3])
            data_list.append(mat['X'][0][4])
            data_list.append(mat['X'][0][5])
            labels = np.squeeze(mat['Y'].astype(np.int64))
        elif args.dataset_name == 'LandUse_21':
            train_x = []
            train_x.append(sparse.csr_matrix(mat['X'][0, 0]).toarray())
            train_x.append(sparse.csr_matrix(mat['X'][0, 1]).toarray())
            train_x.append(sparse.csr_matrix(mat['X'][0, 2]).toarray())
            data_list = train_x
            labels = np.squeeze(mat['Y'].astype(np.int64))
        elif args.dataset_name == 'Scene15':
            data_list.append(mat['X'][0][0].T)
            data_list.append(mat['X'][0][1].T)
            data_list.append(mat['X'][0][2].T)
            labels = np.squeeze(mat['gt'].astype(np.int64))
        elif args.dataset_name == 'CCV':
            data_list.append(mat['X'][0][0])
            data_list.append(mat['X'][0][1])
            data_list.append(mat['X'][0][2])
            labels = np.squeeze(mat['Y'].astype(np.int64))
        elif args.dataset_name == 'Caltech-5V':
            data_list.append(mat['X1'])
            data_list.append(mat['X2'])
            data_list.append(mat['X3'])
            data_list.append(mat['X4'])
            data_list.append(mat['X5'])
            labels = np.squeeze(mat['Y'].astype(np.int64))    

        elif args.dataset_name == '3V_Fashion_MV':
            data_list.append(mat['X1'].reshape(mat['X1'].shape[0], mat['X1'].shape[1] * mat['X1'].shape[2]))
            data_list.append(mat['X2'].reshape(mat['X2'].shape[0], mat['X2'].shape[1] * mat['X2'].shape[2]))
            data_list.append(mat['X3'].reshape(mat['X3'].shape[0], mat['X3'].shape[1] * mat['X3'].shape[2]))
            labels = np.array(np.squeeze(mat['Y'])).astype(np.int64)
        elif args.dataset_name == 'NUSWIDEOBJ':
            data_list.append(mat['X'][0][0])
            data_list.append(mat['X'][0][1])
            data_list.append(mat['X'][0][2])
            data_list.append(mat['X'][0][3])
            data_list.append(mat['X'][0][4])
            labels = np.squeeze(mat['Y'].astype(np.int64))

    args.num_views = len(data_list)
    args.num_classes = len(np.unique(labels))
    args.dims_list = [data.shape[1] for data in data_list]

    if labels.min() == 1:
        Y = labels - 1
    else:
        Y = labels
    X = [[] for _ in range(args.num_views)]
    for i in range(args.num_views):
        if isinstance(data_list[i], scipy.sparse.spmatrix):
            data_list[i] = data_list[i].toarray()
        X[i] = skp.minmax_scale(data_list[i]).astype(np.float32)

    return X, Y, args.dims_list, args.num_classes

class NC_MultiViewDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, split_idx, noise_seed, noise_ratio):
        self.rng = np.random.default_rng(noise_seed)
        self.X = X
        self.Y = Y
        original_data = [self.X[i][split_idx] for i in range(len(self.X))]
        original_labels = self.Y[split_idx]
        self.data_list, self.labels, self.indicators, _ = NC_MultiViewDataset.NoiseCorrespondence_inject(original_data, original_labels, noise_ratio, self.rng)

    @staticmethod
    def NoiseCorrespondence_inject(data, labels, noise_ratio, rng):
        n_samples = len(labels)
        m = len(data)
        noisy_data = [np.copy(modality) for modality in data]
        noisy_labels = np.copy(labels)
        noise_indicator = np.ones((n_samples, m), dtype=np.int64)
        num_noisy_samples = int(noise_ratio * n_samples)

        if num_noisy_samples == 0:
            return noisy_data, noisy_labels, noise_indicator, np.array([], dtype=np.int64)

        indices_to_corrupt = rng.choice(n_samples, size=num_noisy_samples, replace=False)
        all_modalities = np.arange(m)
        num_to_keep = int(np.ceil(m / 2.0))
        
        for sample_idx in indices_to_corrupt:
            modalities_to_keep = rng.choice(all_modalities, size=num_to_keep, replace=False)
            modalities_to_corrupt = np.setdiff1d(all_modalities, modalities_to_keep)
            noise_indicator[sample_idx, modalities_to_corrupt] = 0

        per_mod_labels = np.tile(labels.reshape(-1, 1), (1, m))

        for mod_idx in range(m):
            noisy_sample_indices = np.where(noise_indicator[:, mod_idx] == 0)[0]

            noisy_samples_data = data[mod_idx][noisy_sample_indices]
            noisy_samples_labels = labels[noisy_sample_indices]

            shuffle_idx = rng.permutation(noisy_samples_data.shape[0])
            shuffled_data = noisy_samples_data[shuffle_idx]
            shuffled_labels = noisy_samples_labels[shuffle_idx]

            noisy_data[mod_idx][noisy_sample_indices] = shuffled_data
            per_mod_labels[noisy_sample_indices, mod_idx] = shuffled_labels

        noise_indicator = (per_mod_labels == noisy_labels.reshape(-1, 1)).astype(np.int64)

        return noisy_data, noisy_labels, noise_indicator, indices_to_corrupt

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        xs = [d[idx] for d in self.data_list]
        y = self.labels[idx] 
        ind = self.indicators[idx]
        xs = [torch.as_tensor(x) for x in xs]
        y = torch.as_tensor(y, dtype=torch.long) 
        ind = torch.as_tensor(ind, dtype=torch.long)
        return xs, y, ind

class ReliabilityEstimator(nn.Module):
    def __init__(self, num_views, feat_dim, num_classes, eps, temperature=0.5):
        super().__init__()
        self.num_views = num_views
        self.num_classes = num_classes
        self.eps = eps
        self.feat_dim = feat_dim
        self.reliability_dim = 2
        self.temperature = temperature

    def _build_router_mlps(self):
        router_in = self.feat_dim + self.reliability_dim
        self.router_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(router_in, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid()
            ) for _ in range(self.num_views)
        ])

    def _compute_entropy(self, logits):
        log_probs = F.log_softmax(logits / self.temperature, dim=-1)
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)
        max_entropy = torch.log(torch.tensor(self.num_classes, dtype=logits.dtype, device=logits.device))
        normalized_entropy = entropy / max_entropy
        log_argument = torch.clamp(1.0 - normalized_entropy, min=self.eps)
        scaled_entropy = -torch.log(log_argument)

        return probs, scaled_entropy  # [B,C], [B,1]

    def _compute_pairwise_agreement(self, probs_list):
        M = len(probs_list)
        agreement_list = []
        for m in range(M):
            p = probs_list[m].clamp_min(self.eps)
            total_sym_kl = 0.0
            cnt = 0
            for j in range(M):
                if j == m:
                    continue
                q = probs_list[j].clamp_min(self.eps)
                kl_pq = F.kl_div(q.log(), p, reduction='none').sum(dim=-1, keepdim=True)
                kl_qp = F.kl_div(p.log(), q, reduction='none').sum(dim=-1, keepdim=True)
                total_sym_kl = total_sym_kl + (kl_pq + kl_qp)
                cnt += 1
            mean_sym_kl = total_sym_kl / max(cnt, 1)
            agreement = mean_sym_kl
            agreement_list.append(agreement)
        return agreement_list

    def _compute_reliability_features(self, logits_list):
        probs_list, entropy_list = [], []
        for logits in logits_list:
            probs, entropy = self._compute_entropy(logits)
            probs_list.append(probs)
            entropy_list.append(entropy)

        agreement_list = self._compute_pairwise_agreement(probs_list)
        reliability_features = []
        for m in range(self.num_views):
            feats = []
            feats.extend([entropy_list[m], agreement_list[m]])
            reliability_features.append(torch.cat(feats, dim=-1))
        return reliability_features

    def _router_forward(self, feature_list, reliability_features=None):
        reliabilities = []
        for m in range(self.num_views):
            if reliability_features is None:
                router_in = feature_list[m]
            else:
                router_in = torch.cat([feature_list[m], reliability_features[m]], dim=-1)
            reliabilities.append(self.router_mlps[m](router_in))  # [B,1]
        return torch.cat(reliabilities, dim=-1)  # [B, num_views]

    def _finalize_forward(self, feature_list, logits_list):
        reliability_features = self._compute_reliability_features(logits_list)
        reliabilities = self._router_forward(feature_list, reliability_features)  # [B, M]

        logits_stack = torch.stack(logits_list, dim=1)  # [B, M, C]

        fused_logits = (reliabilities.unsqueeze(-1) * logits_stack).sum(dim=1)  # [B, C]
        return logits_list, fused_logits, reliabilities

class MultiViewBackbone(ReliabilityEstimator):
    def __init__(self, args):
        super().__init__(num_views=len(args.dims_list), feat_dim=args.feat_dim, num_classes=args.num_classes, eps=args.eps, temperature=args.temperature)

        self.encoder_list = nn.ModuleList()
        for input_dim in args.dims_list:
            self.encoder_list.append(nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(1024, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(1024, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(1024, self.feat_dim)
            ))

        self.classifier_list = nn.ModuleList()
        for _ in range(self.num_views):
            self.classifier_list.append(nn.Linear(self.feat_dim, self.num_classes))

        self._build_router_mlps()

    def forward(self, views):
        feature_list, logits_list = [], []
        for i, view in enumerate(views):
            features = self.encoder_list[i](view)
            logits = self.classifier_list[i](features)
            feature_list.append(features)
            logits_list.append(logits)

        return self._finalize_forward(feature_list, logits_list)


def get_arguments():
    parser = argparse.ArgumentParser(description='BML')
    parser.add_argument('--dataset_name', default='Caltech-5V', type=str)
    parser.add_argument('--augment_ratio', default=0.5, type=float)
    parser.add_argument('--lambda_w', default=1.0, type=float)
    parser.add_argument('--epochs', default=600, type=int)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--learning_rate', default=0.002, type=float)
    parser.add_argument('--feat_dim', type=int, default=128)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature for reliability estimation (lower=sharper)')
    parser.add_argument('--log_path', default='logs/', type=str)
    parser.add_argument('--dataset_path', default='datasets/multi-view-datasets/', type=str)
    return parser.parse_args()


def train_one_seed(args, seed, X, Y):
    set_seed(seed)
    args.seed = seed

    train_idx, test_idx = train_test_split(
        np.arange(len(Y)),
        test_size=0.2,
        stratify=Y,
        random_state=seed
    )

    model = MultiViewBackbone(args).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-8)
    ce_criterion = nn.CrossEntropyLoss()

    # ---- Train ----
    for epoch in range(args.epochs):
        model.train()
        running_loss_cls = 0.0
        running_loss_align = 0.0
        running_loss_total = 0.0

        train_dataset = NC_MultiViewDataset(X, Y, train_idx, noise_seed=epoch, noise_ratio=args.augment_ratio)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        for batch in train_loader:
            view_data_list = [v.cuda() for v in batch[0]]
            labels = batch[1].cuda()
            clean_indicators = batch[2].cuda()

            _, fused_logits, reliabilities = model(view_data_list)
            loss_cls = ce_criterion(fused_logits, labels)

            loss_align = F.binary_cross_entropy(reliabilities, clean_indicators.float())
            lambda_w = args.lambda_w * (1.0 - epoch / args.epochs)
            total_loss = loss_cls + lambda_w * loss_align

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            running_loss_align += loss_align.item()
            running_loss_total += total_loss.item()
            running_loss_cls += loss_cls.item()
        scheduler.step()
        
        # SWA: maintain running average during last 20% of training
        swa_start = int(args.epochs * 0.8)
        if epoch == swa_start:
            swa_model = {name: param.clone().detach() for name, param in model.named_parameters()}
            swa_n = 1
        elif epoch > swa_start:
            swa_n += 1
            for name, param in model.named_parameters():
                swa_model[name] = (swa_model[name] * (swa_n - 1) + param.clone().detach()) / swa_n
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            logging.info(f"[Seed {seed}] Epoch {epoch+1} | Total loss: {running_loss_total/len(train_loader):.4f} | CLS loss: {running_loss_cls/len(train_loader):.4f} | "
                            f"Align loss: {running_loss_align/len(train_loader):.4f}")

    # ---- Test ----
    # Load SWA weights if available
    if 'swa_model' in locals():
        for name, param in model.named_parameters():
            param.data.copy_(swa_model[name])
        logging.info(f"Loaded SWA weights (averaged over last {args.epochs - swa_start} epochs)")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    noise_ratios = np.linspace(0.0, 1.0, 11)
    model.eval()
    seed_accuracies = []
    with preserve_rng_states(), torch.no_grad():
        for noise_ratio in noise_ratios:
            test_dataset = NC_MultiViewDataset(X, Y, test_idx, noise_seed=seed, noise_ratio=noise_ratio)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

            all_preds = []
            all_labels = []

            for batch in test_loader:
                view_data_list = [v.to(device) for v in batch[0]]
                labels = batch[1].to(device)

                _, fused_logits, _ = model(view_data_list)

                preds = torch.argmax(fused_logits, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            seed_accuracies.append(accuracy_score(all_labels, all_preds))

    return seed_accuracies


if __name__ == '__main__':
    args = to_begin()
    X, Y, dims_list, num_classes = load_multiviewdata(args)

    all_results = {}
    noise_ratios = np.linspace(0.0, 1.0, 11)

    for seed in args.seeds:
        seed_accuracies = train_one_seed(args, seed, X, Y)
        all_results[seed] = seed_accuracies
        logging.info(f"Seed {seed} done.")

    c_width = 6
    header = f" {'Seed':^{c_width}} |"
    for ratio in noise_ratios:
        header += f" {ratio:^{c_width}.1f} |"

    total_width = len(header)
    separator = "=" * total_width
    title_str = f"{args.dataset_name} evaluated complete. ✅"

    result_lines = []
    result_lines.append("\n" + separator)
    result_lines.append(title_str.center(total_width))
    result_lines.append(separator)
    result_lines.append(header)
    result_lines.append("-" * total_width)

    seeds = sorted(all_results.keys())
    acc_array = np.round(np.array([all_results[seed] for seed in seeds]) * 100.0, 2)

    for i, seed in enumerate(seeds):
        row = f" {seed:^{c_width}} |"
        for acc in acc_array[i]:
            row += f" {acc:^{c_width}.2f} |"
        result_lines.append(row)

    result_lines.append("-" * total_width)

    mean_accs = np.mean(acc_array, axis=0)
    std_accs = np.std(acc_array, axis=0)

    mean_row = f" {'MEAN':^{c_width}} |"
    std_row = f" {'STD':^{c_width}} |"
    for mean, std in zip(mean_accs, std_accs):
        mean_row += f" {mean:^{c_width}.2f} |"
        std_row += f" {std:^{c_width}.2f} |"

    result_lines.append(mean_row)
    result_lines.append(std_row)
    result_lines.append(separator)

    logging.info("\n".join(result_lines))