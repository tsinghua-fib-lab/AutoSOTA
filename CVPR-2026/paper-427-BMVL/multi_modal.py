import warnings
warnings.filterwarnings("ignore")
import argparse
import os
import sys
import random
import logging
import time
import json
import pickle
import hashlib
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from PIL import Image
from transformers import BertTokenizer, BertModel
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


class SUN_R_D_T_dataset(Dataset):
    def __init__(self, args, mode='train', tokenizer=None, max_len=77, transform=None, noise_ratio=0.0, noise_seed=0):
        self.args = args
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        key_data = (mode, args.LOAD_SIZE, max_len)
        key_string = repr(key_data)
        base_cache_key = hashlib.md5(key_string.encode('utf-8')).hexdigest()

        cache_dir = os.path.join(args.dataset_path, 'cache/')
        os.makedirs(cache_dir, exist_ok=True)
        self.base_cache_path = os.path.join(cache_dir, f"{base_cache_key}.pkl")
        
        if os.path.exists(self.base_cache_path):
            self._load_from_base_cache(self.base_cache_path)
            print(f"Loaded {len(self.A_list_cached)} base samples from cache.")
        else:
            print("Base cache not found. Building dataset from scratch...")
            data_dir = os.path.join(args.dataset_path, mode)
            json_path = os.path.join(args.dataset_path, f"{mode}.json")
            self._build_base_cache(data_dir, json_path)
            self._save_to_base_cache(self.base_cache_path)

        self.A_list, self.B_list, self.C_list_tokenized, self.img_names, self.labels, self.noise_indicator = self.NoiseCorrespondence_inject(A_list_clean=self.A_list_cached, B_list_clean=self.B_list_cached, C_list_clean=self.C_list_cached, labels_clean=self.labels_cached, names_clean=self.names_cached, noise_ratio=noise_ratio, noise_seed=noise_seed)

    def NoiseCorrespondence_inject(self, A_list_clean, B_list_clean, C_list_clean, labels_clean, names_clean, noise_ratio, noise_seed):
        if noise_ratio == 0:
            noise_indicator = np.ones((len(labels_clean), 3), dtype=np.int64)
            return A_list_clean.copy(), B_list_clean.copy(), C_list_clean.copy(), names_clean.copy(), labels_clean.copy(), noise_indicator
        
        _rng = np.random.default_rng(noise_seed)
        N = len(labels_clean)
        m = 3
        num_noisy = int(noise_ratio * N)

        A_list_noisy = A_list_clean.copy()
        B_list_noisy = B_list_clean.copy()
        C_list_noisy = C_list_clean.copy()
        names_noisy  = names_clean.copy()
        labels_noisy = labels_clean.copy()
        
        corruption_mask = np.ones((N, m), dtype=np.int64)
        indices_to_corrupt = _rng.choice(N, size=num_noisy, replace=False)

        for idx in indices_to_corrupt:
            mod_to_corrupt = _rng.choice(m, size=1)
            corruption_mask[idx, mod_to_corrupt] = 0

        shuffled_labels_per_mod = {0: {}, 1: {}, 2: {}}
        
        # A (RGB)
        idx_A = np.where(corruption_mask[:, 0] == 0)[0]
        if idx_A.size > 0:
            src_A_imgs   = [A_list_clean[i] for i in idx_A]
            src_A_labels = labels_clean[idx_A]
            perm = _rng.permutation(len(idx_A))
            for dst_pos, src_pos in enumerate(perm):
                dst_idx = idx_A[dst_pos]
                src_idx = idx_A[src_pos]
                A_list_noisy[dst_idx] = src_A_imgs[src_pos]
                shuffled_labels_per_mod[0][dst_idx] = int(src_A_labels[src_pos])

        # B (Depth)
        idx_B = np.where(corruption_mask[:, 1] == 0)[0]
        if idx_B.size > 0:
            src_B_imgs   = [B_list_clean[i] for i in idx_B]
            src_B_labels = labels_clean[idx_B]
            perm = _rng.permutation(len(idx_B))
            for dst_pos, src_pos in enumerate(perm):
                dst_idx = idx_B[dst_pos]
                src_idx = idx_B[src_pos]
                B_list_noisy[dst_idx] = src_B_imgs[src_pos]
                shuffled_labels_per_mod[1][dst_idx] = int(src_B_labels[src_pos])

        # C (Text)
        idx_C = np.where(corruption_mask[:, 2] == 0)[0]
        if idx_C.size > 0:
            src_C_texts  = [C_list_clean[i] for i in idx_C]
            src_C_labels = labels_clean[idx_C]
            perm = _rng.permutation(len(idx_C))
            for dst_pos, src_pos in enumerate(perm):
                dst_idx = idx_C[dst_pos]
                src_idx = idx_C[src_pos]
                C_list_noisy[dst_idx] = src_C_texts[src_pos]
                shuffled_labels_per_mod[2][dst_idx] = int(src_C_labels[src_pos])

        per_mod_labels = np.tile(labels_clean.reshape(-1, 1), (1, m))
        
        for dst_idx, src_label in shuffled_labels_per_mod[0].items():
            per_mod_labels[dst_idx, 0] = src_label
        for dst_idx, src_label in shuffled_labels_per_mod[1].items():
            per_mod_labels[dst_idx, 1] = src_label
        for dst_idx, src_label in shuffled_labels_per_mod[2].items():
            per_mod_labels[dst_idx, 2] = src_label
            
        noise_indicator_final = (per_mod_labels == labels_noisy.reshape(-1, 1)).astype(np.int64)

        return A_list_noisy, B_list_noisy, C_list_noisy, names_noisy, labels_noisy, noise_indicator_final

    def _load_from_base_cache(self, cache_path):
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        
        self.A_list_cached = cached_data['A_list']
        self.B_list_cached = cached_data['B_list']
        self.C_list_cached = cached_data['C_list_tokenized']
        self.labels_cached = cached_data['labels']
        self.names_cached = cached_data['names']
        self.classes = cached_data['classes']
        self.class_to_idx = cached_data['class_to_idx']
        self.int_to_class = cached_data['int_to_class']

    def _save_to_base_cache(self, cache_path):
        data_to_cache = {
            'A_list': self.A_list_cached,
            'B_list': self.B_list_cached,
            'C_list_tokenized': self.C_list_cached,
            'labels': self.labels_cached,
            'names': self.names_cached,
            'classes': self.classes,
            'class_to_idx': self.class_to_idx,
            'int_to_class': self.int_to_class
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(data_to_cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _build_base_cache(self, data_dir, json_path):
        self.classes = [d.name for d in os.scandir(data_dir) if d.is_dir()]
        self.classes.sort()
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.int_to_class = dict(zip(range(len(self.classes)), self.classes))
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data_items = json.load(f)

        A_list, B_list, C_list, names, labels = [], [], [], [], []

        for item in data_items:
            rgb_relative_path = item.get('RGB_path')
            text_description = item.get('Description')

            rgb_full_path = os.path.join(data_dir, rgb_relative_path)
            class_name = rgb_relative_path.split(os.path.sep)[0]
            label = self.class_to_idx[class_name]

            depth_relative_path = rgb_relative_path.replace(f"{os.path.sep}RGB{os.path.sep}", f"{os.path.sep}Depth{os.path.sep}")
            depth_relative_path = depth_relative_path.replace("_RGB_", "_Depth_")
            depth_full_path = os.path.join(data_dir, depth_relative_path)
            
            img_name = os.path.basename(rgb_full_path)

            A = Image.open(rgb_full_path).convert('RGB')
            B = Image.open(depth_full_path).convert('RGB')
            C_text = text_description

            w_A, h_A = A.size
            if w_A > self.args.FINE_SIZE:
                A = A.resize((self.args.LOAD_SIZE, self.args.LOAD_SIZE), Image.BICUBIC)
                B = B.resize((self.args.LOAD_SIZE, self.args.LOAD_SIZE), Image.BICUBIC)
            
            tokenized = self.tokenizer(C_text, add_special_tokens=True, padding='max_length', truncation=True, max_length=self.max_len, return_tensors=None)
            
            A_list.append(A)
            B_list.append(B)
            C_list.append(tokenized)
            names.append(img_name)
            labels.append(label)

        self.A_list_cached = A_list
        self.B_list_cached = B_list
        self.C_list_cached = C_list
        self.labels_cached = np.array(labels, dtype=np.int64)
        self.names_cached = names

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        A_pil = self.A_list[index]
        B_pil = self.B_list[index]
        C_tokenized = self.C_list_tokenized[index]
        img_name = self.img_names[index]
        label = self.labels[index]
        noise_indicator = self.noise_indicator[index]

        if self.transform is not None:
            A_tensor = self.transform(A_pil.copy())
            B_tensor = self.transform(B_pil.copy())
        else:
            A_tensor = A_pil
            B_tensor = B_pil

        txt_tensor = torch.tensor(C_tokenized['input_ids'], dtype=torch.long)
        mask_tensor = torch.tensor(C_tokenized['attention_mask'], dtype=torch.long)
        seg_tensor = torch.tensor(C_tokenized['token_type_ids'], dtype=torch.long)

        return {'A': A_tensor, 'B': B_tensor, 'txt': txt_tensor, 'segment': seg_tensor, 'mask': mask_tensor, 'img_name': img_name, 'label': label, 'idx': index, 'noise_indicator': noise_indicator}


class ImageEncoder(nn.Module):
    def __init__(self, args):
        super(ImageEncoder, self).__init__()
        self.args = args
        torch.hub.set_dir(self.args.resnet_model_path)
        model = torchvision.models.resnet18(pretrained=True)
        modules = list(model.children())[:-1]
        self.model = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        out = self.model(x)
        out = self.pool(out)
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()
        return out

class BertEncoder(nn.Module):
    def __init__(self, args):
        super(BertEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_model_path)

    def forward(self, txt, mask, segment):
        last_hidden_state, pooled_output = self.bert(
            input_ids=txt,
            attention_mask=mask,
            token_type_ids=segment,
            return_dict=False,
        )
        return pooled_output

class ReliabilityEstimator(nn.Module):
    def __init__(self, num_views, feat_dim, num_classes, eps):
        super().__init__()
        self.num_views = num_views
        self.num_classes = num_classes
        self.eps = eps
        self.feat_dim = feat_dim
        self.reliability_dim = 2

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
        log_probs = F.log_softmax(logits, dim=-1)
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
            router_in = torch.cat([feature_list[m], reliability_features[m]], dim=-1)
            reliabilities.append(self.router_mlps[m](router_in))  # [B,1]
        return torch.cat(reliabilities, dim=-1)  # [B, num_views]

    def _finalize_forward(self, feature_list, logits_list):
        reliability_features = self._compute_reliability_features(logits_list)
        reliabilities = self._router_forward(feature_list, reliability_features)  # [B, M]

        logits_stack = torch.stack(logits_list, dim=1)  # [B, M, C]

        fused_logits = (reliabilities.unsqueeze(-1) * logits_stack).sum(dim=1)  # [B, C]
        return logits_list, fused_logits, reliabilities

class SUN_R_D_T_Backbone(ReliabilityEstimator):
    def __init__(self, args):
        super().__init__(num_views=3, feat_dim=512, num_classes=args.num_classes, eps=args.eps)

        self.rgbenc   = ImageEncoder(args)
        self.depthenc = ImageEncoder(args)
        self.txtenc = BertEncoder(args)
        self.feat_dim = 512
        self.txt_dim = 768

        self.rgb_head   = nn.Sequential(nn.Linear(self.feat_dim, self.num_classes))
        self.depth_head = nn.Sequential(nn.Linear(self.feat_dim, self.num_classes))
        self.txt_head = nn.Sequential(nn.Linear(self.txt_dim, self.num_classes))
        self.txt_2_img = nn.Sequential(nn.Linear(self.txt_dim, 512))

        self._build_router_mlps()

    def forward(self, rgb, depth, txt, mask, segment):
        rgb_feat_map   = self.rgbenc(rgb)
        depth_feat_map = self.depthenc(depth)
        txt_feat = self.txtenc(txt, mask, segment)

        rgb_feat   = torch.flatten(rgb_feat_map, start_dim=1)   # [B, feat_dim]
        depth_feat = torch.flatten(depth_feat_map, start_dim=1) # [B, feat_dim]

        rgb_logits   = self.rgb_head(rgb_feat)       # [B, C]
        depth_logits = self.depth_head(depth_feat)   # [B, C]
        txt_logits = self.txt_head(txt_feat)  # [B, C]

        logits_list = [rgb_logits, depth_logits, txt_logits]
        feature_list = [rgb_feat, depth_feat, self.txt_2_img(txt_feat)]

        return self._finalize_forward(feature_list, logits_list)


def get_arguments():
    parser = argparse.ArgumentParser(description='BML')
    parser.add_argument('--dataset_name', default='SUN-R-D-T', type=str)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument('--augment_ratio', default=0.5, type=float)
    parser.add_argument('--lambda_w', default=50.0, type=float)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--LOAD_SIZE", type=int, default=256)
    parser.add_argument("--FINE_SIZE", type=int, default=224)
    parser.add_argument('--log_path', default='logs/', type=str)
    parser.add_argument("--dataset_path", type=str, default="datasets/SUN-R-D-T/")
    parser.add_argument("--resnet_model_path", type=str, default="weights/resnet/")
    parser.add_argument("--bert_model_path", type=str, default="weights/google-bert/bert-base-uncased/")
    return parser.parse_args()


def train_one_seed(args, seed, tokenizer, train_tf, test_tf):
    set_seed(seed)
    args.seed = seed

    model = SUN_R_D_T_Backbone(args).cuda()

    bert_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('txtenc.bert.'):
            bert_params.append(param)
        else:
            other_params.append(param)

    lr_bert = 2e-5
    lr_other = 1e-4
    param_groups = [
        {'params': bert_params, 'lr': lr_bert, 'weight_decay': 0.01},
        {'params': other_params, 'lr': lr_other, 'weight_decay': 0.01}
    ]
    optimizer = optim.AdamW(param_groups)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-10)

    ce_criterion = nn.CrossEntropyLoss()

    # ---- Train ----
    for epoch in range(args.epochs):
        running_loss_cls = 0.0
        running_loss_align = 0.0
        running_loss_total = 0.0

        train_dataset = SUN_R_D_T_dataset(args, mode='train', tokenizer=tokenizer, max_len=77, transform=train_tf, noise_ratio=args.augment_ratio, noise_seed=epoch)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)

        model.train()
        for batch in train_loader:
            rgb, depth, tgt = batch['A'].cuda(), batch['B'].cuda(), batch['label'].cuda()
            clean_indicators = batch['noise_indicator'].cuda()
            txt, mask, segment = batch['txt'].cuda(), batch['mask'].cuda(), batch['segment'].cuda()

            _, fused_logits, reliabilities = model(rgb, depth, txt, mask, segment)

            loss_cls = ce_criterion(fused_logits, tgt)
            loss_align = F.binary_cross_entropy(reliabilities, clean_indicators.float())
            total_loss = loss_cls + args.lambda_w * loss_align

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss_cls += loss_cls.item()
            running_loss_align += loss_align.item()
            running_loss_total += total_loss.item()

        scheduler.step()
        if (epoch + 1) % 20 == 0 or epoch == 0:
            logging.info(f"[Seed {seed}] Epoch {epoch+1} | Total loss: {running_loss_total/len(train_loader):.4f} | CLS loss: {running_loss_cls/len(train_loader):.4f} | "
                            f"Align loss: {running_loss_align/len(train_loader):.4f}")

    # ---- Test ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    noise_ratios = np.linspace(0.0, 1.0, 11)
    model.eval()
    seed_accuracies = []

    with preserve_rng_states(), torch.no_grad():
        for noise_ratio in noise_ratios:
            test_dataset = SUN_R_D_T_dataset(args, mode='test', tokenizer=tokenizer, max_len=77, transform=test_tf, noise_ratio=noise_ratio, noise_seed=seed)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)

            all_preds = []
            all_labels = []

            for batch in test_loader:
                rgb, depth, tgt = batch['A'].cuda(), batch['B'].cuda(), batch['label'].cuda()
                txt, mask, segment = batch['txt'].cuda(), batch['mask'].cuda(), batch['segment'].cuda()
                _, fused_logits, _ = model(rgb, depth, txt, mask, segment)
                preds = torch.argmax(fused_logits, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(tgt.cpu().numpy())

            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            seed_accuracies.append(accuracy_score(all_labels, all_preds))

            del test_dataset, test_loader

    logging.info(f"[Seed {seed}] Test done. Acc@noise=0.0: {seed_accuracies[0]*100:.2f}%")
    return seed_accuracies


if __name__ == "__main__":
    args = to_begin()
    args.num_classes = 19

    mean = [0.4951, 0.3601, 0.4587]
    std  = [0.1474, 0.1950, 0.1646]
    train_tf = T.Compose([
        T.Resize((args.LOAD_SIZE, args.LOAD_SIZE)),
        T.RandomCrop((args.FINE_SIZE, args.FINE_SIZE)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tf = T.Compose([
        T.Resize((args.FINE_SIZE, args.FINE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    tokenizer = BertTokenizer.from_pretrained(args.bert_model_path, do_lower_case=True)

    all_results = {}
    noise_ratios = np.linspace(0.0, 1.0, 11)

    for seed in args.seeds:
        seed_accuracies = train_one_seed(args, seed, tokenizer, train_tf, test_tf)
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
