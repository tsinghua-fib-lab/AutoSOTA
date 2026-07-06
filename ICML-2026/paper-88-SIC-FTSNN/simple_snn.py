import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from spikingjelly.activation_based import surrogate, neuron, functional, encoding
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import argparse
import seaborn as sns

from fault_injection import build_fault_manager, get_fault_map
from benchmarks import ECOCHead, install_softsnn, install_router_from_mask, autoroute_with_mask, \
    attach_slot_activity_tracker, install_astro_auto, install_falvolt_auto, install_lifa_auto
from algorithmic_fragmentation import batch_dynamic_fragments, batch_manual_fragments, agg_conf_logits, FragNorm
from learnable_fragmentation import GlobalMultiLineFrags, DynamicGlobalStaticMultiLineFrags, DynamicGlobalMultiLineFragsMerge, DynamicGlobalMultiLineFragsMoE
from surrogate_encoders import SurrogatePoissonEncoder

from utils import ZBiasAdder, _choose_hw_autofold, seq1d_to_2d_autofold, FragmentEnergyTracker

dtype = torch.float

# ===== args (all kept) =====
def str2bool(v):
    if isinstance(v, bool): return v
    v = v.lower()
    if v in ("1", "true", "t", "yes", "y"): return True
    if v in ("0", "false", "f", "no", "n"): return False
    raise argparse.ArgumentTypeError("Boolean expected.")

parser = argparse.ArgumentParser()
# Network parameters
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--dataset", type=str, default="image", choices=["image", "sequential"])
parser.add_argument("--data_path", type=str, default="propdata/MNIST")
parser.add_argument("--num_steps", type=int, default=4)
parser.add_argument("--num_epochs", type=int, default=50)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--limit", type=float, default=1.0)  # it's the boundary of synaptic weights!
parser.add_argument("--bias", type=bool, default=True)
# Z Bias
parser.add_argument("--ZBias", type=bool, default=False)
parser.add_argument("--z_bias_value", type=float, default=50.0)
parser.add_argument("--z_bias_ratio", type=float, default=1.0)  # % of neurons
parser.add_argument("--z_bias_fraction", type=float, default=0.0)  # % of neurons with outlier bias
parser.add_argument("--outlier_z_bias", type=float, default=50.0)  # outlier bias value
parser.add_argument("--bias_start_epoch", type=float, default=5)
parser.add_argument("--bias_target_layer", nargs='+', metavar="PATTERN", default=None)  # e.g., ['fc1']
parser.add_argument("--bias_apply_to_all", type=bool, default=True)
# Faults
parser.add_argument("--Fault", type=bool, default=False)
parser.add_argument("--fault_type", default="stuck", choices=["stuck", "random", "connectivity"])
parser.add_argument("--fault_dist", default="sporadic", choices=["sporadic", "clustered"])
parser.add_argument("--fault_ratio", type=float, default=0.1)  # 10.79%, sa0 : sa1 = 1.75% : 9.04%
parser.add_argument("--noise_std", type=float, default=0.5)
parser.add_argument("--fault_start_epoch", type=int, default=5)
# Benchmarks
parser.add_argument("--ECOC", type=str2bool, default=False)
parser.add_argument("--Soft", type=str2bool, default=False)
parser.add_argument("--Routing", type=str2bool, default=False)
parser.add_argument("--Astrocyte", type=str2bool, default=False)
parser.add_argument("--Falvolt", type=str2bool, default=False)
parser.add_argument("--LIFA", type=str2bool, default=False)
# Proposed
parser.add_argument("--Frag", type=str2bool, default=False)      # Fragmentation function
parser.add_argument("--Learnable", type=str2bool, default=False) # Learnable division line
parser.add_argument("--Dynamic", type=str2bool, default=False)   # Dynamically changing the number of fragments
# ETC
parser.add_argument("--gpu_num", type=int, default=0)
parser.add_argument("--plot", type=bool, default=False)

args = parser.parse_args()

batch_size = args.batch_size
dataset = args.dataset
is_sequential = (dataset == "sequential")
data_path = args.data_path
num_steps = args.num_steps
num_epochs = args.num_epochs
learning_rate = args.learning_rate
weight_limit = args.limit
bias = args.bias
# Z Bias
ZBias_on = args.ZBias
z_bias = args.z_bias_value
z_bias_ratio = args.z_bias_ratio
z_bias_fraction = args.z_bias_fraction
outlier_z_bias = args.outlier_z_bias
bias_start_epoch = args.bias_start_epoch
bias_target_layer = args.bias_target_layer
bias_apply_to_all = args.bias_apply_to_all
# Faults
Fault_on = args.Fault
fault_type = args.fault_type
fault_dist = args.fault_dist
fault_ratio = args.fault_ratio
noise_std = args.noise_std
fault_start_epoch = args.fault_start_epoch
# Benchmarks
ECOC_on = args.ECOC
Soft_on = args.Soft
Routing_on = args.Routing
Astro_on = args.Astrocyte
Falvolt_on = args.Falvolt
LIFA_on = args.LIFA
# Proposed
Frag_on = args.Frag
Learnable_on = args.Learnable
Dynamic_on = args.Dynamic
# ETC
gpu_num = args.gpu_num
plot = args.plot

if gpu_num != 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print("gpu: ", gpu_num)


class UCIHARDataset(Dataset):
    def __init__(self, data_path, split="train"):
        self.split = split
        base = os.path.join(data_path, split, "Inertial Signals")
        signal_names = ["total_acc_x", "total_acc_y", "total_acc_z",
                        "body_acc_x", "body_acc_y", "body_acc_z",
                        "body_gyro_x", "body_gyro_y", "body_gyro_z"]
        signals = []
        for name in signal_names:
            filename = os.path.join(base, f"{name}_{split}.txt")
            data = np.loadtxt(filename)
            signals.append(data)
        self.data = np.stack(signals, axis=2)  # [num_samples, 128, 9]
        self.data = torch.tensor(self.data, dtype=torch.float)
        label_path = os.path.join(data_path, split, f"y_{split}.txt")
        labels = np.loadtxt(label_path, dtype=int)
        self.targets = torch.tensor(labels - 1, dtype=torch.long)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


# Define a transform
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))])

# Load MNIST/FashionMNIST dataset
if not is_sequential:
    if data_path == "propdata/MNIST":
        train_set = datasets.MNIST(data_path, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(data_path, train=False, download=True, transform=transform)
        input_dim = 28 * 28
        num_classes = 10
    elif data_path == "propdata/FMNIST":
        train_set = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)
        input_dim = 28 * 28
        num_classes = 10
    else:
        raise ValueError("Invalid data path for image dataset!")
else:
    train_set = UCIHARDataset(data_path, split="train")
    test_set = UCIHARDataset(data_path, split="test")
    input_dim = 128 * 9
    num_classes = 6

# Create DataLoaders
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)

# Store loss history for future plotting
loss_hist = []
energy_hist = []
est_hist = []
counter = 0
epoch = 0

# Choose 2D shape for fragmentation modules
IMG_H, IMG_W = 28, 28
SEQ_EFF_LEN = None

if is_sequential and (Frag_on or Learnable_on or Dynamic_on):
    raw_len = input_dim  # e.g., 128*9 for UCIHAR

    # (optional) pixel budget: set 0 to disable
    autofold_max_pixels = 0          # 예: 1024로 두면 32x32로 유도 가능
    autofold_aspect_max = 4.0
    min_side = 15                    # 너 fragmentation kernel_size가 15라서 기본값으로 추천

    eff_len = min(raw_len, autofold_max_pixels) if autofold_max_pixels > 0 else raw_len
    IMG_H, IMG_W = _choose_hw_autofold(eff_len, min_side=min_side, aspect_max=autofold_aspect_max)
    SEQ_EFF_LEN = eff_len

    # IMPORTANT: MLP input_dim must match flattened (H*W)
    input_dim = IMG_H * IMG_W

    print(f"[AutoFold] raw_len={raw_len}, eff_len={eff_len} -> (H,W)=({IMG_H},{IMG_W}), input_dim={input_dim}")

# Define Network
class Net(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        functional.set_backend(self, backend='cupy')

        # Initialize layers
        if Frag_on or Learnable_on or Dynamic_on:
            self.fn = FragNorm(num_features=input_dim, time_aggregate=False, affine=True,
                               track_running_stats=False, momentum=0.1, eps=1e-5)
        self.fn1 = FragNorm(num_features=1024, time_aggregate=False, affine=True,
                            track_running_stats=False, momentum=0.1, eps=1e-5)
        self.fc1 = nn.Linear(input_dim, 1024, bias=bias)
        self.lif1 = neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.Sigmoid(alpha=4.0))
        self.fn2 = FragNorm(num_features=512, time_aggregate=False, affine=True,
                            track_running_stats=False, momentum=0.1, eps=1e-5)
        self.fc2 = nn.Linear(1024, 512, bias=bias)
        self.lif2 = neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.Sigmoid(alpha=4.0))
        self.fn3 = FragNorm(num_features=128, time_aggregate=False, affine=True,
                            track_running_stats=False, momentum=0.1, eps=1e-5)
        self.fc3 = nn.Linear(512, 128, bias=bias)
        self.lif3 = neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.Sigmoid(alpha=4.0))
        self.fc4 = nn.Linear(128, num_classes, bias=bias)
        self.lif4 = neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.Sigmoid(alpha=4.0))

    def forward(self, x):
        if Frag_on or Learnable_on or Dynamic_on:
            x = self.fn(x)
        x = self.fc1(x)
        if Frag_on or Learnable_on or Dynamic_on:
            x = self.fn1(x)
        x = self.lif1(x)
        x = self.fc2(x)
        if Frag_on or Learnable_on or Dynamic_on:
            x = self.fn2(x)
        x = self.lif2(x)
        x = self.fc3(x)
        if Frag_on or Learnable_on or Dynamic_on:
            x = self.fn3(x)
        x = self.lif3(x)
        x = self.fc4(x)
        x = self.lif4(x)

        return x


# Load the network onto CUDA if available
net = Net(input_dim=input_dim, num_classes=num_classes).to(device)

# === Z Bias ===
if ZBias_on:
    bias_adder = ZBiasAdder(
        base_bias=z_bias,
        apply_fraction=z_bias_ratio,
        outlier_bias=outlier_z_bias,
        outlier_fraction=z_bias_fraction,
        outlier_mode='override',  # base + outlier
        start_epoch=bias_start_epoch,
        target_patterns=bias_target_layer,
        apply_to_all=bias_apply_to_all,
        seed=7
    )
    bias_adder.attach(net, verbose=True)
else:
    bias_adder = None

# === Benchamrk hooks ===
if ECOC_on:
    ecoc = ECOCHead(num_classes=num_classes, bit_values=(0.0, 1.0))
    net = ecoc.patch_last_linear(net)
else:
    ecoc = None

if Soft_on:
    bounder = install_softsnn(net, mode="bnp2", per="channel", symmetric=True)
else:
    bounder = None

# === Fault injection ===
if Fault_on:
    fault_mgr = build_fault_manager(
        net,
        ratio=fault_ratio,
        fault_type=fault_type,
        distribution=fault_dist,
        stuck_at=weight_limit,
        noise_std=noise_std,
        limit=weight_limit,
        include_bias=True,
    )
else:
    fault_mgr = None

if Fault_on and Routing_on:
    stuck_map = get_fault_map(fault_mgr, include_bias=False)
    router, pairs = install_router_from_mask(net, stuck_map, arch="mlp")
    handles, _ = attach_slot_activity_tracker(net, beta=0.9)
    router.swap_frac = 0.0
    router.soft_beta = 1.0
else:
    stuck_map, router, pairs = None, None, None

if Fault_on and Astro_on:
    astro = install_astro_auto(net, fault_mgr, start_epoch=fault_start_epoch)
else:
    astro = None

if Fault_on and Falvolt_on:
    falvolt = install_falvolt_auto(
        net,
        fault_mgr,
        start_epoch=fault_start_epoch,
        clamp=(0.3, 2.0),
        include_bias=False,
        verbose=True
    )
else:
    falvolt = None

if Fault_on and LIFA_on:
    lifa = install_lifa_auto(
        net,
        fault_mgr,
        start_epoch=fault_start_epoch,
        arch="mlp",
    )
else:
    lifa = None

if Frag_on or Learnable_on or Dynamic_on:
    power_cfg = dict(
        mode="rms",
        target=1.0,
        per_channel=False,
        use_mask=True,
        max_gain=6.0,
        detach_stats=True
    )
else:
    power_cfg = None


# Define weight clipper
def _is_bn_like(m: nn.Module) -> bool:
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        return True
    name = m.__class__.__name__.lower()
    if 'batchnorm' in name:
        return True
    return hasattr(m, 'running_mean') and hasattr(m, 'running_var')


class ParameterClipper:
    def __init__(self, limit=weight_limit, clip_bn=True, bn_limit=weight_limit):
        self.limit = limit
        self.clip_bn = clip_bn
        self.bn_limit = bn_limit

    def __call__(self, module):
        import torch.nn as nn

        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            if getattr(module, "weight", None) is not None:
                module.weight.data.clamp_(-self.limit, self.limit)
            if getattr(module, "bias", None) is not None:
                module.bias.data.clamp_(-self.limit, self.limit)

        elif self.clip_bn:
            if _is_bn_like(module):

                for wname in ('weight', 'gamma'):
                    p = getattr(module, wname, None)
                    if p is not None:
                        p.data.clamp_(-self.bn_limit, self.bn_limit)
                for bname in ('bias', 'beta'):
                    p = getattr(module, bname, None)
                    if p is not None:
                        p.data.clamp_(-self.bn_limit, self.bn_limit)


# Print batch accuracy
def print_batch_accuracy(net_output, net_targets):
    train_pred = ecoc.decode(net_output, metric='euclidean') if ecoc is not None else net_output.argmax(1)
    acc = np.mean((net_targets == train_pred).detach().cpu().numpy())
    print(f"Train set accuracy for a single minibatch: {acc * 100:.2f}%")


# Print training information
def train_printer(predicted, targets):
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.4f}")
    print_batch_accuracy(predicted, targets)
    print("\n")


def _get_attr_by_path(root, dotted):
    obj = root
    parts = dotted.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    return getattr(obj, parts[-1])  # Tensor param/buffer


# Define optimizer, scheduler, and loss function
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
warmup_epochs_lr = 5
warmup_lr = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs_lr)
cosine_lr = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs_lr, eta_min=1e-6)
scheduler = SequentialLR(optimizer, schedulers=[warmup_lr, cosine_lr], milestones=[warmup_epochs_lr])
loss_fn = nn.MSELoss()
# loss_fn = nn.CrossEntropyLoss()
encoder = encoding.PoissonEncoder()

if Learnable_on:
    importance_cfg = {
        "measure": "combo",
        "axis_metric": "gini",
        "cut_scheme": "equal_mass",
        "combo_cfg": {
            # 논문 수식(LoG+Sobel+Var) 표기와 맞추려면 1.0/1.0/1.0가 가장 직관적
            "w_log": 1.0,
            "w_sobel": 1.0,
            "w_var": 1.0,

            # LoG 멀티스케일은 구현 선택(논문은 scale을 명시하진 않음)
            "sigmas": [1.0, 2.0, 4.0],
            "alpha": [0.5, 0.3, 0.2],
            "log_kernel_size": 9,

            "var_sigma": 1.5,
            "var_kernel_size": 9,
        },
    }

    learnable_frags = GlobalMultiLineFrags(
        H=IMG_H, W=IMG_W, num_steps=num_steps,
        n_angles=180,
        importance_cfg=importance_cfg,
        power_norm=power_cfg,
        balance_metric="mse",
        balance_weight=0.01,
        line_sep_weight=1e-3,
        line_sep_cos_thr=0.995,
        line_sep_offset_margin=0.03,
        line_cross_weight=1e-3,
        sharpness=None,
        hard_forward=True,
        hard_eval=True,
        auto_init=True,
        overlap=True,
        kernel_size=15,
        overlap_iter=3,
    ).to(device)

    encoder = SurrogatePoissonEncoder(grad_mode="expected", hard=False, prob_from="raw").to(device)

    params = list(net.parameters()) + list(learnable_frags.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    warmup_epochs_lr = 5
    warmup_lr = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs_lr)
    cosine_lr = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs_lr, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_lr, cosine_lr], milestones=[warmup_epochs_lr])
    dynamic_frags = None

elif Dynamic_on:
    importance_cfg = {
        "measure": "combo",
        "axis_metric": "gini",
        "cut_scheme": "equal_mass",
        "combo_cfg": {
            # 논문 수식(LoG+Sobel+Var) 표기와 맞추려면 1.0/1.0/1.0가 가장 직관적
            "w_log": 1.0,
            "w_sobel": 1.0,
            "w_var": 1.0,

            # LoG 멀티스케일은 구현 선택(논문은 scale을 명시하진 않음)
            "sigmas": [1.0, 2.0, 4.0],
            "alpha": [0.5, 0.3, 0.2],
            "log_kernel_size": 9,

            "var_sigma": 1.5,
            "var_kernel_size": 9,
        },
    }

    # dynamic_frags = DynamicGlobalStaticMultiLineFrags(
    #     H=IMG_H, W=IMG_W,
    #     candidates=(2, 4, 8),
    #     init_num_steps=num_steps,
    #     direction="horizontal",  # 또는 vertical / diag_lr / diag_rl
    #     gumbel_tau=1.0,
    #     gumbel_hard=True,
    #     warmup_iters=500,
    #     importance_cfg=importance_cfg,
    #     power_norm=power_cfg,
    #     hard_forward=True,
    #     hard_eval=True,
    #     overlap=True,
    #     kernel_size=15,
    #     overlap_iter=3,
    #     balance_metric="mse",
    #     balance_weight=0.01,
    #     line_sep_weight=1e-3,
    #     line_sep_cos_thr=0.995,
    #     line_sep_offset_margin=0.03,
    #     line_cross_weight=1e-3,
    #     auto_init=True,  # signature 호환용(정적 모듈에서는 사용 안 함)
    # ).to(device)

    dynamic_frags  = DynamicGlobalMultiLineFragsMoE(
        H=IMG_H, W=IMG_W,
        candidates=(2, 4, 8),
        init_num_steps=num_steps,     # 시작 bias
        gumbel_tau=1.0,
        gumbel_hard=True,
        warmup_iters=500,             # 초반엔 init_num_steps 고정하고 싶으면

        importance_cfg=importance_cfg,
        power_norm=power_cfg,

        hard_forward=True,
        hard_eval=True,

        overlap=True,
        kernel_size=15,
        overlap_iter=3,

        balance_metric="mse",
        balance_weight=0.01,

        line_sep_weight=1e-3,         # 분절선 중복 방지
        line_sep_cos_thr=0.995,
        line_sep_offset_margin=0.03,

        line_cross_weight=1e-3,       # 선 교차 방지

        auto_init=True,               # 첫 배치로 입력-only 앵커 초기화
    ).to(device)

    encoder = SurrogatePoissonEncoder(grad_mode="expected", hard=False, prob_from="raw").to(device)

    params = list(net.parameters()) + list(dynamic_frags.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    warmup_epochs_lr = 5
    warmup_lr = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs_lr)
    cosine_lr = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs_lr, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_lr, cosine_lr], milestones=[warmup_epochs_lr])
    learnable_frags = None

else:
    learnable_frags = None
    dynamic_frags = None


# ===== Fault Injection preparation (Conv/Linear only to avoid BN params) =====
weight_param_names, bias_param_names = [], []
weight_shapes, bias_shapes = [], []

for name, module in net.named_modules():
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.LSTM, nn.GRU)):
        # weight exists
        weight_param_names.append(f"{name}.weight")
        weight_shapes.append(module.weight.data.shape)
        if bias and (module.bias is not None):
            bias_param_names.append(f"{name}.bias")
            bias_shapes.append(module.bias.data.shape)

# Outer training loop
# import time
# def _fmt_time(sec):
#     s = int(round(sec))
#     m, s = divmod(s, 60)
#     h, m = divmod(m, 60)
#     return f"{h:02d}:{m:02d}:{s:02d}"
# if torch.cuda.is_available():
#     torch.cuda.synchronize()
# __train_t0 = time.perf_counter()

for epoch in range(num_epochs):
    if ZBias_on:
        bias_adder.current_epoch = epoch

    iter_counter = 0
    train_batch = iter(train_loader)
    if epoch >= 1:
        scheduler.step()

    net.train()
    for data, targets in train_batch:
        data = data.to(device)
        targets = targets.to(device)
        if not is_sequential:
            data = data.view(batch_size, 1, 28, 28)
        else:
            dmin = data.amin(dim=(1, 2), keepdim=True)
            dmax = data.amax(dim=(1, 2), keepdim=True)
            data = (data - dmin) / (dmax - dmin + 1e-8)

            # convert sequential -> image-like [B,1,IMG_H,IMG_W] when using fragmentation
            if Frag_on or Learnable_on or Dynamic_on:
                data = seq1d_to_2d_autofold(data, H=IMG_H, W=IMG_W, out_len=SEQ_EFF_LEN, pad_value=0.0)

        target_onehot = nn.functional.one_hot(targets, num_classes).float()
        # Label smoothing: 0.05 uniform noise
        eps = 0.05
        target_onehot = target_onehot * (1 - eps) + eps / num_classes

        # inner training loop (spike timing)
        logits_t = None
        if Frag_on:
            # data = batch_manual_fragments(data, num_steps, overlap=True, direction="horizontal",
            #                               kernel_size=15, overlap_iter=3, power_norm=power_cfg)
            data = batch_dynamic_fragments(data, num_steps, overlap=True, method="combo", weak_model=net,
                                           kernel_size=15, overlap_iter=3, per_image=False,
                                           power_norm=power_cfg)  # MNIST => overlap=False, FMNIST => overlap=True
            output = []
            for step in range(num_steps):
                input = data[:, step].view(batch_size, -1)  # [B,C]
                spikes = encoder(input).float()
                output.append(net(spikes))
            output = torch.stack(output, dim=0)
            output = agg_conf_logits(output, tau=2.0, time_major=True)

        elif Learnable_on:
            data = learnable_frags(data)  # [B, T, C, H, W]
            output = []
            for step in range(num_steps):
                input = data[:, step].view(batch_size, -1)  # [B,C]
                spikes = encoder(input).float()
                output.append(net(spikes))
            output = torch.stack(output, dim=0)
            output = agg_conf_logits(output, tau=2.0, time_major=True)

        elif Dynamic_on:
            data = dynamic_frags(data, output_mode="mix", sample_steps=True)  # [B, Tmax, C, H, W]
            num_steps = data.size(1)
            output = []
            for step in range(num_steps):
                input = data[:, step].view(batch_size, -1)
                spikes = encoder(input).float()
                output.append(net(spikes))
            output = torch.stack(output, dim=0)  # [T,B,K]
            output = agg_conf_logits(output, tau=2.0, time_major=True)

        else:
            output = 0
            for step in range(num_steps):
                spikes = encoder(data.view(batch_size, -1)).float()
                output += net(spikes)
            output /= num_steps

        if ECOC_on:
            loss_val = ecoc.loss_ce(output, targets, metric="euclidean", temp=1.0, squared=True)
            # loss_val = torch.sqrt(ecoc.loss_mse(output, targets) + 1e-6)
        elif Soft_on and epoch == 0:
            bounder.capture_snapshot(net)
            bounder.activate()
        elif Frag_on:
            loss_val = torch.sqrt(loss_fn(output, target_onehot) + 1e-6)
        elif Learnable_on:
            loss_val = (torch.sqrt(loss_fn(output, target_onehot) + 1e-6) + learnable_frags.aux_loss()
                        + learnable_frags.sep_loss() + learnable_frags.cross_loss())
        elif Dynamic_on:
            loss_val = (torch.sqrt(loss_fn(output, target_onehot) + 1e-6) + dynamic_frags.aux_loss()
                        + dynamic_frags.sep_loss() + dynamic_frags.cross_loss())
        elif Fault_on and Astro_on:
            loss_val = torch.sqrt(loss_fn(output, target_onehot) + 1e-6) + astro(epoch)
        elif Fault_on and Falvolt_on:
            loss_val = torch.sqrt(loss_fn(output, target_onehot) + 1e-6) + falvolt(epoch)
        elif Fault_on and LIFA_on:
            loss_val = torch.sqrt(loss_fn(output, target_onehot) + 1e-6) + lifa(epoch)
        else:
            loss_val = torch.sqrt(loss_fn(output, target_onehot) + 1e-6)
            # loss_val = loss_fn(output, targets)


        # gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        functional.reset_net(net)

        # store loss history for future plotting
        loss_hist.append(loss_val.item())

        with torch.no_grad():
            # apply faults to the network parameters
            if Fault_on and epoch >= fault_start_epoch:
                fault_mgr.apply_(net)
                if Routing_on and epoch == fault_start_epoch:
                    autoroute_with_mask(router, pairs, stuck_map)
                # optimizer initialization with adjusted learning rate
                if epoch == fault_start_epoch and iter_counter == 0:
                    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

            # applying weight constraint
            net.apply(ParameterClipper())

            # print train/test loss/accuracy
            if counter % 50 == 0:
                train_printer(output.view(batch_size, -1), targets)

            counter += 1
            iter_counter += 1

# if torch.cuda.is_available():
#     torch.cuda.synchronize()
# __train_elapsed = time.perf_counter() - __train_t0
# print(f"[TIMER] Trained {num_epochs} epochs in {_fmt_time(__train_elapsed)} "
#       f"({__train_elapsed:.3f} s, ~{__train_elapsed/num_epochs:.3f} s/epoch)")

total = 0
correct = 0
w_targets = torch.tensor([], dtype=dtype).to(device)
w_predicted = torch.tensor([], dtype=dtype).to(device)

# Test the network
# frag_energy = FragmentEnergyTracker(layout="btc")

with torch.no_grad():
    if ZBias_on:
        bias_adder.current_epoch = epoch

    net.eval()
    if Learnable_on:
        learnable_frags.eval()
    elif Dynamic_on:
        dynamic_frags.eval()

    for m in net.modules():
        if _is_bn_like(m):
            m.train()

    if Fault_on and fault_type == "random":
        fault_mgr.apply_(net)

    for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)

        if not is_sequential:
            data = data.view(batch_size, 1, 28, 28)
        else:
            dmin = data.amin(dim=(1, 2), keepdim=True)
            dmax = data.amax(dim=(1, 2), keepdim=True)
            data = (data - dmin) / (dmax - dmin + 1e-8)

            if Frag_on or Learnable_on or Dynamic_on:
                data = seq1d_to_2d_autofold(data, H=IMG_H, W=IMG_W, out_len=SEQ_EFF_LEN, pad_value=0.0)

        target_onehot = nn.functional.one_hot(targets, num_classes).float()
        # Label smoothing: 0.05 uniform noise
        eps = 0.05
        target_onehot = target_onehot * (1 - eps) + eps / num_classes

        # forward pass
        if Frag_on:
            # data = batch_manual_fragments(data, num_steps, overlap=True, direction="horizontal",
            #                               kernel_size=15, overlap_iter=3, power_norm=power_cfg)
            data = batch_dynamic_fragments(data, num_steps, overlap=True, method="combo", weak_model=net,
                                           kernel_size=15, overlap_iter=3, per_image=False,
                                           power_norm=power_cfg)  # MNIST => overlap=False, FMNIST => overlap=True
            test_output = []
            for step in range(num_steps):
                input = data[:, step].view(batch_size, -1)
                spikes = encoder(input).float()
                test_output.append(net(spikes))
            test_output = torch.stack(test_output, dim=0)
            test_output = agg_conf_logits(test_output, tau=2.0, time_major=True)

        elif Learnable_on:
            data = learnable_frags(data)  # [B, T, C, H, W]
            # frag_energy.update(data)
            test_output = []
            for step in range(num_steps):
                input = data[:, step].view(batch_size, -1)  # [B,C]
                spikes = encoder(input).float()
                test_output.append(net(spikes))
            test_output = torch.stack(test_output, dim=0)
            test_output = agg_conf_logits(test_output, tau=2.0, time_major=True)

        elif Dynamic_on:
            data = dynamic_frags(data, output_mode="selected", sample_steps=False)
            # frag_energy.update(data)
            num_steps = data.size(1)
            test_output = []
            for step in range(num_steps):
                input = data[:, step].view(batch_size, -1)
                spikes = encoder(input).float()
                test_output.append(net(spikes))
            test_output = torch.stack(test_output, dim=0)
            test_output = agg_conf_logits(test_output, tau=2.0, time_major=True)

        else:
            test_output = 0
            for step in range(num_steps):
                spikes = encoder(data.view(batch_size, -1)).float()
                test_output += net(spikes)
            test_output /= num_steps

        # calculate total accuracy
        test_pred = ecoc.decode(test_output, metric='euclidean') if ecoc is not None else test_output.argmax(1)

        total += targets.size(0)
        correct += (targets == test_pred).sum().item()
        w_targets = torch.cat((w_targets, targets), dim=0)
        w_predicted = torch.cat((w_predicted, test_pred), dim=0)

        # reset the network
        functional.reset_net(net)

print(f"Total correctly classified test set images: {correct}/{total}")
print(f"Test Set Accuracy: {100 * correct / total:.2f}%")
# print(frag_energy.format(title=None, joiner=", "))

# Confusion Matrix
w_targets = w_targets.detach().cpu().numpy()
w_predicted = w_predicted.detach().cpu().numpy()
cm = confusion_matrix(w_targets, w_predicted)

# Plotting results
if plot:
    loss_fig = plt.figure(facecolor="w", figsize=(10, 5))
    plt.plot(loss_hist)
    plt.title("Loss Curves")
    plt.xlabel("Iteration")
    plt.ylabel("Train Loss")
    plt.tick_params(axis='both', direction='in')
    plt.show()

    cm_fig = plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
    plt.xlabel("Predictions")
    plt.ylabel("Targets")
    plt.title("Confusion Matrix")
    plt.show()