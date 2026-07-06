import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from spikingjelly.activation_based import surrogate, neuron, functional, encoding
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import argparse
import seaborn as sns

from utils import TDBatchNorm, ZBiasAdder, FragmentEnergyTracker  # tdBN
from fault_injection import build_fault_manager, get_fault_map
from benchmarks import ECOCHead, install_softsnn, install_router_from_mask, autoroute_with_mask, attach_slot_activity_tracker, install_astro_auto, install_falvolt_auto, install_lifa_auto
from algorithmic_fragmentation import batch_dynamic_fragments, batch_manual_fragments, agg_conf_logits, fragmentation_loss
from learnable_fragmentation import GlobalMultiLineFrags, DynamicGlobalStaticMultiLineFrags, DynamicGlobalMultiLineFragsMerge, DynamicGlobalMultiLineFragsMoE
from surrogate_encoders import SurrogatePoissonEncoder

dtype = torch.float

# ===== args (all kept) =====
def str2bool(v):
    if isinstance(v, bool): return v
    v = v.lower()
    if v in ("1","true","t","yes","y"): return True
    if v in ("0","false","f","no","n"): return False
    raise argparse.ArgumentTypeError("Boolean expected.")

parser = argparse.ArgumentParser()
# Network parameters
parser.add_argument("--train_batch_size", type=int, default=100)
parser.add_argument("--test_batch_size", type=int, default=100)
parser.add_argument("--data_path", type=str, default="propdata/CIFAR10")  # choose: propdata/CIFAR10 or propdata/CIFAR100
parser.add_argument("--num_steps", type=int, default=2)
parser.add_argument("--num_epochs", type=int, default=50)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--limit", type=float, default=1.0)             # it's the boundary of synaptic weights!
parser.add_argument("--bias", type=bool, default=False)
parser.add_argument("--vgg_depth", type=int, choices=[7, 11, 15], default=7)
# Z Bias
parser.add_argument("--ZBias", type=bool, default=False)
parser.add_argument("--z_bias_value", type=float, default=100.0)
parser.add_argument("--z_bias_ratio", type=float, default=0.5)      # % of neurons
parser.add_argument("--z_bias_fraction", type=float, default=0.0)  # % of neurons with outlier bias
parser.add_argument("--outlier_z_bias", type=float, default=100.0)  # outlier bias value
parser.add_argument("--bias_start_epoch", type=float, default=5)
parser.add_argument("--bias_target_layer", nargs='+', metavar="PATTERN", default=None)  # e.g., ['fc1']
parser.add_argument("--bias_apply_to_all", type=bool, default=True)
# Faults
parser.add_argument("--Fault", type=bool, default=False)
parser.add_argument("--fault_type", default="stuck", choices=["stuck", "random", "connectivity"])
parser.add_argument("--fault_dist", default="sporadic", choices=["sporadic", "clustered"])
parser.add_argument("--fault_ratio", type=float, default=0.1)       # 10.79%, sa0 : sa1 = 1.75% : 9.04%
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

train_batch_size = args.train_batch_size
test_batch_size = args.test_batch_size
data_path = args.data_path
num_steps = args.num_steps
num_epochs = args.num_epochs
learning_rate = args.learning_rate
weight_limit = args.limit
bias = args.bias
vgg_depth = args.vgg_depth
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

# ===== dataset & transforms =====
# Expect CIFAR10 / CIFAR100 (RGB 3×32×32). If MNIST is accidentally used, convert to 3ch/32x32 for compatibility.
if data_path.endswith("CIFAR10"):
    in_ch, H, W = 3, 32, 32
    num_classes = 10
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    ])
    train_set = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
    test_set  = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
elif data_path.endswith("CIFAR100"):
    in_ch, H, W = 3, 32, 32
    num_classes = 100
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    ])
    train_set = datasets.CIFAR100(data_path, train=True, download=True, transform=transform)
    test_set  = datasets.CIFAR100(data_path, train=False, download=True, transform=transform)
else:
    raise ValueError("Invalid data_path. Use propdata/CIFAR10 or propdata/CIFAR100.")

train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, drop_last=True)
test_loader  = DataLoader(test_set,  batch_size=test_batch_size, shuffle=False, drop_last=True)

# ===== loss/opt/scheduler/encoder =====
loss_fn = nn.MSELoss()
# loss_fn = nn.CrossEntropyLoss()
encoder  = encoding.PoissonEncoder()

# ===== VGG configs (CIFAR-friendly) =====
def get_vgg_cfg(depth):
    # compact VGG variants for CIFAR
    if depth == 7:
        return [64, 'M', 128, 'M', 256, 'M'], 256
    elif depth == 11:
        return [64, 'M', 128, 'M', 256, 256, 'M'], 256
    elif depth == 15:
        return [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M'], 256
    else:
        raise ValueError("vgg_depth must be one of {7,11,15}")

# ===== VGG-SNN =====
class VGG_SNN(nn.Module):
    def __init__(self, in_channels, num_classes, depth=11, bias=True):
        super().__init__()
        functional.set_backend(self, backend='cupy')

        cfg, last_c = get_vgg_cfg(depth)
        self.features, first_conv, first_lif, pool_count = self._make_layers(cfg, in_channels, bias)

        # remember for OnlineFaultMask
        self.first_conv = first_conv
        self.first_lif  = first_lif

        # robustly fix spatial size to 4x4 regardless of pool count
        self.fixpool = nn.AdaptiveAvgPool2d((4, 4))

        # classifier (two-layer MLP like simple_snn)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(last_c * 4 * 4, 512, bias=bias),
            neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.ATan()),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes, bias=bias),
            neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.ATan()),
        )

    def _make_layers(self, cfg, in_ch, bias):
        layers = []
        first_conv = None
        first_lif  = None
        pool_count = 0
        cur_ch = in_ch
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                pool_count += 1
            else:
                conv = nn.Conv2d(cur_ch, v, kernel_size=3, padding=1, bias=bias)
                bn   = TDBatchNorm(v)        # tdBN you provided
                lif  = neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.ATan())
                layers += [conv, bn, lif]
                if first_conv is None:
                    first_conv = conv
                    first_lif  = lif
                cur_ch = v
        return nn.Sequential(*layers), first_conv, first_lif, pool_count

    def forward(self, x):
        x = self.features(x)
        x = self.fixpool(x)
        x = self.classifier(x)
        return x

# ===== Net hooks =====
net = VGG_SNN(in_channels=in_ch, num_classes=num_classes, depth=vgg_depth, bias=bias).to(device)

# ===== Z Bias =====
if ZBias_on:
    bias_adder = ZBiasAdder(
        base_bias=z_bias,
        apply_fraction=z_bias_ratio,
        outlier_bias=outlier_z_bias,
        outlier_fraction=z_bias_fraction,
        outlier_mode='override',
        start_epoch=bias_start_epoch,
        target_patterns=bias_target_layer,
        apply_to_all=bias_apply_to_all,
        seed=7
    )
    bias_adder.attach(net, verbose=True)
else:
    bias_adder = None

# ===== Benchmark hooks =====
if ECOC_on:
    ecoc = ECOCHead(num_classes=num_classes, bit_values=(0.0, 1.0))
    net = ecoc.patch_last_linear(net)
else:
    ecoc = None

if Soft_on:
    bounder = install_softsnn(net, mode="bnp2", per="channel", symmetric=True)
else:
    bounder = None

# ===== Fault Injection =====
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
    router, pairs = install_router_from_mask(net, stuck_map, arch="vgg")
    handles, _ = attach_slot_activity_tracker(net, beta=0.9)
    router.swap_frac = 0.0
else:
    stuck_map, router, pairs = None, None, None

if Fault_on and Astro_on:
    astro = install_astro_auto(net, fault_mgr, start_epoch=fault_start_epoch)
else:
    astro = None

if Fault_on and Falvolt_on:
    falvolt = install_falvolt_auto(
        net,
        fault_mgr,  # build_fault_manager(...) 결과
        start_epoch=fault_start_epoch,  # FalVolt 시작 에폭 (예: 4)
        clamp=(0.3, 2.0),  # v_threshold 클램프 범위
        include_bias=False,  # 보통 False 권장
        verbose=True
    )
else:
    falvolt = None

if Fault_on and LIFA_on:
    lifa = install_lifa_auto(
        net,
        fault_mgr,
        start_epoch=fault_start_epoch,
        arch="vgg",
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

# ===== Parameter clipper (same as simple_snn.py) =====
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
                # weight / bias 또는 gamma / beta 둘 다 커버
                for wname in ('weight', 'gamma'):
                    p = getattr(module, wname, None)
                    if p is not None:
                        p.data.clamp_(-self.bn_limit, self.bn_limit)
                for bname in ('bias', 'beta'):
                    p = getattr(module, bname, None)
                    if p is not None:
                        p.data.clamp_(-self.bn_limit, self.bn_limit)

# ===== Optimizer & Scheduler =====
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.75)

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
        H=32, W=32, num_steps=num_steps,
        n_angles=180,
        importance_cfg=importance_cfg,
        power_norm=power_cfg,
        balance_metric="mse",
        balance_weight=0.01,
        line_sep_weight=1e-3,
        line_sep_cos_thr=0.995,
        line_sep_offset_margin=0.03,
        line_cross_weight=1e-3,
        sharpness=5.0,
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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.75)
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
    #     H=32, W=32,
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
        H=32, W=32,
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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.75)
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

# ===== monitors & logs =====
loss_hist = []
energy_hist = []
est_hist = []
counter = 0
epoch = 0

def print_batch_accuracy(net_output, net_targets):
    train_pred = ecoc.decode(net_output, metric='euclidean') if ecoc is not None else net_output.argmax(1)
    acc = np.mean((net_targets == train_pred).detach().cpu().numpy())
    print(f"Train set accuracy for a single minibatch: {acc * 100:.2f}%")

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

# ===== training =====
# import time
# def _fmt_time(sec):
#     s = int(round(sec))
#     m, s = divmod(s, 60)
#     h, m = divmod(m, 60)
#     return f"{h:02d}:{m:02d}:{s:02d}"
# if torch.cuda.is_available():
#     torch.cuda.synchronize()
# __train_t0 = time.perf_counter()

# ===== training =====
for epoch in range(num_epochs):
    if ZBias_on and bias_adder is not None:
        bias_adder.current_epoch = epoch

    iter_counter = 0
    train_batch = iter(train_loader)
    if epoch >= 1:
        scheduler.step()

    net.train()

    for data, targets in train_batch:
        data = data.to(device)                     # [B,3,32,32] (or converted)
        targets = targets.to(device)
        target_onehot = nn.functional.one_hot(targets, num_classes).float()

        # temporal loop
        logits_t = None
        if Frag_on:
            # data = batch_manual_fragments(data, num_steps, overlap=True, direction="horizontal",
            #                               kernel_size=15, overlap_iter=3, power_norm=power_cfg)
            data = batch_dynamic_fragments(data, num_steps, overlap=True, method="combo", weak_model=net,
                                           kernel_size=15, overlap_iter=3, per_image=False, power_norm=power_cfg)
            output = []
            for step in range(num_steps):
                input = data[:, step]
                spikes = encoder(input).float()
                output.append(net(spikes))
            output = torch.stack(output, dim=0)  # [B,T,C]

        elif Learnable_on:
            data = learnable_frags(data)  # [B, T, C, H, W]
            output = []
            for step in range(num_steps):
                input = data[:, step]
                spikes = encoder(input).float()
                output.append(net(spikes))
            output = torch.stack(output, dim=0)

        elif Dynamic_on:
            data = dynamic_frags(data, output_mode="mix", sample_steps=True)  # [B, Tmax, C, H, W]
            num_steps = data.size(1)
            output = []
            for step in range(num_steps):
                input = data[:, step]
                spikes = encoder(input).float()
                output.append(net(spikes))
            output = torch.stack(output, dim=0)

        else:
            output = 0
            for step in range(num_steps):
                spikes = encoder(data).float()
                output += net(spikes)
            output /= num_steps

        # RMSE (ecos same as simple_snn)
        if ECOC_on:
            loss_val = ecoc.loss_ce(output, targets, metric="euclidean", temp=1.0, squared=True)
            # loss_val = torch.sqrt(ecoc.loss_mse(output, targets) + 1e-6)
        elif Soft_on and epoch == 0:
            bounder.capture_snapshot(net)
            bounder.activate()
        elif Frag_on:
            loss_val = fragmentation_loss(output, targets, mode="rmse")
        elif Learnable_on:
            loss_val = (fragmentation_loss(output, targets, mode="rmse") + learnable_frags.aux_loss() +
                        learnable_frags.sep_loss() + learnable_frags.cross_loss())
        elif Dynamic_on:
            loss_val = (fragmentation_loss(output, targets, mode="rmse") + dynamic_frags.aux_loss() +
                        dynamic_frags.sep_loss() + dynamic_frags.cross_loss())
        elif Fault_on and Astro_on:
            loss_val = torch.sqrt(loss_fn(output, target_onehot) + 1e-6) + astro(epoch)
        elif Fault_on and Falvolt_on:
            loss_val = torch.sqrt(loss_fn(output, target_onehot) + 1e-6) + falvolt(epoch)
        elif Fault_on and LIFA_on:
            loss_val = torch.sqrt(loss_fn(output, target_onehot) + 1e-6) + lifa(epoch)
        else:
            loss_val = torch.sqrt(loss_fn(output, target_onehot) + 1e-6)
            # loss_val = loss_fn(output, targets)


        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        functional.reset_net(net)  # reset LIF states

        loss_hist.append(loss_val.item())

        with torch.no_grad():
            if Fault_on and epoch >= fault_start_epoch:
                fault_mgr.apply_(net)
                if Routing_on and epoch == fault_start_epoch:
                    autoroute_with_mask(router, pairs, stuck_map)
                # optimizer initialization with adjusted learning rate
                if epoch == fault_start_epoch and iter_counter == 0:
                    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

            net.apply(ParameterClipper())

            if counter % 50 == 0:
                if Frag_on or Learnable_on or Dynamic_on:
                    output = agg_conf_logits(output, tau=2.0, time_major=True)
                train_printer(output.view(train_batch_size, -1), targets)

            counter += 1
            iter_counter += 1

# ===== testing =====
# if torch.cuda.is_available():
#     torch.cuda.synchronize()
# __train_elapsed = time.perf_counter() - __train_t0
# print(f"[TIMER] Trained {num_epochs} epochs in {_fmt_time(__train_elapsed)} "
#       f"({__train_elapsed:.3f} s, ~{__train_elapsed/num_epochs:.3f} s/epoch)")

total = 0
correct = 0
w_targets = torch.tensor([], dtype=dtype).to(device)
w_predicted = torch.tensor([], dtype=dtype).to(device)

# ===== testing =====
# frag_energy = FragmentEnergyTracker(layout="btc")

with torch.no_grad():
    if ZBias_on and bias_adder is not None:
        bias_adder.current_epoch = epoch

    net.eval()

    for m in net.modules():
        if _is_bn_like(m):
            m.train()

    if Fault_on and fault_type == "random":
        fault_mgr.apply_(net)

    for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)
        target_onehot = nn.functional.one_hot(targets, num_classes).float()

        if Frag_on:
            # data = batch_manual_fragments(data, num_steps, overlap=True, direction="horizontal",
            #                               kernel_size=15, overlap_iter=3, power_norm=power_cfg)
            data = batch_dynamic_fragments(data, num_steps, overlap=True, method="combo", weak_model=net,
                                           kernel_size=15, overlap_iter=3, per_image=False, power_norm=power_cfg)
            test_output = []
            for step in range(num_steps):
                input = data[:, step]
                spikes = encoder(input).float()
                test_output.append(net(spikes))
            test_output = torch.stack(test_output, dim=0)
            test_output = agg_conf_logits(test_output, tau=2.0, time_major=True)

        elif Learnable_on:
            data = learnable_frags(data)  # [B, T, C, H, W]
            # frag_energy.update(data)
            test_output = []
            for step in range(num_steps):
                input = data[:, step]
                spikes = encoder(input).float()
                test_output.append(net(spikes))
            test_output = torch.stack(test_output, dim=0)
            test_output = agg_conf_logits(test_output, tau=2.0, time_major=True)

        elif Dynamic_on:
            data = dynamic_frags(data, output_mode="selected", sample_steps=True)  # [B, Tmax, C, H, W]
            # frag_energy.update(data)
            num_steps = data.size(1)
            test_output = []
            for step in range(num_steps):
                input = data[:, step]
                spikes = encoder(input).float()
                test_output.append(net(spikes))
            test_output = torch.stack(test_output, dim=0)
            test_output = agg_conf_logits(test_output, tau=2.0, time_major=True)

        else:
            test_output = 0
            for step in range(num_steps):
                spikes = encoder(data).float()
                test_output += net(spikes)
            test_output /= num_steps

        test_pred = ecoc.decode(test_output, metric='euclidean') if ecoc is not None else test_output.argmax(1)
        total += targets.size(0)
        correct += (targets == test_pred).sum().item()
        w_targets = torch.cat((w_targets, targets), dim=0)
        w_predicted = torch.cat((w_predicted, test_pred), dim=0)

        functional.reset_net(net)

# === After the test loop ===
print(f"Total correctly classified test set images: {correct}/{total}")
print(f"Test Set Accuracy: {100 * correct / total:.2f}%")
# print(frag_energy.format(title=None, joiner=", "))

# Confusion Matrix
w_targets = w_targets.detach().cpu().numpy()
w_predicted = w_predicted.detach().cpu().numpy()
cm = confusion_matrix(w_targets, w_predicted)

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
