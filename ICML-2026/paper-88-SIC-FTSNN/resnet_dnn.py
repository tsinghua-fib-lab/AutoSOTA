import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import argparse
import seaborn as sns
from torchvision import transforms, datasets

from fault_injection import build_fault_manager, get_fault_map
from benchmarks import ECOCHead, install_softsnn, install_router_from_mask, autoroute_with_mask, attach_slot_activity_tracker, install_astro_auto, install_falvolt_auto, install_lifa_auto
from algorithmic_fragmentation import batch_dynamic_fragments, batch_manual_fragments, agg_conf_logits, fragmentation_loss
from learnable_fragmentation import GlobalMultiLineFrags, DynamicGlobalMultiLineFragsMerge, DynamicGlobalMultiLineFragsMoE
from utils import TDBatchNorm, make_cifar_loaders, ZBiasAdder

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torchvision._internally_replaced_utils import load_state_dict_from_url

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

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
parser.add_argument("--data_path", type=str, default="propdata/CIFAR100")  # choose: propdata/CIFAR10 or propdata/CIFAR100 or tiny-imagenet-200
parser.add_argument("--use_imagenet", type=str, default="false")  # true/false
parser.add_argument("--num_steps", type=int, default=2)
parser.add_argument("--num_epochs", type=int, default=50)
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--limit", type=float, default=100.0)             # it's the boundary of synaptic weights!
parser.add_argument("--bias", type=bool, default=False)
parser.add_argument("--resnet_depth", type=int, choices=[18, 34], default=18)
# Z Bias
parser.add_argument("--ZBias", type=bool, default=False)
parser.add_argument("--z_bias_value", type=float, default=100.0)
parser.add_argument("--z_bias_ratio", type=float, default=1.0)      # % of neurons
parser.add_argument("--z_bias_fraction", type=float, default=0.0)  # % of neurons with outlier bias
parser.add_argument("--outlier_z_bias", type=float, default=100.0)  # outlier bias value
parser.add_argument("--bias_start_epoch", type=float, default=2)
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
resnet_depth = args.resnet_depth
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
use_imagenet = str(args.use_imagenet).lower() in {"1","true","t","yes","y"}

if use_imagenet:
    train_tf = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_ds = datasets.ImageFolder(f"{data_path}/train", transform=train_tf)
    test_ds = datasets.ImageFolder(f"{data_path}/val", transform=test_tf)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=test_batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    in_ch, H, W, num_classes = 3, 64, 64, 200

else:
    train_loader, _, meta = make_cifar_loaders(
        data_root=data_path,
        dataset="cifar10" if data_path.endswith("CIFAR10") else "cifar100",
        batch_size=train_batch_size,
        num_workers=4,
        preset="none",
        normalize_for_poisson=True
    )

    _, test_loader, _ = make_cifar_loaders(
        data_root=data_path,
        dataset="cifar10" if data_path.endswith("CIFAR10") else "cifar100",
        batch_size=test_batch_size,
        num_workers=4,
        preset="none",
        normalize_for_poisson=True
    )

    in_ch, H, W, num_classes = meta["in_ch"], meta["H"], meta["W"], meta["num_classes"]

# ===== loss/opt/scheduler/encoder =====
# loss_fn = nn.MSELoss()
loss_fn = nn.CrossEntropyLoss()

# ===== ResNet-SNN for CIFAR (20/32/44) =====

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, single_step_neuron: callable = None, **kwargs):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.sn1 = single_step_neuron(**kwargs)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.sn2 = single_step_neuron(**kwargs)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        out = self.sn2(out)

        return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, single_step_neuron: callable = None, **kwargs):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.sn1 = single_step_neuron(**kwargs)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.sn2 = single_step_neuron(**kwargs)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.sn3 = single_step_neuron(**kwargs)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sn2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        out = self.sn3(out)

        return out

class SpikingResNet(nn.Module):
    def __init__(self, block, layers, num_classes=200, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, single_step_neuron: callable = None, **kwargs):
        super(SpikingResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.sn1 = single_step_neuron(**kwargs)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], single_step_neuron=single_step_neuron, **kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], single_step_neuron=single_step_neuron,
                                       **kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], single_step_neuron=single_step_neuron,
                                       **kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], single_step_neuron=single_step_neuron,
                                       **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, single_step_neuron: callable = None, **kwargs):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, single_step_neuron, **kwargs))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, single_step_neuron=single_step_neuron, **kwargs))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

def _spiking_resnet(arch, block, layers, num_classes, pretrained, progress, norm_layer, single_step_neuron, **kwargs):
    model = SpikingResNet(block, layers, num_classes=num_classes, norm_layer=norm_layer, single_step_neuron=single_step_neuron, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def spiking_resnet18(
        pretrained=False, progress=True,
        norm_layer=None, single_step_neuron=None,
        num_classes=10, **kwargs
):

    return _spiking_resnet('resnet18',  # arch name
                           pretrained=pretrained, progress=progress,
                           block=BasicBlock, layers=[2, 2, 2, 2],
                           num_classes=num_classes, norm_layer=norm_layer,
                           single_step_neuron=single_step_neuron, **kwargs
                           )

def spiking_resnet34(
        pretrained=False, progress=True,
        norm_layer=None, single_step_neuron=None,
        num_classes=10, **kwargs
):

    return _spiking_resnet('resnet34',  # arch name
                           pretrained=pretrained, progress=progress,
                           block=BasicBlock, layers=[3, 4, 6, 3],
                           num_classes=num_classes, norm_layer=norm_layer,
                           single_step_neuron=single_step_neuron, **kwargs
                           )

# ===== Net, ECOC/ZBias hooks =====
if resnet_depth == 18:
    net = spiking_resnet18(
        pretrained=False, progress=True,
        norm_layer=TDBatchNorm,
        single_step_neuron=nn.ReLU,
        num_classes=num_classes,
    ).to(device)
elif resnet_depth == 34:
    net = spiking_resnet34(
        pretrained=False, progress=True,
        norm_layer=TDBatchNorm,
        single_step_neuron=nn.ReLU,
        num_classes=num_classes,
    ).to(device)
else:
    raise ValueError("Invalid resnet_depth. Choose 18 or 34.")

# ===== Z Bias =====
if ZBias_on:
    bias_adder = ZBiasAdder(
        base_bias=z_bias,
        apply_fraction=z_bias_ratio,        # 80% 뉴런만 base 적용
        outlier_bias=outlier_z_bias,
        outlier_fraction=z_bias_fraction,     # 그중 5%는 큰 값
        outlier_mode='override',        # base + outlier
        start_epoch=bias_start_epoch,
        target_patterns=bias_target_layer,
        apply_to_all=bias_apply_to_all,         # 전체 레이어 대상
        seed=7
    )
    bias_adder.attach(net, verbose=True)
else:
    bias_adder = None

# === Benchmark hooks ===
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
fault_mgr = None
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

if Fault_on and Routing_on:
    stuck_map = get_fault_map(fault_mgr, include_bias=False)
    router, pairs = install_router_from_mask(net, stuck_map, arch="resnet")
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
        arch="resnet",
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
        sharpness=None,
        hard_forward=True,
        hard_eval=True,
        auto_init=True,
        overlap=True,
        kernel_size=15,
        overlap_iter=3,
    ).to(device)

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
            data = batch_dynamic_fragments(data, num_steps, overlap=True, method="combo", weak_model=None,
                                           kernel_size=15, overlap_iter=3,  per_image=False, power_norm=power_cfg)
            output = []
            for step in range(num_steps):
                input = data[:, step]
                output.append(net(input.float()))
            output = torch.stack(output, dim=0)  # [B,T,C]

        elif Learnable_on:
            data = learnable_frags(data)  # [B, T, C, H, W]
            output = []
            for step in range(num_steps):
                input = data[:, step]
                output.append(net(input.float()))
            output = torch.stack(output, dim=0)
            logits_t = output

        elif Dynamic_on:
            data = dynamic_frags(data, output_mode="mix", sample_steps=True)  # [B, Tmax, C, H, W]
            num_steps = data.size(1)
            output = []
            for step in range(num_steps):
                input = data[:, step]
                output.append(net(input.float()))
            output = torch.stack(output, dim=0)

        else:
            output = 0
            for step in range(num_steps):
                output += net(data.float())
            output /= num_steps

        # RMSE (ecos same as simple_snn)
        if ECOC_on:
            loss_val = ecoc.loss_ce(output, targets, metric="euclidean", temp=1.0, squared=True)
            # loss_val = torch.sqrt(ecoc.loss_mse(output, targets) + 1e-6)
        elif Soft_on and epoch == 0:
            bounder.capture_snapshot(net)
            bounder.activate()
        elif Frag_on:
            loss_val = fragmentation_loss(output, targets, mode="ce")
        elif Learnable_on:
            loss_val = (fragmentation_loss(output, targets, mode="ce") + learnable_frags.aux_loss() +
                        learnable_frags.sep_loss() + learnable_frags.cross_loss())
        elif Dynamic_on:
            loss_val = (fragmentation_loss(output, targets, mode="ce") + dynamic_frags.aux_loss() +
                        dynamic_frags.sep_loss() + dynamic_frags.cross_loss())
        elif Fault_on and Astro_on:
            loss_val = loss_fn(output, target_onehot) + astro(epoch)
        elif Fault_on and Falvolt_on:
            loss_val = loss_fn(output, target_onehot) + falvolt(epoch)
        elif Fault_on and LIFA_on:
            loss_val = loss_fn(output, target_onehot) + lifa(epoch)
        else:
            loss_val = loss_fn(output, target_onehot)
            # loss_val = loss_fn(output, targets)


        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

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
            data = batch_dynamic_fragments(data, num_steps, overlap=True, method="combo", weak_model=None,
                                           kernel_size=15, overlap_iter=3, per_image=False, power_norm=power_cfg)
            test_output = []
            for step in range(num_steps):
                input = data[:, step]
                test_output.append(net(input.float()))
            test_output = torch.stack(test_output, dim=0)
            test_output = agg_conf_logits(test_output, tau=2.0, time_major=True)

        elif Learnable_on:
            data = learnable_frags(data)  # [B, T, C, H, W]
            test_output = []
            for step in range(num_steps):
                input = data[:, step]
                test_output.append(net(input.float()))
            test_output = torch.stack(test_output, dim=0)
            test_output = agg_conf_logits(test_output, tau=2.0, time_major=True)

        elif Dynamic_on:
            data = dynamic_frags(data, output_mode="selected", sample_steps=True)  # [B, Tmax, C, H, W]
            num_steps = data.size(1)
            test_output = []
            for step in range(num_steps):
                input = data[:, step]
                test_output.append(net(input.float()))
            test_output = torch.stack(test_output, dim=0)
            test_output = agg_conf_logits(test_output, tau=2.0, time_major=True)

        else:
            test_output = 0
            for step in range(num_steps):
                test_output += net(data.float())
            test_output /= num_steps

        test_pred = ecoc.decode(test_output, metric='euclidean') if ecoc is not None else test_output.argmax(1)
        total += targets.size(0)
        correct += (targets == test_pred).sum().item()
        w_targets = torch.cat((w_targets, targets), dim=0)
        w_predicted = torch.cat((w_predicted, test_pred), dim=0)


# === After the test loop ===
print(f"Total correctly classified test set images: {correct}/{total}")
print(f"Test Set Accuracy: {100 * correct / total:.2f}%")

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
