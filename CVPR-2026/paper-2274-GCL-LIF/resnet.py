import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import OnlineNeuron, LIFNeuron
from typing import Literal

def conv3x3(in_planes, out_planes, use_ternary=False, stride=1):
    if use_ternary is False:
        return BinaryConv(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)
    else:
        return TernaryConv(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)

def conv1x1(in_planes, out_planes, use_ternary=False, stride=1):
    if use_ternary is False:
        return BinaryConv(in_planes, out_planes, kernel_size=1, stride=stride, padding=0)
    else:
        return TernaryConv(in_planes, out_planes, kernel_size=1, stride=stride, padding=0)

    
class BinaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1):
        super(BinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.rand((self.shape)) * 0.001, requires_grad=True)
        self.test_time = False
        
    def forward(self, x):
        if self.test_time is False:
            real_weights = self.weight
            scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
            scaling_factor = scaling_factor.detach()
            binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
            cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
            binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
            y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)
            return y
        else:
            return F.conv2d(x, self.weight, stride=self.stride, padding=self.padding)


class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        if self.training:
            out_forward = torch.sign(x)
            mask1 = x < -1
            mask2 = x < 0
            mask3 = x < 1
            out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
            out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
            out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
            out = out_forward.detach() - out3.detach() + out3
            return out
        else:
            return torch.sign(x)


class TwnQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, clip_val):
        ctx.save_for_backward(input, clip_val)
        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])
        m = input.norm(p=1).div(input.nelement())
        thres = 0.7 * m
        pos = (input > thres).float()
        neg = (input < -thres).float()
        mask = (input.abs() > thres).float()
        alpha = (mask * input).abs().sum() / mask.sum()
        result = alpha * pos - alpha * neg

        return result

    @staticmethod
    def backward(ctx, grad_output):
        input, clip_val = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None

   
class TernaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1):
        super(TernaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.rand((self.shape)) * 0.001, requires_grad=True)
        self.test_time = False
        self.register_buffer("weight_clip_val", torch.tensor([-1., 1.]))
        self.quantizer = TwnQuantizer
        
    def forward(self, x):
        if self.test_time is False:  
            ternary_weights = self.quantizer.apply(self.weight, self.weight_clip_val)
            return F.conv2d(x, ternary_weights, stride=self.stride, padding=self.padding)
        else:
            return F.conv2d(x, self.weight, stride=self.stride, padding=self.padding)
        

class BasicBlock(nn.Module):

    def __init__(self, T, inplanes, planes, use_ternary=False, stride=1, downsample=None, use_eca=0, mem_bn=False, parallel_mode=False, mode: Literal['vanilla','compression']='vanilla', weight_mode: Literal['vanilla','compression']='compression', use_lif: Literal['vanilla','online','online-learnable','none']='none'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv3x3(inplanes, planes, use_ternary, stride) if weight_mode == 'compression' else nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.attn1 = ECAAttention() if use_eca > 0 else nn.Identity()
        self.conv2 = nn.Sequential(
            conv3x3(planes, planes, use_ternary) if weight_mode == 'compression' else nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )   
        self.attn2 = ECAAttention() if use_eca > 0 else nn.Identity()         
        self.downsample = downsample
        is_online = True if ('online' in use_lif) else False
        is_learnable = True if ('learnable' in use_lif) else False
        self.act1 = OnlineNeuron(T, mem_bn, inplanes, False, mode) if use_lif == 'none' else LIFNeuron(T, online_training=is_online, is_learnable=is_learnable)
        self.act2 = OnlineNeuron(T, mem_bn, planes, parallel_mode, mode) if use_lif == 'none' else LIFNeuron(T, online_training=is_online, is_learnable=is_learnable)
        self.inplanes = inplanes
        self.planes = planes
        self.use_eca = use_eca

    def forward(self, x):
        identity = x  
        if self.downsample is not None:
            identity = self.downsample(x)
        if self.use_eca < 2:
            out = self.attn1(self.conv1(self.act1(x)))
            out = self.attn2(self.conv2(self.act2(out))) + identity
        else:
            if self.inplanes != self.planes:
                out = self.attn1(self.conv1(self.act1(x)))
            else:
                out = self.attn1(x + self.conv1(self.act1(x)))
            out = self.attn2(out + self.conv2(self.act2(out))) + identity
        return out
                
        
class ECAAttention(nn.Module):

    def __init__(self, kernel_size=5):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.gap(x) 
        y = y.squeeze(-1).permute(0, 2, 1)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.permute(0, 2, 1).unsqueeze(-1)
        return x * y.expand_as(x)


class ResNet(nn.Module):
    def __init__(self, T, block, layers, use_ternary=False, num_classes=1000, use_dvs=False, use_resnet19=False, use_eca=0, mem_bn=False, parallel_mode=False, mode: Literal['vanilla','compression']='vanilla', weight_mode: Literal['vanilla','compression']='compression', use_lif: Literal['vanilla','online','online-learnable','none']='none'):
        super(ResNet, self).__init__()
        self.T = T
        self.use_dvs = use_dvs
        self.inplanes = 64
        self.use_ternary = use_ternary
        self.num_classes = num_classes
        self.use_eca = use_eca
        self.mem_bn = mem_bn
        self.parallel_mode = parallel_mode
        self.mode = mode
        self.weight_mode = weight_mode
        self.use_lif = use_lif
        self.skip_name = ['conv1']

        is_online = True if ('online' in self.use_lif) else False
        is_learnable = True if ('learnable' in self.use_lif) else False

        if num_classes == 1000:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(self.inplanes)
        elif self.use_dvs is True:
            self.conv1 = nn.Conv2d(2, self.inplanes, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.inplanes)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.inplanes)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        if use_resnet19 is True:
            self.layer1 = nn.Sequential()
            self.layer2 = self._make_layer(block, 128, layers[0], self.use_ternary, stride=2 if self.use_dvs is True else 1)
            self.layer3 = self._make_layer(block, 256, layers[1], self.use_ternary, stride=2)
            self.layer4 = self._make_layer(block, 512, layers[2], self.use_ternary, stride=2)
            self.fc = nn.Sequential(
                nn.Linear(512, 256),
                OnlineNeuron(T, mode=self.mode) if self.use_lif == 'none' else LIFNeuron(T, online_training=is_online, is_learnable=is_learnable),
                nn.Linear(256, num_classes)
            )
        else:
            self.layer1 = self._make_layer(block, 64, layers[0], self.use_ternary)
            self.layer2 = self._make_layer(block, 128, layers[1], self.use_ternary, stride=2)
            self.layer3 = self._make_layer(block, 256, layers[2], self.use_ternary, stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], self.use_ternary, stride=2)
            self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, planes, blocks, use_ternary, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, use_ternary, stride) if self.weight_mode == 'compression' else nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes)
            )

        layers = []
        layers.append(block(self.T, self.inplanes, planes, use_ternary, stride, downsample, self.use_eca, self.mem_bn, self.parallel_mode, self.mode, self.weight_mode, self.use_lif))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.T, self.inplanes, planes, use_ternary, 1, None, self.use_eca, self.mem_bn, self.parallel_mode, self.mode, self.weight_mode, self.use_lif))
        return nn.Sequential(*layers)

    def snn_forward_impl(self, x):
        x = self.bn1(self.conv1(x))
        if self.use_dvs is False:
            if self.num_classes >= 200:
                x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

        
    def forward(self, x):
        return self.snn_forward_impl(x)


def _resnet(T, block, layers, **kwargs):
    model = ResNet(T, block, layers, **kwargs)
    return model

def resnet18(T, **kwargs):
    return _resnet(T, BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet19(T, **kwargs):
    return _resnet(T, BasicBlock, [3, 3, 2], **kwargs)

def resnet34(T, **kwargs):
    return _resnet(T, BasicBlock, [3, 4, 6, 3], **kwargs)