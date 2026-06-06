import copy
from scipy.stats import special_ortho_group
from functional import *

class DepthwiseSeparableConv(torch.nn.Sequential):
    def __init__(self, chin, chout, dk, stride, padding,padding_mode='reflect', bias=False):
        super().__init__(
            # Depthwise convolution
            torch.nn.Conv2d(chin, chin, kernel_size=dk, stride=stride, padding=padding, bias=False, groups=chin, padding_mode=padding_mode),
            # Pointwise convolution
            torch.nn.Conv2d(chin, chout, kernel_size=1, bias=bias),
        )
        

class ChannelProjector(torch.nn.Module):
    def __init__(self, in_dims: int, out_dims: int, trainnable=False):
        super(ChannelProjector, self).__init__()
        if trainnable:
            orth_mat = torch.nn.Parameter(torch.randn(out_dims, in_dims))
            self.register_parameter('orth_mat', orth_mat)
        else:
            if in_dims >= out_dims:
                flag = True
                dim1 = in_dims
                dim2 = out_dims
            else:   
                dim1 = out_dims
                dim2 = in_dims
                flag = False
            orth_mat = special_ortho_group.rvs(dim1)[:, :dim2]
            orth_mat = torch.from_numpy(orth_mat).float()
            if flag:
                orth_mat = torch.transpose(orth_mat, 0, 1)
            self.register_buffer('orth_mat',  orth_mat.clone().detach())
    
    def forward(self, x):
        x = torch.permute(x, (0, 2, 3, 1)) # (B, H, W, C)
        x = torch.matmul(self.orth_mat, x.unsqueeze(-1)).squeeze(-1)
        x = torch.permute(x, (0, 3, 1, 2)) # (B, C, H, W)
        return x
    
    
class EnergyConv2dEnsemble(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride=1, padding=1, padding_mode="reflect", conv_type='standard', pooling='MaxPool2d', pooling_config={'kernel_size': 2, 'stride': 2},  local_grad=True, pre_backward=False, sampling_len=20, dropout=0.20, consistency_mode='feature', diversity_mode='feature', consistency_factor=0.5, diversity_factor=0.5, num_classes=10, projecting=False, projecting_dim=None, is_last=False, is_first=False, inference_mode='sampling', transform=None, use_bn=True, act_first=True, conv_bias=False, consistent_dropout=False, projector_training=False):
    

        super(EnergyConv2dEnsemble, self).__init__()
        if sampling_len < 1:
            assert consistency_mode == 'supervised', "If sampling_len < 1, the consistency mode must be supervised"
        
        if conv_type == 'standard':
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, bias=conv_bias)
        elif conv_type == 'depthwise':
            self.conv = DepthwiseSeparableConv(in_channels, out_channels, kernel_size, stride, padding, padding_mode=padding_mode)
        else:
            raise ValueError(f"conv_type {conv_type} is not supported. Use 'standard' or 'depthwise'.")
        
        
        self.bn = torch.nn.BatchNorm2d(in_channels, affine=False) if use_bn else torch.nn.Identity()
        self.pool = getattr(torch.nn, pooling)(**pooling_config)
        self.act = torch.nn.ReLU()
        
        if dropout is not None:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = torch.nn.Identity()      
        
        if projecting:
            if projecting_dim is None:
                projecting_dim = num_classes
            self.projector = ChannelProjector(in_dims=out_channels, out_dims=projecting_dim, trainnable=projector_training)
        else:
            self.projector = torch.nn.Identity()
        self.transform = transform
        
        self.local_grad = local_grad # Used to indicate whether to use local gradient or not (BP)
        self.pre_backward = pre_backward
        self.sampling_len = sampling_len
        self.consistency_mode = consistency_mode
        self.diversity_mode = diversity_mode
        self.consistency_factor = consistency_factor
        self.diversity_factor = diversity_factor
        self.num_classes = num_classes
        self.is_last = is_last
        self.is_first = is_first
        self.inference_mode = inference_mode
        self.act_first = act_first
        self.consistent_dropout = consistent_dropout
    
    def standard_forward(self, x: torch.Tensor):
        x = self.conv(x)
        if self.act_first:
            x = self.act(x)
            x = self.pool(x)
        else:
            x = self.pool(x)
            x = self.act(x)
        return x
    
    def generate_self_samples(self, x: torch.Tensor):
        x = x.unsqueeze(0).expand(self.sampling_len, -1, -1, -1, -1)
        x = x.reshape(-1, *x.shape[2:])
        return x
        
    def forward(self, x: torch.Tensor, label=None):
        x = self.bn(x)
        if self.local_grad:
            if self.is_first: 
                if not isinstance(self.dropout, torch.nn.Identity) and self.sampling_len > 1:
                    x = self.generate_self_samples(x)
                    self.dropout.train()
                    x = self.dropout(x)
            else:
                if not isinstance(self.dropout, torch.nn.Identity) and self.consistent_dropout:
                    #self.dropout.train()
                    x = self.dropout(x)
            x = self.standard_forward(x)
            if self.training:
                loss = local_loss(self, self.projector(x), label=label)
                if self.pre_backward:
                    loss.backward()
            x = x.detach()
            
            if self.training:
                return x, loss
            else:
                return x
        else:
            return self.standard_forward(x)

class EnergyCNN(torch.nn.Module):
    def __init__(self, conv_config: list, linear_drop=0.5, linear_dims=24576, num_classes=10, sampling_len=20, local_grad=True, **kwargs):
        super(EnergyCNN, self).__init__()
        self.energy_blocks = torch.nn.ModuleList()
        for i, config in enumerate(conv_config):
            conv = EnergyConv2dEnsemble(**config)
            self.energy_blocks.append(conv)
        
        flatten = torch.nn.Flatten()
        fc1 = torch.nn.Linear(linear_dims, num_classes)
        self.classifier = torch.nn.Sequential(*[flatten, torch.nn.Dropout(linear_drop) if linear_drop is not None else torch.nn.Identity(), fc1])
        
        self.sampling_len = sampling_len
        self.num_classes = num_classes
        self.local_grad = local_grad
        
        for key, value in kwargs.items():
            setattr(self, key, value)
        
    def forward(self, x: torch.Tensor, label=None):
        if self.local_grad:
            loss = 0
            for block in self.energy_blocks:
                x = block(x, label=label)
                if isinstance(x, tuple):
                    x, loss1 = x
                    loss += loss1
            x = self.classifier(x)
            if self.training:
                if self.sampling_len > 1:
                    x = x.reshape(self.sampling_len, -1, *x.shape[1:])
                    x = torch.einsum('ijk, ijk ->jk', x, x) / (self.sampling_len -1) # E[x^2]
                return x, loss
            else:
                if self.sampling_len > 1 and getattr(self.energy_blocks[-1], 'inference_mode', 'usual') == 'sampling':
                    x = x.reshape(self.sampling_len, -1, *x.shape[1:])
                    x = torch.einsum('ijk, ijk ->jk', x, x) / (self.sampling_len -1) # E[x^2]
                elif self.sampling_len > 1 and getattr(self.energy_blocks[-1], 'inference_mode', 'usual') == 'average':
                    x = x.reshape(self.sampling_len, -1, *x.shape[1:])
                    x = torch.mean(x, dim=0)
                return x
        else:
            for block in self.energy_blocks:
                x = block(x)
            x = self.classifier(x)
            return x


class AuxiliaryEnergyModel(torch.nn.Module):
    def __init__(self, trained_model: torch.nn.ModuleList, next_model, **kwargs):
        super(AuxiliaryEnergyModel, self).__init__()
            
        if len(trained_model) == 0:
            self.trained_model = torch.nn.ModuleList().append(torch.nn.Identity())
        else:
            self.trained_model = trained_model
        self.next_model = next_model
        for param in self.trained_model.parameters():
            param.requires_grad = False
        self.trained_model.eval()
        
        for key, value in kwargs.items():
            setattr(self, key, value)
        
    def forward(self, x, y=None):
        self.trained_model.eval()
        for conv in self.trained_model:
            x = conv(x)
        if getattr(self.next_model, 'local_grad', False):
            x, loss = self.next_model(x, y)
            return x, loss
        else:
            x = self.next_model(x)
            if self.trained_model[-1].is_last:
                if self.trained_model[-1].sampling_len > 1 and self.trained_model[-1].inference_mode == 'sampling':
                    x = x.reshape(self.sampling_len, -1, *x.shape[1:])
                    x = torch.einsum('ijk, ijk ->jk', x, x) / (self.sampling_len -1)
                elif self.sampling_len > 1 and self.trained_model[-1].inference_mode == 'average':
                    x = x.reshape(self.sampling_len, -1, *x.shape[1:])
                    x = torch.mean(x, dim=0) 
            return x