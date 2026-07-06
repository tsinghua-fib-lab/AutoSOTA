from torch import nn
from collections import OrderedDict
from enum import IntEnum
import torch
from torch.nn import functional as F

def power_series_matrix_logarithm_trace(Fx, x, k, n):
    """
    Fast-boi Tr(Ln(d(Fx)/dx)) using power-series approximation
    biased but fast
    :param Fx: output of f(x)
    :param x: input
    :param k: number of power-series terms  to use
    :param n: number of Hitchinson's estimator samples
    :return: Tr(Ln(I + df/dx))
    """
    # trace estimation including power series
    #print(Fx.shape)
    outSum = Fx.sum(dim=0)
    dim = list(outSum.shape)
    dim.insert(0, n)
    dim.insert(0, x.size(0))
    u = torch.randn(dim).to(x.device)
   # print(u.shape)
    trLn = 0
    for j in range(1, k + 1):
        if j == 1:
            vectors = u
        # compute vector-jacobian product
        vectors = [torch.autograd.grad(Fx, x, grad_outputs=vectors[:, i],
                                       retain_graph=True, create_graph=True)[0] for i in range(n)]
        #print(vectors[0].shape)
        # compute summand
        #print(torch.stack(vectors).shape)
        vectors = torch.stack(vectors, dim=1)
        #
        # print(vectors.shape)
        vjp4D = vectors.view(x.size(0), n, 1, -1)
        u4D = u.view(x.size(0), n, -1, 1)
        summand = torch.matmul(vjp4D, u4D)
        # add summand to power series
        if (j + 1) % 2 == 0:
            trLn += summand / float(j)
        else:
            trLn -= summand / float(j)
    trace = trLn.mean(dim=1).squeeze()
    return trace,vectors
def compute_log_det(inputs, outputs):
    #print(outputs.shape)
    batch_size = outputs.size(0)
    outVector = torch.sum(outputs,0).view(-1)
    outdim = outVector.size()[0]
    #print(outVector.shape)
    jac = [torch.autograd.grad(outVector[i], inputs,
                                     retain_graph=True, create_graph=True)[0].view(batch_size, outdim) for i in range(outdim)]
    jac=torch.stack(jac,dim=1)
    #print(jac)
    log_det = torch.log(abs(torch.stack([torch.det(jac[i]) for i in range(batch_size)], dim=0)))
    #print(log_det.mean())
    #print(jac[0]==jac[1])
    return log_det, jac
def power_series_full_jac_exact_trace(Fx, x, k):
    """
    Fast-boi Tr(Ln(d(Fx)/dx)) using power-series approximation with full
    jacobian and exact trace
    
    :param Fx: output of f(x)
    :param x: input
    :param k: number of power-series terms  to use
    :return: Tr(Ln(I + df/dx))
    """
    _, jac = compute_log_det(x, Fx)
    jacPower = jac
    summand = torch.zeros_like(jacPower)
    for i in range(1, k+1):
        if i > 1:
            jacPower = torch.matmul(jacPower, jac)
        if (i + 1) % 2 == 0:
            summand += jacPower / (float(i))
        else: 
            summand -= jacPower / (float(i)) 
    trace = torch.diagonal(summand, dim1=1, dim2=2).sum(1)
    return trace,jac
class Residual(nn.Module):  #@save
    def __init__(self, input_channels, mid_channels,
                 use_1x1conv=False, strides=1,act=1,kernel_size=5,padding=2):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, mid_channels,
                               kernel_size=kernel_size, padding=padding, stride=strides,bias=False)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels*2,
                               kernel_size=kernel_size, padding=padding,bias=False)
        self.conv3 = nn.Conv2d(mid_channels*2, mid_channels*4,
                               kernel_size=kernel_size, padding=padding, stride=strides,bias=False)
        self.conv4 = nn.Conv2d(mid_channels*4, input_channels,
                               kernel_size=kernel_size, padding=padding,bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels,eps=1e-4)
        self.bn2 = nn.BatchNorm2d(input_channels,eps=1e-4)
        self.bn3 = nn.BatchNorm2d(mid_channels*2,eps=1e-4)
        self.bn4 = nn.BatchNorm2d(input_channels,eps=1e-4)
        if act==1:
            self.activ=nn.ReLU(True)
        if act==2:
            self.activ=nn.LeakyReLU(True)
        if act==3:
            self.activ=nn.ELU()
        self.mu=nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.sigma=nn.Parameter(torch.tensor(0.5), requires_grad=False)
        


    def forward(self, X):
        Y = self.activ((self.conv1(X)))
        Y = (self.conv2(Y))
        Y=self.activ(Y)
        #Y =self.activ(X+Y)
        Y = self.activ((self.conv3(Y)))
        Y = (self.conv4(Y))
        return Y
    
    from torch import nn
class AENET(nn.Module):  #@save
    def __init__(self, input_channels, mid_channels,
                 use_1x1conv=False, strides=1,act=1,kernel_size=5,padding=2):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, mid_channels,
                               kernel_size=kernel_size, padding=padding, stride=strides)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels*2,
                               kernel_size=kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(mid_channels*2, mid_channels*4,
                               kernel_size=kernel_size, padding=padding, stride=strides)
        self.conv4 = nn.Conv2d(mid_channels*4, input_channels,
                               kernel_size=kernel_size, padding=padding)
        self.deconv1= nn.Conv2d( input_channels,mid_channels,
                               kernel_size=kernel_size, padding=padding)
        self.deconv2= nn.Conv2d(mid_channels,mid_channels*2,
                               kernel_size=kernel_size, padding=padding)
        self.deconv3= nn.Conv2d(mid_channels*2,mid_channels*4,
                               kernel_size=kernel_size, padding=padding)
        self.deconv4= nn.Conv2d(mid_channels*4,input_channels,
                               kernel_size=kernel_size, padding=padding)
        
        
        self.bn1 = nn.BatchNorm2d(mid_channels,eps=1e-4)
        self.bn2 = nn.BatchNorm2d(input_channels,eps=1e-4)
        self.bn3 = nn.BatchNorm2d(mid_channels*2,eps=1e-4)
        self.bn4 = nn.BatchNorm2d(input_channels,eps=1e-4)
        if act==1:
            self.activ=nn.ReLU(True)
        if act==2:
            self.activ=nn.LeakyReLU(True)
        if act==3:
            self.activ=nn.ELU()

    def forward(self, X):
        Y = self.activ((self.conv1(X)))
        Y = (self.conv2(Y))
        Y=self.activ(Y)
        #Y =self.activ(X+Y)
        Y = self.activ((self.conv3(Y)))
        Y = (self.conv4(Y))
        return X+Y    
    def decode(self,X):
        Y = self.activ((self.conv1(X)))
        Y = (self.conv2(Y))
        Y=self.activ(Y)
        #Y =self.activ(X+Y)
        Y = self.activ((self.conv3(Y)))
        Y = self.activ((self.conv4(Y))+X)
        Y = self.activ((self.deconv1(Y)))
        Y = (self.deconv2(Y))
        Y=self.activ(Y)
        #Y =self.activ(X+Y)
        Y = self.activ((self.deconv3(Y)))
        Y = (self.deconv4(Y))
        return Y

class TwoLayer(nn.Module):
    def __init__(self,
                 input_dim=2,
                 num_classes=1, 
                 num_hidden_nodes=20,
                 act=1,
                 alpha=0.01,
                 bia=False
    ):

        super(TwoLayer, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_hidden_nodes = num_hidden_nodes
        
        if act==1:
            activ=nn.ReLU(True)
        if act==2:
            activ=nn.LeakyReLU(alpha,True)
        if act==3:
            activ=nn.ELU()

        self.feature_extractor = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(self.input_dim, self.num_hidden_nodes,bias=bia)),
            ('relu1', activ)]))
        self.size_final = self.num_hidden_nodes
        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.size_final, self.num_classes,bias=bia))]))

    def forward(self, input):
        features = self.feature_extractor(input)
        logits = self.classifier(features.view(-1, self.size_final))
        return logits
        
    def half_forward_start(self, input):
        return self.feature_extractor(input)

    def half_forward_end(self, input):
        return self.classifier(input.view(-1, self.size_final))



class FourLayer(nn.Module):
    def __init__(self,
                 input_dim=2,
                 num_classes=1, 
                 num_hidden_nodes=2,
                 act=1,
                 bia=False
    ):

        super(FourLayer, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_hidden_nodes = num_hidden_nodes
        if act==1:
            activ=nn.ReLU(True)
        if act==2:
            activ=nn.LeakyReLU(True)
        if act==3:
            activ=nn.ELU()
        if act==4:
            activ=nn.Sigmoid()
        if act==5:
            activ=nn.Tanh()

        self.layer1 = nn.Linear(self.input_dim, self.num_hidden_nodes,bias=bia)
        #self.bn1 = nn.BatchNorm1d(self.num_hidden_nodes)
        self.elu1 = activ
        #self.dropout1 = nn.Dropout(0.5)
        self.layer2 = nn.Linear(self.num_hidden_nodes, int(self.num_hidden_nodes),bias=bia)
        #self.bn2 = nn.BatchNorm1d(int(self.num_hidden_nodess))
        self.elu2 = activ
        self.dropout2 = nn.Dropout(0.5)
        #self.layer3 = nn.Linear(int(self.num_hidden_nodes/2), int(self.num_hidden_nodes/4),bias=bia)
        #self.bn3 = nn.BatchNorm1d(int(self.num_hidden_nodes))
        self.elu3 = activ
        self.layer3 = nn.Linear(int(self.num_hidden_nodes),int(self.num_hidden_nodes),bias=bia)
        #self.dropout3 = nn.Dropout(0.1)
        self.layer4 = nn.Linear(int(self.num_hidden_nodes),self.num_classes,bias=bia)
        #nn.init.orthogonal_(self.layer1.weight)
        #nn.init.orthogonal_(self.layer2.weight)
        #nn.init.orthogonal_(self.layer3.weight)
        #nn.init.orthogonal_(self.layer4.weight)

    def forward(self, input):
        #self.apply_spectral_norm()
        out=input
        #if self.head==False:
        #   out=self.elu1(out)
        out = self.layer1(out)
        #out = self.bn1(out)
        out = self.elu1(out)
        #out = self.dropout1(out)
        out = self.layer2(out)
        #out = self.bn2(out)
        out = self.elu2(out)
        #out = self.dropout2(out)
        out = self.layer3(out)
        #out = self.bn3(out)
        out = self.elu3(out)
        #out = self.dropout3(out)
        out = self.layer4(out)
        return out
    
    def apply_spectral_norm(self):
        """
        限制每一层的权重谱范数。如果某一层的谱范数大于 1,则对其权重进行归一化。
        """
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            if isinstance(layer, nn.Linear):
                weight = layer.weight.data  # 获取权重
                #print(weight.shape)
                # 计算谱范数（最大奇异值）
                spectral_norm = torch.linalg.norm(weight, ord=2)
                if spectral_norm > 1:
                    # 对权重进行归一化
                    layer.weight.data = weight / spectral_norm *0.9
class FourLayer_IRN(nn.Module):
    def __init__(self,
                 input_dim=2,
                 num_classes=1, 
                 num_hidden_nodes=2,
                 act=1,
                 bia=False,
                 k=5,
                 n=1,
                 head=False
    ):

        super(FourLayer, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_hidden_nodes = num_hidden_nodes
        self.k=k
        self.head=head
        self.n=n
        if act==1:
            activ=nn.ReLU(True)
        if act==2:
            activ=nn.LeakyReLU(True)
        if act==3:
            activ=nn.ELU()
        if act==4:
            activ=nn.Sigmoid()
        if act==5:
            activ=nn.Tanh()

        self.layer1 = nn.Linear(self.input_dim, self.num_hidden_nodes,bias=bia)
        #self.bn1 = nn.BatchNorm1d(self.num_hidden_nodes)
        self.elu1 = activ
        #self.dropout1 = nn.Dropout(0.5)
        self.layer2 = nn.Linear(self.num_hidden_nodes, int(self.num_hidden_nodes),bias=bia)
        #self.bn2 = nn.BatchNorm1d(int(self.num_hidden_nodess))
        self.elu2 = activ
        self.dropout2 = nn.Dropout(0.5)
        #self.layer3 = nn.Linear(int(self.num_hidden_nodes/2), int(self.num_hidden_nodes/4),bias=bia)
        #self.bn3 = nn.BatchNorm1d(int(self.num_hidden_nodes))
        self.elu3 = activ
        self.layer3 = nn.Linear(int(self.num_hidden_nodes),self.num_hidden_nodes,bias=bia)
        #self.dropout3 = nn.Dropout(0.1)
        self.layer4 = nn.Linear(int(self.num_hidden_nodes),self.num_classes,bias=bia)

    def forward(self, input,sldj=None):
        self.apply_spectral_norm()
        out=input
        #if self.head==False:
        #   out=self.elu1(out)
        out = self.layer1(out)
        #out = self.bn1(out)
        out = self.elu1(out)
        #out = self.dropout1(out)
        out = self.layer2(out)
        #out = self.bn2(out)
        out = self.elu2(out)
        #out = self.dropout2(out)
        out = self.layer3(out)
        #out = self.bn3(out)
        out = self.elu3(out)
        #out = self.dropout3(out)
        out = self.layer4(out)
        #trace,_=power_series_matrix_logarithm_trace(out,input,self.k,self.n)
        trace,_=power_series_full_jac_exact_trace(out,input,10)
        sldj+=trace
        return out,sldj
    
    def apply_spectral_norm(self):
        """
        限制每一层的权重谱范数。如果某一层的谱范数大于 1,则对其权重进行归一化。
        """
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            if isinstance(layer, nn.Linear):
                weight = layer.weight.data  # 获取权重
                #print(weight.shape)
                # 计算谱范数（最大奇异值）
                spectral_norm = torch.linalg.norm(weight, ord=2)
                if spectral_norm > 1:
                    # 对权重进行归一化
                    layer.weight.data = weight / spectral_norm *0.9


class _IRN(nn.Module):
   
    def __init__(self, input_dim, mid_dim,act=1,bia=False,k=5,n=1):
        super(_IRN, self).__init__()
        self.input_dim=input_dim
        self.mid_dim=mid_dim
        self.act=act
        self.bia=bia
        self.k=k
        self.n=n
        self.in_couplings = nn.ModuleList([
            FourLayer(self.input_dim,self.input_dim,self.mid_dim,self.act,self.bia,self.k,self.n,True),
            FourLayer(self.input_dim,self.input_dim,self.mid_dim,self.act,self.bia,self.k,self.n),
            FourLayer(self.input_dim,self.input_dim,self.mid_dim,self.act,self.bia,self.k,self.n),
            FourLayer(self.input_dim,self.input_dim,self.mid_dim,self.act,self.bia,self.k,self.n),
            FourLayer(self.input_dim,self.input_dim,self.mid_dim,self.act,self.bia,self.k,self.n),
            FourLayer(self.input_dim,self.input_dim,self.mid_dim,self.act,self.bia,self.k,self.n),
            FourLayer(self.input_dim,self.input_dim,self.mid_dim,self.act,self.bia,self.k,self.n),
            FourLayer(self.input_dim,self.input_dim,self.mid_dim,self.act,self.bia,self.k,self.n),
            FourLayer(self.input_dim,self.input_dim,self.mid_dim,self.act,self.bia,self.k,self.n),
            FourLayer(self.input_dim,self.input_dim,self.mid_dim,self.act,self.bia,self.k,self.n),
            FourLayer(self.input_dim,self.input_dim,self.mid_dim,self.act,self.bia,self.k,self.n),
            FourLayer(self.input_dim,self.input_dim,self.mid_dim,self.act,self.bia,self.k,self.n),
            FourLayer(self.input_dim,self.input_dim,self.mid_dim,self.act,self.bia,self.k,self.n),
            FourLayer(self.input_dim,self.input_dim,self.mid_dim,self.act,self.bia,self.k,self.n),
            FourLayer(self.input_dim,self.input_dim,self.mid_dim,self.act,self.bia,self.k,self.n),
            FourLayer(self.input_dim,self.input_dim,self.mid_dim,self.act,self.bia,self.k,self.n)
        ])

    def forward(self, x, sldj, reverse=False):
            #print(x.shape)
            #print(x.shape)
            for coupling in self.in_couplings:
                y, sldj = coupling(x, sldj)
                x=x+y
            #if not self.is_last_block:
            #    # Squeeze -> 3x coupling (channel-wise)
            #    x = squeeze_2x2(x, reverse=False)
            #        x, sldj = coupling(x, sldj, reverse)
            #    print(x.shape)
            #    x = squeeze_2x2(x, reverse=True)
            #    print(x.shape)

                # Re-squeeze -> split -> next block
            #    x = squeeze_2x2(x, reverse=False, alt_order=True)
            #    x, x_split = x.chunk(2, dim=1)
            #    x, sldj = self.next_block(x, sldj, reverse)
            #    x = torch.cat((x, x_split), dim=1)
            #    x = squeeze_2x2(x, reverse=True, alt_order=True)

            return x, sldj

class SixLayer(nn.Module):
    def __init__(self,
                 input_dim=2,
                 num_classes=1, 
                 num_hidden_nodes=20,
                 act=1,
                 alpha=0.01,
                 bia=False
    ):

        super(SixLayer, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_hidden_nodes = num_hidden_nodes
        
        if act==1:
            activ=nn.ReLU(True)
        if act==2:
            activ=nn.LeakyReLU(alpha,True)
        if act==3:
            activ=nn.ELU()

        self.feature_extractor = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.input_dim, self.num_hidden_nodes,bias=bia)),
            ('relu1', activ),
            ('fc2', nn.Linear(self.num_hidden_nodes, int(self.num_hidden_nodes/2),bias=bia)),
            ('relu2', activ),
            ('fc3', nn.Linear(int(self.num_hidden_nodes/2), int(self.num_hidden_nodes/4),bias=bia)),
            ('relu3', activ),
            ]))
        self.size_final = int(self.num_hidden_nodes/4)

        self.classifier = nn.Sequential(OrderedDict([
            ('fc4', nn.Linear(int(self.num_hidden_nodes/4), int(self.num_hidden_nodes/8),bias=bia)),
            ('relu4', activ),
            ('fc5', nn.Linear(int(self.num_hidden_nodes/8), int(self.num_hidden_nodes/16),bias=bia)),
            ('relu5', activ),
            ('fc6', nn.Linear(int(self.num_hidden_nodes/16), self.num_classes,bias=bia))]))
        # self.lamda = nn.Parameter(0 * torch.ones([1, 1]))
        # self.inp_lamda = nn.Parameter(0 * torch.ones([1, 1]))


    def forward(self, input):
        features = self.feature_extractor(input)
        logits = self.classifier(features.view(-1, self.size_final))
        return logits
        
    def half_forward_start(self, input):
        return self.feature_extractor(input)

    def half_forward_end(self, input):
        return self.classifier(input.view(-1, self.size_final))
    

class ResidualBlock(nn.Module):
        def __init__(self, in_dim,out_dim,latent_dim,act,bia):
         super(ResidualBlock, self).__init__()
        #self.in_norm = nn.BatchNorm2d(in_channels)
         self.fc1= nn.Linear(in_dim,latent_dim,bia)
         self.bn1=nn.BatchNorm1d(latent_dim)
         #self.fc2=nn.Linear(latent_dim,out_dim,bia)
         self.fc2=nn.Linear(latent_dim,int(latent_dim),bia)
         self.bn2=nn.BatchNorm1d(int(latent_dim))
         self.fc3=nn.Linear(int(latent_dim),int(latent_dim),bia)
         self.bn3=nn.BatchNorm1d(int(latent_dim))
         self.fc4=nn.Linear(int(latent_dim),out_dim,bia)
         self.bn4=nn.BatchNorm1d(out_dim)
         #self.fc2=nn.Linear(latent_dim,in_dim)
         if act==1:
            self.activ=nn.ReLU(True)
         if act==2:
            self.activ=nn.LeakyReLU(True)
         if act==3:
            self.activ=nn.ELU()
         if act==4:
             self.activ=nn.Sigmoid()
         if act==5:
             self.activ=nn.Tanh()

         #self.out_norm = nn.BatchNorm2d(out_channels)
         #self.out_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)

        def forward(self, x):
         x = torch.cat((x, -x), dim=1)
         #res=x
         x=(self.activ(self.fc1(x)))
         #x=(self.activ(self.fc2(x)))
         #x=(self.activ(self.fc3(x)))
         x=self.fc4(x)
         #x=self.fc4(x)
         return x

class MaskType(IntEnum):
    CHECKERBOARD = 0
    CHANNEL_WISE = 1
    QUARTER=2


def checkerboard_mask(input_dim, reverse=False, dtype=torch.float32,
                      device=None, requires_grad=False):
    """Get a checkerboard mask, such that no two entries adjacent entries
    have the same value. In non-reversed mask, top-left entry is 0.

    Args:
        height (int): Number of rows in the mask.
        width (int): Number of columns in the mask.
        reverse (bool): If True, reverse the mask (i.e., make top-left entry 1).
            Useful for alternating masks in RealNVP.
        dtype (torch.dtype): Data type of the tensor.
        device (torch.device): Device on which to construct the tensor.
        requires_grad (bool): Whether the tensor requires gradient.


    Returns:
        mask (torch.tensor): Checkerboard mask of shape (1, 1, height, width).
    """
    reverse=reverse%2
    checkerboard = [i % 2 for i in range(input_dim)]
    mask = torch.tensor(checkerboard, dtype=dtype, device=device, requires_grad=requires_grad)

    if reverse==1:
        mask = 1 - mask

    # Reshape to (1, 1, height, width) for broadcasting with tensors of shape (B, C, H, W)
    mask = mask.view(1,input_dim)
    return mask


def one_of_four_mask(input_dim, reverse=0, dtype=torch.float32,
                      device=None, requires_grad=False):
    """Get a checkerboard mask, such that no two entries adjacent entries
    have the same value. In non-reversed mask, top-left entry is 0.

    Args:
        height (int): Number of rows in the mask.
        width (int): Number of columns in the mask.
        reverse (bool): If True, reverse the mask (i.e., make top-left entry 1).
            Useful for alternating masks in RealNVP.
        dtype (torch.dtype): Data type of the tensor.
        device (torch.device): Device on which to construct the tensor.
        requires_grad (bool): Whether the tensor requires gradient.


    Returns:
        mask (torch.tensor): Checkerboard mask of shape (1, 1, height, width).
    """
    tensor = torch.ones(input_dim)
    num_zeros = input_dim // 4
    if reverse==0:
      tensor.view(-1)[:num_zeros] = 0
    if reverse==1:
        tensor.view(-1)[num_zeros:num_zeros*2] = 0
    if reverse==2:
        tensor.view(-1)[num_zeros*2:num_zeros*3] = 0
    if reverse==2:
        tensor.view(-1)[num_zeros*3:] = 0
    mask = tensor.view(1,input_dim)
    return mask.to('cuda')


class CouplingLayer(nn.Module):
    """Coupling layer in RealNVP.

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the `s` and `t` network.
        num_blocks (int): Number of residual blocks in the `s` and `t` network.
        mask_type (MaskType): One of `MaskType.CHECKERBOARD` or `MaskType.CHANNEL_WISE`.
        reverse_mask (bool): Whether to reverse the mask. Useful for alternating masks.
    """
    def __init__(self, input_dim, mid_dim, mask_type, reverse_mask,act):
        super(CouplingLayer, self).__init__()

        # Save mask info
        self.mask_type = mask_type
        self.reverse_mask = reverse_mask
        self.input_dim=input_dim
        self.mid_dim=mid_dim
        # Build scale and translate network
        self.st_net = ResidualBlock(self.input_dim*2, self.input_dim*2,self.mid_dim,act=act,bia=False)
        self.rescale = nn.utils.weight_norm(Rescale(self.input_dim))

    def forward(self, x, sldj=None):
            if self.mask_type == MaskType.CHECKERBOARD:
                b = checkerboard_mask(self.input_dim, self.reverse_mask, device=x.device)
            if self.mask_type==MaskType.QUARTER:
                b=one_of_four_mask(self.input_dim, self.reverse_mask, device=x.device)
            x_b = x * b
            st = self.st_net(x_b)
            s, t = st.chunk(2, dim=1)
            s=torch.tanh(s)
            s=self.rescale(s)
            s = s * (1 - b)
            t = t * (1 - b)
            #print(s.shape)
            exp_s = s.exp()
            x = x* exp_s+t
            sldj += s.reshape(s.size(0), -1).sum(-1)
            return x, sldj
    def reverse(self, x, sldj=None):
        # x here is y in math
        if self.mask_type == MaskType.CHECKERBOARD:
            b = checkerboard_mask(self.input_dim, self.reverse_mask, device=x.device)
        elif self.mask_type == MaskType.QUARTER:
            b = one_of_four_mask(self.input_dim, self.reverse_mask, device=x.device)
        else:
            raise ValueError("Unsupported mask type")

        x_b = x * b
        st = self.st_net(x_b)
        s, t = st.chunk(2, dim=1)

        s = torch.tanh(s)
        s = self.rescale(s)

        s = s * (1 - b)
        t = t * (1 - b)

        exp_neg_s = (-s).exp()
        x = (x - t) * exp_neg_s

        if sldj is not None:
            sldj = sldj - s.reshape(s.size(0), -1).sum(-1)

        return x, sldj
    
class Rescale(nn.Module):
    """Per-channel rescaling. Need a proper `nn.Module` so we can wrap it
    with `torch.nn.utils.weight_norm`.

    Args:
        num_channels (int): Number of channels in the input.
    """
    def __init__(self, input_dim):
        super(Rescale, self).__init__()
        self.weight = nn.Parameter(torch.ones(input_dim))

    def forward(self, x):
        x = self.weight *x
        return x

class _RealNVP(nn.Module):
   
    def __init__(self, input_dim, mid_dim,masktype,act=1,mu=0,sigma=1,learn=False):
        super(_RealNVP, self).__init__()
        self.input_dim=input_dim
        self.mid_dim=mid_dim
        self.mu=nn.Parameter(mu, requires_grad=learn)
        self.sigma=nn.Parameter(sigma, requires_grad=learn)
        self.alpha=nn.Parameter(torch.rand(sigma.shape),requires_grad=learn)
        if sigma.shape==1:
            self.alpha=nn.Parameter(torch.ones(sigma.shape),requires_grad=False)
        #temp=torch.rand(sigma.shape)
        #self.alpha=nn.Parameter(torch.exp(temp)/torch.sum(torch.exp(temp)),requires_grad=True)

        self.in_couplings = nn.ModuleList([
            CouplingLayer(self.input_dim,self.mid_dim, masktype, reverse_mask=0,act=act),
            CouplingLayer(self.input_dim,self.mid_dim,masktype, reverse_mask=1,act=act),
            #CouplingLayer(self.input_dim,self.mid_dim,masktype, reverse_mask=0,act=act),
            #ouplingLayer(self.input_dim,self.mid_dim,masktype, reverse_mask=1,act=act),
            #CouplingLayer(self.input_dim,self.mid_dim,masktype, reverse_mask=0,act=act),
            #CouplingLayer(self.input_dim,self.mid_dim,masktype, reverse_mask=1,act=act),
            #CouplingLayer(self.input_dim,self.mid_dim,masktype, reverse_mask=0,act=act),
            #CouplingLayer(self.input_dim,self.mid_dim,masktype, reverse_mask=1,act=act),
            #CouplingLayer(self.input_dim,self.mid_dim,masktype, reverse_mask=0,act=act),
            #CouplingLayer(self.input_dim,self.mid_dim,masktype, reverse_mask=1,act=act),
            #CouplingLayer(self.input_dim,self.mid_dim,masktype, reverse_mask=0,act=act),
            #CouplingLayer(self.input_dim,self.mid_dim,masktype, reverse_mask=1,act=act),
            #CouplingLayer(self.input_dim,self.mid_dim,masktype, reverse_mask=0,act=act),
            #CouplingLayer(self.input_dim,self.mid_dim,masktype, reverse_mask=1,act=act),
            #CouplingLayer(self.input_dim,self.mid_dim,masktype, reverse_mask=0,act=act),
            #CouplingLayer(self.input_dim,self.mid_dim,masktype, reverse_mask=1,act=act),
            #CouplingLayer(self.input_dim,self.mid_dim,masktype, reverse_mask=0,act=act),
            #CouplingLayer(self.input_dim,self.mid_dim,masktype, reverse_mask=1,act=act)
            #CouplingLayer(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=False)
        ])

        #if self.is_last_block:
         #   print("LAST BLOCK")
            #self.in_couplings.append(
            #    CouplingLayer(in_channels, mid_channels, num_blocks, masktype, reverse_mask=True))
        #else:
        #    self.out_couplings = nn.ModuleList([
        #        CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks, MaskType.CHANNEL_WISE, reverse_mask=False),
        #        CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks, MaskType.CHANNEL_WISE, reverse_mask=True),
        #        CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks, MaskType.CHANNEL_WISE, reverse_mask=False)
        #    ])
            #self.next_block = _RealNVP(scale_idx + 1, num_scales, 2 * in_channels, 2 * mid_channels, num_blocks,masktype)

    def forward(self, x, sldj, reverse=False):
            #print(x.shape)
            #print(x.shape)
            for coupling in self.in_couplings:
                x, sldj = coupling(x, sldj)

            #if not self.is_last_block:
            #    # Squeeze -> 3x coupling (channel-wise)
            #    x = squeeze_2x2(x, reverse=False)
            #        x, sldj = coupling(x, sldj, reverse)
            #    print(x.shape)
            #    x = squeeze_2x2(x, reverse=True)
            #    print(x.shape)

                # Re-squeeze -> split -> next block
            #    x = squeeze_2x2(x, reverse=False, alt_order=True)
            #    x, x_split = x.chunk(2, dim=1)
            #    x, sldj = self.next_block(x, sldj, reverse)
            #    x = torch.cat((x, x_split), dim=1)
            #    x = squeeze_2x2(x, reverse=True, alt_order=True)

            return x, sldj
    
    def reverse(self, x, sldj=None):
        # invert in reverse order
        for coupling in reversed(self.in_couplings):
            x, sldj = coupling.reverse(x, sldj)
        return x, sldj
    
class _RealNVPsim(nn.Module):
   
    def __init__(self, input_dim, mid_dim,masktype,act=1,mu=0,sigma=1,learn=False):
        super(_RealNVPsim, self).__init__()
        self.input_dim=input_dim
        self.mid_dim=mid_dim
        self.mu=nn.Parameter(mu, requires_grad=learn)
        self.sigma=nn.Parameter(sigma, requires_grad=learn)
        self.alpha=nn.Parameter(torch.rand(sigma.shape),requires_grad=learn)
        if sigma.shape==1:
            self.alpha=nn.Parameter(torch.ones(sigma.shape),requires_grad=False)
        #temp=torch.rand(sigma.shape)
        #self.alpha=nn.Parameter(torch.exp(temp)/torch.sum(torch.exp(temp)),requires_grad=True)

        self.in_couplings = nn.ModuleList([
            CouplingLayer(self.input_dim,self.mid_dim, masktype, reverse_mask=0,act=act),
            CouplingLayer(self.input_dim,self.mid_dim,masktype, reverse_mask=1,act=act),
            CouplingLayer(self.input_dim,self.mid_dim,masktype, reverse_mask=0,act=act),
            CouplingLayer(self.input_dim,self.mid_dim,masktype, reverse_mask=1,act=act),
            CouplingLayer(self.input_dim,self.mid_dim,masktype, reverse_mask=0,act=act),
            #CouplingLayer(self.input_dim,self.mid_dim,masktype, reverse_mask=1,act=act),
            #CouplingLayer(self.input_dim,self.mid_dim,masktype, reverse_mask=0,act=act),
            #CouplingLayer(self.input_dim,self.mid_dim,masktype, reverse_mask=1,act=act),
            #CouplingLayer(self.input_dim,self.mid_dim,masktype, reverse_mask=0,act=act),
            #CouplingLayer(self.input_dim,self.mid_dim,masktype, reverse_mask=1,act=act),
            #CouplingLayer(self.input_dim,self.mid_dim,masktype, reverse_mask=0,act=act),
            #CouplingLayer(self.input_dim,self.mid_dim,masktype, reverse_mask=1,act=act),
            #CouplingLayer(self.input_dim,self.mid_dim,masktype, reverse_mask=0,act=act),
            #CouplingLayer(self.input_dim,self.mid_dim,masktype, reverse_mask=1,act=act),
            #CouplingLayer(self.input_dim,self.mid_dim,masktype, reverse_mask=0,act=act),
            #CouplingLayer(self.input_dim,self.mid_dim,masktype, reverse_mask=1,act=act),
            #CouplingLayer(self.input_dim,self.mid_dim,masktype, reverse_mask=0,act=act),
            #CouplingLayer(self.input_dim,self.mid_dim,masktype, reverse_mask=1,act=act)
            #CouplingLayer(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=False)
        ])

        #if self.is_last_block:
         #   print("LAST BLOCK")
            #self.in_couplings.append(
            #    CouplingLayer(in_channels, mid_channels, num_blocks, masktype, reverse_mask=True))
        #else:
        #    self.out_couplings = nn.ModuleList([
        #        CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks, MaskType.CHANNEL_WISE, reverse_mask=False),
        #        CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks, MaskType.CHANNEL_WISE, reverse_mask=True),
        #        CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks, MaskType.CHANNEL_WISE, reverse_mask=False)
        #    ])
            #self.next_block = _RealNVP(scale_idx + 1, num_scales, 2 * in_channels, 2 * mid_channels, num_blocks,masktype)

    def forward(self, x, sldj, reverse=False):
            #print(x.shape)
            #print(x.shape)
            for coupling in self.in_couplings:
                x, sldj = coupling(x, sldj)

            #if not self.is_last_block:
            #    # Squeeze -> 3x coupling (channel-wise)
            #    x = squeeze_2x2(x, reverse=False)
            #        x, sldj = coupling(x, sldj, reverse)
            #    print(x.shape)
            #    x = squeeze_2x2(x, reverse=True)
            #    print(x.shape)

                # Re-squeeze -> split -> next block
            #    x = squeeze_2x2(x, reverse=False, alt_order=True)
            #    x, x_split = x.chunk(2, dim=1)
            #    x, sldj = self.next_block(x, sldj, reverse)
            #    x = torch.cat((x, x_split), dim=1)
            #    x = squeeze_2x2(x, reverse=True, alt_order=True)

            return x, sldj

class IndependentFeatureNetwork(nn.Module):
    def __init__(self,input_dim):
        super(IndependentFeatureNetwork, self).__init__()
        # 每个特征独立的映射层
        self.feature_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, 10),  # 将1维特征映射到10维
                nn.LeakyReLU(),
                nn.Linear(10, 10),  # 保持10维
                nn.LeakyReLU(),
                nn.Linear(10, 1)  # 再压缩到1维
            ) for _ in range(input_dim)  # 输入有10个特征，每个特征独立
        ])

    def forward(self, x):
        # 对每个特征独立处理
        outputs = []
        for i in range(x.size(1)):  # 遍历每个特征
            feature = x[:, i:i+1]  # 提取第i个特征，保持维度
            feature_output = self.feature_layers[i](feature)  # 单独通过其对应的子网络
            outputs.append(feature_output)
        # 将所有独立特征的输出合并
        return torch.cat(outputs, dim=1)
    


class NormalizingFlow(nn.Module):

    def __init__(self, dim, flow_length):
        super().__init__()

        self.transforms = nn.Sequential(*(
            PlanarFlow(dim) for _ in range(flow_length)
        ))

    def forward(self, z,sldj):
        for transform in self.transforms:
            #print(transform)
            z,temp= transform(z)
            sldj+=temp
        zk = z

        return zk, sldj


class PlanarFlow(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.weight = nn.Parameter(torch.Tensor(1, dim))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.scale = nn.Parameter(torch.Tensor(1, dim))
        self.tanh = nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self):

        self.weight.data.uniform_(-0.01, 0.01)
        self.scale.data.uniform_(-0.01, 0.01)
        self.bias.data.uniform_(-0.01, 0.01)

    def forward(self, z):

        activation = F.linear(z, self.weight, self.bias)
        psi = (1 - self.tanh(activation) ** 2) * self.weight
        det_grad = 1 + torch.mm(psi, self.scale.t())
        #torch.log((det_grad.abs())+ 1e-7)
        #sldj+=torch.log((det_grad.abs())+ 1e-7)
        return z + self.scale * self.tanh(activation),torch.log((det_grad.abs())+ 1e-7)


class PlanarFlowLogDetJacobian(nn.Module):
    """A helper class to compute the determinant of the gradient of
    the planar flow transformation."""

    def __init__(self, affine):
        super().__init__()

        self.weight = affine.weight
        self.bias = affine.bias
        self.scale = affine.scale
        self.tanh = affine.tanh

    def forward(self, z):

        activation = F.linear(z, self.weight, self.bias)
        psi = (1 - self.tanh(activation) ** 2) * self.weight
        det_grad = 1 + torch.mm(psi, self.scale.t())
        return torch.log((det_grad.abs())+ 1e-7)