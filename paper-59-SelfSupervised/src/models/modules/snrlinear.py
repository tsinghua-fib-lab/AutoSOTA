import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Linear

#define _l2normalization
def _l2normalize(v, eps=1e-12):
    return v / (torch.norm(v) + eps)

def max_singular_value(W, u=None, Ip=1):
    """
    power iteration for weight parameter
    """
    #xp = W.data
    if not Ip >= 1:
        raise ValueError("Power iteration should be a positive integer")
    if u is None:
        u = torch.FloatTensor(1, W.size(0)).normal_(0, 1).cuda()
    _u = u
    for _ in range(Ip):
        _v = _l2normalize(torch.matmul(_u, W.detach()), eps=1e-12)
        _u = _l2normalize(torch.matmul(_v, torch.transpose(W.detach(), 0, 1)), eps=1e-12)
    sigma = torch.sum(F.linear(_u, torch.transpose(W.detach(), 0, 1)) * _v)
    return sigma, _u

class SNLinear(Linear):
    r"""Applies a linear transformation to the incoming data: :math:`y = Ax + b`
       Args:
           in_features: size of each input sample
           out_features: size of each output sample
           bias: If set to False, the layer will not learn an additive bias.
               Default: ``True``
       Shape:
           - Input: :math:`(N, *, in\_features)` where :math:`*` means any number of
             additional dimensions
           - Output: :math:`(N, *, out\_features)` where all but the last dimension
             are the same shape as the input.
       Attributes:
           weight: the learnable weights of the module of shape
               `(out_features x in_features)`
           bias:   the learnable bias of the module of shape `(out_features)`

           W(Tensor): Spectrally normalized weight

           u (Tensor): the right largest singular value of W.
       """
    def __init__(self, in_features, out_features, bias=True):
        super(SNLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('u', torch.Tensor(1, out_features).normal_())

    @property
    def W_(self):
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, _u = max_singular_value(w_mat, self.u)
        self.u.copy_(_u)
        return self.weight / sigma

    def forward(self, input):
        return F.linear(input, self.W_, self.bias)