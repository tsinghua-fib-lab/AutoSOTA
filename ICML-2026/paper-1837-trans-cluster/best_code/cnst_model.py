import torch
torch.set_printoptions(precision=2)
from torch import nn
import torch.nn.functional as F

class CModel(nn.Module):
  """
  Transformer implementation of LLoyd's algorithm
  No learnable parameters
  """

  def __init__(
    self,
    ndims,
    nclusters,
    itemp,
    osa=True,
    ldp=True,
  ):
    super().__init__()  # Parent class initialization comes AFTER attribute assignment

    self.osa = osa
    self.ldp = ldp

    # self-attention on X
    self.SAX = Attn(
      D=ndims+nclusters,
      Q=ndims,
      K=ndims,
      V=-nclusters,
      qsign=1,
      ksign=1,
      vsign=-1,
      inv_temp=itemp,
      )
    # cross-attention on X
    self.CAX = Attn(
      D=ndims+nclusters,
      Q=ndims,
      K=ndims,
      V=-nclusters,
      qsign=1,
      ksign=1,
      vsign=1,
      inv_temp=itemp,
      )
    # self-attention on cluster centers
    self.SAC = Attn(
      D=ndims+nclusters,
      Q=-nclusters,
      K=-nclusters,
      V=ndims,
      qsign=1,
      ksign=1,
      vsign=-1,
      inv_temp=itemp,
      )
    # cross-attention on cluster centers
    self.CAC = Attn(
      D=ndims+nclusters,
      Q=-nclusters,
      K=-nclusters,
      V=ndims,
      qsign=1,
      ksign=1,
      vsign=1,
      inv_temp=itemp,
      )

  def forward(self, XX, CC):

    XX = XX + self.SAX(XX, XX, diag_attn=self.osa) + self.CAX(XX, CC)
    CC = CC + self.SAC(CC, CC, diag_attn=self.osa) + self.CAC(CC, XX, ldp=self.ldp)

    return XX, CC

class Attn(nn.Module):
  def __init__(
    self,
    D: int,
    Q: int,
    K: int,
    V: int,
    qsign: int,
    ksign: int,
    vsign: int,
    inv_temp: float,
  ):
    super().__init__()
    self.demb = D

    def getmat(din, ind, sgn):
      ret = nn.Linear(din, din, bias=False)
      ret.weight.data.zero_()
      blk = sgn * torch.eye(abs(ind))
      if ind > 0:
        ret.weight.data[:ind, :ind] = blk
      elif ind < 0:
        ret.weight.data[ind:, ind:] = blk
      return ret

    # Query
    self.qind = Q
    self.qmat = getmat(self.demb, self.qind, sgn=qsign)
    # print("Q:\n", self.qmat)
    # Key
    self.kind = K
    self.kmat = getmat(self.demb, self.kind, sgn=ksign)
    # print("K:\n", self.kmat)
    # Value
    self.vind = V
    self.vmat = getmat(self.demb, self.vind, sgn=vsign)
    # print("V:\n", self.vmat)
    self.gamma = inv_temp

  def forward(
    self,
    q: torch.FloatTensor,
    k: torch.FloatTensor,
    ldp: bool = False,
    diag_attn: bool = False,
  ):
    Q, K, V = self.qmat(q), self.kmat(k), self.vmat(k)
    if diag_attn:
      return V
    if ldp:
      qk = Q @ K.T
      attn = qk * (1.0 / qk.sum(axis=1))[:, None]
    else:
      Qsqnorm = torch.square(torch.linalg.vector_norm(Q, ord=2, dim=1, keepdims=True))
      Ksqnorm = torch.square(torch.linalg.vector_norm(K, ord=2, dim=1, keepdim=True))
      sqdist = 2 * Q @ K.T - Qsqnorm - Ksqnorm.T
      attn = F.softmax(self.gamma * sqdist, dim=-1)
    return attn @ V
