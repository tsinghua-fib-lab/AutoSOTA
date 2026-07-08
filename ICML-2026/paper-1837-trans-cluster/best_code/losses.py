import torch
from torch.nn import Module
torch.set_printoptions(precision=4)
import torch.nn.functional as F

ADICT = {
  "softmax": F.softmax,
}


def kmeans_obj(samples: torch.Tensor, centers: torch.Tensor):
  assert samples.shape[-1] == centers.shape[-1]
  sqdist = torch.square(torch.cdist(samples, centers))
  # compute assignments
  sqd, _ = torch.min(sqdist, dim=1)
  kmeans_obj = torch.sum(sqd)
  return kmeans_obj


def kmeans_obj_batched(samples: torch.Tensor, centers: torch.Tensor):
  assert len(samples.shape) == len(centers.shape) == 3
  assert samples.shape[-1] == centers.shape[-1]
  assert samples.shape[0] == centers.shape[0]
  sqdist = torch.square(torch.cdist(samples, centers))
  # compute assignments
  sqd, _ = torch.min(sqdist, dim=2)
  kmeans_obj = torch.sum(sqd, dim=1)
  return kmeans_obj


class SoftKMObj(Module):
  def __init__(self, gamma: float = 1.0, act: str = "softmax", logloss: bool = False):
    super().__init__()
    self.gamma = gamma
    assert act in ADICT.keys()
    self.activation = ADICT[act]
    self.logloss = logloss

  def forward(self, samples: torch.Tensor, centers: torch.Tensor):
    assert len(samples.shape) == len(centers.shape) == 3
    assert samples.shape[-1] == centers.shape[-1]
    assert samples.shape[0] == centers.shape[0]
    sqdist = torch.square(torch.cdist(samples, centers))
    weights = self.activation(-self.gamma * sqdist, dim=-1)
    sobjs = torch.einsum("bnk, bnk -> bn", weights, sqdist)
    ret = torch.sum(sobjs, dim=1)
    return torch.log(ret) if self.logloss else ret


if __name__ == "__main__":
  n, k, d = 5, 3, 7
  X = torch.rand(n, d)
  C = torch.rand(k, d)
  Xsqnorm = torch.square(torch.linalg.vector_norm(X, ord=2, dim=1, keepdim=True))
  csqnorm = torch.square(torch.linalg.vector_norm(C, ord=2, dim=1, keepdim=True))
  XC = X @ C.T
  sqdist = Xsqnorm - 2 * XC + csqnorm.T
  sqdist2 = torch.square(torch.cdist(X, C))
  print(f"Manual - cdist diff: {torch.sum(torch.abs(sqdist - sqdist2))}")
  # batched compute
  bsz = 2
  XX = torch.rand(bsz, n, d)
  CC = torch.rand(bsz, k, d)
  sqdist = [torch.square(torch.cdist(X, C)) for (X, C) in zip(XX, CC)]
  sqdist2 = torch.square(torch.cdist(XX, CC))
  kmobjs = []
  # True kmeans obj
  kmobjs_batched = kmeans_obj_batched(XX, CC)
  # kmeans obj upperbound via softmax
  sko = SoftKMObj(gamma=2)
  smobjs = sko(XX, CC)
  for i in range(bsz):
    diff = torch.sum(torch.abs(sqdist[i] - sqdist2[i]))
    kmobj = kmeans_obj(XX[i], CC[i])
    kmobjs += [kmobj]
    print(
      f"batch {i+1} diff: {diff:.4f}, kmeans-obj: {kmobj:.4f}, "
      f"softmin-kmeans-obj: {smobjs[i]:.4f}"
    )
  kmobjs = torch.tensor(kmobjs)
  assert torch.all(kmobjs_batched == kmobjs), (
    f"kmeans: {kmobjs}\n"
    f"kmeans-batched: {kmobjs_batched}\n"
    f"kmeans obj diff: {torch.sum(torch.abs(kmobjs - kmobjs_batched))}"
  )
  assert torch.all(kmobjs_batched <= smobjs), (
    f"True objs: {kmobjs_batched}\n"
    f"Softmin objs via softmax: {smobjs}"
  )
