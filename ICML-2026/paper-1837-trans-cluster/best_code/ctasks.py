import random
import numpy as np

import torch
from torch import nn
from torch.distributions import Cauchy, Gumbel, Laplace, LogNormal, Normal
from torch.distributions import Categorical, MixtureSameFamily, Distribution, Independent


DMAP = {
  'normal': Normal,
  'cauchy': Cauchy,
  'gumbel': Gumbel,
  'laplace': Laplace,
  'lognormal': LogNormal,
}


class ClusteringTasks:
  def __init__(
      self,
      klb: int,
      kub: int,
      dims: int,
      nlb: int,
      nub: int,
      dlist: list[str],
      xlb: float = 0.0,
      xub: float = 1.0,
      scale: float = 0.1,
      equal_mix: bool = True,
      equal_scales: bool = True,
  ):
    self.klist = [klb] if klb == kub else np.arange(klb, kub+1).tolist()
    self.dims = dims
    self.nlist = [nlb] if nlb == nub else np.arange(nlb, nub+1).tolist()
    self.dists = [DMAP[d] for d in dlist]
    self.xlb, self.xub = xlb, xub
    self.dist_scale = scale
    self.equal_mix = equal_mix
    self.equal_scales = equal_scales
    print(f"Cuda available: {torch.cuda.is_available()}")
    self.GPUE = torch.cuda.is_available()
    if self.GPUE:
      self.DEVICE = torch.cuda.current_device()
      print(f"[ctask] Found device: {self.DEVICE}")



  def sample_batch(self, bsz: int, same_dist_batch: bool = True):
    k = random.choice(self.klist)
    n = random.choice(self.nlist)
    dist = random.choice(self.dists)
    XX, CC = [], []
    for i in range(bsz):
      # sample distribution to sample from in this task
      if not same_dist_batch:
        dist = random.choice(self.dists)
      X, C = self.sample_task(n, self.dims, k, dist, self.xlb, self.xub)
      XX += [X]
      CC += [C]
    XX = torch.stack(XX)
    CC = torch.stack(CC)
    return XX, CC

  def sample_task(
      self,
      npoints: int,
      ndims: int,
      nclusters: int,
      dist: type[Cauchy | Gumbel | Laplace | LogNormal | Normal],
      xlb: float = 0.0,
      xub: float = 1.0,
  ):
    with torch.no_grad():
      # create mixture distribution
      cweights = torch.ones(nclusters,) if self.equal_mix else torch.rand(nclusters,)
      scale = self.dist_scale * (
        torch.ones(nclusters, ndims) if self.equal_scales
        else torch.rand(nclusters, ndims)
      )
      loc = torch.rand(nclusters, ndims)
      if self.GPUE:
        loc = loc.to(self.DEVICE)
        scale = scale.to(self.DEVICE)
        cweights = cweights.to(self.DEVICE)
      mix = Categorical(cweights)
      comp = Independent(dist(loc=loc, scale=scale), 1)
      dmm = MixtureSameFamily(mix, comp)
      # sample from mixture distribution
      X = dmm.sample((npoints,))
      # scale samples to [0,1]^d
      minX, _ = X.min(dim=0)
      X = X - minX
      maxX, _ = X.max(dim=0)
      X = X / maxX
      if torch.any(torch.isnan(X)):
        print(f'[WARN] found nan in data, replacing with zero')
        X = torch.nan_to_num(X, nan=0.0)
      # permute samples, and select initial centers randomly
      X = X[torch.randperm(npoints)]
      C = X[:nclusters].clone().detach()
      return X, C

if __name__ == "__main__":
  print('test ....')
  SEED = 5487
  # Set seeds for experiment
  RNG = np.random.RandomState(SEED)
  torch.manual_seed(SEED)
  np.random.seed(SEED)
  random.seed(SEED)

  fname = 'data_test.pdf'
  n, k, d = 1000, 5, 2
  scale, eqmix, eqscale = 0.05, True, True
  nrows, ncols = 2, 6
  bsz = nrows * ncols
  # dists = ['normal']
  dists = list(DMAP.keys())
  ctasks = ClusteringTasks(
    k, k, d, n, n, dists,
    scale=scale, equal_mix=eqmix, equal_scales=eqscale
  )
  XX, CC = ctasks.sample_batch(bsz, False)
  print(f"Batch size: {XX.shape}, {CC.shape}")
  print(XX.device, CC.device)

  import matplotlib.pyplot as plt

  fig, axs = plt.subplots(nrows, ncols, figsize=(ncols, nrows), sharex=True, sharey=True)
  for i in range(nrows):
    for j in range(ncols):
      ax = axs[i, j]
      bidx = i * ncols + j
      X = XX[bidx].to('cpu')
      C = CC[bidx].to('cpu')
      ax.scatter(X[:, 0], X[:, 1], marker='.', color='black', s=1)
      ax.scatter(C[:, 0], C[:, 1], marker='^', color='red')
      ax.axis('square')

  fig.tight_layout()
  fig.savefig(fname)
