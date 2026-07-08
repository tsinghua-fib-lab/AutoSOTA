import time

import numpy as np
import torch
torch.set_printoptions(precision=4)

from entmax import sparsemax

def lloyds_iters(
    samples: torch.Tensor,
    init_centers: torch.Tensor,
    niters: int,
    use_cuda: bool = True,
):
  ccenters = init_centers.clone().detach()
  nclusters, ndims = init_centers.shape
  assert samples.shape[1] == ndims
  nsamples = samples.shape[0]
  kmeans_obj = []
  X = samples
  if use_cuda:
    X = X.to('cuda')
    ccenters = ccenters.to('cuda')
  ltime = time.time()
  for i in range(niters+1):
    # compute points to center distance
    sqdist = torch.square(torch.cdist(X, ccenters))
    # compute assignments
    sqd, cassign = torch.min(sqdist, dim=1)
    # print(cassign)
    kmeans_obj += [torch.sum(sqd).item()]
    # compute new centers
    ccenters = torch.stack([
      torch.mean(X[cassign == i], dim=0)
      for i in range(nclusters)
    ])
    if torch.any(torch.isnan(ccenters)):
      print(
        f'[WARN] ({i+1}/{niters}) found nan in lloyds centers, '
        f'replacing with random samples from the dataset'
      )
      nancs = ccenters[torch.isnan(ccenters)]
      nncs = len(nancs) // ndims
      rcs = X[torch.randperm(nsamples)[:nncs]].clone().detach()
      ccenters[torch.isnan(ccenters)] = rcs.reshape(-1)
      assert not torch.any(torch.isnan(ccenters))
  kmeans_obj = torch.tensor(kmeans_obj)
  print(
    f"{niters} Lloyd's iterations completed in "
    f"{time.time()-ltime:0.3f} seconds"
  )
  return kmeans_obj.to('cpu')


def lloyds_iters_batched(
    samples: torch.Tensor,
    init_centers: torch.Tensor,
    niters: int,
    use_cuda: bool = True,
):
  assert len(samples.shape) == len(init_centers.shape) == 3
  ccenters = init_centers.clone().detach()
  bsz, nclusters, ndims = init_centers.shape
  assert samples.shape[2] == ndims
  assert samples.shape[0] == bsz
  nsamples = samples.shape[1]
  kmeans_obj = []
  X = samples
  if use_cuda:
    X = X.to('cuda')
    ccenters = ccenters.to('cuda')
  ltime = time.time()
  if torch.any(torch.isnan(ccenters)):
    print(f'found nan in initial lloyds centers')
  for i in range(niters+1):
    # compute points to center distance
    sqdist = torch.square(torch.cdist(X, ccenters))
    # compute assignments
    sqd, cassign = torch.min(sqdist, dim=2)
    kmeans_obj += [torch.sum(sqd, dim=1).clone().detach()]
    ccenters = torch.stack([torch.stack([
      torch.mean(X[b][cassign[b] == i], dim=0)
      for i in range(nclusters)
    ]) for b in range(bsz) ])
    if torch.any(torch.isnan(ccenters)):
      print(
        f'[WARN] ({i+1}/{niters}) found nan in lloyds centers, '
        f'replacing with random points from the dataset'
      )
      for bidx, (XX, CC) in enumerate(zip(X, ccenters)):
        if torch.any(torch.isnan(CC)):
          print(f"[WARN] - found nan in task {bidx+1}/{bsz}")
          nancs = CC[torch.isnan(CC)]
          nncs = len(nancs) // ndims
          rcs = XX[torch.randperm(nsamples)[:nncs]].clone().detach()
          CC[torch.isnan(CC)] = rcs.reshape(-1)
      assert not torch.any(torch.isnan(ccenters))
      # ccenters = torch.nan_to_num(
      #   ccenters, nan=0.5, posinf=1.0, neginf=0.0
      # )
  kmeans_obj = torch.stack(kmeans_obj)
  assert not torch.any(torch.isnan(kmeans_obj)), (
    'found nan in kmeans objs'
  )
  print(
    f"{niters} Lloyd's iterations completed in "
    f"{time.time()-ltime:0.3f} seconds"
  )
  return kmeans_obj.to('cpu')


def trimmed_iters(
    samples: torch.Tensor,
    init_centers: torch.Tensor,
    niters: int,
    use_cuda: bool = True,
    quantile: float = 0.95,
):
  ccenters = init_centers.clone().detach()
  nclusters, ndims = init_centers.shape
  assert samples.shape[1] == ndims
  nsamples = samples.shape[0]
  kmeans_obj = []
  X = samples
  if use_cuda:
    X = X.to('cuda')
    ccenters = ccenters.to('cuda')
  ltime = time.time()
  for i in range(niters+1):
    # compute points to center distance
    sqdist = torch.square(torch.cdist(X, ccenters))
    # compute assignments
    sqd, cassign = torch.min(sqdist, dim=1)
    # print(cassign)
    kmeans_obj += [torch.sum(sqd).item()]
    # compute per-cluster thresholds
    thres = torch.tensor([
      (
        torch.quantile(sqd[cassign == i], q=quantile)
        if torch.sum(cassign == i) > 0 else 0.0
      )
      for i in range(nclusters)
    ])
    for i in range(nclusters):
      if torch.sum(cassign == i) == 0:
        continue
      pidxs = torch.arange(nsamples, device=cassign.device)[cassign == i]
      dists = sqd[pidxs]
      inliers = dists < thres[i]
      tpidxs = pidxs[inliers]
      dpidxs = torch.arange(nsamples, device=cassign.device)[(cassign == i) & (sqd < thres[i])]
    # compute new centers
    ccenters = torch.stack([
      (
        torch.mean(X[(cassign == i) & (sqd < thres[i])], dim=0)
        if torch.sum(cassign == i) > 0 else
        torch.ones_like(ccenters[i]) * torch.nan
      )
      for i in range(nclusters)
    ])
    if torch.any(torch.isnan(ccenters)):
      print(
        f'[WARN] ({i+1}/{niters}) found nan in trimmed kmeans centers, '
        f'replacing with random samples from the dataset'
      )
      nancs = ccenters[torch.isnan(ccenters)]
      nncs = len(nancs) // ndims
      rcs = X[torch.randperm(nsamples)[:nncs]].clone().detach()
      ccenters[torch.isnan(ccenters)] = rcs.reshape(-1)
      assert not torch.any(torch.isnan(ccenters))
  kmeans_obj = torch.tensor(kmeans_obj)
  print(
    f"{niters} trimmed kmeans iterations completed in "
    f"{time.time()-ltime:0.3f} seconds"
  )
  return kmeans_obj.to('cpu')


def rkm_iters(
    samples: torch.Tensor,
    init_centers: torch.Tensor,
    niters: int,
    use_cuda: bool = True,
    inv_temp: float = 0.01,
):
  ccenters = init_centers.clone().detach()
  nclusters, ndims = init_centers.shape
  assert samples.shape[1] == ndims
  nsamples = samples.shape[0]
  kmeans_obj = []
  X = samples
  if use_cuda:
    X = X.to('cuda')
    ccenters = ccenters.to('cuda')
  ltime = time.time()
  for i in range(niters+1):
    # compute points to center distance
    sqdist = torch.square(torch.cdist(X, ccenters))
    # compute assignments
    sqd, cassign = torch.min(sqdist, dim=1)
    # print(cassign)
    kmeans_obj += [torch.sum(sqd).item()]
    # compute new centers
    ccenters = torch.stack([
      (
        sparsemax(-inv_temp * sqd[cassign == i] / np.sqrt(ndims)) @ X[cassign == i]
        if torch.sum(cassign == i) > 0 else
        torch.ones_like(ccenters[i]) * torch.nan
        # torch.zeros_like(ccenters[i])
      )
      for i in range(nclusters)
    ])
    if torch.any(torch.isnan(ccenters)):
      print(
        f'[WARN] ({i+1}/{niters}) found nan in robust kmeans centers, '
        f'replacing with random samples from the dataset'
      )
      nancs = ccenters[torch.isnan(ccenters)]
      nncs = len(nancs) // ndims
      rcs = X[torch.randperm(nsamples)[:nncs]].clone().detach()
      ccenters[torch.isnan(ccenters)] = rcs.reshape(-1)
      assert not torch.any(torch.isnan(ccenters))
  kmeans_obj = torch.tensor(kmeans_obj)
  print(
    f"{niters} robust kmeans iterations completed in "
    f"{time.time()-ltime:0.3f} seconds"
  )
  return kmeans_obj.to('cpu')


if __name__ == "__main__":
  print('test ....')
  SEED = 153476998
  # Set seeds for experiment
  import numpy as np
  import random
  RNG = np.random.RandomState(SEED)
  torch.manual_seed(SEED)
  np.random.seed(SEED)
  random.seed(SEED)


  from ctasks import ClusteringTasks as CT
  n, d, k = 512, 25, 32
  dlist = ['normal']
  ct = CT(k, k, d, n, n, dlist)

  XX, CC = ct.sample_batch(bsz=32)
  print(f"XX: {XX.shape}, CC: {CC.shape}")
  niters = 5
  bret = lloyds_iters_batched(XX, CC, niters)
  # print(f"Batched return:\n{bret}")
  for bidx, (X, C) in enumerate(zip(XX, CC)):
    print(f"Task {bidx+1} ...")
    ret = lloyds_iters(X, C, niters)
    # print(f"[{bidx+1}] Ind return:\n{ret}")

  n, k, d = 500, 8, 12
  X = torch.rand(n, d)
  C = X[:k, :].clone().detach()
  ret = lloyds_iters(X, C, 3)

  bsz = 5
  XX = torch.rand(bsz, n, d)
  CC = XX[:, :k, :].clone().detach()
  ret = []
  niters = 6
  for i in range(bsz):
    print(f"batch {i+1}")
    ret += [lloyds_iters(XX[i], CC[i], niters)]
  ret = torch.stack(ret).T
  ret2 = lloyds_iters_batched(XX, CC, niters)
  assert torch.all(ret == ret2)
