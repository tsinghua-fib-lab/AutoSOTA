import torch
import time


def kmeans_obj(samples, centers):
  assert samples.shape[-1] == centers.shape[-1]
  Xsqnorm = torch.square(torch.linalg.vector_norm(samples, ord=2, dim=1, keepdim=True))
  csqnorm = torch.square(torch.linalg.vector_norm(centers, ord=2, dim=1, keepdim=True))
  XC = samples @ centers.T
  sqdist = Xsqnorm - 2 * XC + csqnorm.T
  # compute assignments
  sqd, _ = torch.min(sqdist, dim=1)
  kmeans_obj = torch.sum(sqd)
  return kmeans_obj


def lloyds_algorithm(X, init_centers, niters, nclusters, ndims, use_cuda=False):
  # run lloyd's by hand and track centers and kmeans loss per-iteration
  lloyd_centers = []
  curr_centers = init_centers
  lloyd_centers = [init_centers]
  lloyd_objs = []

  if use_cuda:
    X = X.to('cuda')
    curr_centers = curr_centers.to('cuda')

  ltime = time.time()
  for i in range(niters):
    # compute points to center distance
    Xsqnorm = torch.square(torch.linalg.vector_norm(X, ord=2, dim=1, keepdim=True))
    csqnorm = torch.square(torch.linalg.vector_norm(curr_centers, ord=2, dim=1, keepdim=True))
    XC = X @ curr_centers.T
    sqdist = Xsqnorm - 2 * XC + csqnorm.T
    # compute assignments
    sqd, cc = torch.min(sqdist, dim=1)
    lloyd_objs += [torch.log(torch.sum(sqd)).to('cpu')]
    # compute new centers
    curr_centers = torch.reshape(
      torch.cat([torch.mean(X[cc == i], dim=0) for i in range(nclusters)]),
      (nclusters, ndims)
    )
    lloyd_centers += [curr_centers.clone().detach().to('cpu')]
  # print(f"Lloyd's takes {time.time() - ltime:.3f} secs")
  lloyd_objs += [torch.log(kmeans_obj(X.to('cpu'), lloyd_centers[-1]))]
  # print("Lloyd's objs\n", lloyd_objs)
  return lloyd_centers, lloyd_objs, cc
