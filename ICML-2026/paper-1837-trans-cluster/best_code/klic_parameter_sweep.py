"""
Executable script to run a parameter sweep in parallel (CPU) using multiprocessing (via tqdm)
"""

import numpy as np
import multiprocessing as mp
import torch
from lloyds import lloyds_algorithm, kmeans_obj
from cnst_model import CModel
import pandas as pd
from tqdm.contrib.concurrent import process_map
import argparse

parser = argparse.ArgumentParser('./klic_param_sweep.py', description='Perform a parameter sweep of k-means transformer and Lloyd algorithm')
parser.add_argument('--nproc', type=int, default=1, help='num parallel processes (int) | default: 1')



def run(args):
  nproc = args.nproc

  # These are fixed parameter sweeps for the paper
  dimensions = np.linspace(1,10,10,dtype=int)
  clusters = np.arange(5,55,5,dtype=int)
  points = np.logspace(2,6,10,dtype=int)
  temperature = [1, 10, 100] #, 100, 1000]
  ub_scale = np.linspace(1,10,10,dtype=int)

  inputs = []
  for nc in clusters:
    # print('clusters', nc)
    for dim in dimensions:
      for ub in ub_scale:
        for point in points:
          inputs.append(tuple(([dim],[nc],[ub],temperature,[point])))
  # chunk size just batches job per child process, this can be increased for greater efficiency
  results = process_map(param_sweep, inputs, max_workers=nproc, chunksize=5)
  results = pd.concat(results)
  results.to_csv('results/klic_parameter_sweep.csv', index=False)

def param_sweep(input_tuple):

  dimensions, clusters, ub_scale, temperature, points = input_tuple

  niters = 10
  scaled = True

  df = {}
  df['Clusters'] = []
  df['Iteration'] = []
  df['Dimensions'] = []
  df['Gamma'] = []
  df['Points'] = []
  df['Loss'] = []
  df['Scale'] = []

  for nc in clusters:
    # print('clusters', nc)
    for dim in dimensions:
      # print('dim', dim)
      for pt in points:
        for ub in ub_scale:
          # print('pt', pt)
          # Ingest / generate data
          true_centers = torch.rand([int(nc), int(dim)])
          X = []
          for loc in true_centers:
            nsamples = pt // nc
            m = torch.distributions.MultivariateNormal(loc=loc, covariance_matrix=0.01 * torch.eye(loc.shape[-1]))
            X += [m.sample() for _ in range(nsamples)]
          X = torch.reshape(torch.cat(X), (len(X), dim))
          if scaled:
            minX, _ = X.min(dim=0)
            assert minX.size(0) == dim
            X = X - minX
            true_centers = true_centers - minX
            maxX, _ = X.max(dim=0)
            X = X / maxX
            true_centers = true_centers / maxX
            X *= ub
            true_centers *= ub

          X = X[torch.randperm(X.size()[0])]
          #
          init_centers = X[:nc].clone().detach()
          #
          #### Run Lloyds
          lloyd_centers, lloyd_objs, cc = lloyds_algorithm(X, init_centers, niters, nc, dim, use_cuda=False)

          for temp in temperature:
            # print('temp', temp)
            #### Run tfmr
            # Pre-process data with labels
            XX = torch.hstack([X.to('cpu'), torch.zeros(X.size()[0], nc)])
            CC = torch.hstack([init_centers.clone().detach(), torch.eye(nc)])
            assert XX.shape[-1] == CC.shape[-1] == dim + nc

            trf_centers = [CC[:, :dim].clone().detach()]

            model = CModel(
                dim,
                nc,
                temp,
                osa=True,
                ldp=True
              )

            with torch.no_grad():
              for i in range(niters):
                XX, CC = model(XX,CC)
                # track centers
                trf_centers += [CC[:, :dim].clone().detach().to('cpu')]
              YY = XX[:, -nc:].clone().detach().to('cpu').numpy().astype(int)
            trf_objs = [torch.log(kmeans_obj(X.to('cpu'), cc)) for cc in trf_centers]

            df['Clusters'].append(nc)
            df['Iteration'].append(-1)
            df['Dimensions'].append(dim)
            df['Gamma'].append(temp)
            df['Points'].append(pt)
            df['Scale'].append(ub)
            df['Loss'].append(trf_objs[-1].detach().numpy() - lloyd_objs[-1].detach().numpy())

  df = pd.DataFrame(df)
  return df


if __name__ == '__main__':
  args = parser.parse_args()
  run(args)
