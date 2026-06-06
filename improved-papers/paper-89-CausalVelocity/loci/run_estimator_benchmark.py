import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from sklearn.preprocessing import SplineTransformer
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from causa.het_ridge import convex_fgls
from causa.iterative_fgls import iterative_fgls

warnings.filterwarnings('ignore')


def set_seed(seed):
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)
      cudnn.deterministic = True
      cudnn.benchmark = False


def experiment(seed, knot_heuristic, result_dir):
    device = 'cpu'
    sample_list = np.logspace(2, 4, 11).astype(int).tolist()
    df = pd.DataFrame(0, index=np.arange(len(sample_list)), 
                      columns=['ifgls', 'mlconv'], dtype='double')
    df.insert(0, "Samples", sample_list, True)
    for samp in range(len(sample_list)):
        set_seed(seed)
        samples = sample_list[samp]
        print('Running with', samples, 'samples')
        x = (torch.rand(samples) - 0.5) * 8.0 * torch.pi
        n = torch.randn(samples)
        y = torch.sin(x) + n * ((4.0 * torch.pi - np.abs(x)) * 0.1 + 0.2)
        
        # define ground truth
        x_test = torch.linspace(-4.0 * torch.pi, 4.0 * torch.pi, 10000)
        gt_mean = torch.sin(x_test).numpy().copy()
        gt_sd = ((4.0 * torch.pi - np.abs(x_test)) * 0.1 + 0.2).numpy().copy()
        gt_dist = Normal(loc=torch.from_numpy(gt_mean), scale=torch.from_numpy(gt_sd))

        # prepare data
        if knot_heuristic == 'sqrt':
            knots = int(np.sqrt(samples))
        elif knot_heuristic == 'lin':
            knots = int(samples / 10)
        else:
            raise ValueError('Invalid knot_heuristic.')
        spline_trans = SplineTransformer(n_knots=knots, degree=5)
        Phi_test = spline_trans.fit_transform(x_test.unsqueeze(-1).numpy())
        Phi = spline_trans.transform(x.unsqueeze(-1).numpy())
        y = y.flatten()
        
        Phi = torch.from_numpy(Phi)
        Phi_test = torch.from_numpy(Phi_test)
        Phi_np = Phi.numpy().copy()
        Phi_test_np = Phi_test.numpy()
        y_np = y.numpy().copy()
        y = y.to(device).double()
        
        # iterative fgls
        takeLog = False
        w1, scale, weight_2, _, _ = iterative_fgls(y_np, Phi_np, Psi=Phi_np, n_steps=100, takeLog=takeLog)
        loc = Phi_test_np @ w1
        if takeLog:
            scale = np.sqrt(np.exp(Phi_test_np @ weight_2))
        else:
            scale = np.sqrt(np.clip(Phi_test_np @ weight_2, a_min=1e-7, a_max=1e7))
        est_dist = Normal(torch.from_numpy(loc), torch.from_numpy(scale))
        df.loc[samp, 'ifgls'] = torch.mean(kl_divergence(gt_dist, est_dist)).item()
        print('IFGLS', df.loc[samp, 'ifgls'])
        
        # Convex FGLS 
        w11, w12, _, _ = convex_fgls(Phi, torch.abs(Phi), y, delta_Phi=1e-6, delta_Psi=1e-6, n_steps=100) 
        eta_1 = Phi_test @ w11
        eta_2 = - torch.abs(Phi_test) @ w12
        scale = torch.sqrt(- 0.5 / eta_2)
        loc = - 0.5 * eta_1 / eta_2
        est_dist = Normal(loc, scale)
        df.loc[samp, 'mlconv'] = torch.mean(kl_divergence(gt_dist, est_dist)).item()
        print('Convex FGLS', df.loc[samp, 'mlconv'])
        
    df.to_csv(result_dir + '/' + f'estimator_benchmark_sine_{knot_heuristic}_{seed}.csv')

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=711, type=int, help='seed')
    parser.add_argument('--knot_heuristic', default='sqrt', choices=['sqrt', 'lin'], type=str, help='Knot choice heuristic')
    parser.add_argument('--result_dir', type=str, default='results')
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    experiment(args.seed, args.knot_heuristic, args.result_dir)
