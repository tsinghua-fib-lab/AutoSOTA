#!/usr/bin/env python3

import argparse
import time

try: import setGPU
except ImportError: pass

import torch
from torch.autograd import Variable

import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

import setproctitle

import models

import os, sys
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--nTrials', type=int, default=5)
    # parser.add_argument('--boardSz', type=int, default=2)
    # parser.add_argument('--batchSz', type=int, default=150)
    parser.add_argument('--Qpenalty', type=float, default=0.1)
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    setproctitle.setproctitle('bamos.sudoku.prof-sparse')

    print('=== nTrials: {}'.format(args.nTrials))
    print('| {:8s} | {:8s} | {:21s} | {:21s} |'.format(
        'boardSz', 'batchSz', 'dense forward (s)', 'sparse forward (s)'))
    for boardSz in [2,3]:
        with open('data/{}/features.pt'.format(boardSz), 'rb') as f:
            X = torch.load(f)
        with open('data/{}/labels.pt'.format(boardSz), 'rb') as f:
            Y = torch.load(f)
        N, nFeatures = X.size(0), int(np.prod(X.size()[1:]))

        for batchSz in [1, 64, 128]:
            dmodel = models.OptNetEq(boardSz, args.Qpenalty, trueInit=True)
            spmodel = models.SpOptNetEq(boardSz, args.Qpenalty, trueInit=True)
            if args.cuda:
                dmodel = dmodel.cuda()
                spmodel = spmodel.cuda()

            dtimes = []
            sptimes = []
            for i in range(args.nTrials):
                Xbatch = Variable(X[i*batchSz:(i+1)*batchSz])
                Ybatch = Variable(Y[i*batchSz:(i+1)*batchSz])
                if args.cuda:
                    Xbatch = Xbatch.cuda()
                    Ybatch = Ybatch.cuda()

                # Make sure buffers are initialized.
                # dmodel(Xbatch)
                # spmodel(Xbatch)

                start = time.time()
                # dmodel(Xbatch)
                dtimes.append(time.time()-start)

                start = time.time()
                spmodel(Xbatch)
                sptimes.append(time.time()-start)

            d_mean_ms = np.mean(dtimes) * 1000.0
            d_std_ms = np.std(dtimes) * 1000.0
            sp_mean_ms = np.mean(sptimes) * 1000.0
            sp_std_ms = np.std(sptimes) * 1000.0
            print('| {:8d} | {:8d} | {:.2f} +/- {:.2f} ms | {:.2f} +/- {:.2f} ms |'.format(
                boardSz, batchSz, d_mean_ms, d_std_ms, sp_mean_ms, sp_std_ms))

if __name__=='__main__':
    main()
