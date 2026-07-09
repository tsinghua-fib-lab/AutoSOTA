#!/usr/bin/env python3

import matplotlib.pyplot as plt
plt.style.use('bmh')
import pandas as pd
import numpy as np

import os

def main():
    workDir1 = "./work/2.dQP/"
    trainF1 = os.path.join(workDir1, 'train.csv')
    testF1 = os.path.join(workDir1, 'test.csv')
    trainDf1 = pd.read_csv(trainF1, sep=',')
    testDf1 = pd.read_csv(testF1, sep=',')

    workDir2 = "./work/2.optnetEq.Qpenalty=0.1/"
    trainF2 = os.path.join(workDir2, 'train.csv')
    testF2 = os.path.join(workDir2, 'test.csv')
    trainDf2 = pd.read_csv(trainF2, sep=',')
    testDf2 = pd.read_csv(testF2, sep=',')

    fig, ax = plotLoss(trainDf1, testDf1, workDir1)
    plotLoss(trainDf2, testDf2, workDir2, fig, ax)

    fig, ax = plotErr(trainDf1, testDf1, workDir1)
    plotErr(trainDf2, testDf2, workDir2, fig, ax)

def plotLoss(trainDf, testDf, workDir, fig=None, ax=None):

    if fig is None:
        fig, ax = plt.subplots(1, 1)

    trainEpoch = trainDf['epoch'].values
    trainLoss = trainDf['loss'].values

    testEpoch = testDf['epoch'].values
    testLoss = testDf['loss'].values

    N = np.argmax(trainEpoch==1.0)
    trainEpoch = trainEpoch[N-1:]
    trainLoss = np.convolve(trainLoss, np.full(N, 1./N), mode='valid') # smooth
    plt.semilogy(trainEpoch, trainLoss, label=workDir + 'Train')
    plt.semilogy(testEpoch, testLoss, label=workDir + 'Test')
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.xlim(xmin=0,xmax=20)
    # ax.set_ylim(1e-4, 1)
    ax.set_ylim(1e-3, 0.5)


    plt.legend()
    for ext in ['pdf', 'png']:
        f = os.path.join(workDir, "loss."+ext)
        fig.savefig(f)
        print("Created {}".format(f))

    return fig,ax

def plotErr(trainDf, testDf, workDir,fig=None,ax=None):
    if fig is None:
        fig, ax = plt.subplots(1, 1)

    trainEpoch = trainDf['epoch'].values
    trainLoss = trainDf['err'].values

    # for batch size 1 does not make sense to smooth error rate in during training ... it's just 0 or 1
    # N = np.argmax(trainEpoch==1.0)
    # trainEpoch = trainEpoch[N-1:]
    # trainLoss = np.convolve(trainLoss, np.full(N, 1./N), mode='valid')
    # plt.plot(trainEpoch, trainLoss, label=workDir + 'Train')
    plt.plot(testDf['epoch'].values, testDf['err'].values, label=workDir + 'Test')
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.xlim(xmin=0,xmax=20)
    ax.set_ylim(0, 1)

    print(testDf['err'].values)

    plt.legend()
    for ext in ['pdf', 'png']:
        f = os.path.join(workDir, "err."+ext)
        fig.savefig(f)
        print("Created {}".format(f))

    return fig,ax

if __name__ == '__main__':
    main()
