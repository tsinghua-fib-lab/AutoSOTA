# Equilibrium status evaluation tool
import math
import numpy as np


# import torch


# equilibrium status evaluate for dynamic by Euclidean
def equilibrium_index_dynamic(test, train, length):
    if train.shape[1] != range(len(test)):
        print("Wrong input!")
    else:
        x = test
        h = length
        y = train[-h:, :]
        sed = 0
        for i in range(train.shape[0] - h, train.shape[0]):
            ed = np.sqrt(np.sum(np.square(x[0, :] - y[i, :])))
            sed = sed + ed
        esi = (2 / math.pi) * math.atan(math.log(sed / h))
        return esi


# equilibrium status evaluate for static by Euclidean
def equilibrium_index_static_Euclidean(data):
    x = data
    h = x.shape[0]
    esi_list = []

    for i in range(data.shape[0]):
        test = x[i]
        train = np.delete(x, i, axis=0)
        sed = 0
        for j in range(train.shape[0]):
            ed = np.sqrt(np.sum(np.square(train[j, :] - test)))
            sed = sed + ed

        index = (2 / math.pi) * (math.atan(math.log(train.shape[1]) * (sed / np.sqrt(2)) / (h - 1)))
        # index = (2 / math.pi) * (math.atan((sed / np.sqrt(2)) / (h - 1)))
        esi_list.append(index)

    esi = np.sum(esi_list) / h

    return esi


'''
def equilibrium_index_static_Euclidean_torch(y, x, size):

    h = y.size()
    esi_list = []

    for i in range(size):
        test = y[i]
        train = x[i]
        sed = 0
        for j in range(size):
            ed = np.sqrt(np.sum(np.square(torch.abs(train[j] - test))))
            sed = sed + ed


        index = (2 / math.pi) * (math.atan(math.log(train.shape[1]) * (sed / np.sqrt(2)) / (h - 1)))
        #index = (2 / math.pi) * (math.atan((sed / np.sqrt(2)) / (h - 1)))
        esi_list.append(index)

    esi = np.sum(esi_list) / h

    return esi
'''


# equilibrium status evaluate for dynamic by difference

def equilibrium_index_static_difference(data):
    x = data
    h = x.shape[0]
    esi_list = []

    for i in range(data.shape[0]):
        test = x[i]
        train = np.delete(x, i, axis=0)
        sd = 0
        for j in range(train.shape[0]):
            d = np.sum(abs(train[j, :] - test))
            sd = sd + d

        index = sd / (2 * (h - 1))
        esi_list.append(index)

    esi = np.sum(esi_list) / h

    return esi


# equilibrium status evaluate for dynamic by Cosine
def equilibrium_index_static_cosine(data):
    x = data
    h = x.shape[0]
    esi_list = []

    for i in range(data.shape[0]):
        test = x[i]
        train = np.delete(x, i, axis=0)
        sc = 0
        for j in range(train.shape[0]):
            cos = (cos_dist(train[j, :], test)) ** 10  # Magnify different
            sc = sc + cos

        index = sc / (h - 1)
        esi_list.append(index)

    esi = 1 - (np.sum(esi_list) / h)

    return esi


def cos_dist(a, b):
    part_up = 0.0
    a_sq = 0.0
    b_sq = 0.0
    for a1, b1 in zip(a, b):
        part_up += a1 * b1
        a_sq += a1 ** 2
        b_sq += b1 ** 2
    part_down = math.sqrt(a_sq * b_sq)
    if part_down == 0.0:
        return None
    else:
        return part_up / part_down


def equilibrium_index_DI(state, esps):
    DI = np.sum(np.abs(state - esps)) / 2

    return DI


def equilibrium_index_TED(state, esps):
    euclidean = np.sqrt(np.sum(np.square(state - esps)))

    ted = np.sqrt(euclidean / np.sqrt(2))

    return ted
