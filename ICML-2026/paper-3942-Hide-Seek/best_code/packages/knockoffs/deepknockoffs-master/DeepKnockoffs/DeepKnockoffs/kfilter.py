import numpy as np

def kfilter(W, offset=1.0, q=0.1):
    """
    Adaptive significance threshold with the knockoff filter
    :param W: vector of knockoff statistics
    :param offset: equal to one for strict false discovery rate control
    :param q: nominal false discovery rate
    :return a threshold value for which the estimated FDP is less or equal q
    """
    t = np.insert(np.abs(W[W!=0]),0,0)
    t = np.sort(t)
    ratio = np.zeros(len(t));
    for i in range(len(t)):
        ratio[i] = (offset + np.sum(W <= -t[i])) / np.maximum(1.0, np.sum(W >= t[i]))
        
    index = np.where(ratio <= q)[0]
    if len(index)==0:
        thresh = float('inf')
    else:
        thresh = t[index[0]]
       
    return thresh
