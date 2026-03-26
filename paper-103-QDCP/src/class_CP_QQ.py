import numpy as np
from scipy.stats import binom
import scipy.special as sc

class Boucle:
    def __init__(self, i0, it):
        self.i0 = i0
        self.it = it
        self.reset()

    def reset(self):
        self.ic = self.i0.copy()
        l = len(self.ic) - 1
        self.ic[l] = self.i0[l] - 1

    def next(self):
        der = len(self.ic) - 1
        for i in reversed(range(len(self.ic))):
            self.ic[i] += 1
            if (self.ic[i] <= self.it[i]):
                break
            self.ic[i] = self.i0[i]
        return self.ic
    
    def hasNext(self):
        n = 0
        for i in range(len(self.ic)):
            if (self.ic[i] == self.it[i]):
                n += 1
        return n != len(self.ic)


class Multi_Boucle:
    def __init__(self, i0, it):
        self.i0 = i0
        self.it = it
        
        self.reset()

    def reset(self):
        self.ic = self.i0.copy()
        l = len(self.ic) - 1
        self.ic[l] = self.i0[l] - 1

    def next(self):
        for i in reversed(range(len(self.ic))):
            self.ic[i] += 1
            if (self.ic[i] <= self.it[i]):
                break
            else:
                self.ic[i] = self.i0[i]
        return self.ic
    
    def hasNext(self):
        n = 0
        for i in range(len(self.ic)):
            if (self.ic[i] == self.it[i]):
                n += 1
        return n != len(self.ic)
    
def sum_hypergeo(aa, bb, n, t, epsi=0):
    m = len(aa)
    # ==========================
    rv1 = binom(n, t)
    prob = []
    for i in range(m):
        prob.append(rv1.cdf(bb[i]) - rv1.cdf(aa[i]-1))

    r1 = 1.
    for i in range(m):
        r1 *= prob[i]
    # ==========================

    # ==========================
    rv2 = binom(m*n, t)
    # ==========================
    
    # ==========================
    gen = []
    for i in range(m):

        # tt = 1 - ( rv1.cdf(aa[i]-1) + (1 - rv1.cdf(bb[i])) )
        tt = rv1.cdf(bb[i]) - rv1.cdf(aa[i]-1)

        pp = []
        for j in np.arange(n, -1, -1):
            if j<=bb[i] and j>=aa[i]:
                pp.append( rv1.pmf(j)/tt  )
            else:
                pp.append(0)
        gen.append(np.poly1d(pp))

    poly = gen[0]
    for i in range(1, len(gen)):
        poly = np.polymul(poly, gen[i])
    # ==========================

    s = 0
    for l in range(np.sum(aa), np.sum(bb) + 1):
        r2 = rv2.pmf(l)

        if r2>epsi:
            # ==========================
            r3 = poly[l]
            # ==========================
            s += r3*r1/r2
    return s

def calc_matrix_M(m, n, t, epsi=.0, mid=False):

    A = np.zeros((m, n))
    for aa in np.arange((m//2)*mid, m):
        for bb in np.arange((n//2)*mid, n):
            
            r = aa+1
            k = bb+1

            s = 0
            for j in range(r, m+1):
                a_range = np.ones(m, dtype=int)*k
                a_range[j:] = 0
                b_range = np.ones(m, dtype=int)*n
                b_range[j:] = k-1
                s += sum_hypergeo(a_range, b_range, n, .5, epsi)*sc.binom(m, j)

            A[aa, bb] = 1-s/(m*n+1)
    return(A)