import os
import numpy as np
import sys

local_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(local_path)


from .Utils.CI_test import CI_Test
from .Utils.zhang_orient import orient_rules
from .Utils.MMB_TC_Z import MMB_TC
from .Utils.subsets1 import subsets1


class MMB_by_MMB:
    def __init__(self,Data, target, alpha, test, p, maxK, verbose = False):
        self.test = test
        self.data = Data
        self.target = target
        self.alpha = alpha
        self.p = p
        self.maxK = maxK
        self.verbose = verbose
        self.all_test = 0
        self.adj_test = 0
        self.mmb_test = 0
        self.pag = np.zeros((p, p), dtype=int)     #  1  '>'; -1  '-'; 2  'o'; 0 NULL
        # i --> j  pag[i, j] = 1, pag[j, i] = -1
        # i o-> j  pag[i, j] = 1, pag[j, i] = 2
        # i o-o j  pag[i, j] = 2, pag[j, i] = 2
        # i <-> j  pag[i, j] = 1, pag[j, i] = 1
        self.is_mmb = np.zeros((p, p), dtype=int)  # is: 1 ,not:-1, not know:0
        self.all_adj = np.empty(p, dtype=object)
        self.all_MMB = np.empty(p, dtype=object)
        self.all_not_adj_in_MMB = np.empty(p, dtype=object)
        self.all_sepset = np.empty((p, p), dtype=object)
        for i in range(p):
            self.all_adj[i] = []
            self.all_MMB[i] = []
            self.all_not_adj_in_MMB[i] = []
        self.done_MMB = np.full(p, False, dtype=bool)
        self.find_edge = np.full((p,p), False, dtype=bool)
        self.waitlist = [target]
        self.donelist = []

    def find_mmb(self,A):
        if self.done_MMB[A]:
            raise ValueError('had found MMB of A')
        self.all_MMB[A], ntest1, self.is_mmb = MMB_TC(self.data, A, self.alpha, self.is_mmb)
        not_MMB_A = np.where(self.is_mmb[A, :] == -1)[0]
        #------The nodes in T and the nodes not in T of MMB are mutually independent, given MMB of T.
        for no_mmb in not_MMB_A:
            if self.all_sepset[A, no_mmb] == None and self.all_sepset[no_mmb, A] == None:
                self.all_sepset[A, no_mmb] = self.all_MMB[A]
                self.all_sepset[no_mmb, A] = self.all_MMB[A]
                self.find_edge[A, no_mmb] = True
                self.find_edge[no_mmb, A] = True

        self.all_test += ntest1
        self.mmb_test += ntest1
        self.done_MMB[A] = True
        if self.verbose: print(f"MMB of {A} : {self.all_MMB[A]}, CI test num:{ntest1}")

    def find_adj_from_MMB(self, A):
        # Find adjacent nodes from MMB
        self.all_adj[A] = self.all_MMB[A]
        tmp_adj = self.all_adj[A]
        for B in tmp_adj:
            if self.find_edge[A, B]:
                if self.pag[A, B] == 0 and self.all_sepset[A, B] != None:
                    self.all_adj[A] = np.setdiff1d(self.all_adj[A], B)
                    self.all_not_adj_in_MMB[A].append(B)

        sepSetSize = 0
        while (len(self.all_adj[A])-1) >= sepSetSize and sepSetSize <= self.maxK:
            for B in self.all_MMB[A]:
                if self.find_edge[A, B]:
                    continue
                self.pag[A, B] = 2
                self.pag[B, A] = 2
                tmp_set = np.setdiff1d(self.all_adj[A], B)
                tmp_sepsets = subsets1(tmp_set, sepSetSize)
                for i in range(len(tmp_sepsets)):
                    sepset = tmp_sepsets[i]
                    CI, p_value = CI_Test(B, A, sepset, self.data, self.alpha, self.test)
                    self.all_test += 1
                    self.adj_test += 1
                    if CI:  # CI=True means B and A are independent and not adjacent
                        self.all_sepset[A, B] = sepset
                        self.all_sepset[B, A] = sepset
                        self.pag[A, B] = 0
                        self.pag[B, A] = 0
                        self.find_edge[A, B] = True
                        self.find_edge[B, A] = True
                        self.all_adj[A] = np.setdiff1d(self.all_adj[A], B)
                        self.all_not_adj_in_MMB[A].append(B)
                        break
            sepSetSize += 1
        for B in self.all_MMB[A]:
            if not self.find_edge[A, B]:
                self.find_edge[A, B] = True
                self.find_edge[B, A] = True

        # find S1 V-structure: B1 -> A <- B2
        if len(self.all_adj[A]) >= 2 :
            for i in range(0, len(self.all_adj[A]) - 1):
                B1 = self.all_adj[A][i]
                for j in range(i + 1, len(self.all_adj[A])):
                    B2 = self.all_adj[A][j]
                    if self.pag[B1, A] == 1 and self.pag[B2, A] == 1:
                        continue
                    if self.find_edge[B1, B2] and self.find_edge[B2, B1]:
                        if (self.pag[B1, B2] == 0) and (self.pag[B2, B1] == 0):
                            if A not in self.all_sepset[B1, B2]:
                                self.pag[B1, A] = 1
                                self.pag[B2, A] = 1
                                if self.verbose: print(f"find V-:{B1}*->{A}<-*{B2}")

    def stop_condition_three(self, T, maxdepth=5, done=None, depth=1):

        if depth > maxdepth:
            return False
        if done is None and depth == 1:
            done = list(np.where((self.pag[T, :] != 2) & (self.pag[:, T] != 2) & (self.pag[T, :] != 0) & (self.pag[:, T] != 0))[0])
        if T not in done: done.append(T)
        adj_T = np.where(self.pag[T, :] != 0)[0]
        adj_T = np.setdiff1d(adj_T, done)
        adj_T = list(adj_T)
        if len(adj_T) == 0:
            if T in self.donelist:
                return True
            else:
                return False

        all_break_T = np.full(len(adj_T), False, dtype=bool)

        for i in range(len(adj_T)):
            if self.pag[T, adj_T[i]] == 1:
                all_break_T[i] = True

        if np.all(all_break_T):  # If the current adj node has been blocked, all paths of T passing through the adj will be blocked.
            return True

        else:
            for i in range(len(adj_T)):
                if not all_break_T[i]:
                    all_break_T[i] = self.stop_condition_three(int(adj_T[i]),  done=done, depth=depth+1)

        if np.all(all_break_T):
            return True
        else:
            return False

    def find_trueV_structure(self,A):
        if not self.done_MMB[A]:
            raise ValueError('not found MMB of A')
        # find S2 V-structure:A -> B <- C
        for C in self.all_not_adj_in_MMB[A]:
            for B in self.all_adj[A]:
                if self.pag[A, B] == 1 and self.pag[C, B] == 1:
                    continue
                if not self.find_edge[B, C]:
                    if not self.done_MMB[B]:
                        self.find_mmb(B)
                        self.find_adj_from_MMB(B)
                        adj = (self.pag[B, C] != 0)
                    else:
                        if C not in self.all_MMB[B]:
                            adj = False
                        else:
                            adj = (self.pag[B, C] != 0)

                else:
                    adj = (self.pag[B, C] != 0)
                if self.pag[A, B] == 1 and self.pag[C, B] == 1:
                    continue
                if (B not in self.all_sepset[A, C]) and (adj):
                    self.pag[A, B] = 1
                    self.pag[C, B] = 1
                    if self.pag[B, C] == 0:
                        self.pag[B, C] = 2
                    self.find_edge[B, C] = True
                    self.find_edge[C, B] = True
                    if self.verbose:print(f"find V- in target({A}) :{A}*->{B}<-*{C}")

    def mmb_by_mmb(self,meek_verbose = False):
        num_calculated = 0
        while len(self.donelist) <= self.p and len(self.waitlist) > 0:  # stop two

            A = self.waitlist.pop(0)
            if A in self.donelist:
                continue
            else:
                self.donelist.append(A)
            if not self.done_MMB[A]:
                self.find_mmb(A)
                self.find_adj_from_MMB(A)

            for B in self.all_MMB[A]:
                if B not in self.waitlist and B not in self.donelist:
                    self.waitlist.append(B)

            # find true V-structures S2
            self.find_trueV_structure(A)
            self.pag = orient_rules(self.pag,self.all_sepset,self.find_edge,detail_out=meek_verbose)
            num_calculated += 1
            if num_calculated > len(self.all_adj[self.target]):
                if 2 not in self.pag[self.target, :] and 2 not in self.pag[:, self.target]:  # stop one
                    if self.verbose: print("stop one")
                    break
                if self.stop_condition_three(self.target):
                    if self.verbose: print("stop three")
                    break

        P = np.where(np.logical_and(self.pag[:, self.target] == 1, self.pag[self.target, :] == -1))[0]
        C = np.where(np.logical_and(self.pag[self.target, :] == 1, self.pag[:, self.target] == -1))[0]
        dis_depth1 = np.where(np.logical_and(self.pag[self.target, :] == 1, self.pag[:, self.target] == 1))[0]
        un = np.where((self.pag[self.target, :] == 2) | (self.pag[:, self.target] == 2))[0]
        ci_test = self.all_test
        
        # For our case : 
        parents = np.where((self.pag[:, self.target] == 1))[0]
        bidir = np.where(np.logical_and(self.pag[self.target, :] == 1, self.pag[:, self.target] == 1))[0]
        bicircle = np.where(np.logical_and((self.pag[self.target, :] == 2), (self.pag[:, self.target] == 2)))[0]
        
        ci_test = self.all_test
        
        return parents, bidir, bicircle, ci_test
    
    