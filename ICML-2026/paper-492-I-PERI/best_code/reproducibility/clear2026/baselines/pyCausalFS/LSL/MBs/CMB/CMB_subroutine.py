# _*_code:utf_8_*_
#!/usr/bin/env python
# date:2019/9/5 20:41
import numpy as np
from baselines.pyCausalFS.CBD.MBs.HITON.HITON_PC import HITON_PC
from baselines.pyCausalFS.CBD.MBs.HITON.HITON_MB import HITON_MB
from baselines.pyCausalFS.LSL.MBs.CMB.CausalSearch import CausalSearch


def CMB_subroutine(Data, T, alaph, IDT, already_calculated_MB, all_MB, is_discrete):

    # already_calculated_MB[T] = 0
    Z = []
    idT3 = []
    idT3_count = 0
    idT4 = []
    idT4_count = 0
    num_ci = 0
    PCT, _, n_c = HITON_PC(Data, T, alaph, is_discrete)
    num_ci += n_c
    IDT, idT3, idT3_count, idT4, idT4_count, n_c1 = CausalSearch(
        Data, T, PCT, Z, IDT, alaph, idT3, idT3_count, idT4, idT4_count, is_discrete)
    num_ci += n_c1
    # step 2:further test variables with idT=4
    for i in range(idT4_count):
        x = idT4[i][0]
        y = idT4[i][1]
        if already_calculated_MB[x] == 1:
            all_MB[x], n_c2 = HITON_MB(Data, x, alaph, is_discrete)
            num_ci += n_c2

            already_calculated_MB[x] = 0
        Z = []
        if x in all_MB.keys():
            Z = [i for i in all_MB[x] if i != T and i != y]
        IDT, idT3, idT3_count, idT4, idT4_count, n_c3 = CausalSearch(
            Data, T, PCT, Z, IDT, alaph, idT3, idT3_count, idT4, idT4_count, is_discrete)
        num_ci += n_c3
        if 4 not in IDT:
            break
    parents = [idx for idx, i in enumerate(IDT[T]) if i == 1]
    for i in range(len(parents)):
        x = parents[i]
        for j in range(len(parents)):
            if j != i:
                y = parents[j]
                for k in range(idT4_count):
                    if idT4[k][0] == x:
                        z = idT4[k][1]
                        for l in range(idT4_count):
                            if l != k:
                                if (idT4[l][0] == y and idT4[l][1] == z) or (idT4[l][0] == z and idT4[l][1] == y):
                                    IDT[T, z] = 1
                    elif idT4[k][1] == x:
                        z = idT4[k][0]
                        for l in range(idT4_count):
                            if l != k:
                                if (idT4[l][0] == y and idT4[l][1] == z) or (idT4[l][0] == z and idT4[l][1] == y):
                                    IDT[T, z] = 1
    for idx, i in enumerate(IDT[T]):
        if i == 4:
            IDT[T, idx] = 3

    return IDT, idT3, idT3_count, PCT, num_ci

    # step 3:resolve variable set with idT=3
