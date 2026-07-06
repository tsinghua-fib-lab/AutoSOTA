# _*_code:utf_8_*_
#!/usr/bin/env python
# date:2019/9/2 16:56
from baselines.pyCausalFS.LSL.MBs.common.condition_independence_test import cond_indep_test


def CausalSearch(
        Data,
        T,
        PCT,
        Z,
        IDT,
        alaph,
        idT3,
        idT3_count,
        idT4,
        idT4_count,
        is_discrete):
    num_ci = 0
    # step 1:Single PC
    if len(PCT) == 1:
        IDT[T, PCT[0]] = 3

    # step 2:Check C2 & C3
    for i in range(len(PCT)):
        for j in range(len(PCT)):
            if i != j:
                x = PCT[i]
                y = PCT[j]
                if x in Z or y in Z:
                    continue
                # print("X is: ",x," y is: ",y," Z is: ", Z)
                pval, _ = cond_indep_test(Data, x, y, Z, is_discrete)
                num_ci += 1
                condition_vars = [i for i in Z]
                condition_vars.append(T)
                condition_vars = sorted(set(condition_vars))
                pval2, _ = cond_indep_test(Data, x, y, condition_vars, is_discrete)
                num_ci += 1
                if pval > alaph and pval2 <= alaph:
                    IDT[T, x] = 1
                    IDT[T, y] = 1
                elif pval <= alaph and pval2 > alaph:
                    if IDT[T, x] == 1:
                        IDT[T, y] = 2
                    elif IDT[T, y] != 2:
                        IDT[T, y] = 3
                    if IDT[T, y] == 1:
                        IDT[T, x] = 2
                    elif IDT[T, x] != 2:
                        IDT[T, x] = 3
                    # add(X,Y)to pairs with idT=3
                    idT3_count += 1
                    idT3.append([x, y])
                else:
                    if (IDT[T, x] == 0 and IDT[T, y] == 0) or (
                            IDT[T, x] == 4 and IDT[T, y] == 4):
                        IDT[T, x] = 4
                        IDT[T, y] = 4
                    # add(X,Y) to pairs with idT=4
                    idT4_count += 1
                    idT4.append([x, y])

    # step 3:identify idT=3 pairs with known parents
    for i in range(len(PCT)):
        x = PCT[i]
        if IDT[T, x] == 1:
            for j in range(idT3_count):
                if idT3[j][0] == x:
                    y = idT3[j][1]
                    IDT[T, y] = 2
                elif idT3[j][1] == x:
                    y = idT3[j][0]
                    IDT[T, y] = 2
    return IDT, idT3, idT3_count, idT4, idT4_count, num_ci
