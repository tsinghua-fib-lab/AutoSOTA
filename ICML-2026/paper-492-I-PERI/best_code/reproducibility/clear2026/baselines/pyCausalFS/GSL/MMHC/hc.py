import numpy as np
from scipy.special import gammaln
import copy

def score_diff(gra1, gra2, data, score_function):
    remain = []
    for tar in data:
        if set(gra1[tar]) != set(gra2[tar]):
            remain.append(tar)
            remain.extend(gra1[tar])
            remain.extend(gra2[tar])
    remain = list(set(remain))
    if score_function == 'bdeu':
        sco_diff = score_bdeu_diff(gra1, gra2, data[remain])
    elif score_function == 'bic':
        sco_diff = score_bic_diff(gra1, gra2, data[remain])
    return sco_diff

# compare the bdeu score between gra1 and gra2, a positive value means bdeu(gra2) > bdeu(gra1)
def score_bdeu_diff(gra1, gra2, data):
    iss = 1
    bdeu_diff = 0
    for tar in data:
        if set(gra1[tar]) != set(gra2[tar]):
            q1 = 1
            q2 = 1
            r = len(data[tar].unique())

            if gra1[tar]:
                N_jk = {k: v[tar].value_counts().to_dict() for k, v in data.groupby(gra1[tar])}

                for par_var in gra1[tar]:
                    q1 = q1 * len(data[par_var].unique())
                alp1_j = iss / q1
                alp1_jk = iss / q1 / r

                for key_par in N_jk:
                    for key_tar in N_jk[key_par]:
                        bdeu_diff = bdeu_diff - gammaln(alp1_jk + N_jk[key_par][key_tar]) + gammaln(alp1_jk)
                    bdeu_diff = bdeu_diff - gammaln(alp1_j) + gammaln(alp1_j + sum(N_jk[key_par].values()))
            else:
                alp1_j = iss
                alp1_jk = iss / r
                N_k = data[tar].value_counts().to_dict()
                bdeu_diff = bdeu_diff - gammaln(alp1_j) + gammaln(alp1_j + len(data))

                for key_tar in N_k:
                    bdeu_diff = bdeu_diff - gammaln(alp1_jk + N_k[key_tar]) + gammaln(alp1_jk)
            if gra2[tar]:
                N_jk = {k: v[tar].value_counts().to_dict() for k, v in data.groupby(gra2[tar])}

                for par_var in gra2[tar]:
                    q2 = q2 * len(data[par_var].unique())

                alp2_j = iss / q2
                alp2_jk = iss / q2 / r

                for key_par in N_jk:
                    for key_tar in N_jk[key_par]:
                        bdeu_diff = bdeu_diff + gammaln(alp2_jk + N_jk[key_par][key_tar]) - gammaln(alp2_jk)
                    bdeu_diff = bdeu_diff + gammaln(alp2_j) - gammaln(alp2_j + sum(N_jk[key_par].values()))
            else:
                alp2_j = iss
                alp2_jk = iss / r
                N_k = data[tar].value_counts().to_dict()

                bdeu_diff = bdeu_diff + gammaln(alp2_j) - gammaln(alp2_j + len(data))

                for key_tar in N_k:
                    bdeu_diff = bdeu_diff + gammaln(alp2_jk + N_k[key_tar]) - gammaln(alp2_jk)
    return bdeu_diff

# compare the bic score between gra1 and gra2, a positive value means bdeu(gra2) > bdeu(gra1)
def score_bic_diff(gra1, gra2, data):
    bic_diff = 0
    for tar in data:
        if set(gra1[tar]) != set(gra2[tar]):
            q1 = 1
            q2 = 1
            r = len(data[tar].unique())

            if gra1[tar]:
                N_jk = {k: v[tar].value_counts().to_dict() for k, v in data.groupby(gra1[tar])}

                for par_var in gra1[tar]:
                    q1 = q1 * len(data[par_var].unique())

                for key_par in N_jk:
                    for key_tar in N_jk[key_par]:
                        bic_diff = bic_diff - N_jk[key_par][key_tar] * np.log(N_jk[key_par][key_tar] / sum(N_jk[key_par].values()))

            else:
                N_k = data[tar].value_counts().to_dict()

                for key_tar in N_k:
                    bic_diff = bic_diff - N_k[key_tar] * np.log(N_k[key_tar] / len(data))

            if gra2[tar]:
                N_jk = {k: v[tar].value_counts().to_dict() for k, v in data.groupby(gra2[tar])}

                for par_var in gra2[tar]:
                    q2 = q2 * len(data[par_var].unique())

                for key_par in N_jk:
                    for key_tar in N_jk[key_par]:
                        bic_diff = bic_diff + N_jk[key_par][key_tar] * np.log(N_jk[key_par][key_tar] / sum(N_jk[key_par].values()))
            else:
                N_k = data[tar].value_counts().to_dict()
                for key_tar in N_k:
                    bic_diff = bic_diff + N_k[key_tar] * np.log(N_k[key_tar] / len(data))

            bic_diff = bic_diff + np.log(data.shape[0]) / 2 * (r - 1) * (q1 - q2)
    return bic_diff

def hc(data, pc, score_function):
    # input:
    # data: training data
    # pc: PC set for variables
    # score_function: score function used in hill climbing algorithm
    # output:
    # gra: a dictionary containing variables with their parents

    gra = {}
    gra_temp = {}
    for node in data:
        gra[node] = []
        gra_temp[node] = []

    diff = 1

    # attempt to find better graph until no difference could make
    while diff > 1e-10:

        diff = 0
        edge_candidate = []
        gra_temp = copy.deepcopy(gra)

        cyc_flag = False

        for tar in data:
            # attempt to add edges
            for pc_var in pc[tar]:
                underchecked = [pc_var]
                checked = []
                while underchecked:
                    if cyc_flag:
                        break
                    underchecked_copy = copy.deepcopy(underchecked)
                    for gra_par in underchecked_copy:
                        if gra[gra_par]:
                            if tar in gra[gra_par]:
                                cyc_flag = True
                                break
                            else:
                                for key in gra[gra_par]:
                                    if key not in checked:
                                        underchecked.append(key)
                        underchecked.remove(gra_par)
                        checked.append(gra_par)

                if cyc_flag:
                    cyc_flag = False
                else:
                    gra_temp[tar].append(pc_var)

                    score_diff_temp = score_diff(gra, gra_temp, data, score_function)
                    if (score_diff_temp - diff > -1e-10):
                        diff = score_diff_temp
                        edge_candidate = [tar, pc_var, 'a']

                    gra_temp[tar].remove(pc_var)

            for par_var in gra[tar]:
                # attempt to reverse edges
                gra_temp[par_var].append(tar)
                gra_temp[tar].remove(par_var)
                underchecked = [tar]
                checked = []
                while underchecked:
                    if cyc_flag:
                        break
                    underchecked_copy = copy.deepcopy(underchecked)
                    for gra_par in underchecked_copy:
                        if gra_temp[gra_par]:
                            if par_var in gra_temp[gra_par]:
                                cyc_flag = True
                                break
                            else:
                                for key in gra_temp[gra_par]:
                                    if key not in checked:
                                        underchecked.append(key)
                        underchecked.remove(gra_par)
                        checked.append(gra_par)

                if cyc_flag:
                    cyc_flag = False
                else:
                    score_diff_temp = score_diff(gra, gra_temp, data, score_function)
                    if score_diff_temp - diff > 1e-10:
                        diff = score_diff_temp
                        edge_candidate = [tar, par_var, 'r']

                gra_temp[par_var].remove(tar)

                # attempt to delete edges
                score_diff_temp = score_diff(gra, gra_temp, data, score_function)
                if (score_diff_temp - diff > -1e-10):
                    diff = score_diff_temp
                    edge_candidate = [tar, par_var, 'd']

                gra_temp[tar].append(par_var)

        # print(diff)
        # print(edge_candidate)

        if edge_candidate:
            if edge_candidate[-1] == 'a':
                gra[edge_candidate[0]].append(edge_candidate[1])
                pc[edge_candidate[0]].remove(edge_candidate[1])
                pc[edge_candidate[1]].remove(edge_candidate[0])
            elif edge_candidate[-1] == 'r':
                gra[edge_candidate[1]].append(edge_candidate[0])
                gra[edge_candidate[0]].remove(edge_candidate[1])
            elif edge_candidate[-1] == 'd':
                gra[edge_candidate[0]].remove(edge_candidate[1])
                pc[edge_candidate[0]].append(edge_candidate[1])
                pc[edge_candidate[1]].append(edge_candidate[0])

    dag = {}
    for var in gra:
        dag[var] = {}
        dag[var]['parents'] = gra[var]
        dag[var]['children'] = []

    return dag