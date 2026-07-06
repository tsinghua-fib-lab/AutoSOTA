import numpy as np
# TAIL = -1
# ARROW = 1
# CIRCLE = 2
# NULL = 0
def updateList(path, set, old_list):  # arguments are all lists
    temp = []
    if len(old_list) > 0:
        temp.extend(old_list)
    temp.extend([path + [s] for s in set])
    return temp

def minDiscPath(pag, a, b, c):

    p = pag.shape[0]
    visited = np.full(p,False,dtype=bool)
    visited[a] = visited[b] = visited[c] = True
    indD = list(np.where((pag[a, :] != 0) & (pag[:, a] == 1))[0])
    for indd in indD:
        if visited[indd] == True:
            indD.remove(indd)

    if len(indD) > 0:
        path_list = updateList([a], indD, [])
        while len(path_list) > 0:
            mpath = path_list[0]
            d = mpath[-1]
            if pag[c][d] == 0 and pag[d][c] == 0:
                mpath.reverse()
                mpath.extend([b, c])
                return mpath
            else:
                pred = mpath[-2]
                path_list = path_list[1:len(path_list)]
                visited[d] = True
                if pag[d][c] == 1 and pag[c][d] == -1 and pag[pred][d] == 1:
                    indR = list(np.where((pag[d, :] != 0) & (pag[:, d] == 1))[0])
                    for indr in indR:
                        visited[indr] = False
                    if len(indR) > 0:
                        path_list = updateList(mpath, indR, path_list)
    return []

def orient_rules(pag, sepset,find_edge,rules = np.full(4,True,dtype=bool), detail_out = True):
    #  1  '>'
    # -1  '-'
    #  2  'o'
    #  0   NULL
    p = pag.shape[0]
    #  R1 - R4
    old_pag = np.zeros((p, p))
    while not np.array_equal(pag, old_pag):
        old_pag = np.copy(pag)
        #--------------R1-------------------------------------------
        if rules[0]:
            #i *-> j
            inds = np.argwhere((pag == 1) & (np.transpose(pag) != 0))
            for ind in inds:
                i = ind[0]
                j = ind[1]
                Ks = np.where((pag[j, :] != 0) & (pag[:, j] == 2) & (pag[i, :] == 0) & (pag[:, i] == 0))[0]
                if len(Ks) > 0:
                    for k in Ks:
                        if not (pag[i, j] == 1 and pag[k, j] == 2 and pag[i, k] == 0 and pag[k, i] == 0):
                            raise ValueError('find error in rule1')
                        if find_edge[i, k] and pag[i, k] == 0 and pag[k, i] == 0 and (j in sepset[i, k]):
                            pag[j, k] = 1
                            pag[k, j] = -1
                            if detail_out:
                                print(f'Rule 1 \n Orient: {i} *-> {j} o-* {k} as: {j} -> {k}')


        # --------------R2-------------------------------------------
        if rules[1]:
            #i *-o k
            inds = np.argwhere((pag == 2) & (np.transpose(pag) != 0))
            for ind in inds:
                i = ind[0]
                k = ind[1]
                Js = np.where(((pag[i, :] == 1) & (pag[:, i] == -1) & (pag[:, k] == 1) & (pag[k, :] != 0)) | ((pag[i, :] == 1) & (pag[:, i] != 0) & (pag[:, k] == 1) & (pag[k, :] == -1)))[0]
                if len(Js) > 0:
                    pag[i, k] = 1
                    for j in Js:
                        if not ((pag[i, j] == 1 and pag[j, i] == -1 and pag[j, k] == 1) or (pag[j, k] == 1 and pag[k, j] == -1 and pag[i, j] == 1)):
                            raise ValueError('find error in rule2')
                    if detail_out:
                        print(f'Rule 2  \n Orient: {i} -> {Js} *-> {k} or {i} *-> {Js} -> {k} with {i} *-o {k} as: {i} *-> {k} ')

        # --------------R3-------------------------------------------
        if rules[2]:
            #j o-* l
            inds = np.argwhere((pag != 0) & (np.transpose(pag) == 2))
            for ind in inds:
                j = ind[0]
                l = ind[1]
                indIK = np.where((pag[j, :] != 0) & (pag[:, j] == 1) & (pag[:, l] == 2) & (pag[l, :] != 0))[0]
                if len(indIK) >= 2:
                    counter = 0
                    while ( (counter < len(indIK) - 1) and (pag[l, j] != 1)):
                        ii = counter + 1
                        while ((ii < len(indIK)) and (pag[l, j] != 1) ):
                            if (pag[indIK[counter], indIK[ii]] == 0) and (pag[indIK[ii], indIK[counter]] == 0) and find_edge[indIK[ii], indIK[counter]] and find_edge[indIK[counter], indIK[ii]] and (l in sepset[indIK[ii], indIK[counter]]) :
                                pag[l, j] = 1
                                if detail_out:
                                    print(f'Orienting edge {l}*-o{j}  to {l}*->{j} with rule 3')
                            ii += 1
                        counter += 1
        # --------------R4-------------------------------------------
        if rules[3]:
            # j o-* k
            inds = np.argwhere((pag != 0) & (np.transpose(pag) == 2))
            while(len(inds) > 0):
                j = inds[0, 0]
                k = inds[0, 1]

                inds = np.delete(inds, 0,axis=0)
                #find all i -> k and i <-* j
                indI = np.where((pag[j, :] == 1) & (pag[:, j] != 0) & (pag[k, :] == -1) & (pag[:, k] == 1))[0]

                while(len(indI) > 0 and pag[k, j] == 2) :
                    i = indI[0]

                    indI = np.delete(indI, 0)
                    done = False
                    while done == False and pag[i][j] != 0 and pag[i][k] != 0 and pag[j][k] != 0:
                        md_path = minDiscPath(pag, i, j, k)
                        if len(md_path) == 0:
                            done = True

                        else:
                            # a path exists
                            if find_edge[md_path[0],md_path[-1]]:
                                if (j in sepset[md_path[0]][md_path[-1]]) or (j in sepset[md_path[-1]][md_path[0]]):
                                    pag[j][k] = 1
                                    pag[k][j] = -1
                                    if detail_out:print(f'Orienting edge {k} *-o{j} to  {k}-->{j} with rule 4')
                                else:
                                    pag[i][j] = pag[j][k] = pag[k][j] = 1
                                    if detail_out:print(f'Orienting edge {i}<->{j}<->{k} with rule 4')
                            done = True

    return pag







