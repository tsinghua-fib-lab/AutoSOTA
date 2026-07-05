# coding=utf-8
# /usr/bin/env python
"""
date: 2019/7/23 16:44
desc: 
"""

import numpy as np


def realMB( kVar, path):
    graph = np.zeros((kVar,kVar))
    parents = [[] for i in range(kVar)]
    children = [[] for i in range(kVar)]
    MB = [[] for i in range(kVar)]
    PC = [[] for i in range(kVar)]
    spouses = [[] for i in range(kVar)]

    i = 0
    with open(path) as fileobject:
        for line in fileobject:
            a = line.split("  ")
            j = 0
            for n in a :
                graph[i,j] = n
                j += 1
            i += 1

    for m in range(kVar):
        parents[m] = [i for i in range(kVar) if graph[i][m] == 1]
        children[m] = [ i for i in range(kVar) if graph[m][i] == 1]

        PC[m] = list(set(parents[m]).union(set(children[m])))

    for m in range(kVar):
        for child in children[m]:
            spouse = parents[int(child)]
            spouses[m] = list(set(spouses[m]).union(set(spouse)))
        if m in spouses[m]:
            spouses[m].remove(m)

    for m in range(kVar):
        MB[m] = list(set(PC[m]).union(set(spouses[m])))

    return MB, PC