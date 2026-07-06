import copy
def cpdag(dag):
    # convert a DAG to a CPDAG
    cpdag = copy.deepcopy(dag)
    for var in cpdag:
        if (len(cpdag[var]['par']) > 1):
            par_temp = copy.deepcopy(cpdag[var]['par'])
            for par in cpdag[var]['par']:
                par_temp.remove(par)
                for par_oth in par_temp:
                    if ((par in cpdag[par_oth]['par']) & (len(cpdag[par_oth]['par']) == 1)) | (par in cpdag[par_oth]['nei']) | ((par_oth in cpdag[par]['par']) & (len(cpdag[par]['par']) == 1)) | (par_oth in cpdag[par]['nei']):
                        if par not in cpdag[var]['nei']:
                            cpdag[var]['nei'].append(par)
                        if var not in cpdag[par]['nei']:
                            cpdag[par]['nei'].append(var)
                        if par in cpdag[var]['par']:
                            cpdag[var]['par'].remove(par)
                        if par_oth not in cpdag[var]['nei']:
                            cpdag[var]['nei'].append(par_oth)
                        if var not in cpdag[par_oth]['nei']:
                            cpdag[par_oth]['nei'].append(var)
                        if par_oth in cpdag[var]['par']:
                            cpdag[var]['par'].remove(par_oth)
        elif (len(cpdag[var]['par']) != 0):
            par = dag[var]['par'][0]
            while (len(dag[par]['par']) == 1):
                par = dag[par]['par'][0]
            if (len(dag[par]['par']) == 0):
                cpdag[var]['nei'].extend(cpdag[var]['par'])
                if var not in cpdag[cpdag[var]['par'][0]]['nei']:
                    cpdag[cpdag[var]['par'][0]]['nei'].append(var)
                cpdag[var]['par'] = []
    return cpdag