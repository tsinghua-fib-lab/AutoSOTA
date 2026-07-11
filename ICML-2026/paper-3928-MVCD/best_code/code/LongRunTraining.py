# this part is to calculate the equilibrium parameter adjustment matrix
from itertools import count
import numpy as np

from Cointegration import cointegration
from EquilibriumIndex import equilibrium_index_TED
from EquilibriumParameter import equilibrium_state_parameter_set


### Version 1 - Not in use
# def L_parameters(esps, state):
#     no_part = len(esps)
#     #print(no_part)
#     l = np.ones(no_part)
#     #print(no_part)
#     for i in range(len(state)):
#         l = (esps - state[i] + l)/2
#         #print(l1)
#     return l

#  #   elif isinstance(n, (int, float)):
#   #      for i in range(n):
#  #           l = (state[i] - state_p[i] + l) / 2
# #        return l


# print(L_parameters(n=4))

### Version 2 - Not in use
# def long_run_equilibrium(esps_0,spss):
#     esps = esps_0
#     i = 0
#     no_part = len(esps)
#     l = np.ones(no_part)

#     while cointegration(esps,spss):

#         l = (esps - spss[i] + l) / 2

#         esps = (esps_0 + l) / 2

#         i += 1

#     return esps

### long run equilibrium training of esps with parameter l
### Algorithm in the main text
def long_run_equilibrium_l(esps_0,spss,choose):

    #print(esps)
    i = 0
    no_part = len(esps_0)  ### number of parts, esps_0 as the initial esps
    # FIXED: removed "esps_0 = np.zeros(no_part)" which incorrectly overwrote the input initial estimate
    l = np.ones(no_part)   ### initialise all values of vector l to 1

    esps = np.copy(esps_0)  ### start from the initial equilibrium estimate

    if choose == 1:
        while True:
            for i in range(len(spss)):
                l = (esps - spss[i] + l) / 2

            esps = esps - (l / 2)
            # print(esps)

            if cointegration(esps,
                             spss) == False:  ### break if esps and spss are in cointegration! Note: False means the rejection of hypothesis, e.g. not in cointegration is false
                break

            i += 1
    elif choose == 2:
        while True:
            for i in range(len(spss)):
                l = (esps - spss[i] + l) / 2

            esps = esps - (l / 2)
            # print(esps)
            EI = equilibrium_index_TED(spss[-1],esps)

            if EI <  0.1:
                break





    return esps


