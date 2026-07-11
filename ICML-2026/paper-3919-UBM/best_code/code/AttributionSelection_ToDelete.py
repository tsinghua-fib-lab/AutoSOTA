# Select the attribution for ESE
import itertools
import numpy as np
from EquilibriumIndex import equilibrium_index_TED
from EquilibriumParameter import feature_distribution
from StateParameter import state_parameter_set

def select_attribute(num_attribute,attribute,state,train_start,h):
    EI = []
    attribute_list = []
    for i in range(num_attribute):
        for set in itertools.combinations(range(num_attribute), i + 1):  # to generate different attribute combinations
            set = np.array(set)
            attribute_list.append(set)
            for j in range(i + 1):
                x = 0
                a = feature_distribution(attribute[:, j], 1)  # correlates[i])
                x += a
                spss = []
                y = 0
                for p in range(train_start, train_start + h):  # train_start is t_0
                    sps = state_parameter_set(state[i])
                    sps = sps.astype(float)
                    EI += equilibrium_index_TED(x, sps)
                ei_raw = y / h
            EI = EI.append(ei_raw)  # Get the EI value under different combinations

    attribution_selection = attribute_list[EI.index(min(EI))]
    return attribution_selection














