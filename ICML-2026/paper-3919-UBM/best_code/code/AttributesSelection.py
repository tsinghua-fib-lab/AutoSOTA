# the function for selecting the attribution
import math

import numpy as np

from EquilibriumIndex import equilibrium_index_TED, equilibrium_index_DI
from EquilibriumParameter import feature_distribution_up_0, equilibrium_state_parameter_set
from Correlation import attribution_correlate_coe   ### Revise package name

def transpose_list(matrix):
    return [list(row) for row in zip(*matrix)]

### using TED function to select the most relevant attributes
def attribution_select(method_choose, attributes, state, list):  #### attribution_select_TED -> attribute_select_TED
    EI = 1  ### Set the initial EI as the maximum value of 1.  The actual EI of an attribute subset would be < 1.
    
    no_part = len(attributes)  ### number of parts --> revise the variable number <<<attributes>>>
    #print(no_part)
    
    alt = subsets(list)  ### list all possible subsets, complexity analysis, it looks like a factorial f(n!)
    #print(alt)
    
    num = len(alt)
    #print(num)


    ### search through all possible subsets to find the one with smallest EI value
    for i in range(1,num):
        selected_attribute = attributes[:,alt[i]]

        number_attribute = len(alt[i])

       # print(alt[i])

        #print(no_attribute)
        
        correlates = attribution_correlate_coe(3, selected_attribute, state[-1])
        sum_cor = sum(abs(correlates))
        x = 0
        for j in range(number_attribute):
            a = feature_distribution_up_0(selected_attribute[:, j], correlates[j])
            #print(sum(a))
            x += a
            #print(x)
        esps = equilibrium_state_parameter_set(sum_cor, number_attribute, no_part, x)
        #print(sum(esps))

        if method_choose == 1:
            EI_value = equilibrium_index_TED(state[-1], esps)  # the EI value calculated by Euclidean distance
        elif method_choose == 2:
            EI_value = equilibrium_index_DI(state[-1], esps)  # the EI value calculated by difference
        else:
            print("Wrong selection!")

        #print(alt[i])

        if EI_value < EI:
            EI = EI_value
            attribute_set = alt[i]
            print(alt[i])
            print(i)
        else:
            EI = EI

    return attribute_set

### using Dis function to select the most relevant attributes
def attribution_select_Dis(attributes, state, list):
    EI = 1
    no_part = len(attributes)
    
    alt = subsets(list)
    #print(alt)
    
    num = len(alt)
    #print(num)
    
    for i in range(1,num):
        at_column = attributes[:,alt[i]]
        no_attribute = len(alt[i])
        #print(no_attribute)
        correlates = attribution_correlate_coe(3, at_column, state[-1])
        sum_cor = sum(abs(correlates))
        x = 0
        for i in range(no_attribute):
            a = feature_distribution_up_0(at_column[:, i], correlates[i])
            #print(sum(a))
            x += a
            #print(x)
        esps = equilibrium_state_parameter_set(sum_cor, no_attribute, no_part, x)
        #print(sum(esps))
        EI_DiS = equilibrium_index_DI(state[-1], esps)
        #print(EI_DiS)
        if EI_DiS < EI:
            EI = EI_DiS
            attributeset = alt[i]
        else:
            EI = EI
            
    return attributeset  ### attributeSet  attribute_set


### ?????? 
def subsets(nums):
    result = [[]]
    for num in nums:
        for element in result[:]:
            x = element[:]
            x.append(num)
            result.append(x)
    return result
