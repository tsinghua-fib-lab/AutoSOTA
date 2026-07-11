# Here, we can calculate equilibrium parameter by
# using attribution parameter
import numpy as np
from sklearn.linear_model import LinearRegression


def equilibrium_parameter(part_coeffs, part, whole):
    part_col_total = [0 for _ in range(part.shape[0])]
    for rows in range(part.shape[0]):
        # Save the current sum of each columns
        part_col_total = [a + x_sum_part for a, x_sum_part in zip(a, part_col_total)]

    ep = [0 for _ in range(len(part_coeffs))]
    for i in range(len(part_coeffs)):
        ep[i] = (part_coeffs[i] * part_col_total[i]) / whole

    return ep


## for covid
def equilibrium_state_parameter_set_distribution(attributes, correlations):
    sumDistribution = 0
    for i in range(attributes.shape[1]):
        at = feature_distribution(attributes[i, :], correlations[i])
        sumDistribution += at

    espsDistribution = ((sumDistribution / attributes.shape[1]) + 1) / attributes.shape[0]

    return espsDistribution

def equilibrium_state_parameter_set_1(sumfat,no_attribution):

    fat_mean = sumfat / no_attribution
    fat_transform = fat_mean - np.min(fat_mean)
    esps = fat_transform / np.sum(fat_transform)

    return esps


def equilibrium_state_parameter_set(sum_cor,no_attribution,no_part,sum_feat):

    y = (np.ones(no_part))/no_part
    #print(y)
    x = (sum_feat / sum_cor) / no_attribution
    #print(max((x+1)/2))

    #esps = y * (1+x)
    #print(sum(z))


    esps = ((x - np.min(x)) / abs(np.sum(np.min(x)))) / no_part
    #print(esps-z)

    return esps



####
def feature_distribution(attribute, correlation):
    maxAttr = np.max(attribute)
    minAttr = np.min(attribute)

    diff = attribute - np.mean(attribute)
    featureDistribution = (diff / (maxAttr - minAttr)) * correlation

    return featureDistribution



def feature_distribution_up_0(attribute, correlation): # when all attributes large then 0
    maxAttr = np.max(attribute)
    minAttr = np.min(attribute)

    diff = attribute - np.mean(attribute)
    featureDistribution = (diff / (maxAttr - 0)) * correlation

    return featureDistribution