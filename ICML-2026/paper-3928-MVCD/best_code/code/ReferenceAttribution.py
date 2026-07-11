from scipy.stats import pearsonr

from EquilibriumParameter import feature_distribution, equilibrium_state_parameter_set_distribution


# degree normolly set over 0.5
def reference_attribute(degree, state, attributes, correlations):
    ra = []
    ra_c = []
    nonra = []
    nonra_c = []
    for i in range(attributes.shape[0]):
        r_row, p_value = pearsonr(state, attributes[i])
        if r_row > degree & p_value <= 0.05:
            ra.append(attributes[i])
            ra_c.append(correlations[i])
        else:
            nonra.append(attributes[i])
            nonra_c.append(correlations[i])

    return ra, ra_c, nonra, nonra_c


def reference_parameter(degree, state, attributes, correlations):
    ra, ra_c, nonra, nonra_c = reference_attribute(degree, state, attributes, correlations)
    rp = []
    for i in range(attributes.shape[1]):
        at = feature_distribution(ra[i, :], ra_c[i])
        rp += at

    return rp
