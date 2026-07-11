import numpy as np

# dynamic parameters
# mainly two paramaters, the Gamma is a parameter set which include n parameters (gamma). n is equal the number of parts.
# also, Gamma can be inherited, or set initially that be I or special set
# theta will be the empty set after adjusting Gamma every time.

gamma_set = []  # dynamic parameter set

theta_set = []  # dynamic adjustment parameter set

# other parameters

num_part = 10


# function for dynamic parameter
# to obtain the first Gamma parameter set. The default is initial set and the number of part is 10.
def initial_gamma_set(ini=True, gamma_set=None, num_part=10):
    ini_gamma_set = np.array([1 for _ in range(num_part)])
    if not ini:
        ini_gamma_set = gamma_set
    return ini_gamma_set


# to obtain the dynamic adjustment parameter
def theta_set(state_set, dynamic_equilibrium_state_set):
    # way 1 coefficient changes
    theta_set = state_set / dynamic_equilibrium_state_set

    # way 2 log transformation difference

    # way 2 Integral difference


    return theta_set


def adjust_gamma_set(gamma_set, theta_set):
    adjust_gamma = (gamma_set * theta_set) / 2

    return adjust_gamma


x = np.array([0.1, 0.2, 0.3, 0.4])
y = np.array([0.3, 0.2, 0.2, 0.3])
print(sum(x / y))
