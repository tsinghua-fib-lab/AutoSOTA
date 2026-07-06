# SPDX-FileCopyrightText: 2022 deanbodenham <https://github.com/deanbodenham>
# SPDX-FileCopyrightText: 2025 Pôle d'Expertise de la Régulation Numérique <contact.peren@finances.gouv.fr>
#
# SPDX-License-Identifier: MIT

from scipy.stats import gamma
from scipy.stats import f
import numpy as np
import warnings

# methods for checking input
from .utilities import checkCoeffAllPositive
from .utilities import checkXAllPositive
from .utilities import getCoeffError
from .utilities import getXError

# for lpb4, there are several methods in utilities
from .utilities import sum_of_powers
from .utilities import update_moments
from .utilities import get_moments_from_cumulants
from .utilities import get_cumulant_vec_vectorised
from .utilities import get_weighted_sum_of_chi_squared_moments
from .utilities import get_lambdatilde_1
from .utilities import deltaNmat_applied
from .utilities import det_deltaNmat
from .utilities import get_lambdatilde_p
from .utilities import get_base_vector
from .utilities import get_ith_coeff_of_Stilde_poly
from .utilities import get_Stilde_poly_coeff
from .utilities import get_VDM_b_vec
from .utilities import get_vandermonde
from .utilities import get_real_poly_roots
from .utilities import gen_and_solve_VDM_system
from .utilities import get_mixed_pval_vec


# Hall-Buckley-Eagleson
def hbe(coeff, x):
    '''Hall-Buckley-Eagleson method

    Computes the cdf of a positively-weighted sum of chi-squared
    random variables with the Hall-Buckley-Eagleson (HBE) method.


    Parameters:
    -----------
    coeff (list or numpy array): The coefficient vector. 
                                 All values must be greater than 0.

    x (list or numpy array or float): The vector of quantile values. 
                                      All values must be greater than 0.


    Returns:
    --------
    The cdf of the x value(s). It is returned as the same type as x, 
    i.e. if x is a list, it is returned as a list; if x is a numpy array
    it is returned as a numpy array.


    Details:
    --------
     * Note that division assumes Python 3, so may not work with Python 2.
     * Depends on numpy libary for the arrays.
     * If lists are passed, they are converted to numpy arrays (and back again).
     * Depends on scipy library for scipy.stats.gamma function.


    Examples:
    ---------
    #Examples taken from Table 18.6 in N. L. Johnson, S. Kotz, N. Balakrishnan.
    #Continuous Univariate Distributions, Volume 1, John Wiley & Sons, 1994.

    # how to load the hbe function from momenthchi2
    from momentchi2 import hbe

    # should give value close to 0.95, actually 0.94908
    hbe([1.5, 1.5, 0.5, 0.5], 10.203)            

    # should give value close to 0.05, but is actually 0.02857
    hbe([1.5, 1.5, 0.5, 0.5], 0.627)            

    # specifying parameters
    hbe(coeff=[1.5, 1.5, 0.5, 0.5], x=10.203)            

    # x is a list, output approx. 0.05, 0.95
    hbe([1.5, 1.5, 0.5, 0.5], [0.627, 10.203])  

    # instead of lists can be numpy arrays 
    # (any list is converted to a numpy arrays inside the function anyway)
    import numpy as
    hbe( np.array([1.5, 1.5, 0.5, 0.5]), np.array([0.627, 10.203]) )  


    References:
    -----------
     * P. Hall. Chi squared approximations to the distribution of a
       sum of independent random variables. The Annals of
       Probability, 11(4):1028-1036, 1983.

     * M. J. Buckley and G. K. Eagleson. An approximation to the
       distribution of quadratic forms in normal random variables.
       Australian Journal of Statistics, 30(1):150-159, 1988.

     * D. A. Bodenham and N. M. Adams. A comparison of efficient 
       approximations for a weighted sum of chi-squared random variables. 
       Statistics and Computing, 26(4):917-928, 2016.
    '''
    # some checking, so that passing lists/arrays does not matter
    if isinstance(coeff, list):
        coeff = np.array(coeff)

    isList = False
    if not isinstance(x, float):
        if isinstance(x, list):
            isList = True
            x = np.array(x)

    # checking values of coeff and x and throwing errors 
    if not checkCoeffAllPositive(coeff):
        raise Exception(getCoeffError(coeff))

    if not checkXAllPositive(x):
        raise Exception(getXError(x))


    # the next two lines work for floats or numpy arrays, but not lists
    K_1 = sum(coeff)
    K_2 = 2 * sum(coeff**2)
    K_3 = 8 * sum(coeff**3)
    nu = 8 * (K_2**3) / (K_3**2)
    k = nu / 2
    theta = 2
    # in the next line x can be a float or numpy array, but not a list
    x = (  (2 * nu / K_2)**(0.5)  ) * (x - K_1) + nu 
    p = gamma.cdf(x, a=k, scale=theta)

    # if x was passed as a list, will return a list
    if isList:
        p = p.tolist()
    return(p)



# Satterthwaite-Welch
def sw(coeff, x):
    '''Satterthwaite-Welch method

    Computes the cdf of a positively-weighted sum of chi-squared
    random variables with the Satterthwaite-Welch (SW) method.


    Parameters:
    -----------
    coeff (list or numpy array): The coefficient vector. 
                                 All values must be greater than 0.

    x (list or numpy array or float): The vector of quantile values. 
                                      All values must be greater than 0.


    Returns:
    --------
    The cdf of the x value(s). It is returned as the same type as x, 
    i.e. if x is a list, it is returned as a list; if x is a numpy array
    it is returned as a numpy array.


    Details:
    --------
     * Note that division assumes Python 3, so may not work with Python 2.
     * Depends on numpy libary for the arrays.
     * If lists are passed, they are converted to numpy arrays (and back again).
     * Depends on scipy library for scipy.stats.gamma function.


    Examples:
    ---------
    #Examples taken from Table 18.6 in N. L. Johnson, S. Kotz, N. Balakrishnan.
    #Continuous Univariate Distributions, Volume 1, John Wiley & Sons, 1994.

    # how to load the sw function from momenthchi2
    from momentchi2 import sw

    # should give value close to 0.95, actually 0.95008
    sw([1.5, 1.5, 0.5, 0.5], 10.203)            

    # should give value close to 0.05, but is actually 0.0657
    sw([1.5, 1.5, 0.5, 0.5], 0.627)            

    # specifying parameters
    sw(coeff=[1.5, 1.5, 0.5, 0.5], x=10.203)            

    # x is a list, output approx. 0.05, 0.95
    sw([1.5, 1.5, 0.5, 0.5], [0.627, 10.203])  

    # instead of lists can be numpy arrays 
    # (any list is converted to a numpy arrays inside the function anyway)
    import numpy as np
    sw( np.array([1.5, 1.5, 0.5, 0.5]), np.array([0.627, 10.203]) )  


    References:
    -----------
     * B. L.Welch. The significance of the difference between two
       means when the population variances are unequal.
       Biometrika, 29(3/4):350-362, 1938.

     * F. E. Satterthwaite. An approximate distribution of estimates
       of variance components. Biometrics Bulletin, 2(6):110-114,

     * G. E. P. Box Some theorems on quadratic forms applied in the
       study of analysis of variance problems, I. Effects of
       inequality of variance in the one-way classification. _The
       Annals of Mathematical Statistics_, 25(2):290-302, 1954.

     * D. A. Bodenham and N. M. Adams. A comparison of efficient 
       approximations for a weighted sum of chi-squared random variables. 
       Statistics and Computing, 26(4):917-928, 2016.
    '''
    # some checking, so that passing lists/arrays does not matter
    if isinstance(coeff, list):
        coeff = np.array(coeff)

    isList = False
    if not isinstance(x, float):
        if isinstance(x, list):
            isList = True
            x = np.array(x)

    # checking values of coeff and x and throwing errors 
    if not checkCoeffAllPositive(coeff):
        raise Exception(getCoeffError(coeff))

    if not checkXAllPositive(x):
        raise Exception(getXError(x))

    w = sum(coeff)
    u = sum(coeff**2) / (w**2)
    k = 0.5 / u
    theta = 2 * u * w
    p = gamma.cdf(x, a=k, scale=theta)

    # if x was passed as a list, will return a list
    if isList:
        p = p.tolist()
    return(p)




# Wood's F method
def wf(coeff, x):
    '''Wood's F method

    Computes the cdf of a positively-weighted sum of chi-squared
    random variables with the Wood F (WF) method.

    Parameters:
    -----------
    coeff (list or numpy array): The coefficient vector. 
                                 All values must be greater than 0.

    x (list or numpy array or float): The vector of quantile values. 
                                      All values must be greater than 0.


    Returns:
    --------
    The cdf of the x value(s). It is returned as the same type as x, 
    i.e. if x is a list, it is returned as a list; if x is a numpy array
    it is returned as a numpy array.


    Details:
    --------
     * Note that division assumes Python 3, so may not work with Python 2.
     * Depends on numpy libary for the arrays.
     * If lists are passed, they are converted to numpy arrays (and back again).
     * Depends on scipy library for scipy.stats.f function.

    Note:
    -----
    There are pathological cases where, for certain
    coefficient vectors (which result in certain cumulant values), the
    Wood F method will be unable to match moments (cumulants) with the
    three-parameter _F_ distribution. In this situation, the HBE
    method is used, and a warning is displayed. A simple example of
    such a pathological case is when the coefficient vector is of
    length 1. Note that these pathological cases are rare; see (Wood,
    1989) in the references.


    Examples:
    ---------
    #Examples taken from Table 18.6 in N. L. Johnson, S. Kotz, N. Balakrishnan.
    #Continuous Univariate Distributions, Volume 1, John Wiley & Sons, 1994.

    # how to load the wf function from momenthchi2
    from momentchi2 import wf

    # should give value close to 0.95, actually 0.951058
    wf([1.5, 1.5, 0.5, 0.5], 10.203)            

    # should give value close to 0.05, but is actually 0.055739
    wf([1.5, 1.5, 0.5, 0.5], 0.627)            

    # specifying parameters
    wf(coeff=[1.5, 1.5, 0.5, 0.5], x=10.203)            

    # x is a list, output approx. 0.05, 0.95
    wf([1.5, 1.5, 0.5, 0.5], [0.627, 10.203])  

    # instead of lists can be numpy arrays 
    # (any list is converted to a numpy arrays inside the function anyway)
    import numpy as np
    wf( np.array([1.5, 1.5, 0.5, 0.5]), np.array([0.627, 10.203]) )  


    # Pathological case; throws warning and calls hbe() method
    wf([0.9], 1) 

    References:
    -----------
     * A. T. A. Wood. An F approximation to the distribution of a
       linear combination of chi-squared variables. Communications
       in Statistics-Simulation and Computation, 18(4):1439-1456,
       1989.

     * D. A. Bodenham and N. M. Adams. A comparison of efficient 
       approximations for a weighted sum of chi-squared random variables. 
       Statistics and Computing, 26(4):917-928, 2016.
    '''
    # some checking, so that passing lists/arrays does not matter
    if isinstance(coeff, list):
        coeff = np.array(coeff)

    isList = False
    if not isinstance(x, float):
        if isinstance(x, list):
            isList = True
            x = np.array(x)

    # checking values of coeff and x and throwing errors 
    if not checkCoeffAllPositive(coeff):
        raise Exception(getCoeffError(coeff))

    if not checkXAllPositive(x):
        raise Exception(getXError(x))

    # the next two lines work for floats or numpy arrays, but not lists
    K_1 = sum(coeff)
    K_2 = 2 * sum(coeff**2)
    K_3 = 8 * sum(coeff**3)
    r_1 = 4 * ( K_2**2 ) * K_1 + K_3 * ( K_2 - (K_1**2) )
    r_2 = K_3 * K_1 - 2 * (K_2**2)

    # check for pathological case, else call WF method
    if (r_1==0) or (r_2==0):
        warnings.warn("Pathological case: either r1 or r2 equals 0: running hbe instead.")
        p = hbe(coeff, x)
    else:
        beta = r_1 / r_2
        alpha_1 = 2 * K_1 * ( K_3 * K_1 + (K_1**2) * K_2 - (K_2**2) )/r_1
        alpha_2 = 3 + 2 * K_2 * ( K_2 + (K_1**2) )/r_2
        x = x * alpha_2/(alpha_1 * beta)
        p = f.cdf(x, dfn = 2 * alpha_1, dfd = 2 * alpha_2)

    # if x was passed as a list, will return a list
    if isList:
        p = p.tolist()
    return(p)




# Lindsay-Pilla-Basak method
def lpb4(coeff, x, p=4):
    '''Lindsay-Pilla-Basak method

    Computes the cdf of a positively-weighted sum of chi-squared
    random variables with the Wood F (WF) method.

    Parameters:
    -----------
    coeff (list or numpy array): The coefficient vector. 
                                 All values must be greater than 0.

    x (list or numpy array or float): The vector of quantile values. 
                                      All values must be greater than 0.

    p (int): Number of moments to match, default value is p=4.


    Returns:
    --------
    The cdf of the x value(s). It is returned as the same type as x, 
    i.e. if x is a list, it is returned as a list; if x is a numpy array
    it is returned as a numpy array.


    Details:
    --------
     * Note that division assumes Python 3, so may not work with Python 2.
     * Depends on numpy libary for the arrays.
     * If lists are passed, they are converted to numpy arrays (and back again).
     * Depends on scipy library for scipy.stats.f function.

    Note:
    -----
    The coefficient vector must of length at least four if default p=4. In
    some cases when the coefficient vector was of length two or three,
    the algorithm would be unable to find roots of a particular
    equation during an intermediate step, and so the algorithm would
    produce nan's/errors. If the coefficient vector is of length less than
    four, the Hall-Buckley-Eagleson method is used (and a warning is displayed).


    Examples:
    ---------
    #Examples taken from Table 18.6 in N. L. Johnson, S. Kotz, N. Balakrishnan.
    #Continuous Univariate Distributions, Volume 1, John Wiley & Sons, 1994.

    # how to load the lpb4 function from momenthchi2
    from momentchi2 import lpb4 

    # should give value close to 0.95, actually 0.9500092
    lpb4([1.5, 1.5, 0.5, 0.5], 10.203)            

    # should give value close to 0.05, but is actually 0.05001144
    lpb4([1.5, 1.5, 0.5, 0.5], 0.627)            

    # specifying parameters
    lpb4(coeff=[1.5, 1.5, 0.5, 0.5], x=10.203)            

    # x is a list, output approx. 0.05, 0.95
    lpb4([1.5, 1.5, 0.5, 0.5], [0.627, 10.203])  

    # instead of lists can be numpy arrays 
    # (any list is converted to a numpy arrays inside the function anyway)
    import numpy as np
    lpb4( np.array([1.5, 1.5, 0.5, 0.5]), np.array([0.627, 10.203]) )  


    # Pathological case; throws warning and calls hbe() method
    lpb4([0.5, 0.3, 0.2], 2.708)

    # Previous example works when you force p=3
    # lpb4([0.5, 0.3, 0.2], 2.708, p=3)


    References:
    -----------
     * B. G. Lindsay, R. S. Pilla, and P. Basak. Moment-based
       approximations of distributions using mixtures: Theory and
       applications. Annals of the Institute of Statistical Mathematics, 
       52(2):215-230, 2000.

     * D. A. Bodenham and N. M. Adams. A comparison of efficient 
       approximations for a weighted sum of chi-squared random variables. 
       Statistics and Computing, 26(4):917-928, 2016.
    '''
    # some checking, so that passing lists/arrays does not matter
    if isinstance(coeff, list):
        coeff = np.array(coeff)

    isList = False
    if not isinstance(x, float):
        if isinstance(x, list):
            isList = True
            x = np.array(x)

    # checking values of coeff and x and throwing errors 
    if not checkCoeffAllPositive(coeff):
        raise Exception(getCoeffError(coeff))

    if not checkXAllPositive(x):
        raise Exception(getXError(x))

    # check pathological case, p must be at least 2...
    if len(coeff) < p:
        message = "Pathological case, less than " + str(p) + " : running hbe instead."
        warnings.warn(message)
        mixed_pval_vec = hbe(coeff, x)
        if isList:
            mixed_pval_vec = mixed_pval_vec.tolist()
        return(mixed_pval_vec)

    #----------------------------------------------------------------#
    # Step 1: Determine/compute the moments m_1(H), ... m_2p(H)
    # compute the first 2p moments for Q = sum coeff chi-squared	
    moment_vec = get_weighted_sum_of_chi_squared_moments(coeff, p)

    #----------------------------------------------------------------#
    # Step 2.1: generate matrices; will generate these later
    
    # Step 2.2: get lambdatilde_1 - this method is exact (no bisection), solves determinant equation
    lambdatilde_1 = get_lambdatilde_1(moment_vec[0], moment_vec[1])

    #----------------------------------------------------------------#
    #Step 3:	Use bisection method (scipy.optimize.brentq) to find lambdatilde_2 
    #and others up to lambdatilde_p; we only need final lambdatilde_p
    bisect_tol = 1e-9
    lambdatilde_p = get_lambdatilde_p(lambdatilde_1, p, moment_vec, bisect_tol)

    #----------------------------------------------------------------#
    #Step 4:
    #Calculate delta_star_lambda_p
    #can already do this using methods in Step 2.1 
    #----------------------------------------------------------------#

    #----------------------------------------------------------------#
    #Step 5:
    #Step 5.1: Compute matrix M_p
    M_p = deltaNmat_applied(lambdatilde_p, moment_vec, p)

    #Step 5.2: Compute polynomial coefficients of the modified M_p matrix 
    mu_poly_coeff_vec = get_Stilde_poly_coeff(M_p)

    #Step 5.3 Compute real part of roots of polynomial given by 
    #         mu_vec = (mu_1, ..., mu_p)	
    mu_roots = get_real_poly_roots(mu_poly_coeff_vec)

    #----------------------------------------------------------------#
    #Step 6: Generate Vandermonde matrix using mu_vec and vector using 
    #        deltastar_i's, to solve for pi_vec = (pi_1, ..., pi_p)
    pi_vec = gen_and_solve_VDM_system(M_p, mu_roots)

    #----------------------------------------------------------------#
    #Step 7: Compute the linear combination (using pi_vec) of the 
    #        i gamma cdfs using parameters lambdatilde_p and mu_i 
    #	     (but need to create scale/shape parameters carefully)
    mixed_pval_vec = get_mixed_pval_vec(x, mu_roots, pi_vec, lambdatilde_p)

    # if x was passed as a list, will return a list
    if isList:
        mixed_pval_vec = mixed_pval_vec.tolist()
    return(mixed_pval_vec)


