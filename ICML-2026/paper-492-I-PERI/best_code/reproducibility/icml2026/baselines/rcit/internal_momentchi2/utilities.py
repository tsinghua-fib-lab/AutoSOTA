# SPDX-FileCopyrightText: 2022 deanbodenham <https://github.com/deanbodenham>
# SPDX-FileCopyrightText: 2025 Pôle d'Expertise de la Régulation Numérique <contact.peren@finances.gouv.fr>
#
# SPDX-License-Identifier: MIT

import math

import numpy as np
from scipy.special import comb
from scipy.optimize import brentq as uniroot
from scipy.stats import gamma


def checkCoeffAllPositive(coeff):
    '''check all entries in coeff vector positive
    '''
    if isinstance(coeff, (int, float)):
        return( coeff > 0)
    return ( all(i >0 for i in coeff) )


def getCoeffError(coeff):
    '''get the error message if there is a coeff error
    '''
    if isinstance(coeff, (int, float)):
        if not (coeff > 0):
            return("coefficient value needs to be strictly > 0.")
    else:
        if not ( all(i >0 for i in coeff) ):
            return("all coefficients need to be strictly > 0.")
    return("unknown error with coefficients.")


def checkXAllPositive(x):
    '''check all entries in x vector positive
    '''
    if isinstance(x, (int, float)):
        return( x > 0)
    return ( all(i > 0 for i in x) )


def getXError(x):
    '''get the error message if there is an error in x vector
    '''
    if isinstance(x, (int, float)):
        if not (x > 0):
            return("x value needs to be strictly > 0.")
    else:
        if not ( all(i > 0 for i in x) ):
            return("all values in x need to be strictly > 0.")
    return("unknown error with x.")


# utilities for Lindsay-Pilla-Basak method
def sum_of_powers(index, v):
    '''v is a vector, and index is a number
    '''
    return(sum(v**index))


def get_cumulant_vec_vectorised(coeff, p):
    '''get the cumulants kappa_1, kappa_2, ..., kappa_2p
    '''
    # from 1:2*p inclusive
    index = np.arange(1, 2*p+1)
    kappa = np.zeros(len(index))
    for i in range(len(index)):
        kappa[i] = ( 2**(index[i]-1) ) * math.factorial(index[i]-1) * sum_of_powers(index[i], coeff)
    return(kappa)


def update_moments(n, moment_vec, cumul_vec):
    '''Updates moments from cumulants.
       Returns the sum of the additional terms/lower products of moments and cumulants
       used in the computation of moments
    '''
    m = np.arange(1, n)
    # comb from scipy is vectorised, and so is indexing in moment_vec
    sum_of_additional_terms = sum( comb(n-1, m-1) * cumul_vec[m-1] * moment_vec[n-m-1] )
    return(sum_of_additional_terms)


def get_moments_from_cumulants(cumul_vec):
     '''get the moment vector from the cumulant vector 
     '''
     #start off by assigning it to cumulant_vec, since moment[n] = cumulant[n] + {other stuff}
     moment_vec = np.copy(cumul_vec)
     #check if more than 1 moment required
     if (len(moment_vec) > 1):
         #can't get rid of this for loop, since updates depend on previous moments
         for n in range(1, len(moment_vec)): 
             #can vectorise this part, I think
             moment_vec[n] = moment_vec[n] + update_moments(n+1, moment_vec, cumul_vec)	
         # end of for
     #end of if
     return(moment_vec) 


def get_weighted_sum_of_chi_squared_moments(coeff, p):
    '''Hides the computation of the cumulants, by just talking about moments.
    '''
    cumul_vec = get_cumulant_vec_vectorised(coeff, p)
    moment_vec = get_moments_from_cumulants(cumul_vec)
    return (moment_vec)


def get_lambdatilde_1(m1, m2):
    '''Returns a first estimate of lambdatilde
    '''
    return ( m2 / (m1**2) - 1 )


def deltaNmat_applied(x, m_vec, N):
    '''Compute the delta_N matrix 
    '''
    Nplus1 = N+1
    # moments 0, 1, ..., 2N
    m_vec = np.append( [1], m_vec[0:(2*N)])

    # these will be the coefficients for the x in (1+c_1*x)*(1+c_2*x)*...
    # want coefficients 0, 0, 1, 2, .., 2N-1 - so 2N+1 in total 
    coeff_vec = np.append([0], np.arange(0, 2*N))*x + 1

    #this computes the terms involving lambda in a vectorised way
    prod_x_terms_vec = 1/ np.cumprod(coeff_vec)

    #going to create matrix over indices i, j
    delta_mat = np.zeros( (Nplus1, Nplus1) ) 
    for i in range(Nplus1):
        for j in range(Nplus1):
            # so index is between 0 and 2N, inclusive
            index = i + j
            delta_mat[i,j] = m_vec[index] * prod_x_terms_vec[index] 
    return(delta_mat)


def det_deltaNmat(x, m_vec, N):
    '''Return the determinant of the deltaNmat
    '''
    return(  np.linalg.det( deltaNmat_applied(x, m_vec, N) )  ) 


def get_lambdatilde_p(lambdatilde_1, p, moment_vec, bisect_tol):
    '''Compute lambdatilde_p, starting from lambdatilde_1 and updating
       needs to use scipy.optimize.brentq, which is imported as uniroot, 
       as a nod to the R function.
    '''
    lambdatilde_vec = np.zeros(p)
    lambdatilde_vec[0] = lambdatilde_1
    # check that p > 1, ans successively compute lambdatilde[i]
    if p > 1:
        for i in range(1, p):
            # actually brentq from scipy.optimize
            # in [a, b]
            root = uniroot( f=det_deltaNmat, a=0, b=lambdatilde_vec[i-1], 
                    args=( moment_vec, i+1) )
            lambdatilde_vec[i] = root
    # extract last value
    lambdatilde_p = lambdatilde_vec[p-1]
    return(lambdatilde_p)

def get_base_vector(n, i):
    '''Return a vector of length n, where all entries are zero
       except for ith entry (i=0, 1, 2, ..., n-1)
    '''
    v = np.zeros(n)
    v[i] = 1
    return(v)


def get_ith_coeff_of_Stilde_poly(i, mat):
    '''Replace nth column of square matrix with 
       base vector i and compute determinant
    '''
    n = mat.shape[0] 
    base_vec = get_base_vector(n, i)
    # last column has index n-1
    mat[:,(n-1)] = base_vec
    return( np.linalg.det(mat) )


def get_Stilde_poly_coeff(M_p):
    '''get polynomial coefficients from (p+1)-dimensional matrix
    '''
    n = M_p.shape[0]
    mu_poly_coeff_vec = np.zeros(n)
    for i in range(n):
        mu_poly_coeff_vec[i] = get_ith_coeff_of_Stilde_poly(i, M_p)
    return mu_poly_coeff_vec


def get_VDM_b_vec(mat):
    '''Simply extracts the first column, and removes last element of last column
       (so column is one element less)
    '''
    b = mat[:,0]
    b = np.delete(b, len(b)-1)
    return(b)


def get_vandermonde(vec):
    '''Generates the van der monde matrix from a vector
    '''
    p = len(vec)
    vdm = np.zeros( (p, p) )
    for i in range(p):
        vdm[i] = vec**i 
    return(vdm)


def get_real_poly_roots(mu_poly_coeff_vec):
    '''Gets real part of complex roots of polynomial with coefficients a,
       where
       a[0] + a[1] * x + ... + a[n-1] * x**(n-1)
       Need to reverse vector to conform with np.roots function
       and then need to reverse again so roos increase in size
    '''
    mu_roots = np.real( np.roots(mu_poly_coeff_vec[::-1]) )
    mu_roots = mu_roots[::-1]
    return(mu_roots)


def gen_and_solve_VDM_system(M_p, mu_roots):
    '''Generates the VDM matrix and solves the linear system.
    '''
    b = get_VDM_b_vec(M_p)
    vdm = get_vandermonde(mu_roots)
    # solve the linear system
    pi_vec = np.linalg.solve(vdm, b)
    return(pi_vec)


def get_mixed_pval_vec(q_vec, mu_vec, pi_vec, lambdatilde_p):
    '''Get sum of weighted gamma cdfs
    '''
    p = len(mu_vec)
    #shape
    alpha = 1/lambdatilde_p
    #NB: scale beta = mu/alpha, as per formulation in Lindsay paper
    beta_vec = mu_vec / alpha

    # need to deal with case of float vs list
    partial_pval_vec = 0
    if not isinstance(q_vec, float):
        partial_pval_vec = np.zeros(len(q_vec))
    
    for i in range(p):
        partial_pval_vec = partial_pval_vec + pi_vec[i] * gamma.cdf(q_vec, a=alpha, scale=beta_vec[i])

    return(partial_pval_vec)

