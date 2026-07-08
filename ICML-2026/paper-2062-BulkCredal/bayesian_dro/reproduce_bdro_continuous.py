import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.stats import dirichlet
from scipy.stats import truncnorm 
from scipy.stats import gamma 
from scipy.stats import expon 
from scipy.stats import rv_discrete
import time
import gurobipy as gp
from gurobipy import GRB
from joblib import Parallel, delayed

number_iteration_theta = 100 
number_iteration_xi = 100
b = 8           # from Section 4.1
h = 3 
M = 50 
L = 100
LARGEST_X = 50  # from Section 4.1: support is 0 <= x <= 50
SMALLEST_X = 0

my_mean = 10    # from Section 4.1
my_std = 10     # from Section 4.1: variance is 100 -> std is 10

def xi_generation(theta, D_xi, random_state = None):
    xi = expon.rvs(scale = 1 / theta, size = D_xi, random_state=random_state)
    return xi
def data_generation(D_data, random_state = None):

    myclip_a, myclip_b = 0, np.inf

    a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
    data = truncnorm.rvs(a, b, loc = my_mean, scale = my_std, size = D_data, random_state=random_state)
    return data
def theta_generation(data, D_theta, random_state = None):

    alpha0, beta0 = 1, 1
    alpha = alpha0 + D_data
    beta = beta0 + np.sum(data)
    theta = gamma.rvs(a = alpha, scale = 1 / beta, size = D_theta, random_state=random_state) 
    return theta
def cost(x, xi):
    return h * np.maximum(0, x - xi) + b * np.maximum(0, xi - x)
def nabla_cost(x, xi):
    if x > xi:
        return h 
    elif x < xi:
        return - b
    else:
        return 0
def ecdf(x):
    x = np.sort(x)
    u, c = np.unique(x, return_counts=True) # c is count, u is unique element
    n = len(x)
    y = (np.cumsum(c) - 0.5)/n
    def interpolate(input_data):
        yinterp = np.interp(input_data, u, y, left=0.0, right=1.0)
        return yinterp
    return interpolate
def cumulative_kl(x, reference_pdf, fraction = 0.9):
    dx = np.diff(np.sort(np.unique(x)))
    ex = np.min(dx)*fraction
    n = len(x)
    P = ecdf(x)
    KL = (1./n)*np.sum(np.log((P(x) - P(x-ex)) / (ex * reference_pdf)))
    return KL

def Bayesian_DRO_1(lam, x, theta_index, epsilon):
    return lam * epsilon + lam * np.log(np.mean(np.exp(cost(x, xi[theta_index, :]) / lam)))
def Bayesian_DRO_2(x, theta_index, epsilon):
    bnds = [(0.01,None)]
    if epsilon <= 0:
        initial_point = 2000
    elif epsilon <= 0.7:
        initial_point = 100
    elif epsilon <= 2:
        initial_point = 50
    elif epsilon <= 2.4:
        initial_point = 40
    else:
        initial_point = 10
    res = optimize.minimize(Bayesian_DRO_1, initial_point, bounds = bnds, args = (x, theta_index, epsilon))
    optimal_lambda = res.x
    optimal_obj = res.fun
    if np.isnan(optimal_obj):
        optimal_obj = Bayesian_DRO_1(np.array([optimal_lambda, x, theta_index, epsilon]))
    return optimal_obj
def Bayesian_DRO_3(x, epsilon):
    temp_value = np.zeros(number_iteration_theta)
    for i in range(number_iteration_theta):
        temp_value[i] = Bayesian_DRO_2(x, i, epsilon)
    value = np.mean(temp_value)
    return value
def main_Bayesian_DRO(epsilon):
    support = np.arange(SMALLEST_X, LARGEST_X + 1, 1)
    value_support = np.zeros(len(support))
    for i in range(len(support)) :
        value_support[i] = Bayesian_DRO_3(support[i], epsilon)
    smallest_index = np.argmin(value_support)
    support_smallest = np.arange(support[smallest_index]-1, support[smallest_index]+1, 0.01)
    value_support_smallest = np.zeros(len(support_smallest))
    for i in range(len(support_smallest)) :
        value_support_smallest[i] = Bayesian_DRO_3(support_smallest[i], epsilon)
    temp = support_smallest[np.argmin(value_support_smallest)]
    if temp <= LARGEST_X:
        optimal_x = temp
    else:
        optimal_x = LARGEST_X
    return optimal_x
def BRO(x):
    return np.mean([np.mean(cost(x, xi[i,:])) for i in range(number_iteration_theta)])
def main_BRO():
    bnds = [(SMALLEST_X, LARGEST_X)]
    initial_point = sol_true
    res = optimize.minimize(BRO, initial_point, bounds = bnds)
    optimal_x = res.x
    return optimal_x[0]

def main_empirical():
    ascend_data = np.sort(data)
    len_data = len(data)
    for i in range(1,len_data+1):
        if (i - 1) / len_data < b / (h+b) and i / len_data >= b / (h+b):
            return ascend_data[i - 1]
        
def main_empirical_DRO_Wasserstein(epsilon, p):
    len_data = len(data)
    if p == 1:
        # put data in ascending order
        ascend_data = np.sort(data)
        for i in range(1,len_data+1):
            if (i - 1) / len_data < b / (h+b) and i / len_data >= b / (h+b):
                return ascend_data[i - 1]
    if p > 1:
        Delta = 1 / (h + b) * (1 / p) ** (1 / (p - 1)) * ((p - 1) / p) * (b ** (p / (p - 1)) - h ** (p / (p - 1)))
        Lambda = (1 / (h + b)) * (b ** (p / (p - 1)) * h + h ** (p / (p - 1)) * b)
        ascend_data = np.sort(data)
        for i in range(1,len_data+1):
            if (i - 1) / len_data < b / (h+b) and i / len_data >= b / (h+b):
                temp = ascend_data[i - 1]
                break
        return temp + Delta * p ** (1 / (p - 1)) * epsilon * (1 / Lambda) ** (1 / p)      
    
def Bayesian_DRO_1_epsilon1(lam, x, theta_index, epsilon_1):
    return lam * epsilon_1 + lam * np.log(np.mean(np.exp(cost(x, xi[theta_index, :]) / lam)))
def Bayesian_DRO_2_epsilon1(x, theta_index, epsilon_1):
    bnds = [(0.01,None)]
    if epsilon_1 <= 0:
        initial_point = 2000
    elif epsilon_1 <= 0.7:
        initial_point = 100
    elif epsilon_1 <= 2:
        initial_point = 50
    elif epsilon_1 <= 2.4:
        initial_point = 40
    else:
        initial_point = 10
    res = optimize.minimize(Bayesian_DRO_1_epsilon1, initial_point, bounds = bnds, args = (x, theta_index, epsilon_1))
    optimal_lambda = res.x
    optimal_obj = res.fun
    if np.isnan(optimal_obj):
        optimal_obj = Bayesian_DRO_1_epsilon1(np.array([optimal_lambda, x, theta_index, epsilon_1]))
    return optimal_obj
def Bayesian_DRO_3_epsilon1(x, epsilon_1):
    temp_value = np.zeros(number_iteration_theta)
    for i in range(number_iteration_theta):
        temp_value[i] = Bayesian_DRO_2_epsilon1(x, i, epsilon_1[i])
    value = np.mean(temp_value)
    return value
def main_Bayesian_DRO_epsilon1():
    epsilon_1 = np.zeros(number_iteration_theta)
    for i in range(number_iteration_theta):
        reference_pdf = expon.pdf(data, scale = 1 / theta[i])
        temp = cumulative_kl(data, reference_pdf)
        epsilon_1[i] = temp   
        
    support = np.arange(SMALLEST_X, LARGEST_X + 1, 1)
    value_support = np.zeros(len(support))
    for i in range(len(support)) :
        value_support[i] = Bayesian_DRO_3_epsilon1(support[i], epsilon_1)
    smallest_index = np.argmin(value_support)
    support_smallest = np.arange(support[smallest_index]-1, support[smallest_index]+1, 0.01)
    value_support_smallest = np.zeros(len(support_smallest))
    for i in range(len(support_smallest)) :
        value_support_smallest[i] = Bayesian_DRO_3_epsilon1(support_smallest[i], epsilon_1)
    temp = support_smallest[np.argmin(value_support_smallest)]
    if temp <= LARGEST_X:
        optimal_x = temp
    else:
        optimal_x = LARGEST_X
    return optimal_x, epsilon_1
    
def Bayesian_DRO_1_epsilon2(lam, x, theta_index, epsilon_2):
    return lam * epsilon_2 + lam * np.log(np.mean(np.exp(cost(x, xi[theta_index, :]) / lam)))
def Bayesian_DRO_2_epsilon2(x, theta_index, epsilon_2):
    bnds = [(0.01,None)]
    if epsilon_2 <= 0:
        initial_point = 2000
    elif epsilon_2 <= 0.7:
        initial_point = 100
    elif epsilon_2 <= 2:
        initial_point = 50
    elif epsilon_2 <= 2.4:
        initial_point = 40
    else:
        initial_point = 10
    res = optimize.minimize(Bayesian_DRO_1_epsilon2, initial_point, bounds = bnds, args = (x, theta_index, epsilon_2))
    optimal_lambda = res.x
    optimal_obj = res.fun
    if np.isnan(optimal_obj):
        optimal_obj = Bayesian_DRO_1_epsilon2(np.array([optimal_lambda, x, theta_index, epsilon_2]))
    return optimal_obj
def Bayesian_DRO_3_epsilon2(x, epsilon_2):
    temp_value = np.zeros(number_iteration_theta)
    for i in range(number_iteration_theta):
        temp_value[i] = Bayesian_DRO_2_epsilon2(x, i, epsilon_2[i])
    value = np.mean(temp_value)
    return value
def main_Bayesian_DRO_epsilon2():
    epsilon_2 = np.zeros(number_iteration_theta)
    for i in range(number_iteration_theta):
        reference_pdf = expon.pdf(data, scale = 1 / theta[i])
        temp = cumulative_kl(data, reference_pdf)
        epsilon_2[i] = temp / 2            
        
    support = np.arange(SMALLEST_X, LARGEST_X + 1, 1)
    value_support = np.zeros(len(support))
    for i in range(len(support)) :
        value_support[i] = Bayesian_DRO_3_epsilon2(support[i], epsilon_2)
    smallest_index = np.argmin(value_support)
    support_smallest = np.arange(support[smallest_index]-1, support[smallest_index]+1, 0.01)
    value_support_smallest = np.zeros(len(support_smallest))
    for i in range(len(support_smallest)) :
        value_support_smallest[i] = Bayesian_DRO_3_epsilon2(support_smallest[i], epsilon_2)
    temp = support_smallest[np.argmin(value_support_smallest)]
    if temp <= LARGEST_X:
        optimal_x = temp
    else:
        optimal_x = LARGEST_X
    return optimal_x, epsilon_2

def calculate_epsilon3(empirical_x, theta_index):
    np.random.seed(0)
    sample_xi = xi_generation(theta[theta_index], L)
    temp_nabla = np.zeros(L)
    for i in range(L):
        temp_nabla[i] = nabla_cost(empirical_x, sample_xi[i])
    temp_reference = expon.pdf(sample_xi, loc = 0, scale = 1 / theta[theta_index])

    m = gp.Model("epsilon_3")
    temp_q = dict()
    temp_q_abs = dict()
    temp_difference = dict()
    for i in range(L):
        temp_q[i] = m.addVar(name="temp_q%s" % str([i])) # probability mass
        temp_q_abs[i] = m.addVar(name="temp_q_abs%s" % str([i]))
        temp_difference[i] = m.addVar(lb = - temp_reference[i], name="temp_difference%s" % str([i]))
        m.addConstr(temp_q[i] >= 1e-4)
        m.addConstr(temp_difference[i] - temp_q[i] == - temp_reference[i])
        m.addGenConstrAbs(temp_q_abs[i], temp_difference[i])
        
    m.addConstr(gp.quicksum(temp_q[i] / temp_reference[i] for i in range(L)) == L)
    m.addConstr(gp.quicksum(temp_q[i] * temp_nabla[i] / temp_reference[i] for i in range(L)) == 0)
    
    obj = gp.quicksum(temp_q_abs[i] for i in range(L))
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()
    initial_point = np.zeros(L)
    for i in range(L):
        initial_point[i] = m.getVarByName("temp_q%s" % str([i])).x
    
    cons=({'type': 'eq','fun': lambda q: np.sum(q / temp_reference) - L},
          {'type': 'eq', 'fun': lambda q: np.sum(q * temp_nabla / temp_reference)}
         )
    bnds = [(0, 1)] * L
    options = {"maxiter": 10000}
    def epsilon3_obj(q, theta_index):
        return np.sum([q[i] / temp_reference[i] * np.log(q[i] / temp_reference[i]) for i in range(L)]) / L

    res = optimize.minimize(epsilon3_obj, initial_point, bounds = bnds, constraints = cons, args = (theta_index), method='SLSQP', options = options)
    optimal_q = res.x
    optimal_obj = res.fun
    if np.isnan(optimal_obj):
        optimal_obj = epsilon3_obj(optimal_q, theta_index)
    return optimal_obj
def Bayesian_DRO_1_epsilon3(lam, x, theta_index, epsilon_3):
    return lam * epsilon_3 + lam * np.log(np.mean(np.exp(cost(x, xi[theta_index, :]) / lam)))
def Bayesian_DRO_2_epsilon3(x, theta_index, epsilon_3):
    bnds = [(0.01,None)]
    if epsilon_3 <= 0:
        initial_point = 2000
    elif epsilon_3 <= 0.7:
        initial_point = 100
    elif epsilon_3 <= 2:
        initial_point = 50
    elif epsilon_3 <= 2.4:
        initial_point = 40
    else:
        initial_point = 10
    res = optimize.minimize(Bayesian_DRO_1_epsilon3, initial_point, bounds = bnds, args = (x, theta_index, epsilon_3))
    optimal_lambda = res.x
    optimal_obj = res.fun
    if np.isnan(optimal_obj):
        optimal_obj = Bayesian_DRO_1_epsilon3(np.array([optimal_lambda, x, theta_index, epsilon_3]))
    return optimal_obj
def Bayesian_DRO_3_epsilon3(x, epsilon_3):
    temp_value = np.zeros(number_iteration_theta)
    for i in range(number_iteration_theta):
        temp_value[i] = Bayesian_DRO_2_epsilon3(x, i, epsilon_3[i])
    value = np.mean(temp_value)
    return value
def main_Bayesian_DRO_epsilon3(empirical_x):
    epsilon_3 = np.zeros(number_iteration_theta)
    for i in range(number_iteration_theta):
        epsilon_3[i] = calculate_epsilon3(empirical_x, i)
    support = np.arange(SMALLEST_X, LARGEST_X + 1, 1)
    value_support = np.zeros(len(support))
    for i in range(len(support)) :
        value_support[i] = Bayesian_DRO_3_epsilon3(support[i], epsilon_3)
    smallest_index = np.argmin(value_support)
    support_smallest = np.arange(support[smallest_index]-1, support[smallest_index]+1, 0.01)
    value_support_smallest = np.zeros(len(support_smallest))
    for i in range(len(support_smallest)) :
        value_support_smallest[i] = Bayesian_DRO_3_epsilon3(support_smallest[i], epsilon_3)
    temp = support_smallest[np.argmin(value_support_smallest)]
    if temp <= LARGEST_X:
        optimal_x = temp
    else:
        optimal_x = LARGEST_X
    return optimal_x, epsilon_3

sol_true = truncnorm.ppf((b - 0) / (h + b), a = - my_mean / my_std, b = np.inf, loc = my_mean, scale = my_std)
D_data = 20
replication = 200
epsilon_set = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1, 1.5, 2, 2.5, 3]
WASSERSTEIN_EPSILON = epsilon_set + list(range(4, 16))

NUM_TEST_OBSERVATIONS = 50
generator = np.random.default_rng(seed=replication)
data_eval = data_generation((replication, NUM_TEST_OBSERVATIONS), random_state=generator)

solution_BRO = np.zeros(replication)

solution_empirical = np.zeros(replication)

solution_Bayesian_DRO = np.zeros([replication, len(epsilon_set)])

solution_epsilon_1 = np.zeros(replication)
epsilon_1 = np.zeros([replication, number_iteration_theta])

solution_epsilon_2 = np.zeros(replication)
epsilon_2 = np.zeros([replication, number_iteration_theta])

solution_epsilon_3 = np.zeros(replication)
epsilon_3 = np.zeros([replication, number_iteration_theta])


solution_empirical_DRO_Wasserstein = np.zeros([replication, len(WASSERSTEIN_EPSILON)])

obj_BRO = np.zeros(replication)

obj_empirical = np.zeros(replication)

obj_Bayesian_DRO = np.zeros([replication, len(epsilon_set), NUM_TEST_OBSERVATIONS])
bdro_in_sample_cost = np.zeros([replication, len(epsilon_set), D_data])

obj_epsilon_1 = np.zeros(replication)

obj_epsilon_2 = np.zeros(replication)

obj_epsilon_3 = np.zeros(replication)

obj_empirical_DRO_Wasserstein = np.zeros([replication, len(WASSERSTEIN_EPSILON), NUM_TEST_OBSERVATIONS])
wass_in_sample_cost = np.zeros([replication, len(WASSERSTEIN_EPSILON), D_data])

for k in range(replication):

    generator = np.random.default_rng(seed=k)
    data = data_generation(D_data, random_state=generator)
    theta = theta_generation(data, number_iteration_theta, random_state=generator)
    xi = np.zeros([number_iteration_theta, number_iteration_xi])
    for i in range(number_iteration_theta):
        xi[i] = xi_generation(theta[i], number_iteration_xi, random_state=generator)
    
    # solution_BRO[k] = main_BRO()
    # obj_BRO[k] = cost(solution_BRO[k], data_eval[k])
    
    # solution_empirical[k] = main_empirical()
    # obj_empirical[k] = cost(solution_empirical[k], data_eval[k])

    # solution_epsilon_1[k], epsilon_1[k,:] = main_Bayesian_DRO_epsilon1()
    # obj_epsilon_1[k] = cost(solution_epsilon_1[k], data_eval[k])          
        
    # solution_epsilon_2[k], epsilon_2[k,:] = main_Bayesian_DRO_epsilon2()
    # obj_epsilon_2[k] = cost(solution_epsilon_2[k], data_eval[k])  
    
    # solution_epsilon_3[k], epsilon_3[k,:] = main_Bayesian_DRO_epsilon3(solution_empirical[k])
    # obj_epsilon_3[k] = cost(solution_epsilon_3[k], data_eval[k])    
  
    print("Starting Wasserstein DRO loop")
    wasserstein_gen = Parallel(n_jobs=-1)(delayed(main_empirical_DRO_Wasserstein)(epsilon, 2) for epsilon in WASSERSTEIN_EPSILON)
    solution_empirical_DRO_Wasserstein[k] = np.array(list(wasserstein_gen))
    for index, epsilon in enumerate(WASSERSTEIN_EPSILON):
        obj_empirical_DRO_Wasserstein[k, index] = cost(solution_empirical_DRO_Wasserstein[k, index], data_eval[k])
        wass_in_sample_cost[k, index] = cost(solution_empirical_DRO_Wasserstein[k, index], data)

    dir_name = "in_sample"

    wasserstein_out_sample_df = pd.DataFrame({
        "replication": [k]*len(WASSERSTEIN_EPSILON) * NUM_TEST_OBSERVATIONS,
        "epsilon": np.repeat(WASSERSTEIN_EPSILON, NUM_TEST_OBSERVATIONS),
        "index": np.tile(np.arange(NUM_TEST_OBSERVATIONS), len(WASSERSTEIN_EPSILON)),
        "wasserstein_dro_sol": np.repeat(solution_empirical_DRO_Wasserstein[k], NUM_TEST_OBSERVATIONS),
        "wasserstein_dro_cost": obj_empirical_DRO_Wasserstein[k].flatten(),
    })

    wasserstein_in_sample_df = pd.DataFrame({
        "replication": [k]*len(WASSERSTEIN_EPSILON) * D_data,
        "epsilon": np.repeat(WASSERSTEIN_EPSILON, D_data),
        "index": np.tile(np.arange(D_data), len(WASSERSTEIN_EPSILON)),
        "wasserstein_dro_sol": np.repeat(solution_empirical_DRO_Wasserstein[k], D_data),
        "wasserstein_dro_cost": wass_in_sample_cost[k].flatten(),
    })

    print("Starting Bayesian DRO loop")
    bayesian_dro_gen = Parallel(n_jobs=-1)(delayed(main_Bayesian_DRO)(epsilon) for epsilon in epsilon_set)
    solution_Bayesian_DRO[k] = np.array(list(bayesian_dro_gen))
    for index, epsilon in enumerate(epsilon_set):
        obj_Bayesian_DRO[k, index] = cost(solution_Bayesian_DRO[k, index], data_eval[k])
        bdro_in_sample_cost[k, index] = cost(solution_Bayesian_DRO[k, index], data)

    bdro_out_sample_df = pd.DataFrame({
        "replication": [k]*len(epsilon_set) * NUM_TEST_OBSERVATIONS,
        "epsilon": np.repeat(epsilon_set, NUM_TEST_OBSERVATIONS),
        "index": np.tile(np.arange(NUM_TEST_OBSERVATIONS), len(epsilon_set)),
        "bayesian_dro_sol": np.repeat(solution_Bayesian_DRO[k], NUM_TEST_OBSERVATIONS),
        "bayesian_dro_cost": obj_Bayesian_DRO[k].flatten(),
    })

    bdro_in_sample_df = pd.DataFrame({
        "replication": [k]*len(epsilon_set) * D_data,
        "epsilon": np.repeat(epsilon_set, D_data),
        "index": np.tile(np.arange(D_data), len(epsilon_set)),
        "bayesian_dro_sol": np.repeat(solution_Bayesian_DRO[k], D_data),
        "bayesian_dro_cost": bdro_in_sample_cost[k].flatten(),
    })
