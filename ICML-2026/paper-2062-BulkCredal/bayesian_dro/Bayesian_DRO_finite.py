import numpy as np
from scipy import optimize
from scipy.stats import dirichlet
from scipy.stats import rv_discrete
import time
try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    gp = None
    GRB = None

b = 10
h = 2
c = 3
number_iteration_theta = 100

# xi has finitely many supports, theta_c is its true weight, generated from uniform summing up to 1
xi_space = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
M = np.max(xi_space)
temp_c = np.random.uniform(size=len(xi_space))
theta_c = list(temp_c / np.sum(temp_c))


def data_generation(D_data):
    return list(np.random.choice(xi_space, D_data, p=theta_c))


def theta_generation(data, D_theta):
    alpha0 = np.int_(np.ones(len(xi_space)))
    number_of_data = len(data)
    count_data = np.int_(np.zeros(len(xi_space)))
    for index, value in enumerate(xi_space):
        count_data[index] = data.count(value)
    alpha = alpha0 + count_data
    theta = dirichlet.rvs(alpha, size=D_theta)
    return theta


def cost(x, xi):
    return h * np.maximum(0, x - xi) + b * np.maximum(0, xi - x) + c * x


def nabla_cost(x, xi):
    # xi is a number
    if x > xi:
        return h + c
    elif x < xi:
        return c - b
    else:
        return c


def Bayesian_DRO_1(lam, x, theta_index, epsilon):
    temp_obj = np.zeros(len(xi_space))
    for index, xi in enumerate(xi_space):
        temp_obj[index] = np.exp(cost(x, xi) / lam) * theta[theta_index, index]
    obj = np.sum(temp_obj)
    log_obj = np.log(obj)
    value = lam * epsilon + lam * log_obj
    return value


def Bayesian_DRO_2(x, theta_index, epsilon):
    bnds = [(0.01, None)]
    if epsilon <= 0:
        initial_point = 2000
    elif epsilon <= 0.1:
        initial_point = 20
    elif epsilon <= 0.3:
        initial_point = 10
    elif epsilon <= 0.7:
        initial_point = 5
    elif epsilon <= 2:
        initial_point = 2
    elif epsilon <= 2.4:
        initial_point = 1
    else:
        initial_point = 0.2
    res = optimize.minimize(
        Bayesian_DRO_1, initial_point, bounds=bnds, args=(x, theta_index, epsilon)
    )
    optimal_lambda = res.x
    optimal_obj = res.fun
    if np.isnan(optimal_obj):
        optimal_obj = Bayesian_DRO_1(
            np.array([optimal_lambda, x, theta_index, epsilon])
        )
    return optimal_obj


def Bayesian_DRO_3(x, epsilon):
    temp_value = np.zeros(number_iteration_theta)
    for i in range(number_iteration_theta):
        temp_value[i] = Bayesian_DRO_2(x, i, epsilon)
    value = np.mean(temp_value)
    return value


def main_Bayesian_DRO(epsilon):
    bnds = [(0, M)]
    initial_point = sol_true
    res = optimize.minimize(Bayesian_DRO_3, initial_point, bounds=bnds, args=(epsilon))
    optimal_x = res.x
    return optimal_x


def BRO(x):
    temp_obj = np.zeros([number_iteration_theta, len(xi_space)])
    obj = np.zeros(number_iteration_theta)
    for i in range(number_iteration_theta):
        for index, xi in enumerate(xi_space):
            temp_obj[i, index] = cost(x, xi) * theta[i, index]
        obj[i] = np.sum(temp_obj[i, :])
    value = np.mean(obj)
    return value


def main_BRO():
    bnds = [(0, M)]
    initial_point = sol_true
    res_BRO = optimize.minimize(BRO, initial_point, bounds=bnds)
    sol_BRO = res_BRO.x
    return sol_BRO


def empirical(x):
    return np.mean(cost(x, np.array(data)))


def main_empirical():
    bnds = [(0, M)]
    initial_point = sol_true
    res_empirical = optimize.minimize(empirical, initial_point, bounds=bnds)
    sol = res_empirical.x
    return sol


def empirical_DRO_Gotoh_1(q, x, epsilon, valid_set):
    # first we solve a maximization problem, but we turn it to minimization
    temp_obj_cost = np.zeros(len(valid_set))
    for index, valid_element in enumerate(valid_set):
        temp_obj_cost[index] = q[index] * cost(x, xi_space[valid_element])
    temp_obj = np.sum(temp_obj_cost)
    return -temp_obj


def empirical_DRO_Gotoh_2(x, epsilon, valid_set):
    temp_len = len(valid_set)
    initial_point = empirical_p[valid_set]
    bnds = [(0, 1)] * temp_len
    cons = (
        {"type": "eq", "fun": lambda q: np.sum(q) - 1},
        {
            "type": "ineq",
            "fun": lambda q: epsilon
            - np.sum(
                [
                    q[index] * (np.log(q[index]) - np.log(empirical_p[valid_element]))
                    for index, valid_element in enumerate(valid_set)
                    if q[index] > 0
                ]
            ),
        },
    )
    res = optimize.minimize(
        empirical_DRO_Gotoh_1,
        initial_point,
        bounds=bnds,
        constraints=cons,
        args=(x, epsilon, valid_set),
    )
    optimal_q = res.x
    optimal_obj = -res.fun
    if np.isnan(optimal_obj):
        optimal_obj = -empirical_DRO_Gotoh_1(optimal_q, x, epsilon, valid_set)
    return optimal_obj


def main_empirical_DRO_Gotoh(epsilon):
    initial_point = sol_true
    bnds = [(0, M)]
    valid_set = list(np.nonzero(empirical_p > 0)[0])
    res = optimize.minimize(
        empirical_DRO_Gotoh_2, initial_point, bounds=bnds, args=(epsilon, valid_set)
    )
    optimal_x = res.x
    return optimal_x


def Bayesian_DRO_1_epsilon1(lam, x, theta_index, epsilon_1):
    temp_obj = np.zeros(len(xi_space))
    for index, xi in enumerate(xi_space):
        temp_obj[index] = np.exp(cost(x, xi) / lam) * theta[theta_index, index]
    obj = np.sum(temp_obj)
    log_obj = np.log(obj)
    value = lam * epsilon_1 + lam * log_obj
    return value


def Bayesian_DRO_2_epsilon1(x, theta_index, epsilon_1):
    bnds = [(0.01, None)]
    if epsilon_1 <= 0:
        initial_point = 2000
    elif epsilon_1 <= 0.1:
        initial_point = 20
    elif epsilon_1 <= 0.3:
        initial_point = 10
    elif epsilon_1 <= 0.7:
        initial_point = 5
    elif epsilon_1 <= 2:
        initial_point = 2
    elif epsilon_1 <= 2.4:
        initial_point = 1
    else:
        initial_point = 0.2
    res = optimize.minimize(
        Bayesian_DRO_1_epsilon1,
        initial_point,
        bounds=bnds,
        args=(x, theta_index, epsilon_1),
    )
    optimal_lambda = res.x
    optimal_obj = res.fun
    if np.isnan(optimal_obj):
        optimal_obj = Bayesian_DRO_1_epsilon1(
            np.array([optimal_lambda, x, theta_index, epsilon_1])
        )
    return optimal_obj


def Bayesian_DRO_3_epsilon1(x, epsilon_1):
    temp_value = np.zeros(number_iteration_theta)
    for i in range(number_iteration_theta):
        temp_value[i] = Bayesian_DRO_2_epsilon1(x, i, epsilon_1[i])
    value = np.mean(temp_value)
    return value


def main_Bayesian_DRO_epsilon1():
    bnds = [(0, M)]
    initial_point = sol_true
    epsilon_1 = np.zeros(number_iteration_theta)
    for j in range(number_iteration_theta):
        epsilon_1[j] = np.sum(
            [
                empirical_p[i] * np.log(empirical_p[i] / theta[j, i])
                for i in range(len(xi_space))
                if empirical_p[i] > 0
            ]
        )
    res = optimize.minimize(
        Bayesian_DRO_3_epsilon1, initial_point, bounds=bnds, args=(epsilon_1)
    )
    optimal_x = res.x
    return optimal_x, epsilon_1


def Bayesian_DRO_1_epsilon2(lam, x, theta_index, epsilon_2):
    temp_obj = np.zeros(len(xi_space))
    for index, xi in enumerate(xi_space):
        temp_obj[index] = np.exp(cost(x, xi) / lam) * theta[theta_index, index]
    obj = np.sum(temp_obj)
    log_obj = np.log(obj)
    value = lam * epsilon_2 + lam * log_obj
    return value


def Bayesian_DRO_2_epsilon2(x, theta_index, epsilon_2):
    bnds = [(0.01, None)]
    if epsilon_2 <= 0:
        initial_point = 2000
    elif epsilon_2 <= 0.1:
        initial_point = 20
    elif epsilon_2 <= 0.3:
        initial_point = 10
    elif epsilon_2 <= 0.7:
        initial_point = 5
    elif epsilon_2 <= 2:
        initial_point = 2
    elif epsilon_2 <= 2.4:
        initial_point = 1
    else:
        initial_point = 0.2
    res = optimize.minimize(
        Bayesian_DRO_1_epsilon2,
        initial_point,
        bounds=bnds,
        args=(x, theta_index, epsilon_2),
    )
    optimal_lambda = res.x
    optimal_obj = res.fun
    if np.isnan(optimal_obj):
        optimal_obj = Bayesian_DRO_1_epsilon2(
            np.array([optimal_lambda, x, theta_index, epsilon_2])
        )
    return optimal_obj


def Bayesian_DRO_3_epsilon2(x, epsilon_1):
    temp_value = np.zeros(number_iteration_theta)
    for i in range(number_iteration_theta):
        temp_value[i] = Bayesian_DRO_2_epsilon2(x, i, epsilon_1[i])
    value = np.mean(temp_value)
    return value


def main_Bayesian_DRO_epsilon2():
    bnds = [(0, M)]
    initial_point = sol_true
    epsilon_2 = np.zeros(number_iteration_theta)
    for j in range(number_iteration_theta):
        epsilon_2[j] = (
            np.sum(
                [
                    empirical_p[i] * np.log(empirical_p[i] / theta[j, i])
                    for i in range(len(xi_space))
                    if empirical_p[i] > 0
                ]
            )
            / 2
        )
    res = optimize.minimize(
        Bayesian_DRO_3_epsilon2, initial_point, bounds=bnds, args=(epsilon_2)
    )
    optimal_x = res.x
    return optimal_x, epsilon_2


def calculate_epsilon3(empirical_x, theta_index):
    temp_nabla = np.zeros(len(xi_space))
    for i in range(len(xi_space)):
        temp_nabla[i] = nabla_cost(empirical_x, xi_space[i])

    m = gp.Model("epsilon_3")
    temp_q = dict()
    for i in range(len(xi_space)):
        temp_q[i] = m.addVar(lb=0, ub=1, name="temp_q%s" % str([i]))  # probability mass
        m.addConstr(temp_q[i] >= 2e-2)
    m.addConstr(gp.quicksum(temp_q[i] for i in range(len(xi_space))) == 1)
    m.addConstr(
        gp.quicksum(temp_q[i] * temp_nabla[i] for i in range(len(xi_space))) == 0
    )
    m.optimize()
    initial_point = np.array(m.x)

    cons = (
        {"type": "eq", "fun": lambda q: np.sum(q) - 1},
        {
            "type": "eq",
            "fun": lambda q: np.sum(
                [q[i] * temp_nabla[i] for i in range(len(xi_space))]
            ),
        },
    )
    bnds = [(0, 1)] * len(xi_space)

    def epsilon3_obj(q, theta_index):
        return np.sum(
            [q[i] * np.log(q[i] / theta[theta_index, i]) for i in range(len(xi_space))]
        )

    res = optimize.minimize(
        epsilon3_obj,
        initial_point,
        bounds=bnds,
        constraints=cons,
        args=(theta_index),
        method="SLSQP",
    )
    optimal_q = res.x
    optimal_obj = res.fun
    if np.isnan(optimal_obj):
        optimal_obj = epsilon3_obj(optimal_q, theta_index)
    return optimal_obj


def Bayesian_DRO_1_epsilon3(lam, x, theta_index, epsilon_3):
    temp_obj = np.zeros(len(xi_space))
    for index, xi in enumerate(xi_space):
        temp_obj[index] = np.exp(cost(x, xi) / lam) * theta[theta_index, index]
    obj = np.sum(temp_obj)
    log_obj = np.log(obj)
    value = lam * epsilon_3 + lam * log_obj
    return value


def Bayesian_DRO_2_epsilon3(x, theta_index, epsilon_3):
    bnds = [(0.01, None)]
    if epsilon_3 <= 0:
        initial_point = 2000
    elif epsilon_3 <= 0.1:
        initial_point = 20
    elif epsilon_3 <= 0.3:
        initial_point = 10
    elif epsilon_3 <= 0.7:
        initial_point = 5
    elif epsilon_3 <= 2:
        initial_point = 2
    elif epsilon_3 <= 2.4:
        initial_point = 1
    else:
        initial_point = 0.2
    res = optimize.minimize(
        Bayesian_DRO_1_epsilon3,
        initial_point,
        bounds=bnds,
        args=(x, theta_index, epsilon_3),
    )
    optimal_lambda = res.x
    optimal_obj = res.fun
    if np.isnan(optimal_obj):
        optimal_obj = Bayesian_DRO_1_epsilon3(
            np.array([optimal_lambda, x, theta_index, epsilon_3])
        )
    return optimal_obj


def Bayesian_DRO_3_epsilon3(x, epsilon_3):
    temp_value = np.zeros(number_iteration_theta)
    for i in range(number_iteration_theta):
        temp_value[i] = Bayesian_DRO_2_epsilon3(x, i, epsilon_3[i])
    value = np.mean(temp_value)
    return value


def main_Bayesian_DRO_epsilon3(empirical_x):
    bnds = [(0, M)]
    initial_point = sol_true
    epsilon_3 = np.zeros(number_iteration_theta)
    for i in range(number_iteration_theta):
        epsilon_3[i] = calculate_epsilon3(empirical_x, i)
    res = optimize.minimize(
        Bayesian_DRO_3_epsilon3, initial_point, bounds=bnds, args=(epsilon_3)
    )
    optimal_x = res.x
    return optimal_x, epsilon_3


sol_true = rv_discrete(name="weighted_uniform", values=(xi_space, theta_c)).ppf(
    (b - c) / (h + b)
)
D_data = 10
replication = 200
epsilon_set = [
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    1,
    1.5,
    2,
    2.5,
    3,
]
data_eval = data_generation(replication)

solution_BRO = np.zeros(replication)

solution_empirical = np.zeros(replication)

solution_Bayesian_DRO = np.zeros([replication, len(epsilon_set)])

solution_epsilon_1 = np.zeros(replication)
epsilon_1 = np.zeros([replication, number_iteration_theta])

solution_epsilon_2 = np.zeros(replication)
epsilon_2 = np.zeros([replication, number_iteration_theta])

solution_epsilon_3 = np.zeros(replication)
epsilon_3 = np.zeros([replication, number_iteration_theta])

solution_empirical_DRO_Gotoh = np.zeros([replication, len(epsilon_set)])

obj_BRO = np.zeros(replication)

obj_empirical = np.zeros(replication)

obj_Bayesian_DRO = np.zeros([replication, len(epsilon_set)])

obj_epsilon_1 = np.zeros(replication)

obj_epsilon_2 = np.zeros(replication)

obj_epsilon_3 = np.zeros(replication)

obj_empirical_DRO_Gotoh = np.zeros([replication, len(epsilon_set)])

for k in range(replication):
    data = data_generation(D_data)
    number_of_data = len(data)
    count_data = np.int_(np.zeros(len(xi_space)))
    for index, value in enumerate(xi_space):
        count_data[index] = data.count(value)
    empirical_p = count_data / number_of_data

    theta = theta_generation(data, number_iteration_theta)

    solution_BRO[k] = main_BRO()[0]
    obj_BRO[k] = cost(solution_BRO[k], data_eval[k])

    solution_empirical[k] = main_empirical()[0]
    obj_empirical[k] = cost(solution_empirical[k], data_eval[k])

    for index, epsilon in enumerate(epsilon_set):
        solution_Bayesian_DRO[k, index] = main_Bayesian_DRO(epsilon)
        obj_Bayesian_DRO[k, index] = cost(solution_Bayesian_DRO[k, index], data_eval[k])

    solution_epsilon_1[k], epsilon_1[k, :] = main_Bayesian_DRO_epsilon1()
    obj_epsilon_1[k] = cost(solution_epsilon_1[k], data_eval[k])

    solution_epsilon_2[k], epsilon_2[k, :] = main_Bayesian_DRO_epsilon2()
    obj_epsilon_2[k] = cost(solution_epsilon_2[k], data_eval[k])

    solution_epsilon_3[k], epsilon_3[k, :] = main_Bayesian_DRO_epsilon3(
        solution_empirical[k]
    )
    obj_epsilon_3[k] = cost(solution_epsilon_3[k], data_eval[k])

    for index, epsilon in enumerate(epsilon_set):
        solution_empirical_DRO_Gotoh[k, index] = main_empirical_DRO_Gotoh(epsilon)
        obj_empirical_DRO_Gotoh[k, index] = cost(
            solution_empirical_DRO_Gotoh[k, index], data_eval[k]
        )
