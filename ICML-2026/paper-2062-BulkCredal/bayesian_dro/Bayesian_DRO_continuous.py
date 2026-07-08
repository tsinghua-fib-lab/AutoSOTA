
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import truncnorm
from scipy.stats import gamma
from scipy.stats import expon
from scipy.special import logsumexp
try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    gp = None
    GRB = None
from joblib import Parallel, delayed

NUMBER_ITERATION_THETA = 10
NUMBER_ITERATION_XI = 10
EPSILON_SET = [
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
b = 8
h = 3
LARGEST_X = 100
SMALLEST_X = 0
L = 100

DGP_MEAN_TRUNCATED_NORMAL = 10
DGP_STD_TRUNCATED_NORMAL = 10

NUM_OBSERVATIONS = 20  # Number of observations from true DGP


def xi_generation(theta, D_xi, random_state = None):
    """Likelihood generation"""
    xi = expon.rvs(scale=1 / theta, size=D_xi, random_state=random_state)
    return xi


def data_generation(num_observations, mean = DGP_MEAN_TRUNCATED_NORMAL, std = DGP_STD_TRUNCATED_NORMAL, random_state = None):
    """True DGP"""
    myclip_a, myclip_b = 0, np.inf

    a, b = (myclip_a - mean) / std, (
        myclip_b - mean
    ) / std
    data = truncnorm.rvs(
        a,
        b,
        loc=mean,
        scale=std,
        size=num_observations,
        random_state=random_state,
    )
    return data


def theta_generation(data, D_theta, random_state = None):
    """Posterior"""
    alpha0, beta0 = 1, 1
    alpha = alpha0 + data.shape[0]
    beta = beta0 + np.sum(data)
    theta = gamma.rvs(a=alpha, scale=1 / beta, size=D_theta, random_state=random_state)
    return theta


def cost(x, xi):
    return h * np.maximum(0, x - xi) + b * np.maximum(0, xi - x)


def nabla_cost(x, xi):
    if x > xi:
        return h
    elif x < xi:
        return -b
    else:
        return 0


def ecdf(x):
    x = np.sort(x)
    u, c = np.unique(x, return_counts=True)  # c is count, u is unique element
    n = len(x)
    y = (np.cumsum(c) - 0.5) / n

    def interpolate(input_data):
        yinterp = np.interp(input_data, u, y, left=0.0, right=1.0)
        return yinterp

    return interpolate


def cumulative_kl(x, reference_pdf, fraction=0.9):
    dx = np.diff(np.sort(np.unique(x)))
    ex = np.min(dx) * fraction
    n = len(x)
    P = ecdf(x)
    KL = (1.0 / n) * np.sum(np.log((P(x) - P(x - ex)) / (ex * reference_pdf)))
    return KL


def Bayesian_DRO_1(lam, xi, x, theta_index, epsilon):
    return lam * epsilon + lam * np.log(
        np.mean(np.exp(cost(x, xi[theta_index, :]) / lam))
    )


def Bayesian_DRO_1_lse(lam, xi, x, theta_index, epsilon):
    """Calculates Bayesian_DRO_1 with Log-Sum-Exp

    Notes:
        This avoids "RuntimeWarning: overflow encountered in exp" warning:
        $$
        \log \\frac{1}{N}\sum_{i=1}^N \exp\left( \\frac{c_i}{\lambda} \\right) =
        \log \left(\\frac{1}{N}\\right) + \log \sum_{i=1}^N \exp\left( \\frac{c_i}{\lambda} \\right)
        $$
    """
    cost_over_lam = cost(x, xi[theta_index, :]) / lam
    return lam * epsilon + lam * (
        np.log(1 / cost_over_lam.shape[0]) + logsumexp(cost_over_lam)
    )


def Bayesian_DRO_2(xi, x, theta_index, epsilon, lse = True):
    bnds = [(0.01, None)]
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
    if lse:
        res = optimize.minimize(
            Bayesian_DRO_1_lse,
            initial_point,
            bounds=bnds,
            args=(xi, x, theta_index, epsilon),
        )
    else:
        res = optimize.minimize(
            Bayesian_DRO_1,
            initial_point,
            bounds=bnds,
            args=(xi, x, theta_index, epsilon),
        )
    optimal_lambda = res.x
    optimal_obj = res.fun
    if np.isnan(optimal_obj):
        if lse:
            optimal_obj = Bayesian_DRO_1_lse(
                np.array([optimal_lambda, xi, x, theta_index, epsilon])
            )
        else:
            optimal_obj = Bayesian_DRO_1(
                np.array([optimal_lambda, xi, x, theta_index, epsilon])
            )
    return optimal_obj


def Bayesian_DRO_3(xi, x, epsilon, lse=True):
    num_iteration_theta = xi.shape[0]
    temp_value = np.zeros(num_iteration_theta)
    for i in range(num_iteration_theta):
        temp_value[i] = Bayesian_DRO_2(xi, x, i, epsilon, lse=lse)
    value = np.mean(temp_value)
    return value


def main_Bayesian_DRO(xi, epsilon, lse: bool = True):
    support = np.arange(SMALLEST_X, LARGEST_X + 1, 1)
    value_support = np.zeros(len(support))
    for i in range(len(support)):
        value_support[i] = Bayesian_DRO_3(xi, support[i], epsilon, lse=lse) 
    smallest_index = np.argmin(value_support)
    support_smallest = np.arange(
        support[smallest_index] - 1, support[smallest_index] + 1, 0.01
    )
    value_support_smallest = np.zeros(len(support_smallest))
    for i in range(len(support_smallest)):
        value_support_smallest[i] = Bayesian_DRO_3(xi, support_smallest[i], epsilon, lse=lse)  
    temp = support_smallest[np.argmin(value_support_smallest)]
    if temp <= LARGEST_X:
        optimal_x = temp
    else:
        optimal_x = LARGEST_X
    return optimal_x


def BRO(x, xi):
    return np.mean([np.mean(cost(x, xi[i, :])) for i in range(NUMBER_ITERATION_THETA)])


def main_BRO(sol_true, xi):
    bnds = [(SMALLEST_X, LARGEST_X)]
    initial_point = sol_true
    res = optimize.minimize(BRO, initial_point, bounds=bnds, args=(xi))
    optimal_x = res.x
    return optimal_x[0]


def main_empirical(data):
    ascend_data = np.sort(data)
    len_data = len(data)
    for i in range(1, len_data + 1):
        if (i - 1) / len_data < b / (h + b) and i / len_data >= b / (h + b):
            return ascend_data[i - 1]


def main_empirical_DRO_Wasserstein(data, epsilon, p):
    len_data = len(data)
    if p == 1:
        # put data in ascending order
        ascend_data = np.sort(data)
        for i in range(1, len_data + 1):
            if (i - 1) / len_data < b / (h + b) and i / len_data >= b / (h + b):
                return ascend_data[i - 1]
    if p > 1:
        Delta = (
            1
            / (h + b)
            * (1 / p) ** (1 / (p - 1))
            * ((p - 1) / p)
            * (b ** (p / (p - 1)) - h ** (p / (p - 1)))
        )
        Lambda = (1 / (h + b)) * (b ** (p / (p - 1)) * h + h ** (p / (p - 1)) * b)
        ascend_data = np.sort(data)
        for i in range(1, len_data + 1):
            if (i - 1) / len_data < b / (h + b) and i / len_data >= b / (h + b):
                temp = ascend_data[i - 1]
                break
        return temp + Delta * p ** (1 / (p - 1)) * epsilon * (1 / Lambda) ** (1 / p)


def Bayesian_DRO_1_epsilon1(lam, x, theta_index, epsilon_1, xi):
    return lam * epsilon_1 + lam * np.log(
        np.mean(np.exp(cost(x, xi[theta_index, :]) / lam))
    )


def Bayesian_DRO_2_epsilon1(x, theta_index, epsilon_1, xi):
    bnds = [(0.01, None)]
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
    res = optimize.minimize(
        Bayesian_DRO_1_epsilon1,
        initial_point,
        bounds=bnds,
        args=(x, theta_index, epsilon_1, xi),
    )
    optimal_lambda = res.x
    optimal_obj = res.fun
    if np.isnan(optimal_obj):
        optimal_obj = Bayesian_DRO_1_epsilon1(
            np.array([optimal_lambda, x, theta_index, epsilon_1, xi])
        )
    return optimal_obj


def Bayesian_DRO_3_epsilon1(x, epsilon_1, xi):
    temp_value = np.zeros(NUMBER_ITERATION_THETA)
    for i in range(NUMBER_ITERATION_THETA):
        temp_value[i] = Bayesian_DRO_2_epsilon1(x, i, epsilon_1[i], xi)
    value = np.mean(temp_value)
    return value


def main_Bayesian_DRO_epsilon1(data, theta, xi):
    epsilon_1 = np.zeros(NUMBER_ITERATION_THETA)
    for i in range(NUMBER_ITERATION_THETA):
        reference_pdf = expon.pdf(data, scale=1 / theta[i])
        temp = cumulative_kl(data, reference_pdf)
        epsilon_1[i] = temp

    support = np.arange(SMALLEST_X, LARGEST_X + 1, 1)
    value_support = np.zeros(len(support))
    for i in range(len(support)):
        value_support[i] = Bayesian_DRO_3_epsilon1(support[i], epsilon_1, xi)
    smallest_index = np.argmin(value_support)
    support_smallest = np.arange(
        support[smallest_index] - 1, support[smallest_index] + 1, 0.01
    )
    value_support_smallest = np.zeros(len(support_smallest))
    for i in range(len(support_smallest)):
        value_support_smallest[i] = Bayesian_DRO_3_epsilon1(
            support_smallest[i], epsilon_1, xi
        )
    temp = support_smallest[np.argmin(value_support_smallest)]
    if temp <= LARGEST_X:
        optimal_x = temp
    else:
        optimal_x = LARGEST_X
    return optimal_x, epsilon_1


def Bayesian_DRO_1_epsilon2(lam, x, theta_index, epsilon_2, xi):
    return lam * epsilon_2 + lam * np.log(
        np.mean(np.exp(cost(x, xi[theta_index, :]) / lam))
    )


def Bayesian_DRO_2_epsilon2(x, theta_index, epsilon_2):
    bnds = [(0.01, None)]
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


def Bayesian_DRO_3_epsilon2(x, epsilon_2):
    temp_value = np.zeros(NUMBER_ITERATION_THETA)
    for i in range(NUMBER_ITERATION_THETA):
        temp_value[i] = Bayesian_DRO_2_epsilon2(x, i, epsilon_2[i])
    value = np.mean(temp_value)
    return value


def main_Bayesian_DRO_epsilon2(data, theta):
    epsilon_2 = np.zeros(NUMBER_ITERATION_THETA)
    for i in range(NUMBER_ITERATION_THETA):
        reference_pdf = expon.pdf(data, scale=1 / theta[i])
        temp = cumulative_kl(data, reference_pdf)
        epsilon_2[i] = temp / 2

    support = np.arange(SMALLEST_X, LARGEST_X + 1, 1)
    value_support = np.zeros(len(support))
    for i in range(len(support)):
        value_support[i] = Bayesian_DRO_3_epsilon2(support[i], epsilon_2)
    smallest_index = np.argmin(value_support)
    support_smallest = np.arange(
        support[smallest_index] - 1, support[smallest_index] + 1, 0.01
    )
    value_support_smallest = np.zeros(len(support_smallest))
    for i in range(len(support_smallest)):
        value_support_smallest[i] = Bayesian_DRO_3_epsilon2(
            support_smallest[i], epsilon_2
        )
    temp = support_smallest[np.argmin(value_support_smallest)]
    if temp <= LARGEST_X:
        optimal_x = temp
    else:
        optimal_x = LARGEST_X
    return optimal_x, epsilon_2


def calculate_epsilon3(empirical_x, theta, theta_index):
    np.random.seed(0)
    sample_xi = xi_generation(theta[theta_index], L)
    temp_nabla = np.zeros(L)
    for i in range(L):
        temp_nabla[i] = nabla_cost(empirical_x, sample_xi[i])
    temp_reference = expon.pdf(sample_xi, loc=0, scale=1 / theta[theta_index])

    m = gp.Model("epsilon_3")
    temp_q = dict()
    temp_q_abs = dict()
    temp_difference = dict()
    for i in range(L):
        temp_q[i] = m.addVar(name="temp_q%s" % str([i]))  # probability mass
        temp_q_abs[i] = m.addVar(name="temp_q_abs%s" % str([i]))
        temp_difference[i] = m.addVar(
            lb=-temp_reference[i], name="temp_difference%s" % str([i])
        )
        m.addConstr(temp_q[i] >= 1e-4)
        m.addConstr(temp_difference[i] - temp_q[i] == -temp_reference[i])
        m.addGenConstrAbs(temp_q_abs[i], temp_difference[i])

    m.addConstr(gp.quicksum(temp_q[i] / temp_reference[i] for i in range(L)) == L)
    m.addConstr(
        gp.quicksum(temp_q[i] * temp_nabla[i] / temp_reference[i] for i in range(L))
        == 0
    )

    obj = gp.quicksum(temp_q_abs[i] for i in range(L))
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()
    initial_point = np.zeros(L)
    for i in range(L):
        initial_point[i] = m.getVarByName("temp_q%s" % str([i])).x

    cons = (
        {"type": "eq", "fun": lambda q: np.sum(q / temp_reference) - L},
        {"type": "eq", "fun": lambda q: np.sum(q * temp_nabla / temp_reference)},
    )
    bnds = [(0, 1)] * L
    options = {"maxiter": 10000}

    def epsilon3_obj(q, theta_index):
        return (
            np.sum(
                [
                    q[i] / temp_reference[i] * np.log(q[i] / temp_reference[i])
                    for i in range(L)
                ]
            )
            / L
        )

    res = optimize.minimize(
        epsilon3_obj,
        initial_point,
        bounds=bnds,
        constraints=cons,
        args=(theta_index),
        method="SLSQP",
        options=options,
    )
    optimal_q = res.x
    optimal_obj = res.fun
    if np.isnan(optimal_obj):
        optimal_obj = epsilon3_obj(optimal_q, theta_index)
    return optimal_obj


def Bayesian_DRO_1_epsilon3(lam, xi, x, theta_index, epsilon_3):
    return lam * epsilon_3 + lam * np.log(
        np.mean(np.exp(cost(x, xi[theta_index, :]) / lam))
    )


def Bayesian_DRO_2_epsilon3(xi, x, theta_index, epsilon_3):
    bnds = [(0.01, None)]
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
            np.array([optimal_lambda, xi, x, theta_index, epsilon_3])
        )
    return optimal_obj


def Bayesian_DRO_3_epsilon3(xi, x, epsilon_3):
    temp_value = np.zeros(NUMBER_ITERATION_THETA)
    for i in range(NUMBER_ITERATION_THETA):
        temp_value[i] = Bayesian_DRO_2_epsilon3(xi, x, i, epsilon_3[i])
    value = np.mean(temp_value)
    return value


def main_Bayesian_DRO_epsilon3(xi, empirical_x, theta):
    epsilon_3 = np.zeros(NUMBER_ITERATION_THETA)
    for i in range(NUMBER_ITERATION_THETA):
        epsilon_3[i] = calculate_epsilon3(empirical_x, theta, i)
    support = np.arange(SMALLEST_X, LARGEST_X + 1, 1)
    value_support = np.zeros(len(support))
    for i in range(len(support)):
        value_support[i] = Bayesian_DRO_3_epsilon3(xi, support[i], epsilon_3)
    smallest_index = np.argmin(value_support)
    support_smallest = np.arange(
        support[smallest_index] - 1, support[smallest_index] + 1, 0.01
    )
    value_support_smallest = np.zeros(len(support_smallest))
    for i in range(len(support_smallest)):
        value_support_smallest[i] = Bayesian_DRO_3_epsilon3(
            xi, support_smallest[i], epsilon_3
        )
    temp = support_smallest[np.argmin(value_support_smallest)]
    if temp <= LARGEST_X:
        optimal_x = temp
    else:
        optimal_x = LARGEST_X
    return optimal_x, epsilon_3


def main():
    print("Hello, world!")
    sol_true = truncnorm.ppf(
        (b - 0) / (h + b), a=-DGP_MEAN_TRUNCATED_NORMAL / DGP_STD_TRUNCATED_NORMAL, b=np.inf, loc=DGP_MEAN_TRUNCATED_NORMAL, scale=DGP_STD_TRUNCATED_NORMAL
    )
    replication = 200

    generator = np.random.default_rng(seed=replication)
    data_eval = data_generation(replication, random_state=generator) # one data point for each replication

    # solution_BRO = np.zeros(replication)

    # solution_empirical = np.zeros(replication)

    solution_Bayesian_DRO = np.zeros([replication, len(EPSILON_SET)])
    solution_Bayesian_DRO_lse = np.zeros([replication, len(EPSILON_SET)])

    # solution_epsilon_1 = np.zeros(replication)
    # epsilon_1 = np.zeros([replication, NUMBER_ITERATION_THETA])

    # solution_epsilon_2 = np.zeros(replication)
    # epsilon_2 = np.zeros([replication, NUMBER_ITERATION_THETA])

    # solution_epsilon_3 = np.zeros(replication)
    # epsilon_3 = np.zeros([replication, NUMBER_ITERATION_THETA])

    # solution_empirical_DRO_Wasserstein = np.zeros([replication, len(EPSILON_SET)])

    # obj_BRO = np.zeros(replication)

    # obj_empirical = np.zeros(replication)

    obj_Bayesian_DRO = np.zeros([replication, len(EPSILON_SET)])
    obj_Bayesian_DRO_lse = np.zeros([replication, len(EPSILON_SET)])

    # obj_epsilon_1 = np.zeros(replication)

    # obj_epsilon_2 = np.zeros(replication)

    # obj_epsilon_3 = np.zeros(replication)

    # obj_empirical_DRO_Wasserstein = np.zeros([replication, len(EPSILON_SET)])
    print("Starting main loop")

    def main_loop(k):
        print()
        print()
        print("### Running replication", k)
        print()
        generator = np.random.default_rng(seed=k)
        data = data_generation(NUM_OBSERVATIONS, random_state=generator)
        theta = theta_generation(data, NUMBER_ITERATION_THETA, random_state=generator)
        xi = np.zeros([NUMBER_ITERATION_THETA, NUMBER_ITERATION_XI])
        for i in range(NUMBER_ITERATION_THETA):
            xi[i] = xi_generation(theta[i], NUMBER_ITERATION_XI, random_state=generator)

        # solution_BRO[k] = main_BRO(sol_true, xi)
        # obj_BRO[k] = cost(solution_BRO[k], data_eval[k])

        # solution_empirical[k] = main_empirical(data)
        # obj_empirical[k] = cost(solution_empirical[k], data_eval[k])

        # without Log-Sum-Exp fix (original Bayesian DRO code)
        for index, epsilon in enumerate(EPSILON_SET):
            print("Bayesian DRO. Epsilon = ", epsilon)
            solution_Bayesian_DRO[k, index] = main_Bayesian_DRO(xi, epsilon, lse=False)
            obj_Bayesian_DRO[k, index] = cost(
                solution_Bayesian_DRO[k, index], data_eval[k]
            )

        # with Log-Sum-Exp adjustment
        print()
        for index, epsilon in enumerate(EPSILON_SET):
            print("Bayesian DRO LSE. Epsilon = ", epsilon)
            solution_Bayesian_DRO_lse[k, index] = main_Bayesian_DRO(xi, epsilon, lse=True)
            obj_Bayesian_DRO_lse[k, index] = cost(
                solution_Bayesian_DRO_lse[k, index], data_eval[k]
            )


        # solution_epsilon_1[k], epsilon_1[k, :] = main_Bayesian_DRO_epsilon1(data, theta, xi)
        # obj_epsilon_1[k] = cost(solution_epsilon_1[k], data_eval[k])

        # solution_epsilon_2[k], epsilon_2[k, :] = main_Bayesian_DRO_epsilon2(data, theta)
        # obj_epsilon_2[k] = cost(solution_epsilon_2[k], data_eval[k])

        # solution_epsilon_3[k], epsilon_3[k, :] = main_Bayesian_DRO_epsilon3(
        #     xi, solution_empirical[k], theta
        # )
        # obj_epsilon_3[k] = cost(solution_epsilon_3[k], data_eval[k])

        # print()
        # for index, epsilon in enumerate(EPSILON_SET):
        #     print("Wasserstein DRO. Epsilon = ", epsilon)
        #     solution_empirical_DRO_Wasserstein[
        #         k, index
        #     ] = main_empirical_DRO_Wasserstein(data, epsilon, 2)
        #     obj_empirical_DRO_Wasserstein[k, index] = cost(
        #         solution_empirical_DRO_Wasserstein[k, index], data_eval[k]
        #     )
        df = pd.DataFrame({
            "replication": [k]*len(EPSILON_SET),
            "epsilon": EPSILON_SET,
            # "wasserstein_dro_sol": solution_empirical_DRO_Wasserstein[k],
            # "wasserstein_dro_cost": obj_empirical_DRO_Wasserstein[k],
            "bayesian_dro_sol": solution_Bayesian_DRO[k],
            "bayesian_dro_cost": obj_Bayesian_DRO[k],
            "bayesian_dro_lse_sol": solution_Bayesian_DRO_lse[k],
            "bayesian_dro_lse_cost": obj_Bayesian_DRO_lse[k],
        })


    Parallel(n_jobs=-1)(delayed(main_loop)(k) for k in range(replication))


if __name__ == "__main__":
    main()
