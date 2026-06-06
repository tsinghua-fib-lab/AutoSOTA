import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.stats import norm

def theta_eps(x, eps):
    return (1 - norm.cdf((eps / x) - (x / 2))) - np.exp(eps) * (1 - norm.cdf((eps / x) + (x / 2)))

def dobrushin(eps, delta, sigma=1.0, C=1.0):
    eps_tild = np.log(1 + (np.exp(eps) - 1) / delta)
    gamma = theta_eps(2 * C / sigma, eps_tild)
    return eps, gamma * delta

def weak_dobrushin(eps, delta, sigma=1.0, C=1.0):
    gamma = 2 * norm.cdf(C / sigma) - 1
    return eps, gamma * delta

def ultra_mixing(eps, delta, sigma=1.0, C=1.0):
    gamma = 1 - np.exp(-2 * (C / sigma) ** 2)
    eps_out = np.log(1 + gamma * (np.exp(eps) - 1))
    delta_out = gamma * delta * np.exp(eps_out - eps)
    return eps_out, delta_out

def renyi2dp(eps_renyi, delta=0.1):
    alpha = np.linspace(1.001, 100, 1000)
    eps_array = eps_renyi - (np.log(delta) + np.log(alpha)) / (alpha - 1) + np.log((alpha - 1) / alpha)
    eps = np.min(eps_array)
    return eps, delta

def pabi_renyi(num_iter, sigma=1.0, C=1.0, lr=1e-3, C_0=10.0, l2_reg=1e-2):
    factor = 1 - lr * l2_reg

    sum_1 = np.sum([factor ** (num_iter - t - 1) for t in range(num_iter)])
    sum_2 = np.sum([factor ** (2 * (num_iter - t - 1)) for t in range(num_iter)])

    eps = 0.5 * ((2 * C_0 * factor ** num_iter + 2 * lr * C * sum_1) ** 2) / (sigma ** 2 * sum_2)
    return eps

def pabi(num_iter, sigma=1.0, C=1.0, lr=1e-4, C_0=5.0, l2_reg=1e-2, delta=0.1):
    eps_renyi = pabi_renyi(num_iter, sigma=sigma, C=C, lr=lr, C_0=C_0, l2_reg=l2_reg)
    return renyi2dp(eps_renyi, delta=delta)

# Function to plot eps_t or delta_t as a function of a single parameter
def plot_parameter_effect(param_name, param_values, update_function, fixed_params):
    eps_values = []
    delta_values = []

    for param_value in param_values:
        params = fixed_params.copy()
        params[param_name] = param_value

        if update_function.__name__ == "pabi":
            eps_t, delta_t = update_function(params['num_iter'], sigma=params['sigma'], C=params['C'], lr=params['lr'], C_0=params['C_0'], l2_reg=params['l2_reg'], delta=params['delta'])
        else:
            eps_t, delta_t = params['eps'], params['delta']
            for i in range(params['num_iter']):
                eps_t, delta_t = update_function(eps_t, delta_t, sigma=params['sigma'], C=params['C'])

        eps_values.append(eps_t)
        delta_values.append(delta_t)

    print("epsilon: {}".format(eps_values))
    print("delta: {}".format(delta_values))
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, eps_values, label=f"eps_t vs {param_name}", marker='o')
    plt.plot(param_values, delta_values, label=f"delta_t vs {param_name}", marker='s')
    plt.xlabel(param_name)
    plt.ylabel("Value")
    plt.title(f"Effect of {param_name} on eps_t and delta_t")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
fixed_params = {
    'num_iter': 10,
    'sigma': 1.0,
    'C': 1.0,
    'lr': 1e-4,
    'C_0': 2.0,
    'l2_reg': 1e-2,
    'delta': 0.1,
    # 'delta': 1.0,
    'eps': 1.0
}

# Select parameter to vary and its range
param_name = 'num_iter'
# param_name = 'lr'
param_values = range(1, 20)  # Example range of num_iter
# param_values = np.logspace(-4, 0, num=10)  # Example range of num_iter

# Plot for PABI
plot_parameter_effect(param_name, param_values, pabi, fixed_params)

# Plot for Dobrushin
# plot_parameter_effect(param_name, param_values, dobrushin, fixed_params)
