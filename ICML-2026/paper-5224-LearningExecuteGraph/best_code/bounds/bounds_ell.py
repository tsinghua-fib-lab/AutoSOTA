import jax
import jax.numpy as jnp
from neural_tangents import stax
import neural_tangents as nt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # Add this import for CSV handling

import sys
import os

from templates import flooding
from templates import flooding_utils

import scienceplots
from matplotlib.ticker import FuncFormatter
jax.config.update('jax_enable_x64', True)

plt.style.use('science')

D=2
L = 15
d = D**2+1
d_G = 6 # graph diameter

l_range = np.arange(1, L+1) 
sigma_mu_square_ratio = np.zeros(L, dtype=np.float64)
iterations = np.zeros(L, dtype=np.float64)
enc_input_dim = np.zeros(L, dtype=np.float64)
all_possible_input = np.zeros(L, dtype=np.float64)

for l in range(1, L+1):

    X, Y_train = flooding.get_dataset(l,d)
    X_train = jnp.eye(len(X))
    Y_train = jnp.array(Y_train, dtype=jnp.float64)
    
    n0 = len(X)
    # Define a 2-layer ReLU network using neural-tangents' stax API
    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(1024),   # First dense layer with 128 units
        stax.Relu(),       # ReLU activation
        stax.Dense(n0)      # Output dense layer with 1 unit
    )

    iterations[l-1] = d_G*(d+4)
    enc_input_dim[l-1] = Y_train.shape[0]
    all_possible_input[l-1] = 2**n0

    all_one = [1 for i in range(l)]

    X_test = flooding.get_sample(l,d)[1]
    X_test['message'][0] = 1
    X_test['my_slot'][0][0] = 1
    _, X_test = flooding_utils.encode_data(X_test, X)
    X_test = jnp.array(X_test, dtype=jnp.float64).reshape(1, -1)
   
    predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, X_train, Y_train)
    y_pred = predict_fn(x_test=X_test, get='ntk', compute_cov=True)

    # Get absolute value of b
    abs_b = max(0,y_pred.covariance[0, 0])

    # Square each element of a
    a_squared = (y_pred.mean[0]) ** 2  

    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-10

    # Calculate the ratio for each element: abs(b) / (a_element^2 + epsilon)
    ratios = abs_b / (a_squared + epsilon)  

    # Find the maximum value
    max_ratio = jnp.max(ratios)

    sigma_mu_square_ratio[l-1] = max_ratio.item()

xrange = np.arange(1, len(sigma_mu_square_ratio)+1)
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1
axs[0].plot(xrange, sigma_mu_square_ratio)
axs[0].set_title('Sigma/Mu Square Ratio')
axs[0].set_xlabel('Index')
axs[0].set_ylabel('Ratio')

# Plot 2
Num_bound = 8*sigma_mu_square_ratio*np.log(2*enc_input_dim*iterations*all_possible_input)
Thr_bound = 8*enc_input_dim*np.log(2*enc_input_dim*iterations*all_possible_input)
axs[1].plot(xrange, Thr_bound, label='Thr_bound')
axs[1].plot(xrange, Num_bound, label='Num_bound')
axs[1].set_yscale('log')
axs[1].set_title('Thr_bound and Num_bound (log scale)')
axs[1].set_xlabel('Index')
axs[1].set_ylabel('Value')
axs[1].legend()

# Create results folder if it doesn't exist
figure_folder = "figures"
if not os.path.exists(figure_folder):
    os.makedirs(figure_folder)
    print(f"Created folder: {figure_folder}")

figure_path = os.path.join(figure_folder, 'ell_plot.png')

# Save the plot to a file
plt.tight_layout()  # Adjust layout to prevent overlapping
plt.savefig(figure_path, dpi=300, bbox_inches='tight')  # Save as PNG
# plt.show()  # Uncomment if you also want to display the plot

# Save data to CSV file
data_dict = {
    'l': l_range,
    'Num_bound': Num_bound,
    'Thr_bound': Thr_bound,
}

# Create DataFrame
df = pd.DataFrame(data_dict)

# Create results folder if it doesn't exist
results_folder = "results"
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
    print(f"Created folder: {results_folder}")

# Save to CSV
csv_filename = 'ell_data.csv'

# Define the file path
file_path = os.path.join(results_folder, csv_filename)

df.to_csv(file_path, index=False)
print(f"Data saved to {csv_filename}")

