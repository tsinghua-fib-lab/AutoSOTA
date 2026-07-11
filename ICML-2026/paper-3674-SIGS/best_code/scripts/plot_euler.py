import numpy as np
import matplotlib.pyplot as plt

# 1. Setup the Grid
N = 200  # Resolution
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)

# 2. The Data Strings
data = {
    "rho": {
        "man": "exp(-0.0887135327298883*sin(pi*x)*sin(pi*y) + 0.503913913782121*sin(pi*x)*sin(2*pi*y) + 0.259377112129216*sin(2*pi*x)*sin(pi*y) + 0.139524166394621*sin(2*pi*x)*sin(2*pi*y))",
        "opt": "exp(0.273218273614685*sin(pi*x)*sin(2*pi*y) + 0.217398243729459*sin(2*pi*x)*sin(pi*y) + 0.22919159522389*sin(2*pi*x)*sin(2*pi*y) + 0.00218307236194415*sin(2*pi*y)*cos(pi*x))"
    },
    "u": {
        "man": "pi*(-0.243231551944921*sin(pi*x)*sin(pi*y) - 0.384609818580336*sin(pi*x)*sin(2*pi*y) - 0.494077541781533*sin(2*pi*x)*sin(pi*y) + 0.517851271572421*sin(2*pi*x)*sin(2*pi*y))",
        "opt": "-0.92927405*sin(pi*x)*sin(pi*y) - sin(pi*x)*sin(2*pi*y) + 0.051755502*sin(pi*x)*cos(3*pi*y) - 1.5016502*sin(2*pi*x)*sin(pi*y) + 1.5556752*sin(2*pi*x)*sin(2*pi*y)"
    },
    "v": {
        "man": "pi*(0.0714991104833803*sin(pi*x)*sin(pi*y) + 0.232632214102786*sin(pi*x)*sin(2*pi*y) - 0.536002830111012*sin(2*pi*x)*sin(pi*y) + 0.664552886020228*sin(2*pi*x)*sin(2*pi*y))",
        "opt": "0.71823139*sin(pi*x)*sin(2*pi*y) + 2.0506258*sin(2*pi*x)*sin(2*pi*y) - 0.30668526*sin(4*pi*x)*sin(pi*y) - 1.222399*sin(pi*y)*cos(pi*x) + sin(pi*y)*cos(3*pi*x)"
    },
    "p": {
        "man": "exp(0.235072445665542*sin(pi*x)*sin(pi*y) - 0.321614651495709*sin(pi*x)*sin(2*pi*y) - 0.355730501034074*sin(2*pi*x)*sin(pi*y) - 0.447733635951432*sin(2*pi*x)*sin(2*pi*y))",
        "opt": "exp(-0.389124179603661*sin(pi*x)*sin(2*pi*y) - 0.354278175986934*sin(2*pi*x)*sin(pi*y) - 0.410140846300394*sin(2*pi*x)*sin(2*pi*y))"
    }
}

def eval_str(expr, X, Y):
    """Safely evaluates the expression string using numpy functions."""
    local_dict = {"sin": np.sin, "cos": np.cos, "exp": np.exp, "pi": np.pi, "x": X, "y": Y}
    return eval(expr, {"__builtins__": None}, local_dict)

# 3. Create Plots
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(10, 16))

variables = ['rho', 'u', 'v', 'p']

for i, var in enumerate(variables):
    # Evaluate fields
    Z_man = eval_str(data[var]['man'], X, Y)
    Z_opt = eval_str(data[var]['opt'], X, Y)


    # Plot Manufactured (Left)
    im1 = axes[i, 0].imshow(Z_man, origin='lower', extent=[0, 1, 0, 1],
                             cmap='RdBu_r')
    axes[i, 0].set_title(f'{var} (Manufactured)')
    axes[i, 0].set_ylabel(var, fontsize=14, rotation=0, labelpad=20)
    fig.colorbar(im1, ax=axes[i, 0], fraction=0.046, pad=0.04)

    # Plot Optimized (Right)
    im2 = axes[i, 1].imshow(Z_opt, origin='lower', extent=[0, 1, 0, 1],
                             cmap='RdBu_r')
    axes[i, 1].set_title(f'{var} (SIGS Optimized)')
    fig.colorbar(im2, ax=axes[i, 1], fraction=0.046, pad=0.04)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('sigs_manufactured_vs_optimized.png', dpi=300)
plt.show()
