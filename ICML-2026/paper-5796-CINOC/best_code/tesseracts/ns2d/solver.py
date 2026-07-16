"""
2D Incompressible Navier-Stokes Solver with Smoke Density (PhiFlow)

Direct copy from official PhiFlow tutorial:
https://tum-pbs.github.io/PhiFlow/Fluids_Tutorial.html

Key: Uses Solve(rank_deficiency=0) to handle pressure solve properly.
"""

from phi.flow import *
import matplotlib.pyplot as plt
import numpy as np


def simulate_smoke_plumes():
    """
    Official PhiFlow smoke simulation pattern.
    
    From: https://tum-pbs.github.io/PhiFlow/Fluids_Tutorial.html
    """
    
    # Grid setup
    smoke = CenteredGrid(0, extrapolation.BOUNDARY, x=64, y=80, bounds=Box(x=64, y=80))
    velocity = StaggeredGrid(0, extrapolation.ZERO, x=64, y=80, bounds=Box(x=64, y=80))
    
    # Inflow locations (multiple smoke sources at bottom)
    INFLOW_LOCATION = tensor([(16, 8), (32, 8), (48, 8)], batch('inflow_loc'), channel(vector='x,y'))
    INFLOW = 0.6 * CenteredGrid(
        Sphere(center=INFLOW_LOCATION, radius=4), 
        extrapolation.BOUNDARY, 
        x=64, y=80, 
        bounds=Box(x=64, y=80)
    )
    
    # Storage
    trajectory = []
    
    N_STEPS = 80
    print(f"Running {N_STEPS} steps...")
    
    for i in range(N_STEPS):
        # 1. Advect smoke + add inflow
        smoke = advect.mac_cormack(smoke, velocity, dt=1) + INFLOW
        
        # 2. Buoyancy force (smoke pushes velocity upward)
        buoyancy_force = smoke * (0, 0.5) @ velocity
        
        # 3. Advect velocity by itself + add buoyancy
        velocity = advect.semi_lagrangian(velocity, velocity, dt=1) + buoyancy_force
        
        # 4. Make incompressible - KEY: use Solve(rank_deficiency=0)
        velocity, _ = fluid.make_incompressible(velocity, (), Solve(rank_deficiency=0))
        
        # Store snapshots  
        if i % 4 == 0:
            # Sum over batch dimension to get single image
            smoke_summed = math.sum(smoke.values, 'inflow_loc')
            trajectory.append(smoke_summed.numpy('x,y'))
        
        if i % 10 == 0:
            print(f"  Step {i}")
    
    return trajectory


def setup_academic_style():
    """Configure matplotlib for academic/conference style."""
    tex_fonts = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 18,
        "font.size": 16,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.titlesize": 20,
        "figure.titlesize": 24,
        "axes.linewidth": 1.5,
        "lines.linewidth": 2.0,
        "grid.alpha": 0.3,
        "grid.linewidth": 1.0,
    }
    plt.rcParams.update(tex_fonts)


def plot_smoke(trajectory, filename='ns2d_phiflow_test.png'):
    """
    Conference-quality visualization of smoke evolution.
    Uses contourf for smooth rendering and academic styling.
    """
    setup_academic_style()
    
    n_snapshots = len(trajectory)
    n_plots = min(6, n_snapshots)
    indices = [int(i * (n_snapshots - 1) / (n_plots - 1)) for i in range(n_plots)]
    
    # Create figure with extra space on right for colorbar
    fig = plt.figure(figsize=(20, 12))
    
    # GridSpec: 2 rows, 4 cols (last col for colorbar)
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.08], wspace=0.25, hspace=0.3)
    
    vmax = max(trajectory[i].max() for i in indices)
    vmax = max(vmax, 0.1)
    
    # Grid for contourf
    Nx, Ny = trajectory[0].shape
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1.25, Ny)  # Domain aspect ratio
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    levels = np.linspace(0, vmax, 50)
    
    cf = None
    for i, idx in enumerate(indices):
        row, col = i // 3, i % 3
        ax = fig.add_subplot(gs[row, col])
        smoke_data = trajectory[idx]
        
        # Use contourf for smooth visualization
        cf = ax.contourf(X, Y, smoke_data, levels=levels, cmap='hot', extend='max')
        
        # Add contour lines for structure
        ax.contour(X, Y, smoke_data, levels=10, colors='white', alpha=0.3, linewidths=0.5)
        
        ax.set_title(f'$t = {idx * 4}$', fontweight='bold')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_aspect('equal')
        
        # Mark inflow locations
        inflow_xs = [0.25, 0.5, 0.75]
        for ix in inflow_xs:
            ax.scatter(ix, 0.1, s=60, c='cyan', marker='^', 
                      edgecolors='white', linewidth=1, zorder=10)
    
    # Colorbar in dedicated column
    cax = fig.add_subplot(gs[:, 3])
    cbar = fig.colorbar(cf, cax=cax, orientation='vertical')
    cbar.set_label(r'Smoke Density $\rho$', fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    
    plt.suptitle('2D Navier-Stokes Smoke Simulation', 
                 fontsize=24, fontweight='bold', y=0.98)
    
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {filename}")
    plt.close()



if __name__ == "__main__":
    print("="*60)
    print("2D Navier-Stokes Smoke Simulation (PhiFlow)")
    print("="*60)
    print("\nUsing official PhiFlow tutorial code")
    print("Grid: 64x80, 3 inflow sources")
    
    trajectory = simulate_smoke_plumes()
    
    print(f"\nCollected {len(trajectory)} snapshots")
    plot_smoke(trajectory)
    
    print("\nDone!")
