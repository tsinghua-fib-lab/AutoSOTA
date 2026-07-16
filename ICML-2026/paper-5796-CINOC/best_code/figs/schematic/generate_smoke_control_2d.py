import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle

# Set random seed for reproducibility
np.random.seed(42)

# Domain parameters for 2D
Nx, Ny = 64, 80  # Grid points
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1.25, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# Resolution: 640 (width) x 1080 (height)
dpi = 100
fig_width = 640 / dpi
fig_height = 1080 / dpi

# ═══════════════════════════════════════════════════════════════════════════════
# ACADEMIC STYLING
# ═══════════════════════════════════════════════════════════════════════════════

def setup_academic_style():
    """Configure matplotlib for academic/conference style."""
    tex_fonts = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 11,
        "font.size": 10,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.titlesize": 12,
        "figure.titlesize": 13,
        "axes.linewidth": 1.2,
        "lines.linewidth": 2.0,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.8,
    }
    plt.rcParams.update(tex_fonts)

def create_target_shape(X, Y, shape_type='circle'):
    """Create target smoke shape (circle or square)."""
    if shape_type == 'circle':
        # Circle centered at (0.5, 0.75)
        cx, cy = 0.5, 0.75
        r = 0.25
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        target = 0.8 * np.exp(-(dist**2) / (2 * (r/3)**2))
        target[dist > r] = 0
    else:
        # Square
        target = np.zeros_like(X)
        mask = (X > 0.3) & (X < 0.7) & (Y > 0.55) & (Y < 0.95)
        target[mask] = 0.7
    return target

def create_initial_smoke(X, Y):
    """Create initial smoke distribution (blob at bottom)."""
    # Smoke blob at bottom center
    cx, cy = 0.5, 0.3
    smoke = 0.9 * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * 0.15**2))
    return smoke

def simulate_smoke_evolution(initial, target, n_actuators=9, n_steps=50):
    """Simulate schematic smoke evolution with kinetic actuators."""
    states = []
    errors = np.zeros((n_steps, Nx, Ny))
    actuator_trajectories = []

    # Kinetic actuators: start in one configuration, end in another
    n_side = int(np.sqrt(n_actuators))

    # Initial positions (3x3 grid, lower)
    actuator_positions_init = []
    for i in range(n_side):
        for j in range(n_side):
            x_pos = 0.2 + 0.6 * i / (n_side - 1)
            y_pos = 0.2 + 0.7 * j / (n_side - 1)  # Start lower
            actuator_positions_init.append([x_pos, y_pos])
    actuator_positions_init = np.array(actuator_positions_init)

    # Final positions (moved to track target)
    actuator_positions_final = []
    for i in range(n_side):
        for j in range(n_side):
            x_pos = 0.25 + 0.5 * i / (n_side - 1)
            y_pos = 0.4 + 0.7 * j / (n_side - 1)  # Moved up to target region
            actuator_positions_final.append([x_pos, y_pos])
    actuator_positions_final = np.array(actuator_positions_final)

    # Evolution: interpolate from initial to target
    for t in range(n_steps):
        alpha = t / (n_steps - 1)
        # Smooth transition
        state = initial * (1 - alpha**0.6) + target * alpha**0.6
        states.append(state)
        errors[t] = np.abs(state - target)

        # Kinetic actuators move over time
        actuator_positions = (1 - alpha) * actuator_positions_init + alpha * actuator_positions_final
        actuator_trajectories.append(actuator_positions)

    return states, errors, actuator_trajectories

def create_smoke_control_schematic(save_path):
    """Create 2D smoke control schematic."""

    setup_academic_style()

    # Create smoke distributions
    initial = create_initial_smoke(X, Y)
    target = create_target_shape(X, Y, shape_type='circle')

    # Simulate evolution with kinetic actuators
    states, errors, actuator_trajectories = simulate_smoke_evolution(initial, target)

    # Create figure with 3 rows (squeezed vertically with careful spacing)
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    gs = fig.add_gridspec(3, 2, hspace=0.50, wspace=0.50,
                         top=0.93, bottom=0.09, left=0.10, right=0.90,
                         width_ratios=[1, 1])

    # Colors
    color_initial = '#2166ac'
    color_target = '#b2182b'

    # ====== Row 1: Initial and Target States ======
    ax1a = fig.add_subplot(gs[0, 0])
    im1a = ax1a.imshow(initial.T, origin='lower', extent=[0, 1, 0, 1.25],
                       cmap='RdBu_r', vmin=0, vmax=1, aspect='auto')
    ax1a.set_title('Initial State', fontsize=11, fontweight='bold')
    ax1a.set_xlabel('x', fontsize=11)
    ax1a.set_ylabel('y', fontsize=11)
    ax1a.set_xticks([0, 0.5, 1])
    ax1a.set_yticks([0, 0.5, 1.0])
    cbar1a = plt.colorbar(im1a, ax=ax1a, fraction=0.046, pad=0.08, aspect=25)
    cbar1a.set_label('Smoke Density', fontsize=9)
    cbar1a.ax.tick_params(labelsize=8)

    ax1b = fig.add_subplot(gs[0, 1])
    im1b = ax1b.imshow(target.T, origin='lower', extent=[0, 1, 0, 1.25],
                       cmap='RdBu_r', vmin=0, vmax=1, aspect='auto')
    # Draw target shape contour
    ax1b.contour(X, Y, target, levels=[0.3], colors='lime', linewidths=2, linestyles='--')
    ax1b.set_title('Target Shape', fontsize=11, fontweight='bold')
    ax1b.set_xlabel('x', fontsize=11)
    ax1b.set_ylabel('y', fontsize=11)
    ax1b.set_xticks([0, 0.5, 1])
    ax1b.set_yticks([0, 0.5, 1.0])
    cbar1b = plt.colorbar(im1b, ax=ax1b, fraction=0.046, pad=0.08, aspect=25)
    cbar1b.set_label('Smoke Density', fontsize=9)
    cbar1b.ax.tick_params(labelsize=8)

    # Add main problem title
    fig.suptitle('Density Transportation', fontsize=20, fontweight='bold', y=0.985)

    # Add panel labels more cleanly
    ax1a.text(0.02, 0.98, '(a)', transform=ax1a.transAxes, fontsize=12,
             fontweight='bold', va='top', ha='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ====== Row 2: Evolution Snapshots ======
    # Show 2 snapshots: early and late
    snapshot_indices = [10, 45]  # Early and late in evolution

    for col_idx, t_idx in enumerate(snapshot_indices):
        ax = fig.add_subplot(gs[1, col_idx])
        state = states[t_idx]
        im = ax.imshow(state.T, origin='lower', extent=[0, 1, 0, 1.25],
                      cmap='RdBu_r', vmin=0, vmax=1, aspect='auto')

        # Overlay kinetic actuator positions at this timestep
        actuator_positions_t = actuator_trajectories[t_idx]
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(actuator_positions_t)))
        for i, (ax_x, ax_y) in enumerate(actuator_positions_t):
            ax.scatter(ax_x, ax_y, c=[colors[i]], s=70,
                      edgecolors='white', linewidths=1.5, zorder=10, marker='o')

        # Draw target contour
        ax.contour(X, Y, target, levels=[0.3], colors='lime',
                  linewidths=1.5, linestyles='--', alpha=0.7)

        time_label = 'Early' if col_idx == 0 else 'Final'
        ax.set_title(f'Evolution ({time_label})', fontsize=11, fontweight='bold')
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('y', fontsize=11)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1.0])

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.08, aspect=25)
        cbar.ax.tick_params(labelsize=8)
        if col_idx == 0:
            cbar.set_label('Smoke Density', fontsize=9)

    # Add panel label for row 2 (on first subplot)
    ax_first_row2 = fig.axes[2]  # First subplot of row 2
    ax_first_row2.text(0.02, 0.98, '(b)', transform=ax_first_row2.transAxes,
                      fontsize=12, fontweight='bold', va='top', ha='left',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ====== Row 3: Space-Time Error (spanning both columns) ======
    ax3 = fig.add_subplot(gs[2, :])
    # Average error over x-direction to create space-time plot
    errors_avg_x = np.mean(errors, axis=1)  # (n_steps, Ny)

    from scipy.ndimage import zoom
    errors_highres = zoom(errors_avg_x, (3, 1), order=1)

    im3 = ax3.imshow(errors_highres.T, aspect='auto', origin='lower',
                    extent=[0, len(states), 0, 1.25], cmap='hot',
                    vmin=0, vmax=np.max(errors_avg_x) * 1.05)
    ax3.set_xlabel('Time Step', fontsize=11)
    ax3.set_ylabel('y Position', fontsize=11)
    ax3.set_title('(c) Space-Time Error Evolution (x-averaged)', fontsize=12, fontweight='bold')
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.025, pad=0.04, aspect=30)
    cbar3.set_label('Error Magnitude', fontsize=10)
    cbar3.ax.tick_params(labelsize=8)

    # Save in multiple formats
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved smoke_control schematic (PNG) to {save_path}")

    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"Saved smoke_control schematic (PDF) to {pdf_path}")

    svg_path = save_path.replace('.png', '.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    print(f"Saved smoke_control schematic (SVG) to {svg_path}")

    plt.close()

# Generate the schematic
create_smoke_control_schematic(
    "/mnt/d/Repositories/Tesseract-Hackathon/figs/schematic/smoke_control.png"
)

print("\n" + "=" * 70)
print("  SMOKE CONTROL SCHEMATIC (2D) GENERATION COMPLETE")
print("=" * 70)
