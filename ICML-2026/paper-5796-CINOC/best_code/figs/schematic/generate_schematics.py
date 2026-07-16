import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch

# Set random seed for reproducibility
np.random.seed(42)

# Domain parameters
N = 100  # Number of spatial points
x = np.linspace(0, 1, N)
T_steps = 50  # Number of time steps for visualization

# Resolution: 640 (width) x 1080 (height) -> convert to inches at 100 DPI
dpi = 100
fig_width = 640 / dpi
fig_height = 1080 / dpi

# ═══════════════════════════════════════════════════════════════════════════════
# ACADEMIC STYLING (Times New Roman / Serif) - matching conference style
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

def create_target_curve(x):
    """Create a smooth asymmetric target curve with boundary conditions = 0"""
    # Create asymmetric curve to demonstrate kinetic actuators
    # Sharp peak on one side, gradual decay on the other
    curve = 0.6 * np.exp(-((x - 0.35) ** 2) / 0.02) + 0.3 * np.exp(-((x - 0.7) ** 2) / 0.06)
    # Add smooth baseline variation
    curve = curve * np.sin(np.pi * x)
    curve[0] = 0
    curve[-1] = 0
    return curve

def create_gaussian_control(x, positions, intensities, sigma=0.05):
    """Create control field as sum of Gaussians"""
    control = np.zeros_like(x)
    for pos, intensity in zip(positions, intensities):
        control += intensity * np.exp(-((x - pos) ** 2) / (2 * sigma ** 2))
    return control

def simulate_evolution(initial, target, n_actuators=3, n_steps=50):
    """Simulate schematic state evolution (simplified)"""
    states = []
    errors = np.zeros((n_steps, len(initial)))

    # Kinetic actuators: positions change over time
    # Start positions
    actuator_positions_init = np.array([0.2, 0.5, 0.8])
    # End positions (moved to track features)
    actuator_positions_end = np.array([0.35, 0.55, 0.75])

    for t in range(n_steps):
        alpha = t / (n_steps - 1)  # Progress from 0 to 1
        # Smooth interpolation with some controlled dynamics
        state = initial * (1 - alpha**0.7) + target * alpha**0.7

        # Add some realistic PDE-like smoothing (enforce BC)
        state[0] = 0
        state[-1] = 0

        states.append(state)
        errors[t] = np.abs(state - target)

    # Create control intensities with kinetic (moving) actuators
    control_snapshots = []
    snapshot_times = [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1]

    for t in snapshot_times:
        alpha = t / (n_steps - 1)
        # Actuators move over time (kinetic behavior)
        actuator_positions = actuator_positions_init * (1 - alpha) + actuator_positions_end * alpha
        # Control intensity decreases over time
        intensities = np.random.uniform(0.4, 1.0, n_actuators) * (1 - alpha * 0.7)
        control = create_gaussian_control(x, actuator_positions, intensities)
        control_snapshots.append((t, control, actuator_positions))

    return states, errors, control_snapshots

def create_smoke_distribution(x):
    """Create smoke concentration profile"""
    # Smoke concentrated in the middle, decaying at boundaries
    smoke = 0.8 * np.exp(-((x - 0.3) ** 2) / 0.03) + 0.5 * np.exp(-((x - 0.7) ** 2) / 0.05)
    # Enforce boundary conditions (ventilation at boundaries)
    smoke[0] = 0
    smoke[-1] = 0
    return smoke

def create_schematic(problem_type, save_path):
    """Create schematic diagram for specified problem type"""

    setup_academic_style()

    if problem_type == "reference_tracking":
        initial = np.zeros_like(x)
        target = create_target_curve(x)
        title = "Reference Tracking Problem"
    elif problem_type == "stabilization":
        initial = create_target_curve(x)
        target = np.zeros_like(x)
        title = "Stabilization Problem"
    elif problem_type == "smoke_control":
        initial = create_smoke_distribution(x)
        target = np.zeros_like(x)  # Clear air
        title = "Smoke Control Problem"
    else:
        raise ValueError("Unknown problem type")

    # Simulate evolution
    states, errors, control_snapshots = simulate_evolution(initial, target)

    # Create figure with 3 subplots (squeezed vertically with careful spacing)
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    gs = fig.add_gridspec(3, 1, hspace=0.45, top=0.93, bottom=0.09, left=0.12, right=0.95)

    # Professional colors matching conference style
    color_initial = '#2166ac'  # Professional blue
    color_target = '#b2182b'   # Professional red
    color_state = color_initial  # Use same blue for state evolution

    # ====== Subplot 1: Initial and Target States ======
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(x, initial, '-', linewidth=2.5, label='Initial State', alpha=0.9, color=color_initial)
    ax1.plot(x, target, '--', linewidth=2.5, label='Target State', alpha=0.9, color=color_target)
    ax1.fill_between(x, initial, alpha=0.15, color=color_initial)
    ax1.fill_between(x, target, alpha=0.15, color=color_target)
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.8, alpha=0.3)
    ax1.set_xlim(0, 1)
    y_max = max(np.abs(initial).max(), np.abs(target).max()) * 1.15
    ax1.set_ylim(-y_max, y_max)
    ax1.set_ylabel('State $z(x)$', fontsize=11)
    ax1.set_xlabel('Space $x$', fontsize=11)
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle='--')
    # Add main problem title at the top (larger)
    fig.suptitle(title, fontsize=20, fontweight='bold', y=0.985)
    ax1.set_title('(a) Initial and Target Configuration',
                  fontsize=12, fontweight='bold', pad=10)

    # ====== Subplot 2: State Evolution + Control ======
    ax2 = fig.add_subplot(gs[1])

    # Plot state evolution with single blue color and varying opacity
    n_plot_states = 15
    indices = np.linspace(0, len(states) - 1, n_plot_states, dtype=int)

    for idx, i in enumerate(indices):
        alpha = 0.15 + 0.85 * (idx / (n_plot_states - 1))  # Increasing opacity (lighter->darker)
        ax2.plot(x, states[i], color=color_state, linewidth=2.0, alpha=alpha)

    # Plot control intensities (Gaussian combinations)
    ax2_twin = ax2.twinx()
    # Use red/orange tones for control matching actuator theme
    colors_control = plt.cm.Reds(np.linspace(0.35, 0.85, len(control_snapshots)))

    for idx, (t, control, positions) in enumerate(control_snapshots):
        alpha_ctrl = 0.3 + 0.7 * (idx / max(1, len(control_snapshots) - 1))
        ax2_twin.fill_between(x, control, alpha=alpha_ctrl * 0.4,
                              color=colors_control[idx])
        # Mark actuator positions (kinetic - they move over time)
        # Place dots at the actual control value at each actuator position
        for pos in positions:
            # Find the control value at this actuator position
            idx_pos = np.argmin(np.abs(x - pos))
            control_value = control[idx_pos]
            ax2_twin.plot(pos, control_value, 'o',
                         color=colors_control[idx], markersize=8, alpha=alpha_ctrl,
                         markeredgecolor='black', markeredgewidth=0.8)

    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.8, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-y_max, y_max)
    ax2.set_ylabel('State Evolution $z(x,t)$', fontsize=11, color=color_state)
    ax2.set_xlabel('Space $x$', fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color_state)

    ax2_twin.set_ylabel('Control Intensity $u(x,t)$', fontsize=11, color='#b2182b')
    ax2_twin.set_ylim(0, 1.3)
    ax2_twin.tick_params(axis='y', labelcolor='#b2182b')

    ax2.set_title('(b) State Evolution and Control Application',
                  fontsize=12, fontweight='bold', pad=10)

    # ====== Subplot 3: Space-Time Error Heatmap ======
    ax3 = fig.add_subplot(gs[2])

    # Use 'hot' colormap matching the heat2D visualization style
    # Interpolate errors to higher resolution for smoother visualization
    from scipy.ndimage import zoom
    errors_highres = zoom(errors, (3, 1), order=1)  # 3x higher temporal resolution

    # CRITICAL: Center the colormap at error=0 by setting vmin=0
    error_max = np.max(errors_highres)

    # Plot error heatmap with 'hot' colormap (matching heat2D style)
    time_axis = np.linspace(0, 1, errors_highres.shape[0])
    im = ax3.imshow(errors_highres, aspect='auto', origin='lower',
                    extent=[0, 1, 0, 1], cmap='hot',
                    vmin=0, vmax=error_max * 1.05)

    ax3.set_xlabel('Space $x$', fontsize=11)
    ax3.set_ylabel('Time $t$', fontsize=11)
    ax3.set_title('(c) Space-Time Error: $|z(x,t) - z_{\\mathrm{target}}(x)|$',
                  fontsize=12, fontweight='bold', pad=10)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label('Error Magnitude', fontsize=10)

    # Save figure in multiple formats
    # PNG
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved {problem_type} schematic (PNG) to {save_path}")

    # PDF
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"Saved {problem_type} schematic (PDF) to {pdf_path}")

    # SVG
    svg_path = save_path.replace('.png', '.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    print(f"Saved {problem_type} schematic (SVG) to {svg_path}")

    plt.close()

# Generate all three schematics
create_schematic("reference_tracking",
                "/mnt/d/Repositories/Tesseract-Hackathon/figs/schematic/reference_tracking.png")
create_schematic("stabilization",
                "/mnt/d/Repositories/Tesseract-Hackathon/figs/schematic/stabilization.png")
create_schematic("smoke_control",
                "/mnt/d/Repositories/Tesseract-Hackathon/figs/schematic/smoke_control.png")

print("\n" + "=" * 70)
print("  SCHEMATIC GENERATION COMPLETE")
print("=" * 70)
print("All three diagrams created in PNG, PDF, and SVG formats:")
print("  • Reference Tracking")
print("  • Stabilization")
print("  • Smoke Control")
print("=" * 70)
