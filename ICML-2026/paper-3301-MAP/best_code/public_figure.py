"""
Public-Friendly Visualization: How Noise Direction Matters for Curved Spaces

This script creates an accessible visualization showing why adding noise perpendicular 
to a curve preserves the original pattern better than adding noise in all directions.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
from scipy.integrate import cumulative_trapezoid

# Set random seed for reproducibility
np.random.seed(42)

# ============================================
# CONFIGURATION
# ============================================

SIGMA = 0.05  # Reduced for tighter distribution
N_SAMPLES = 100000
N_DISCRETIZE = 3000

# Spiral parameters
SPIRAL_N_TURNS = 1.5
SPIRAL_T_MIN = 0.25
SPIRAL_T_MAX = 1.0
SPIRAL_SCALE = 1.0

# Visual parameters
SCATTER_ALPHA = 0.15  # Increased from 0.02 for visibility
SCATTER_SIZE = 12  # Increased for lifted distribution visibility
RIBBON_WIDTH = 0.18  # Increased from 0.05 for density clarity
FONT_SIZE = 38
TITLE_FONT_SIZE = 42
LABEL_FONT_SIZE = 34

# Colors - vibrant and accessible
CURVE_COLOR = 'black'
SCATTER_COLOR = '#3498db'  # Bright blue
RIBBON_CMAP = 'hot'        # High contrast: dark (low) to bright yellow/white (high)
ARROW_COLOR = '#e74c3c'    # Bright red for visibility

# Density parameters (2-component mixture - well-separated sharp peaks)
# This makes distortion much more visually obvious
DENSITY_MEANS = [0.25, 0.75]  # Two well-separated peaks
DENSITY_STDS = [0.04, 0.04]   # Sharp, narrow peaks
DENSITY_WEIGHTS = [1.0, 1.0]  # Equal heights

# ============================================
# MANIFOLD CLASS
# ============================================

class SpiralManifold:
    """An Archimedean spiral - a simple curved line."""
    
    def __init__(self, n_turns=2.5, t_min=0.15, t_max=1.0, scale=1.0):
        self.n_turns = n_turns
        self.t_min = t_min
        self.t_max = t_max
        self.scale = scale
        self.omega = 2 * np.pi * n_turns
        self._build_arc_length_table()
    
    def point(self, t):
        """Get point at parameter t."""
        t = np.asarray(t)
        theta = self.omega * t
        r = self.scale * t
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        if t.ndim == 0:
            return np.array([x, y])
        return np.stack([x, y], axis=1)
    
    def get_tangent(self, t):
        """Unit tangent vector (direction along the curve)."""
        t = np.asarray(t)
        theta = self.omega * t
        dx = self.scale * (np.cos(theta) - self.omega * t * np.sin(theta))
        dy = self.scale * (np.sin(theta) + self.omega * t * np.cos(theta))
        if t.ndim == 0:
            tangent = np.array([dx, dy])
            return tangent / np.linalg.norm(tangent)
        tangents = np.stack([dx, dy], axis=1)
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        return tangents / norms
    
    def get_normal(self, t):
        """Unit normal vector (perpendicular to the curve)."""
        tangent = self.get_tangent(t)
        if tangent.ndim == 1:
            return np.array([-tangent[1], tangent[0]])
        return np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    
    def _build_arc_length_table(self, n_points=5000):
        """Build lookup table for arc length."""
        t_vals = np.linspace(self.t_min, self.t_max, n_points)
        points = self.point(t_vals)
        diffs = np.diff(points, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        self._arc_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])
        self._t_table = t_vals
        self.total_length = self._arc_lengths[-1]
    
    def arc_length(self, t):
        """Arc length from t_min to parameter t."""
        t = np.asarray(t)
        return np.interp(t, self._t_table, self._arc_lengths)
    
    def project(self, points):
        """Project points onto the spiral (nearest point)."""
        spiral_pts, t_vals = self.discretize(N_DISCRETIZE)
        tree = cKDTree(spiral_pts)
        _, indices = tree.query(points)
        return spiral_pts[indices], t_vals[indices]
    
    def discretize(self, n_points):
        """Return n_points evenly spaced on the curve."""
        t = np.linspace(self.t_min, self.t_max, n_points)
        return self.point(t), t

# ============================================
# SAMPLING FUNCTIONS
# ============================================

def mixture_of_gaussians_pdf(s, means, stds, weights):
    """Evaluate mixture of Gaussians PDF."""
    s = np.asarray(s)
    pdf = np.zeros_like(s, dtype=float)
    for mu, sigma, w in zip(means, stds, weights):
        pdf += w * np.exp(-0.5 * ((s - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    return pdf


def sample_from_manifold(manifold, n_samples, means, stds, weights):
    """Sample points from a non-uniform pattern along the curve."""
    s_min, s_max = 0, manifold.total_length
    
    # Sample using inverse CDF method
    s_fine = np.linspace(s_min, s_max, 10000)
    pdf_fine = mixture_of_gaussians_pdf(s_fine, means, stds, weights)
    cdf_fine = cumulative_trapezoid(pdf_fine, s_fine, initial=0)
    cdf_fine /= cdf_fine[-1]
    
    u = np.random.rand(n_samples)
    s_samples = np.interp(u, cdf_fine, s_fine)
    t_samples = np.interp(s_samples, manifold._arc_lengths, manifold._t_table)
    
    points = manifold.point(t_samples)
    normals = manifold.get_normal(t_samples)
    
    return points, normals, t_samples, s_samples


def add_isotropic_noise(points, sigma):
    """Add noise in all directions."""
    noise = sigma * np.random.randn(*points.shape)
    return points + noise


def add_normal_noise(points, normals, sigma):
    """Add noise only perpendicular to the curve."""
    noise_magnitude = sigma * np.random.randn(len(points))
    noise = noise_magnitude[:, np.newaxis] * normals
    return points + noise


def project_and_get_arc_length(manifold, points):
    """Project points back to curve and return positions."""
    projected_points, t_values = manifold.project(points)
    arc_lengths = manifold.arc_length(t_values)
    return projected_points, arc_lengths


def estimate_density_on_manifold(arc_lengths, total_length, n_eval=200):
    """Estimate density pattern from samples."""
    s_eval = np.linspace(0, total_length, n_eval)
    if len(arc_lengths) < 2:
        return s_eval, np.zeros_like(s_eval)
    kde = gaussian_kde(arc_lengths, bw_method='silverman')
    density = kde(s_eval)
    return s_eval, density

# ============================================
# VISUALIZATION FUNCTIONS
# ============================================

def plot_density_ribbon(ax, manifold, s_eval, density, width=RIBBON_WIDTH, 
                        color=SCATTER_COLOR, vmax=None):
    """Plot density as a colored ribbon along the curve using single color with varying opacity."""
    t_eval = np.interp(s_eval, manifold._arc_lengths, manifold._t_table)
    points = manifold.point(t_eval)
    normals = manifold.get_normal(t_eval)
    
    upper = points + width/2 * normals
    lower = points - width/2 * normals
    
    if vmax is None:
        vmax = density.max() if density.max() > 0 else 1
    
    # Normalize density to [0, 1] for alpha
    normalized_density = density / vmax
    
    # Apply power transform for more dramatic contrast
    # This makes high densities much more visible
    normalized_density = normalized_density ** 0.4  # Exponent < 1 emphasizes peaks
    
    for i in range(len(s_eval) - 1):
        # Use opacity to represent density with wider range
        alpha = 0.1 + 0.9 * normalized_density[i]  # Range from 0.1 to 1.0
        polygon = plt.Polygon(
            [upper[i], upper[i+1], lower[i+1], lower[i]],
            facecolor=color,
            alpha=alpha,
            edgecolor='none'
        )
        ax.add_patch(polygon)
    
    # Outline
    ax.plot(upper[:, 0], upper[:, 1], 'k-', linewidth=0.5, alpha=0.3)
    ax.plot(lower[:, 0], lower[:, 1], 'k-', linewidth=0.5, alpha=0.3)


def plot_scatter(ax, points, color=SCATTER_COLOR, alpha=SCATTER_ALPHA, s=SCATTER_SIZE):
    """Plot scatter of points."""
    ax.scatter(points[:, 0], points[:, 1], c=color, alpha=alpha, s=s, edgecolors='none')


def plot_curve(ax, manifold, n_points=500, color=CURVE_COLOR, linewidth=3):
    """Plot the curve."""
    points, _ = manifold.discretize(n_points)
    ax.plot(points[:, 0], points[:, 1], color=color, linewidth=linewidth, zorder=10)


def plot_perpendicular_arrows(ax, manifold, n_arrows=8, scale=0.25, color=ARROW_COLOR):
    """Plot arrows showing perpendicular direction."""
    total_length = manifold.total_length
    s_positions = np.linspace(0, total_length, n_arrows + 2)[1:-1]
    t_positions = np.interp(s_positions, manifold._arc_lengths, manifold._t_table)
    
    points = manifold.point(t_positions)
    normals = manifold.get_normal(t_positions)
    
    for i in range(len(t_positions)):
        p = points[i]
        n = normals[i]
        # Arrows in both directions
        ax.annotate('', xy=p + scale * n, xytext=p,
                   arrowprops=dict(arrowstyle='->', color=color, lw=3, zorder=5))
        ax.annotate('', xy=p - scale * n, xytext=p,
                   arrowprops=dict(arrowstyle='->', color=color, lw=3, zorder=5))


def setup_axes(ax, manifold, padding=0.35):
    """Set up axis limits."""
    points, _ = manifold.discretize(500)
    x_min, x_max = points[:, 0].min() - padding, points[:, 0].max() + padding
    y_min, y_max = points[:, 1].min() - padding, points[:, 1].max() + padding
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.axis('off')


# ============================================
# MAIN EXPERIMENT
# ============================================

print("Creating public-friendly visualization...")
print("Generating samples and applying noise...")

# Create spiral
spiral = SpiralManifold(n_turns=SPIRAL_N_TURNS, t_min=SPIRAL_T_MIN, 
                        t_max=SPIRAL_T_MAX, scale=SPIRAL_SCALE)

# Scale density parameters
means = [m * spiral.total_length for m in DENSITY_MEANS]
stds = [s * spiral.total_length for s in DENSITY_STDS]

# Sample original pattern
orig_points, orig_normals, orig_t, orig_s = sample_from_manifold(
    spiral, N_SAMPLES, means, stds, DENSITY_WEIGHTS
)

# Add noise
iso_noisy = add_isotropic_noise(orig_points, SIGMA)
normal_noisy = add_normal_noise(orig_points, orig_normals, SIGMA)

# Project back
iso_proj, iso_s = project_and_get_arc_length(spiral, iso_noisy)
normal_proj, normal_s = project_and_get_arc_length(spiral, normal_noisy)

# Estimate densities
s_eval = np.linspace(0, spiral.total_length, 200)
orig_density = mixture_of_gaussians_pdf(s_eval, means, stds, DENSITY_WEIGHTS)
_, iso_density = estimate_density_on_manifold(iso_s, spiral.total_length)
_, normal_density = estimate_density_on_manifold(normal_s, spiral.total_length)

# Normalize
orig_density /= np.trapezoid(orig_density, s_eval)

# Set consistent color scale
vmax_ref = orig_density.max()

# ============================================
# CREATE FIGURE
# ============================================

print("Creating figure...")

fig, axes = plt.subplots(1, 3, figsize=(22, 8))

# Panel 1: Original constrained distribution
ax = axes[0]
plot_density_ribbon(ax, spiral, s_eval, orig_density, vmax=vmax_ref)
plot_scatter(ax, orig_points, color=SCATTER_COLOR, alpha=1.0, s=150)
plot_curve(ax, spiral, color=CURVE_COLOR, linewidth=3)
setup_axes(ax, spiral)
ax.set_title('Original\nDistribution\n(satisfying constraint)', fontsize=TITLE_FONT_SIZE, pad=20, fontweight='bold')

# Panel 2: Lifted distribution (perturbed away from constraint)
ax = axes[1]
plot_scatter(ax, normal_noisy, color=SCATTER_COLOR, alpha=0.2, s=SCATTER_SIZE)
plot_curve(ax, spiral, color=CURVE_COLOR, linewidth=4)
setup_axes(ax, spiral)
ax.set_title('Lifted\nDistribution\n(easier to learn)', fontsize=TITLE_FONT_SIZE, pad=20, fontweight='bold')

# Panel 3: Projected back (same distribution as original)
ax = axes[2]
plot_density_ribbon(ax, spiral, s_eval, normal_density, vmax=vmax_ref)
plot_scatter(ax, normal_proj, color=SCATTER_COLOR, alpha=1.0, s=150)
plot_curve(ax, spiral, color=CURVE_COLOR, linewidth=3)
setup_axes(ax, spiral)
ax.set_title('Projected\nDistribution\n(recovered)', fontsize=TITLE_FONT_SIZE, pad=20, fontweight='bold')

# Add legend annotations on the first panel, lower and spaced apart
ax = axes[0]
# Blue dot symbol to match sample color
ax.text(0.06, 0.03, '●', fontsize=LABEL_FONT_SIZE, color=SCATTER_COLOR,
    transform=ax.transAxes, verticalalignment='center')
ax.text(0.16, 0.03, 'Data samples', fontsize=LABEL_FONT_SIZE-6,
    transform=ax.transAxes, verticalalignment='center')
ax.text(0.06, -0.05, '—', fontsize=LABEL_FONT_SIZE,
    transform=ax.transAxes, verticalalignment='center')
ax.text(0.16, -0.05, 'Constraint', fontsize=LABEL_FONT_SIZE-6,
    transform=ax.transAxes, verticalalignment='center')

# Layout
plt.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.10, 
                    wspace=0.12, hspace=0.15)

# Save
fig.savefig('public_figure.pdf', bbox_inches='tight', dpi=300)
fig.savefig('public_figure.svg', bbox_inches='tight')
print("\nFigure saved as 'public_figure.pdf' and 'public_figure.svg'")

plt.show()

print("\n" + "="*60)
print("VISUALIZATION COMPLETE")
print("="*60)

