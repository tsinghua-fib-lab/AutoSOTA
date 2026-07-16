# 2D Heat Equation Control with Obstacles

This example demonstrates Differentiable Predictive Control (DPC) for the 2D heat equation with **static circular obstacles** that agents must navigate around while controlling the temperature field. Both centralized and decentralized controller architectures are implemented.

## Overview

- **Problem**: Control a 2D heat PDE on a 32×32 grid using 16 mobile actuators
- **Obstacles**: 3 fixed circular obstacles strategically placed to challenge navigation
- **Training**: Pure JAX implementation (no Tesseract dependency for faster iteration)
- **Data**: Reuses pre-generated Gaussian Random Field (GRF) dataset from `../heat2D/data/`

### Obstacle Configuration

Three circular obstacles (radius = 0.08) are placed at:
- **Left**: (0.15, 0.50) - blocks left side
- **Right**: (0.85, 0.50) - blocks right side
- **Bottom**: (0.50, 0.15) - blocks bottom center

Agents are initialized in a 4×4 grid from [0.26, 0.26] to [0.74, 0.74], ensuring no overlap with obstacles.

### Collision Penalties

Enhanced collision penalties ensure robust avoidance:
- **Agent-agent collision**: λ = 20.0 (R_safe = 0.08)
- **Agent-obstacle collision**: λ = 50.0 (R_safe + obstacle_radius)
- **Boundary violations**: λ = 100.0
- **Tracking error**: λ = 5.0
- **Control effort**: λ = 0.001
- **Acceleration smoothness**: λ = 0.1

## Directory Structure

```
heat2D_obstacles/
├── centralized/
│   ├── train.py           # Training script
│   ├── visualize.py       # Static visualization
│   ├── animate.py         # Animation generation
│   ├── dynamics_dual.py   # PDE dynamics wrapper
│   └── data_utils.py      # Data loading utilities
├── decentralized/
│   ├── train.py           # Training script
│   ├── visualize.py       # Static visualization
│   ├── animate.py         # Animation generation
│   ├── dynamics_dual.py   # PDE dynamics wrapper
│   └── data_utils.py      # Data loading utilities
├── data/
│   ├── generate_dataset.py        # Dataset generation script
│   └── heat2d_dataset_32x32.npz   # Pre-generated GRF data (5000 samples)
└── README.md              # This file
```

## Prerequisites

Ensure you have:
- JAX with GPU support (recommended) or CPU
- Python 3.8+
- Required packages: `jax`, `flax`, `optax`, `matplotlib`, `numpy`, `tqdm`

The data has already been generated and is available at `data/heat2d_dataset_32x32.npz`.

---

## Usage

### 1. Data Generation (Optional - Already Done)

The dataset has already been generated with 5000 (initial, target) pairs using Gaussian Random Fields. If you need to regenerate:

```bash
cd data
python generate_dataset.py
```

**Parameters:**
- Grid size: 32×32
- Samples: 5000
- Length scales: 0.25 (init), 0.4 (target)
- Output: `heat2d_dataset_32x32.npz`

---

### 2. Training

#### Centralized Controller (Global Sensing)

**Quick Test** (1 sample, 10 epochs, ~7 seconds):
```bash
cd centralized
python train.py --test
```

**Full Training** (5000 samples, 500 epochs, ~30-60 minutes):
```bash
cd centralized
python train.py
```

**Outputs:**
- `centralized_params_heat2d_obstacles.msgpack` - Trained parameters
- `training_metrics_heat2d_obstacles.png` - Loss curves (6 subplots):
  - Total Loss
  - Tracking Loss
  - Effort Loss
  - Agent-Agent Collision Loss
  - Agent-Obstacle Collision Loss
  - Boundary Loss

---

#### Decentralized Controller (Local Sensing)

**Quick Test** (1 sample, 10 epochs, ~7 seconds):
```bash
cd decentralized
python train.py --test
```

**Full Training** (5000 samples, 500 epochs, ~30-60 minutes):
```bash
cd decentralized
python train.py
```

**Outputs:**
- `decentralized_params_heat2d_obstacles.msgpack` - Trained parameters
- `training_metrics_heat2d_obstacles_decentralized.png` - Loss curves (6 subplots)

---

### 3. Visualization (Static Plots)

Generate publication-quality static visualizations comparing controlled vs uncontrolled evolution.

#### Centralized
```bash
cd centralized
python visualize.py
```

**Outputs:**
- `heat2d_obstacles_centralized_visualization.pdf` - Vector graphics (publication quality)
- `heat2d_obstacles_centralized_visualization.png` - Raster image (high resolution)

**Figure Layout:**
- **3 rows × 6 timesteps**:
  - Row 1: Uncontrolled evolution (with obstacles shown)
  - Row 2: DPC controlled evolution (agents + obstacles)
  - Row 3: Tracking error (agents + obstacles)
- **Bottom row**: Time-series metrics (MSE, Agent Speed, Control Intensity)

#### Decentralized
```bash
cd decentralized
python visualize.py
```

**Outputs:**
- `heat2d_obstacles_decentralized_visualization.pdf`
- `heat2d_obstacles_decentralized_visualization.png`

---

### 4. Animation (Videos)

Generate 10-second animations showing the full control trajectory.

#### Centralized
```bash
cd centralized
python animate.py
```

**Outputs:**
- `heat2d_obstacles_animation.gif` - Animated GIF (150 DPI, ~2-5 MB)
- `heat2d_obstacles_animation.mp4` - MP4 video (200 DPI, requires ffmpeg)

**Animation Layout (2×2):**
- Top-left: Uncontrolled evolution (with obstacles)
- Top-right: DPC controlled evolution (agents colored by control intensity + obstacles)
- Bottom-left: Tracking error field (agents + obstacles)
- Bottom-right: MSE time series comparison

#### Decentralized
```bash
cd decentralized
python animate.py
```

**Outputs:**
- `heat2d_obstacles_decentralized_animation.gif`
- `heat2d_obstacles_decentralized_animation.mp4`

**Note:** If MP4 encoding fails, ensure `ffmpeg` is installed:
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

---

## Complete Workflow Examples

### Centralized: Test → Full Train → Visualize → Animate
```bash
# Navigate to centralized directory
cd examples/heat2D_obstacles/centralized

# 1. Quick test to verify setup (~7 seconds)
python train.py --test

# 2. Full training (~30-60 minutes)
python train.py

# 3. Generate static visualization
python visualize.py

# 4. Generate animation
python animate.py
```

### Decentralized: Test → Full Train → Visualize → Animate
```bash
# Navigate to decentralized directory
cd examples/heat2D_obstacles/decentralized

# 1. Quick test to verify setup (~7 seconds)
python train.py --test

# 2. Full training (~30-60 minutes)
python train.py

# 3. Generate static visualization
python visualize.py

# 4. Generate animation
python animate.py
```

### Parallel Training (Both Architectures)
```bash
# In one terminal
cd examples/heat2D_obstacles/centralized && python train.py

# In another terminal
cd examples/heat2D_obstacles/decentralized && python train.py
```

---

## Expected Results

### Training Metrics

After successful training, you should observe:

| Loss Component | Initial | Final | Description |
|---------------|---------|-------|-------------|
| **Tracking (l_track)** | ~0.1-0.5 | ~0.001-0.01 | MSE between controlled and target field |
| **Effort (l_effort)** | ~50-200 | ~0.001-0.01 | Control energy usage |
| **Agent-Agent Collision** | ~0.01 | ~0.0 | Agents maintain separation (R_safe = 0.08) |
| **Agent-Obstacle Collision** | ~0.1 | ~0.0 | Agents avoid obstacles |
| **Boundary Violations** | ~0.0 | ~0.0 | Agents stay in [0.02, 0.98]² domain |

### Performance Comparison

- **Centralized**: Better tracking performance (global information)
- **Decentralized**: Zero-shot scalability (transfer to different agent counts)
- **Both**: Successfully navigate around obstacles without collisions

---

## Implementation Details

### Key Differences from Standard heat2D

1. **Obstacle Collision Loss**: New penalty term for agent-obstacle distances
   ```python
   # Agent-obstacle distance calculation
   diff_obst = xi_traj[:, :, None, :] - obstacle_centers[None, None, :, :]
   dists_obst = jnp.sqrt(jnp.sum(diff_obst**2, axis=-1) + 1e-8)
   safety_distances = R_safe + obstacle_radii[None, None, :]
   l_coll_obstacles = jnp.mean(jnp.maximum(0, safety_distances - dists_obst) ** 2)
   ```

2. **Increased Collision Weights**: Stronger penalties for collision avoidance
   - Agent-agent: 1.0 → 20.0 (20× increase)
   - Agent-obstacle: 50.0 (new, strong penalty)

3. **Obstacle Rendering**: All visualization scripts render obstacles as gray circles

4. **No Tesseract Dependency**: Pure JAX implementation for faster iteration

### Architecture

- **Centralized Policy**: CNN branch processes full 32×32 error field
- **Decentralized Policy**: Each agent processes 8-point local window (resampled to 20 points)
- **Trunk Network**: Fourier features for spatial encoding
- **Fusion Layer**: Combines branch + trunk → (u, v) per agent

---

## Troubleshooting

### Common Issues

**1. "No module named 'data_utils'"**
```bash
# Ensure you're running from the correct directory
cd examples/heat2D_obstacles/centralized  # or decentralized
python train.py
```

**2. "Parameters not found"**
```bash
# Train the model first
python train.py
# Then visualize
python visualize.py
```

**3. "Out of memory" during training**
```bash
# Reduce batch size in train.py:
batch_size = 8  # instead of 16
```

**4. Animation encoding fails**
```bash
# Install ffmpeg
sudo apt-get install ffmpeg  # Linux
brew install ffmpeg          # macOS
```

**5. Training diverges (NaN loss)**
```bash
# Try reducing learning rate in train.py:
lr_schedule = optax.exponential_decay(5e-4, 2000, 0.5)  # instead of 1e-3
```

---

## Customization

### Modify Obstacles

Edit the `OBSTACLES` array in `train.py`, `visualize.py`, and `animate.py`:

```python
# Example: Add a 4th obstacle
OBSTACLES = jnp.array([
    [0.15, 0.50, 0.08],   # Left
    [0.85, 0.50, 0.08],   # Right
    [0.50, 0.15, 0.08],   # Bottom
    [0.50, 0.85, 0.10],   # Top (larger radius)
])
```

### Adjust Penalties

In `train.py`, modify the loss weights:

```python
total_loss = 5.0 * l_track + \
             0.001 * l_effort + \
             100.0 * l_bound + \
             20.0 * l_coll_agents + \      # Increase for stricter collision avoidance
             50.0 * l_coll_obstacles + \   # Increase for harder obstacles
             0.1 * l_accel
```

### Change Grid Resolution

In `train.py`:

```python
n_grid = 64  # Increase resolution (will need to regenerate data)
```

Then regenerate data:

```bash
cd data
python generate_dataset.py  # Modify script to use n_grid=64
```

---

## File Descriptions

### Training Scripts (`train.py`)
- Loads GRF dataset from `../data/`
- Initializes 16 agents in 4×4 grid
- Trains DeepONet policy via gradient descent
- Saves trained parameters as `.msgpack`
- Generates training metrics plot

### Visualization Scripts (`visualize.py`)
- Loads trained parameters
- Generates single test scenario (GRF seed=1234)
- Runs controlled and uncontrolled trajectories
- Creates 3-row comparison plot with obstacles
- Saves as PDF (vector) and PNG (raster)

### Animation Scripts (`animate.py`)
- Loads trained parameters
- Generates 2×2 animated comparison
- Renders obstacles on all field plots
- Saves as GIF and MP4
- Duration: 10 seconds @ 30fps

### Dynamics Wrapper (`dynamics_dual.py`)
- Wraps PDE solver (from `tesseracts/solverHeat2D_*/solver.py`)
- Implements `unroll_controlled()` for trajectory rollouts
- Pure JAX implementation (no Tesseract overhead)

### Data Utilities (`data_utils.py`)
- `generate_grf_2d()`: Creates 2D Gaussian Random Field with Dirichlet BCs
- `load_dataset()`: Loads pre-generated `.npz` file
- `get_training_data()`: Smart loading with fallback to generation

---

## Citation

If you use this code, please cite:

```bibtex
@article{your_paper,
  title={Differentiable Predictive Control for PDE Systems with Obstacles},
  author={Your Name},
  journal={Conference/Journal},
  year={2024}
}
```

---

## Related Examples

- `../heat2D/` - 2D heat equation without obstacles
- `../heat1d/` - 1D heat equation (simpler baseline)
- `../fkpp1d/` - 1D Fisher-KPP (nonlinear reaction-diffusion)
- `../ks1d/` - 1D Kuramoto-Sivashinsky (chaotic dynamics)

---

## Contact

For questions or issues:
- Open an issue in the repository
- Check `../CLAUDE.md` for project overview and architecture details
