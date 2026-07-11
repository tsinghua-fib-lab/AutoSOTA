---
jupytext:
  main_language: python
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell}
:tags: [hide-cell]

import os
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import tree_util

from jax_md_mod import custom_partition
from jax_md import space, energy, partition

import optax

import matplotlib.pyplot as plt

from chemtrain.data import preprocessing
from chemtrain.trainers import ForceMatching

base_path = Path(os.environ.get("DATA_PATH", "./data"))
```

# Force Matching



Force matching is a bottom-up method to derive coarse-grained potentials
$U_\theta$ from atomistic reference data.
In a variational formulation, the approach learns a set of parameters $\theta$
by optimizing the error $\chi^2$ of predicted coarse forces
$\mathbf F_I^\theta(\mathbf R)$ on the coarse-grained sites
$\mathbf R = M(\mathbf r)$ [^Noid2008]

```{math}
\chi^2 = \frac{1}{3N}\left\langle \sum_{I=1}^N \left| \mathbf{\hat{F}}_I^\text{AT} - \mathbf{F}_I^\theta(\mathbf{R})\right|^2 \right\rangle_\text{AT}.
```

## Load Data

In this example, we use reference data from an all-atomistic simulation of
ethane. We obtained this data in the example
[Prior Simulation](./prior_simulation.md).

```{image} ../_static/ethane.png
:align: center
:alt: Ethane
```

<br>

```{code-cell}
train_ratio = 0.5

box = jnp.asarray([1.0, 1.0, 1.0])

all_forces = preprocessing.get_dataset(base_path / "forces_ethane.npy")
all_positions = preprocessing.get_dataset(base_path / "positions_ethane.npy")
```

## Compute Mapping

The reference data contains only fine-grained forces $\mathbf f_i$ and positions
$\mathbf r_i$.
Thus, we must define a mapping $M$ that derives the
positions of the coarse-grained sites $\mathcal I_I$ and the forces acting on
them [^Noid2008]

```{math}
\mathbf R_I = \sum_{i \in \mathcal I_I} c_{Ii} \mathbf r_i.
```

We select the two carbon atoms $C_1$ and $C_2$ as locations of the
coarse-grained sites $\mathcal I_1$ and $\mathcal I_2$ and neglect the hydrogen
atoms.
We then compute the effective coarse-grained forces from the atomistic forces
via the corresponding linear mapping [^Noid2008]

```{math}
\mathbf{F}_I = \sum_{i \in \mathcal I_I} \frac{d_{Ii}}{c_{Ii}} \mathbf f_i.
```

```{code-cell}
# Center of Mass (COM) mapping
displacement_fn, shift_fn = space.periodic_general(box, fractional_coordinates=True)

# Scale the position data into fractional coordinates
position_dataset = preprocessing.scale_dataset_fractional(all_positions, box)

masses = jnp.asarray([15.035, 1.011, 1.011, 1.011])

weights = jnp.asarray([
    [1, 0.0000, 0, 0, 0, 0.000, 0.000, 0.000],
    [0.0000, 1, 0.000, 0.000, 0.000, 0, 0, 0]
])

position_dataset, force_dataset = preprocessing.map_dataset(
    position_dataset, displacement_fn, shift_fn, weights, weights, all_forces 
)
```

## Setup Model

As a coarse-grained potential model, we choose a simple spring bond

```{math}
    U(\mathbf R) = \frac{1}{2} k_B (|\mathbf R_1 - \mathbf R_2| - b_0)^2.
```

To ensure that the model parameters remain positive during optimization,
we transform them into a constraint space
$\theta_1 = \log b_0,\ \theta_2= \log k_B$.

```{code-cell}
r_init = position_dataset[0, ...]

displacement_fn, shift_fn = space.periodic_general(box, fractional_coordinates=True)
neighbor_fn = custom_partition.masked_neighbor_list(displacement_fn, r_cutoff=1.0)

nbrs_init = neighbor_fn.allocate(r_init)

init_params = {
    "log_b0": jnp.log(0.11),
    "log_kb": jnp.log(5000.0)
}

def energy_fn_template(energy_params):
    harmonic_energy_fn = energy.simple_spring_bond(
        displacement_fn, bond=jnp.asarray([[0, 1]]),
        length=jnp.exp(energy_params["log_b0"]),
        epsilon=jnp.exp(energy_params["log_kb"]),
        alpha=2.0
    )
    
    return harmonic_energy_fn    

def force_fn_template(energy_params):
    neg_energy_fn = lambda r, **kwargs: -energy_fn_template(energy_params)(r, **kwargs)
    return jax.grad(neg_energy_fn, argnums=0)

@jax.value_and_grad
def test_loss_fn(params, r, f):
    return jnp.mean(jnp.sum((f - force_fn_template(params)(r, neighbor=nbrs_init)) ** 2, axis=-1))

sample_idx = 0

print(f"Energy with initial params is {energy_fn_template(init_params)(position_dataset[sample_idx, ...], neighbor=nbrs_init)}")
print(f"Forces with initial params are\n{force_fn_template(init_params)(position_dataset[sample_idx, ...], neighbor=nbrs_init)}")
print(f"Parameter gradients on initial sample are\n{test_loss_fn(init_params, position_dataset[sample_idx, ...], force_dataset[sample_idx, ...])[1]}")
```

## Analytical Solution

As our model relies only on the magnitude of the displacement between $C_1$ and $C_2$,
we compute this distance and plot it.

```{code-cell}
disp = jax.vmap(displacement_fn)(position_dataset[:, 0, :], position_dataset[:, 1, :])
dist_CC = jnp.sqrt(jnp.sum(disp ** 2, axis=-1))

plt.figure()
plt.hist(dist_CC, bins=100)
plt.xlabel("Distance C_1 - C_2 [nm]")
plt.ylabel("Count")
```

Indeed, the distance between the two carbon atoms is approximately
Gaussian distributed.
Hence, the choice of a harmonic potential model is reasonable.

However, we want to check whether our force data supports this hypothesis.
Therefore, we project the forces onto the displacement vector between the two
carbon atoms.

```{code-cell}
disp_dir = disp / dist_CC[:, None]
force_proj = jnp.einsum('ijk, i...k->ij', force_dataset, disp_dir)

plt.figure()
plt.scatter(dist_CC, force_proj[:, 0], color="r", s=1)
plt.scatter(dist_CC, force_proj[:, 1], color="b", s=1)
plt.xlabel("Distance C_1 - C_2 [nm]")
plt.ylabel("Projected Force")
```

We see that also the force reference data is quite noisy, but still correlates
with the distance between the coarse-grained sites.

```{math}

\mathbf F_I = (-1)^I k_B (|\mathbf{R}_1 - \mathbf{R}_2| - b_0) \frac{\mathbf{R}_1 - \mathbf{R}_2}{|\mathbf{R}_1 - \mathbf{R}_2|}.
```

Since this relationship is linear, we might estimate the parameters of the model
via a linear regression fit. 

```{code-cell}
# Least squares solution
lhs = jnp.stack((dist_CC, jnp.ones_like(dist_CC)), axis=-1)
rhs = -force_proj[:, (0,)]

kb, c = jnp.linalg.lstsq(lhs, rhs, rcond=None)[0]
b0 = -c / kb

print(f"Estimated potential parameters are {kb[0] :.1f} kJ/mol/nm^2 and {b0[0] :.3f} nm")
```

## Setup Optimizer

```{code-cell}
subsample = 25
batch_per_device = 10
epochs = 20
initial_lr = 0.02
lr_decay = 0.05

lrd = int(position_dataset.shape[0] / subsample / batch_per_device * epochs)
lr_schedule = optax.exponential_decay(initial_lr, lrd, lr_decay)
optimizer = optax.chain(
    optax.scale_by_adam(0.9, 0.95),
    optax.scale_by_schedule(lr_schedule),
    # Flips the sign of the update for gradient descend
    optax.scale_by_learning_rate(1.0),
)
```

## Setup Force Matching

```{code-cell}
force_matching = ForceMatching(
    init_params=init_params, energy_fn_template=energy_fn_template,
    nbrs_init=nbrs_init, optimizer=optimizer, batch_per_device=batch_per_device,
)

# We can provide numpy arrays to initialize the datasets for training,
# validation, and testing in a single step
force_matching.set_datasets({
    "F": force_dataset[::subsample, :, :],
    "R": position_dataset[::subsample, :, :],
}, train_ratio=train_ratio)

```

```{code-cell}
:tags: [hide-output]

force_matching.train(epochs, checkpoint_freq=1000)
```

```{code-cell}
# We can also provide completely new samples for a single stage, e.g., testing
force_matching.set_dataset({
    "F": force_dataset[1::subsample, :, :],
    "R": position_dataset[1::subsample, :, :],
}, stage = "testing")

mae_error = force_matching.evaluate_mae_testset()
print(mae_error)
```

## Results

```{code-cell}
plt.plot(force_matching.train_losses)
plt.xticks(ticks=range(0, epochs + 1, 5))
plt.xlabel("Epoch")
plt.ylabel("Force Error")
```

Finally, we compare the values obtained from a least-squares fit to those
obtained from force-matching.

```{code-cell}
pred_parameters = tree_util.tree_map(jnp.exp, force_matching.params)

b0_err = jnp.abs(b0[0] - pred_parameters["log_b0"])
kb_err = jnp.abs(kb[0] - pred_parameters["log_kb"])

print(f"Force matching predicted {pred_parameters['log_b0']:.3f} nm and {pred_parameters['log_kb']:.1f} kJ/mol/nm^2")
print(f"Least squares predicted {b0[0]:.3f} nm and {kb[0]:.1f} kJ/mol/nm^2")
print(f"Absolute error in b0 is {b0_err:.3f} nm and in kb is {kb_err:.1f} kJ/mol/nm^2")
```

## Further Reading

### Examples

- [Alanine Dipeptide in Implicit Water](../examples/CG_alanine_dipeptide.ipynb)

### Publications

1. Stephan Thaler, Maximilian Stupp, Julija Zavadlav; *Deep coarse-grained potentials via relative entropy minimization*. J. Chem. Phys. 28 December 2022; 157 (24): 244103. <https://doi.org/10.1063/5.0124538>

## References

[^Noid2008]: W. G. Noid, Jhih-Wei Chu, Gary S. Ayton, Vinod Krishna, Sergei Izvekov, Gregory A. Voth, Avisek Das, Hans C. Andersen; *The multiscale coarse-graining method. I. A rigorous bridge between atomistic and coarse-grained models*. J. Chem. Phys. 28 June 2008; 128 (24): 244114. https://doi-org.eaccess.tum.edu/10.1063/1.2938860
