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
from jax import random

import jax_md_mod
from jax_md import space, simulate, partition
from jax_md_mod import custom_quantity
from jax_md_mod.model import prior

import numpy as onp

from matplotlib import pyplot as plt

from chemtrain import ensemble, quantity

base_path = Path(os.environ.get("DATA_PATH", "./data"))
```

# Prior Simulation

The potential $U(\mathbf r)$ describes all interactions between particles of
a system.
However, not all these interactions are simple to parametrize [^Das2009].
For example, short range interactions are crucial to ensure that particles do
not overlap, but require quickly increasing forces at small particle distances.
Hence, a common approach separates the full potential in to a learnable bias
$\Delta U_\theta$ and a part kept fixed $U^\text{prior}$ during the variational
procedure

```{math}
U(\mathbf r) = U^\text{prior}(\mathbf r) + \Delta U_\theta(\mathbf r)
```

Since the fixed potential manifests beliefs before seeing any data, it is
a frequently used termed *prior potential* as in Bayesian statistics.
Likewise, $\Delta$*-learning* refers to only learning an additive
correction on top of the prior potential.

## Setup Force Field

As a prior, we want to use a classical force field.
Therefore, we define the potential parameters in a force field file of the following form:

```{code-cell}
ff_path = base_path / "ethane.toml"

with open(ff_path, "r") as f:
    print(f.read())
    
force_field = prior.ForceField.load_ff(ff_path)
```

## Setup Topology

We defined the energies associated with each bond, angle, and dihedral.
However, to compute the energy for a given set of coordinates, we must define
which atoms form these bonds, angles, and dihedral angles.

The bonds are already part of the PDB file we generated via open-babel.
Therefore, we can traverse the graph and extract all atoms
connected by simple paths of length 2 and 3.
Therefore, we find the angles and dihedral angles.

```{code-cell}
import mdtraj

conf_path = base_path / "ethane.pdb"

unv = mdtraj.load(conf_path, standard_names=False)
top = unv.top

topology = prior.Topology.from_mdtraj(top, mapping=force_field.mapping(by_name=True))
```

We can also plot the initial conformation. 

```{code-cell}
:tags: [hide-input]

box = jnp.asarray([1.0, 1.0, 1.0])
r_init = unv.xyz[0]

ax = plt.figure().add_subplot(projection='3d')

# Plot the atoms and assign the correct color to the species
ax.scatter(r_init[:, 0], r_init[:, 1], r_init[:, 2], c=topology.get_atom_species())

for idx1, idx2 in topology.get_bonds()[0]:
    ax.plot(r_init[(idx1, idx2), 0], r_init[(idx1, idx2), 1], r_init[(idx1, idx2), 2], color="k")

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

centers = r_init.mean(axis=0)
rmin = centers - 0.25
rmax = centers + 0.25

ax.set_xlim3d([rmin[0], rmax[2]])
ax.set_ylim3d([rmin[1], rmax[2]])
ax.set_zlim3d([rmin[2], rmax[2]])

ax.view_init(elev=20., azim=-35, roll=0)
```

## Setup Prior Energy

Now, we can combine the force field and topology into a function that generates
concrete energies for a given set of coordinates.

The force field does not act directly on the particle position.
Instead, it acts on the displacement between the particles, respecting the
periodic boundaries via the minimum image convention.
Thus, we first have to initialize this periodical space.

```{code-cell}
r_init = jnp.asarray(r_init)

displacement_fn, shift_fn = space.periodic_general(box, fractional_coordinates=False)
neighbor_fn = partition.neighbor_list(displacement_fn, box, 1.0)

nbrs_init = neighbor_fn.allocate(r_init)

prior_energy_fn = prior.init_prior_potential(displacement_fn, nonbonded_type="lennard_jones")(topology, force_field)

print(f"Energy on initial configuration: {prior_energy_fn(r_init, nbrs_init)}")
```

## Setup Simulation

To stabilize the simulation, we re-partition the masses from the carbon atoms
to the hydrogen atoms.


```{code-cell}
timings = ensemble.sampling.process_printouts(
    time_step=0.001, total_time=1e3, t_equilib=1e2,
    print_every=0.1, t_start=0.0
)

init_ref_state, sim_template = ensemble.sampling.initialize_simulator_template(
    simulate.nvt_langevin, shift_fn=shift_fn, nbrs=nbrs_init,
    init_with_PRNGKey=True, extra_simulator_kwargs={"kT": 2.56, "gamma": 1.0, "dt": 0.001}
)

mass = force_field.get_nonbonded_params(topology.get_atom_species())[0][:, 0]

reference_state = init_ref_state(
    random.PRNGKey(11), r_init,
    energy_or_force_fn=prior_energy_fn,
    init_sim_kwargs={"mass": mass, "neighbor": nbrs_init}
)
```

## Simulate

With the space, force function, and simulator set up, we can now compute a trajectory.
Following, we evaluate the potential energy, forces, and root mean
square distance (rmsd) for this trajectory for every sampled conformation.

```{code-cell}
quantities = {
    "energy": lambda state, *args, **kwargs: prior_energy_fn(state.position, *args, **kwargs),
    "rmsd": custom_quantity.init_rmsd(r_init, displacement_fn, box),
    "force": lambda state, *args, **kwargs: -jax.grad(prior_energy_fn)(state.position, *args, **kwargs)
}

simulate_fn = ensemble.sampling.trajectory_generator_init(
    simulator_template=sim_template,
    energy_fn_template=lambda _: prior_energy_fn,
    ref_timings=timings,
    quantities=quantities,
)

traj_state = simulate_fn(None, reference_state)
```

## Results

We save the energies, forces, and positions for later use, e.g., in coarse-graining applications.

```{code-cell}
# We save the force, energy, and position computations for later
onp.save(base_path / "forces_ethane.npy", traj_state.aux["force"])
onp.save(base_path / "energies_ethane.npy", traj_state.aux["energy"])
onp.save(base_path / "positions_ethane.npy", traj_state.trajectory.position)
```

```{code-cell}
disp = jax.vmap(displacement_fn)(
    traj_state.trajectory.position[:, 0, :],
    traj_state.trajectory.position[:, 1, :]
)
dist_CC = jnp.sqrt(jnp.sum(jnp.square(disp), axis=-1))

plt.hist(dist_CC, bins=100)
plt.xlabel("Distance C_1 - C_2 [nm]")
plt.ylabel("Frequency")
```

```{code-cell}
plt.plot(timings.t_production_end[::10], traj_state.aux["energy"][::10])
plt.xlabel("Time [ps]")
plt.ylabel("Energy [kJ/mol]")
```

```{code-cell}
plt.plot(timings.t_production_end[::10], traj_state.aux["rmsd"][::10])
plt.xlabel("Time [ps]")
plt.ylabel("RMSD [nm^2]")
```

## References

[^Das2009]: Das, A.; Andersen, H. C. The Multiscale Coarse-Graining Method. III. A Test of Pairwise Additivity of the Coarse-Grained Potential and of New Basis Functions for the Variational Calculation. _The Journal of Chemical Physics_ **2009**, _131_ (3), 034102. [https://doi.org/10.1063/1.3173812](https://doi.org/10.1063/1.3173812).
