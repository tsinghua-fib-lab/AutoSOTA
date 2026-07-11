# Getting Started

## Example Setup

Using an exported model in ``chemtrain-deploy`` requires a correct
[installation](#chemtrain-deploy_installation) of ``LAMMPS`` with the ``JCN``
plugin.
Additionally, exporting a model requires a working installation of ``chemtrain``.

The following LAMMPS script describes how to use an exported model in a
simulation. A simple example on how to export a model can be found in the
documentation of {class}`chemtrain.deploy.exporter.Exporter`.

``input.lmp``:
```text
# Note: Plugins are loaded automatically if the paths in the environment
#       variables are set correctly

# 1) Basic settings. The units defined here must correspond to the units used
#    in the model.
units real
dimension 3
atom_style atomic
boundary p p p

neighbor 	2.0 bin
neigh_modify 	every 1 delay 0 check yes once no

# Must be adjusted model-dependent
comm_modify cutoff 7.0

# 2) Create random positions
region simulation_box block -25 25 -25 25 -25 25

create_box 1 simulation_box

create_atoms 1 random 4170 341341 simulation_box

# 3) Simulation settings
mass 1 18

# Loads the previously exported model. ``comm off`` selects its ordinary
# ``default`` variant and is also the default when the option is omitted.
pair_style chemtrain_deploy cpu 0.95 comm off # <backend> <memory_fraction>
pair_coeff * * model.ptb 1.1 1.1 # <path_to_model> <buffer_multiplier> <buffer_multiplier>

# A model exported with ``export(communication=True)`` also contains a
# distributed ``comm`` variant. Select it on a GPU with Newton pair forces on:
# newton on
# pair_style chemtrain_deploy cuda 0.95 comm on
# Requesting ``comm on`` without that variant, or with Newton off, is an error.

# 4) Visualization
thermo 10
thermo_style custom step temp pe ke etotal press

# 5) Run
min_style quickmin
minimize 1.0e-3 1.0e-5 10000 10000

neigh_modify 	every 10 delay 0 check yes

# 6) Visualization
thermo 50
dump mydmp all atom 100 dump.lammpstrj

# Must reset the velocity or the simulation explodes instantaneously
velocity all create 100.0 4928459 rot no dist gaussian

# 7) Run
fix         1 all nve
fix         2 all langevin 300.0 300.0 $(100. * dt) 1530917

timestep 1
run 10000
```

Similar to JAX, visible devices can be set using the environment variable
``CUDA_VISIBLE_DEVICES``. The plugin will automatically use the first visible
device. Therefore, the command

```bash
mpirun -n 8 lmp -in input.lmp
```

will run the simulation on the first visible device.
