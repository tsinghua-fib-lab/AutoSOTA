import functools
import importlib
import numpy as onp

import jax
import jax.numpy as jnp

from jax_md_mod import (
    custom_space,
    custom_partition,
    custom_quantity
)
from jax_md import partition

from chemtrain import learn

from ase.calculators import calculator


class ChemtrainCalculator(calculator.Calculator):
    """ASE Calculator to deploy chemtrain models."""

    implemented_properties = ["energy", "forces"]

    def __init__(self,
                 energy_fn_template,
                 params,
                 r_cutoff: float = 0.5,
                 has_aux: bool = False,
                 capacity_multiplier: float = 1.25,
                 neighbor_list_format: str = 'sparse',
                 scale_energy: float = 96.485,
                 scale_pos: float = 10.0,
                 scale_charge: float = 11.7871,
                 **kwargs):
        super().__init__(**kwargs)

        self.scale_energy = scale_energy
        self.scale_pos = scale_pos
        self.scale_charge = scale_charge
        
        self.params = params
        self.energy_fn_template = energy_fn_template

        self.displacement_fn, _ = custom_space.general_space(
            box=jnp.eye(3), periodic=jnp.array([True, True, True], dtype=bool),
            wrapped=True
        )

        self.num_atoms = 0
        self.max_atoms = 0

        self.r_cutoff = r_cutoff
        self.capacity_multiplier = capacity_multiplier

        if neighbor_list_format == "sparse":
            neighbor_list_format = partition.Sparse
        elif neighbor_list_format == "dense":
            neighbor_list_format = partition.Dense
        else:
            raise ValueError(
                f"neighbor_list_format must be 'sparse' or 'dense', got "
                f"{neighbor_list_format}"
            )
        
        self.neighbor_list = custom_partition.masked_neighbor_list(
            self.displacement_fn, self.r_cutoff,
            capacity_multiplier=self.capacity_multiplier,
            format=neighbor_list_format,
        )
        self.nbrs = None
        
        self.has_aux = has_aux

        self.results = {}
        



    def initialize(self, atoms, overflow: bool = False):

        # Recompile if overflow occurred
        recompile = overflow or (self.nbrs is None)

        # Padd the atoms to the next exponent of 2
        self.num_atoms = len(atoms)
        if self.num_atoms > self.max_atoms:
            print(f"Increase the number of atoms from {self.max_atoms} to {self.num_atoms}")
            self.max_atoms = int(self.capacity_multiplier * self.num_atoms)
            recompile = True

        self.data = {
            # Global properties
            "periodic": ~jnp.asarray(atoms.get_pbc(), dtype=bool),
            "box": jnp.asarray([[atoms.get_cell()[j, i] for j in range(3)] for i in range(3)]) / self.scale_pos,
            "total_charge": jnp.sum(atoms.get_initial_charges()) * self.scale_charge,
            # Atomic properties
            "species": jnp.zeros(
                    (self.num_atoms,), dtype=jnp.int32
                ).at[:len(atoms)].set(
                    jnp.asarray(atoms.get_atomic_numbers(), dtype=jnp.int32)
                ),
            "mask": jnp.zeros(
                    (self.num_atoms,), dtype=bool
                ).at[:len(atoms)].set(
                    jnp.ones((len(atoms),), dtype=bool)
                ),
        }

        print(f"Recompile {recompile} with overflow {overflow}")

        # Initialize a new neighbor list and a new model
        if recompile:
            self.nbrs = self.neighbor_list.allocate(
                jnp.asarray(atoms.get_positions()) / self.scale_pos, **self.data
            )

                     # This feature extractor enables to evaluate the energy function
            # only once for all computations involving the energy and forces.
            feature_fns = {
                "energy_and_force": custom_quantity.energy_force_wrapper(
                    self.energy_fn_template, has_aux=self.has_aux
                )
            }

            # These are common quantities to train on. The energy function is not
            # necessary, since forces and energy are pre-extracted
            quantities = {
                "F": custom_quantity.force_wrapper(None),
                "U": custom_quantity.energy_wrapper(None),
                "overflow": lambda _, neighbor, **kwargs: neighbor.did_buffer_overflow,
            }


            self.model = jax.jit(learn.force_matching.init_model(
                self.nbrs, quantities=quantities,
                feature_extract_fns=feature_fns
            ))



    def calculate(
        self, atoms=None, properties=['energy'],
        system_changes=calculator.all_changes
    ):
        """Do the calculation.

        properties: list of str
            List of what needs to be calculated.  Can be any combination
            of 'energy', 'forces', 'stress', 'dipole', 'charges', 'magmom'
            and 'magmoms'.
        system_changes: list of str
            List of what has changed since last calculation.  Can be
            any combination of these six: 'positions', 'numbers', 'cell',
            'pbc', 'initial_charges' and 'initial_magmoms'.

        Subclasses need to implement this, but can ignore properties
        and system_changes if they want.  Calculated properties should
        be inserted into results dictionary like shown in this dummy
        example::

            self.results = {'energy': 0.0,
                            'forces': np.zeros((len(atoms), 3)),
                            'stress': np.zeros(6),
                            'dipole': np.zeros(3),
                            'charges': np.zeros(len(atoms)),
                            'magmom': 0.0,
                            'magmoms': np.zeros(len(atoms))}

        The subclass implementation should first call this
        implementation to set the atoms attribute and create any missing
        directories.
        """
        super().calculate(atoms, properties, system_changes)

        if any(change in system_changes for change in [
                    'numbers', 'cell', 'pbc', 'initial_charges'
        ]) or len(atoms) > self.max_atoms:
            self.initialize(atoms)

        obs = {
            "R": jnp.asarray(atoms.get_positions()) / self.scale_pos,
            "overflow": jnp.array(False, dtype=bool),
            "U": jnp.array(0.0, dtype=jnp.float32),
            "F": jnp.zeros((self.max_atoms, 3), dtype=jnp.float32)
        } | self.data
        obs = jax.tree.map(functools.partial(jnp.expand_dims, axis=0), obs)

        quantities = None
        for _ in range(2):  
            quantities = self.model(self.params, obs)
    
            print(f"Computed quantities: {quantities}")

            if not any(quantities['overflow']):
                break

            print("Detected overflow!")
            self.initialize(atoms, overflow=True)
    
        else:
            raise RuntimeError(
                "Neighbor list overflowed even after re-initialization."
            )
    
        self.results["energy"] = onp.asarray(quantities['U'][0]) / self.scale_energy
        self.results["forces"] = onp.asarray(quantities['F'][0, :self.num_atoms, :]) / self.scale_energy / self.scale_pos
