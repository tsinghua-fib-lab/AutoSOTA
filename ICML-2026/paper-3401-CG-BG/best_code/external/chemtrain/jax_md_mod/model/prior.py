# Copyright 2023 Multiscale Modeling of Fluid Materials, TU Munich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module provides a simple classical force-field with pairwise non-bonded
interactions and bonded, angular, and dihedral interactions.

Creating a force field consists of three simple steps. First, the force field
parameters are read from a TOML file:

.. code:: python

   force_field = prior.ForceField.load_ff(ff_path)

Then, the force field is combined with a topology. This topoly can be infered
from a graph, e.g., provided by the package `mdtraj`:

.. code:: python

   top = mdtraj.load_topology(top_path)
   topology = prior.Topology.from_mdtraj(top, mapping=force_field.mapping(by_name=True))

Finally, topology and force field parameters are combined to create a potential:

.. code:: python

   prior_energy_fn_template = prior.init_prior_potential(
       displacement_fn, nonbonded_type="lennard_jones"
   )
   energy_fn = energy_fn_template(force_field, topology)

For a more detailed example, see the :doc:`/algorithms/prior_simulation`
example.

"""

import functools
import importlib
import re
import itertools
from io import StringIO
from typing import List, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as onp
from jax import tree_util
from jax.typing import ArrayLike
from jax_md import energy

from jax_md_mod.model import sparse_graph
from jax_md_mod import custom_energy


@tree_util.register_pytree_node_class
class ForceField:
    """Parameter set for classical molecular dynamics potentials.

    This class simplifies access to classical molecular dynamics potential
    parameters. This class is registered as a pytree with parameters as leaves,
    thus enable computing gradients with respect to the force field parameters.

    To construct a force field from a TOML file, see :func:`ForceField.load_ff`.

    """

    def __init__(self, data=None, lookup=None, mapping=None):
        self._data = data
        self._mapping = mapping
        self._lookup = lookup

    @classmethod
    def load_ff(cls, fname: str):
        """Loads a force field from a toml file.

        This is the recommended way to create a force field.

        Args:
            fname: Force field parameter file

        Returns:
            Returns a force field instance with parameters read from the file.

        """
        tomli = importlib.import_module("tomli")

        data = {
            "nonbonded": None,
            "bonded": {}
        }

        lookup = {
            "nonbonded": None,
            "bonded": {}
        }

        with open(fname, "rb") as f:
            ff = tomli.load(f)

        # Parse the contents using numpy
        non_bonded = onp.genfromtxt(
            StringIO(ff["nonbonded"]["atomtypes"]),
            dtype=None, delimiter=",", encoding="UTF-8"
        )

        mapping = {
            name: index for name, index, *_ in
            non_bonded
        }

        num_species = max(mapping.values()) + 1

        # Create lookup matrices. Entries unspecified by the force field
        # are assigned a lookup index of -1. This enables to mask out
        # not existing parametrizations.
        nonbonded_lookup = onp.full(num_species, -1, dtype=int)
        nonbonded_data = onp.zeros((num_species, 3))
        for idx, particle in enumerate(non_bonded):
            i, _, *params = particle
            s = mapping[i]
            nonbonded_lookup[s] = idx
            nonbonded_data[idx, :] = onp.asarray(params)

        data["nonbonded"] = jnp.asarray(nonbonded_data)
        lookup["nonbonded"] = jnp.asarray(nonbonded_lookup)

        bonds = onp.genfromtxt(
            StringIO(ff["bonded"]["bondtypes"]),
            dtype=None, delimiter=",", encoding="UTF-8", autostrip=True,
            comments="#"
        ).reshape((-1,))

        # Fill up to indicate which bonds are not provided by the force field
        bond_lookup = onp.full(
            (num_species, num_species), -1, dtype=int)
        bond_data = onp.zeros((max([len(bonds), 1]), 2))
        for idx, bond in enumerate(bonds):
            i, j, *params = bond
            s1 = mapping[i]
            s2 = mapping[j]

            bond_lookup[s1, s2] = idx
            bond_lookup[s2, s1] = idx
            bond_data[idx, :] = onp.asarray(params)

        data["bonded"]["bonds"] = jnp.asarray(bond_data)
        lookup["bonded"]["bonds"] = jnp.asarray(bond_lookup)

        angles = onp.genfromtxt(
            StringIO(ff["bonded"]["angletypes"]),
            dtype=None, delimiter=",", encoding="UTF-8", autostrip=True
        ).reshape((-1,))

        angle_lookup = onp.full(
            (num_species, num_species, num_species), -1, dtype=int)
        angle_data = onp.zeros((max([len(angles), 1]), 2))
        for idx, angle in enumerate(angles):
            i, j, k, *params = angle
            s1 = mapping[i]
            s2 = mapping[j]
            s3 = mapping[k]

            angle_lookup[s1, s2, s3] = idx
            angle_lookup[s3, s2, s1] = idx
            angle_data[idx, :] = onp.asarray(params)

        data["bonded"]["angles"] = jnp.asarray(angle_data)
        lookup["bonded"]["angles"] = jnp.asarray(angle_lookup)

        # Maximum number of dihedral terms
        max_terms = 4
        dihedrals = onp.genfromtxt(
            StringIO(ff["bonded"]["dihedraltypes"]),
            dtype=None, delimiter=",", encoding="UTF-8", autostrip=True
        ).reshape((-1,))

        dihedrals_lookup = onp.full(
            (num_species, num_species, num_species, num_species, max_terms), -1, dtype=int)
        dihedral_data = onp.zeros((max([len(dihedrals), 1]), 1))
        for idx, dihedral in enumerate(dihedrals):
            i, j, k, l, phase, kd, nd = dihedral
            s1 = mapping[i]
            s2 = mapping[j]
            s3 = mapping[k]
            s4 = mapping[l]

            dihedrals_lookup[s1, s2, s3, s4, nd - 1] = idx
            dihedrals_lookup[s4, s3, s2, s1, nd - 1] = idx

            # We assume that the phase shift only assumes values of multiples of
            # pi (i.e. 0 or 180 deg). To simply learn the phase shift, we
            # reformulate the dihedral potential function from
            # kd * (1 + cos(psi - phase)) to |kd| + kd * cos(psi) and choose
            # kd < 0 for a phase shift of 180.0 deg and kd > 0 otherwise.

            if phase >= 90.:
                dihedral_data[idx, :] = -onp.asarray(kd)
            else:
                dihedral_data[idx, :] = onp.asarray(kd)

        data["bonded"]["dihedrals"] = jnp.asarray(dihedral_data)
        lookup["bonded"]["dihedrals"] = jnp.asarray(dihedrals_lookup)

        data["lj14_scaling"] = ff["nonbonded"].get("lj14_scaling", 1.0)

        return cls(data, lookup, mapping)

    def write_ff(self, fname: str):
        """Writes the parameters of the force field back into a toml file.

        Args:
            fname: Path, where the force field should be stored.

        """
        tomli_w = importlib.import_module("tomli_w")

        reverse_mapping = {
            value: key for key, value in self._mapping.items()
        }

        atom_data = [
            (i, self._lookup["nonbonded"][i])
            for i in range(self.max_species)
            if self._lookup["nonbonded"][i] >= 0
        ]
        atomtypes = "# name,    species,    mass,        sigma,      epsilon\n"
        atomtypes += "\n".join(
            ",".join([
                f"{reverse_mapping[i]}".rjust(5, " "),
                f"{i}".rjust(5, " "),
                f"{self._data['nonbonded'][idx][0] :.3f}".rjust(7, " "),
                f"{self._data['nonbonded'][idx][1] :.5e}".rjust(9, " "),
                f"{self._data['nonbonded'][idx][2] :.5e}".rjust(9, " "),
            ])
            for i, idx in atom_data
        )
        atomtypes += "\n"


        bond_data = [
            (i, j, self._lookup["bonded"]["bonds"][i, j])
            for i, j in itertools.product(range(self.max_species), repeat=2)
            if self._lookup["bonded"]["bonds"][i, j] >= 0 and i <= j
        ]
        bondtypes = "#    i,    j,    b0,    kb\n"
        bondtypes += "\n".join(
            ",".join([
                f"{reverse_mapping[i]}".rjust(5, " "),
                f"{reverse_mapping[j]}".rjust(5, " "),
                f"{self._data['bonded']['bonds'][idx][0] :.5f}".rjust(7, " "),
                f"{self._data['bonded']['bonds'][idx][1] :.1f}".rjust(7, " "),
            ])
            for i, j, idx in bond_data
        )
        bondtypes += "\n"

        angle_data = [
            (i, j, k, self._lookup["bonded"]["angles"][i, j, k])
            for i, j, k in itertools.product(range(self.max_species), repeat=3)
            if self._lookup["bonded"]["angles"][i, j, k] >= 0 and i <= k
        ]
        angletypes = "#    i,    j,    k,    th0,    kth\n"
        angletypes += "\n".join(
            ",".join([
                f"{reverse_mapping[i]}".rjust(5, " "),
                f"{reverse_mapping[j]}".rjust(5, " "),
                f"{reverse_mapping[k]}".rjust(5, " "),
                f"{self._data['bonded']['angles'][idx][0] :.3f}".rjust(7, " "),
                f"{self._data['bonded']['angles'][idx][1] :.3f}".rjust(7, " "),
            ])
            for i, j, k, idx in angle_data
        )
        angletypes += "\n"

        max_terms = self._lookup["bonded"]["dihedrals"].shape[-1]
        dihedraltypes = "#    i,    j,    k,    l,    phase,    kd    pn\n"
        for nd in range(max_terms):
            dihedral_idx = [
                idx if idx[0] < idx[3] else idx[::-1]
                for idx in itertools.product(range(self.max_species), repeat=4)
            ]

            # Remove double elements and undefined entries
            dihedral_idx = set(dihedral_idx)
            dihedral_data = [
                (i, j, k, l, self._lookup["bonded"]["dihedrals"][i, j, k, l, nd])
                for i, j, k, l in dihedral_idx
                if self._lookup["bonded"]["dihedrals"][i, j, k, l, nd] >= 0
            ]

            dihedraltypes += "\n".join(
                ",".join([
                    f"{reverse_mapping[i]}".rjust(5, " "),
                    f"{reverse_mapping[j]}".rjust(5, " "),
                    f"{reverse_mapping[k]}".rjust(5, " "),
                    f"{reverse_mapping[l]}".rjust(5, " "),
                    f"{180. * (self._data['bonded']['dihedrals'][idx][0] <= 0) :.2f}".rjust(7, " "),
                    f"{onp.abs(self._data['bonded']['dihedrals'][idx][0]) :.3f}".rjust(7, " "),
                    f"{nd + 1}".rjust(4, " "),
                ])
                for i, j, k, l, idx in dihedral_data
            )
            dihedraltypes += "\n"

        ff_params = {
            "bonded": {
                "bondtypes": bondtypes,
                "angletypes": angletypes,
                "dihedraltypes": dihedraltypes
            },
            "nonbonded": {
                "atomtypes": atomtypes
            }
        }

        with open(fname, "wb") as f:
            tomli_w.dump(ff_params, f, multiline_strings=True)

    def get_data(self, keys: Union[str, Tuple[Union[str, Tuple[str]]]] = None):
        """Returns the trainable parameters.

        It is possible to select only a subset of the trainable parameters.
        For example, by selecting only the bond parameters of the bonded
        interactions:

        .. code :: python

           >>> selection = (
           ...     ("bonded", ("bonds",)),
           ... )
           >>> ff.get_data(selection)

        Args:
            keys: Key, or tuple structure of keys corresponding to the partial
                data that should be returned.

        Returns:
            Returns the learnable interaction parameters.

        """
        if keys is None:
            return self._data
        if isinstance(keys, str):
            return {keys: self._data[keys]}

        data = {}
        for key in keys:
            if isinstance(key, tuple):
                data[key[0]] = {}
                for subkey in key[1]:
                    data[key[0]][subkey] = self._data[key[0]][subkey]
            else:
                data[key] = self._data[key]

        return data

    def set_data(self, data):
        """Update the learnable interaction parameters.

        In combination with `get_data`, `set_data` enables to learn only
        parts of the force field.

        Args:
            data: Dictionary, corresponding to the internal data structure.

        Returns:
            Returns a new instance of the force field.

        """
        # TODO: Check whether this avoids leaking tracers
        new_data = {}
        for key, item in self._data.items():
            if isinstance(item, dict):
                new_data[key] = {k: v for k, v in self._data[key].items()}
                if key in data.keys():
                    new_data[key].update(data[key])
            else:
                if key in data.keys():
                    new_data[key] = data[key]
                else:
                    new_data[key] = item

        new_lookup = {
            "bonded": {**self._lookup["bonded"]},
            "nonbonded": self._lookup["nonbonded"]
        }
        new_mapping = {**self._mapping}

        new_force_field = ForceField(new_data, new_lookup, new_mapping)
        return new_force_field

    @property
    def max_species(self):
        """Maximum number of species."""
        return max(self._mapping.values()) + 1

    def mapping(self, by_name=False, renaming_pattern: List[Tuple[str, str]] = None):
        """Returns a function that maps atom data into a species number.

        Args:
            by_name: If true, then the atom name (e. g. ``"CA"``) is used as an
                identifier. Otherwise, the element symbol (e. g. ``"C"``) is
                used to look up the species number.
            renaming_pattern: Translate the atom names between different naming
                conventions, e.g. from PDB to AMBER.

        Returns:
            Returns a mapping function that maps atom data into a species
            number.

        """
        @rename_atoms(lookup_table=renaming_pattern)
        def mapping_fn(symbol: int, is_water: bool, name: str = "", residue: str = "", **kwargs):
            if by_name:
                if residue == "ALA" and name == "CB":
                    print(f"Rename ALA CB to CH3")
                    name = "CH3"
                if residue == "NME" and name == "C":
                    print(f"Rename NME C to CH3")
                    name = "CH3"

                return self._mapping[name]

            if is_water:
                return self._mapping[symbol + "W"]
            else:
                return self._mapping[symbol]

        return mapping_fn

    def get_nonbonded_params(self, s):
        """Get the nonbonded parameters for a species."""
        idx = self._lookup["nonbonded"][s]
        data = self._data["nonbonded"][idx]
        valid = idx >= 0
        return data, valid

    def get_bond_params(self, s1, s2):
        """Get the parameters of a bond between two species."""
        idx = self._lookup["bonded"]["bonds"][s1, s2]
        data = self._data["bonded"]["bonds"][idx]
        valid = idx >= 0
        return data, valid

    def get_angle_params(self, s1, s2, s3):
        """Get the parameters of an angle between three species."""
        idx = self._lookup["bonded"]["angles"][s1, s2, s3]
        data = self._data["bonded"]["angles"][idx]
        valid = idx >= 0
        return data, valid

    def get_dihedral_params(self, s1, s2, s3, s4, multiplicity=None):
        """Get the parameters of an angle between four species."""
        # Automatically set the maximum multiplicity
        if multiplicity is None:
            max_terms = self._lookup["bonded"]["dihedrals"].shape[-1]
            multiplicity = jnp.arange(max_terms)
            params = jax.vmap(
                self.get_dihedral_params,
                in_axes=(None, None, None, None, 0),
                out_axes=-1)(s1, s2, s3, s4, multiplicity)
            return params

        idx = self._lookup["bonded"]["dihedrals"][s1, s2, s3, s4, multiplicity]
        data = self._data["bonded"]["dihedrals"][idx]
        valid = idx >= 0
        return data, valid, jnp.full_like(s1, multiplicity + 1)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        structure, lookup, mapping = aux_data
        data = tree_util.tree_unflatten(structure, children)
        return cls(data, lookup, mapping)

    def tree_flatten(self):
        children, structure = tree_util.tree_flatten(self._data)
        aux_data = (structure, self._lookup, self._mapping)
        return children, aux_data

@tree_util.register_pytree_node_class
class Topology:
    """Class documenting the topology of a system.

    This pytree stores the topological information of a system and provides
    utilities, to create this information e.g. from an `mdtraj` topology.

    For a simple way to load a topology from a molecular graph, see
    :func:`Topology.from_mdtraj`.

    Args:
        num_particles: The number of particles in the system
        species: Integer or array of species number.
        bond_idx: Array of shape `[B, 2]` with indices of the atoms that span
            the :math:`B` bonds.
        angle_idx: Array of shape `[A, 3]` with indices of the atoms that span
            the :math:`A` angles.
        dihedral_idx: Array of shape `[D, 4]` with indices of the atoms that
            span the :math:`D` dihedral angles.
        improper_dihedral_idx: Not used.
    """

    def __init__(self,
                 num_particles: int,
                 species: ArrayLike = None,
                 bond_idx: ArrayLike = None,
                 angle_idx: ArrayLike = None,
                 dihedral_idx: ArrayLike = None,
                 improper_dihedral_idx: ArrayLike = None):
        if bond_idx is None:
            bond_idx = jnp.zeros((0, 2), dtype=jnp.int_)
        if angle_idx is None:
            angle_idx = jnp.zeros((0, 3), dtype=jnp.int_)
        if dihedral_idx is None:
            dihedral_idx = jnp.zeros((0, 4), dtype=jnp.int_)
        if improper_dihedral_idx is None:
            improper_dihedral_idx = jnp.zeros((0, 4), dtype=jnp.int_)

        self._num_particles = num_particles
        self._species = species
        self._bond_idx = bond_idx
        self._angle_idx = angle_idx
        self._dihedral_idx = dihedral_idx
        self._improper_dihedral_idx = improper_dihedral_idx

    def get_padded_topology(self,
                            num_particles: int,
                            max_species: int = 1,
                            max_bonds: int = 0,
                            max_angles: int = 0,
                            max_dihedrals: int = 0,
                            max_improper_dihedrals: int = 0):
        """Returns a new topology with padded graph data.

        When dealing with systems that have a different number of bonds, etc.
        padding the topologies enables to still batch-process the systems.
        """

        # Create a new data structure by padding the lists
        # TODO: Check that no files have less angles than given.
        #       Pad all the indices and replace the old invalid particle indices
        #       by the new num_atoms
        # TODO: Replace the invalid indices by the new num_particles
        raise NotImplementedError("Padding not yet implemented")

    def get_atom_species(self, idx=None):
        """Returns the species numbers for the atoms of the system."""
        if self._species is None:
            return None

        if idx is None:
            idx = jnp.arange(self._num_particles)
        if self._species.ndim == 0:
            species = jnp.full_like(idx, self._species, dtype=jnp.int_)
        else:
            species = self._species[idx]
        species = jnp.where(
            idx < self._num_particles, species, -1
        )
        return species

    def get_bonds(self):
        """Returns the indices, species and mask for all angles."""
        idxs = self._bond_idx
        if idxs.ndim == 1:
            return None, None, None

        species = jnp.stack(
            (
                self.get_atom_species(idxs[:, 0]),
                self.get_atom_species(idxs[:, 1])
            ), axis=-1
        )
        mask = idxs[:, 0] < self._num_particles
        return idxs, species, mask

    def get_angles(self):
        """Returns the indices, species and mask for all angles."""
        idxs = self._angle_idx
        if idxs.ndim == 1:
            return None, None, None

        species = jnp.stack(
            (
                self.get_atom_species(idxs[:, 0]),
                self.get_atom_species(idxs[:, 1]),
                self.get_atom_species(idxs[:, 2])
            ), axis=-1
        )
        mask = idxs[:, 0] < self._num_particles
        return idxs, species, mask

    def get_dihedrals(self):
        """Returns the indices, species and mask for all angles."""
        idxs = self._dihedral_idx
        if idxs.ndim == 1:
            return None, None, None

        species = jnp.stack(
            (
                self.get_atom_species(idxs[:, 0]),
                self.get_atom_species(idxs[:, 1]),
                self.get_atom_species(idxs[:, 2]),
                self.get_atom_species(idxs[:, 3])
            ), axis=-1
        )
        mask = idxs[:, 0] < self._num_particles
        return idxs, species, mask

    def prune_topology(self, force_field: ForceField):
        """Remove all bonds, angles, etc. not parametrized by the force field.

        Args:
            force_field: Force field to identify, which bonds, etc. are
                parametrized.

        Returns:
            Returns a topology with potentially reduced size.
        """

        bonds_idx, species, mask = self.get_bonds()
        mask = onp.logical_and(mask, force_field.get_bond_params(*species.T)[1])
        bonds_idx = bonds_idx[mask]

        angle_idx, species, mask = self.get_angles()
        mask = onp.logical_and(mask, force_field.get_angle_params(*species.T)[1])
        angle_idx = angle_idx[mask]

        dihedral_idx, species, mask = self.get_dihedrals()
        ff_mask = force_field.get_dihedral_params(*species.T)[1]
        mask = onp.logical_and(mask, ff_mask.any(axis=1))
        dihedral_idx = dihedral_idx[mask]

        top = Topology(
            self._num_particles, self._species, bonds_idx, angle_idx,
            dihedral_idx, None
        )
        return top

    @classmethod
    def concatenate(cls, *topologies):
        """Concatenate the topologies. """
        raise NotImplementedError(
            "Concatenating topologies is not yet implemented.")

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data, *children)

    def tree_flatten(self):
        children = (
            self._species, self._bond_idx, self._angle_idx,
            self._dihedral_idx, self._improper_dihedral_idx,
        )
        aux_data = (
            self._num_particles,
        )
        return children, aux_data

    @classmethod
    def from_mdtraj(cls, topology, mapping=None):
        """Creates a topology instance from a ``mdtraj`` topology instance.

        This function uses mdtraj to create a graph-representation of the
        system. It then discovers bonds, angles and dihedral angles by searching
        for all simple paths between tuples with corresponding length.

        Args:
            topology: `mdtraj` topology object
            mapping: Function to map the atom data, e.g., atom symbol, to a
                species number.

        Returns:
            Returns a JAX topology based on the ``mdtraj`` topology.

        """

        graph = topology.to_bondgraph()
        nx = importlib.import_module("networkx")

        if mapping is None:
            mapping = lambda **kwargs: kwargs["number"]

        species = [
            mapping(
                number=n.element.number, name=n.name,
                symbol=n.element.symbol, code=n.residue.code, residue=n.residue.name,
                is_water=n.residue.is_water, is_protein=n.residue.is_protein
            )
            for n in graph.nodes
        ]

        # Retrieve topology from graph
        bonds = []
        angles = []
        dihedrals = []

        # Discover all molecules (connected graphs)
        for mol in nx.k_edge_subgraphs(graph, 1):
            # Find all bonds, angles and dihedrals between pairs of atoms
            for atm1, atm2 in itertools.combinations(mol, 2):
                if not nx.has_path(graph, atm1, atm2):
                    # Pairs not part of the same molecule
                    continue

                all_paths = nx.all_simple_paths(graph, atm1, atm2, cutoff=4)
                for path in all_paths:
                    # Sort the path and always start with the lowest numbered
                    # species
                    if species[path[0].index] > species[path[-1].index]:
                        path.reverse()

                    # Identify bonds, angles and dihedrals
                    if len(path) == 2:
                        bonds.append([a.index for a in path])
                    elif len(path) == 3:
                        angles.append([a.index for a in path])
                    elif len(path) == 4:
                        dihedrals.append([a.index for a in path])

        return cls(
            len(species),
            species=jnp.asarray(species),
            bond_idx=jnp.asarray(bonds),
            angle_idx=jnp.asarray(angles),
            dihedral_idx=jnp.asarray(dihedrals)
        )


# Functions to read out potential parameters for all bonds, angles, etc. of the
# system

def init_bond_potential(displacement_fn, topology, force_field):
    """Initializes all bond-potentials for the given topology. """
    bond_idx, bond_species, bond_mask = topology.get_bonds()
    if bond_idx is None:
        return None, None

    bond_params, mask = force_field.get_bond_params(
        bond_species[:, 0], bond_species[:, 1])

    # Mask out bonds by setting their energy to zero
    b0 = bond_params[:, 0]
    kb = jnp.where(
        jnp.logical_and(mask, bond_mask), bond_params[:, 1], 0.0
    )
    bond_potential = energy.simple_spring_bond(
        displacement_fn, bond_idx, length=b0, epsilon=kb)
    return bond_potential, jnp.logical_and(mask, bond_mask)


def init_angle_potential(displacement_fn, topology, force_field):
    """Initializes all angle-potentials for the given topology. """
    angle_idx, angle_species, angle_mask = topology.get_angles()
    if angle_idx is None:
        return None, None

    angle_params, mask = force_field.get_angle_params(
        angle_species[:, 0], angle_species[:, 1], angle_species[:, 2])
    # Mask out bonds by setting their energy to zero
    th0 = angle_params[:, 0]
    kth = jnp.where(
        jnp.logical_and(mask, angle_mask), angle_params[:, 1], 0.0
    )
    angle_potential = custom_energy.harmonic_angle(
        displacement_fn, angle_idx, th0=th0, kth=kth)
    return angle_potential, jnp.logical_and(mask, angle_mask)


def init_dihedral_potential(displacement_fn, topology, force_field):
    """Initializes all dihedral-potentials for the given topology. """
    dihedral_idx, dihedral_species, dihedral_mask = topology.get_dihedrals()
    if dihedral_idx is None:
        return None, None

    data, mask, mult = force_field.get_dihedral_params(
        dihedral_species[:, 0], dihedral_species[:, 1], dihedral_species[:, 2],
        dihedral_species[:, 3]
    )

    kd = data[:, 0, :]
    phase = jnp.zeros_like(kd)

    # Mask out bonds by setting their energy to zero if
    # - the term with corresponding multiplicity is undefined
    # - the dihedral angle does not exist
    kd = jnp.where(jnp.logical_and(mask, dihedral_mask[:, None]), kd, 0.0)

    dihedral_potential = custom_energy.periodic_dihedral(
        displacement_fn, dihedral_idx, phase_angle=phase, force_constant=kd,
        multiplicity=mult
    )
    return dihedral_potential, jnp.logical_and(jnp.any(mask, axis=1), dihedral_mask)


def init_nonbonded_potential(displacement_fn,
                             topology,
                             force_field: ForceField,
                             nonbonded_type = "repulsion",
                             r_onset=0.45, r_cutoff=0.5):
    """Initializes the non-bonded interactions for the given topology. """
    species = topology.get_atom_species()
    nonbonded_params, mask = force_field.get_nonbonded_params(species)

    # Box size is not necessary, as we don't generate a neighbor list.
    # We use the Lorentz-Berthelot combining rules.
    kwargs = dict(
        initialize_neighbor_list=False, box_size=0.0,
        sigma=(lambda s1, s2: 0.5 * (s1 + s2), nonbonded_params[:, 1]),
        epsilon=(lambda e1, e2: jnp.sqrt(e1 * e2), mask * nonbonded_params[:, 2])
    )

    # TODO: Provide onset and cutoff arguments

    if nonbonded_type == "repulsion":
        return custom_energy.generic_repulsion_neighborlist(
            displacement_fn, **kwargs, r_onset=r_onset, r_cutoff=r_cutoff)
    if nonbonded_type == "truncated_lennard_jones":
        # No additional truncation necessary
        return custom_energy.truncated_lennard_jones_neighborlist(
            displacement_fn, **kwargs)
    if nonbonded_type == "lennard_jones":
        return custom_energy.customn_lennard_jones_neighbor_list(
            displacement_fn, **kwargs, r_onset=r_onset, r_cutoff=r_cutoff)

    raise NotImplementedError(
        f"Nonbonded type {nonbonded_type} is not a valid choice. Choose from"
        f"'repulsion', 'truncated_lennard_jones' or 'lennard_jones'."
    )


def init_prior_potential(displacement_fn,
                         mask_bonded: bool = True,
                         nonbonded_type: str = "repulsion",
                         r_onset: float = 0.45,
                         r_cutoff: float = 0.5):
    """Initializes the prior template.

    Args:
        displacement_fn: `jax_md` displacement function
        mask_bonded: Excluded non-bonded interactions between bonded atoms via
            masking them in the neighbor list. This might be ineffective if the
            number of bonds is very large.
        nonbonded_type: Type of the nonbonded interactions. Possible values are
            ``"repulsion"``, ``"lennard_jones"``, ``"truncated_lennard_jones"``.
        r_onset: Distance to start smoothing the nonbonded interactions to 0.
        r_cutoff: Cutoff distance of the neighbor list.

    """
    def prior_potential_template(topology: Topology, force_field: ForceField):
        # Bonded
        bond_potential, bond_mask = init_bond_potential(
            displacement_fn, topology, force_field)
        angle_potential, angle_mask = init_angle_potential(
            displacement_fn, topology, force_field)
        dihedral_potential, dihedral_mask = init_dihedral_potential(
            displacement_fn, topology, force_field)

        # Nonbonded
        nonbonded_potential_fn = init_nonbonded_potential(
            displacement_fn, topology, force_field, nonbonded_type,
            r_onset=r_onset, r_cutoff=r_cutoff
        )

        def potential_fn(position, neighbor=None, **kwargs):
            pot = 0.0

            if bond_potential is not None:
                pot += bond_potential(position)
            if angle_potential is not None:
                pot += angle_potential(position)
            if dihedral_potential is not None:
                pot += dihedral_potential(position)

            if mask_bonded:
                # Exclude pairs connected via bonds and angles from the
                # neighbor list to exclude them also from beeing considered
                # in the nonbonded potential computation
                neighbor = sparse_graph.subtract_topology_from_neighbor_list(
                    neighbor, topology, bond_mask=bond_mask,
                    angle_mask=angle_mask, dihedral_mask=dihedral_mask
                )
                pot += nonbonded_potential_fn(position, neighbor)
            else:
                raise NotImplementedError(
                    "Currently, only masking is implemented to exclude "
                    "non-bonded interactions from bonded neighbors."
                )

            return pot
        return potential_fn
    return prior_potential_template


# Helper functions to train the force field.


def constrain_ff_params(unconstrained_data):
    """Maps unconstrained force field data onto a constrained space.

    The constrained space enforces the following properties:
      - Force constants are positive :math:`k_B, k_\\theta, k_D > 0`
      - Bond lengths are positive :math:`b_0> 0`
      - Angles are subject to :math:`\\theta, \\theta_D \\in [0, 180]`

    Args:
        unconstrained_data: Data obtained by `force_field.get_data()` to be
            constrained.

    Returns:
        Returns values mapped onto the constrained space.

    """
    constrained_params = {}
    if "bonded" in unconstrained_data.keys():
        constrained_params["bonded"] = {}
        if "bonds" in unconstrained_data["bonded"].keys():
            constrained_params["bonded"]["bonds"] = jnp.log(
                unconstrained_data["bonded"]["bonds"]
            )
        if "angles" in unconstrained_data["bonded"].keys():
            constrained_params["bonded"]["angles"] = jnp.stack((
                jnp.arctanh(unconstrained_data["bonded"]["angles"][:, 0] / 90. - 1.),
                jnp.log(unconstrained_data["bonded"]["angles"][:, 1])
            ), axis=-1)
        if "dihedrals" in unconstrained_data["bonded"].keys():
            constrained_params["bonded"]["dihedrals"] = unconstrained_data["bonded"]["dihedrals"]
    if "nonbonded" in unconstrained_data.keys():
        constrained_params["nonbonded"] = jnp.log(unconstrained_data["nonbonded"])

    return constrained_params

def unconstrain_ff_params(constrained_data):
    """Recovers the unconstrained data from the constrained space.

    This function maps the parameters back onto the unconstrained space.
    Additionally, the mapping stops the backpropagation of gradients for
    the dihedral multiplicity.

    Args:
        constrained_data: Force field parameters in the constrained space.

    """
    unconstrained_params = {}
    if "bonded" in constrained_data.keys():
        unconstrained_params["bonded"] = {}
        if "bonds" in constrained_data["bonded"].keys():
            unconstrained_params["bonded"]["bonds"] = jnp.exp(
                constrained_data["bonded"]["bonds"]
            )
        if "angles" in constrained_data["bonded"].keys():
            unconstrained_params["bonded"]["angles"] = jnp.stack((
                90. * (1 + jnp.tanh(constrained_data["bonded"]["angles"][:, 0])),
                jnp.exp(constrained_data["bonded"]["angles"][:, 1])
            ), axis=-1)
        if "dihedrals" in constrained_data["bonded"].keys():
            unconstrained_params["bonded"]["dihedrals"] = constrained_data["bonded"]["dihedrals"]
    if "nonbonded" in constrained_data.keys():
        unconstrained_params["nonbonded"] = jnp.exp(constrained_data["nonbonded"])

    return unconstrained_params


# Helper function to rename e.g. from PDB atom names to AMBER atom names


PDB_TO_AMBER = [
    ("H.+", "HC"), ("H", "H"), ("CH3", "CT"), ("C[A-B]+", "CT"), ("C", "C"),
    ("N", "N"), ("O", "O")
]


def lookup_fn(name, lookup_table):
    """Searches for a matching pattern and return the alternative atom name. """
    for pattern, alt_name in lookup_table:
        if re.fullmatch(pattern, name) is not None:
            return alt_name
    return f"UNDEFINED({name})"


def rename_atoms(lookup_table=None):
    """Renames the atoms before passing them to the mapping."""
    def decorator(fun):
        if lookup_table is None:
            return fun

        @functools.wraps(fun)
        def wrapper(**kwargs):
            name = kwargs.pop("name")
            alt_name = lookup_fn(name, lookup_table)
            return fun(name=alt_name, **kwargs)
        return wrapper
    return decorator
