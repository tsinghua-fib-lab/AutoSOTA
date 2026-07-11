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

from pathlib import Path

import numpy as onp
import pytest

import mdtraj

from jax import tree_util

from jax_md_mod.model.prior import Topology, ForceField

class TestForceField:

    @pytest.mark.parametrize("sample_conf", [
            (
                "heavy_2_7nm.gro",
                "heavy_untrained.toml",
                True
            ),
        ])
    def test_io(self, tmp_path, datafiles, sample_conf):
        # Test whether loading and saving of a force field works by comparing
        # the loaded values for multiple topologies

        top, ff_file, by_name = sample_conf
        top = datafiles / top
        ff_file = datafiles / ff_file
        top = mdtraj.load_topology(top)

        org_force_field = ForceField.load_ff(ff_file)
        topology = Topology.from_mdtraj(top, org_force_field.mapping(by_name=by_name))
        org_parameters = (
            org_force_field.get_bond_params(*topology.get_bonds()[1].T),
            org_force_field.get_angle_params(*topology.get_angles()[1].T),
            org_force_field.get_dihedral_params(*topology.get_dihedrals()[1].T)
        )

        # Test whether saving and loading changes the force field parameters
        new_ff_file = tmp_path / Path(ff_file).name
        org_force_field.write_ff(new_ff_file)
        new_force_field = ForceField.load_ff(new_ff_file)

        new_parameters = (
            new_force_field.get_bond_params(*topology.get_bonds()[1].T),
            new_force_field.get_angle_params(*topology.get_angles()[1].T),
            new_force_field.get_dihedral_params(*topology.get_dihedrals()[1].T)
        )

        assert tree_util.tree_structure(org_parameters) == tree_util.tree_structure(new_parameters)
        for old, new in zip(tree_util.tree_leaves(org_parameters), tree_util.tree_leaves(new_parameters)):
            assert onp.all(onp.isclose(old, new))


class TestTopology:

    @pytest.mark.skip
    def test_pruning(self):
        pass

        # Test whether pruning works

    @pytest.mark.skip
    def test_concatenation(self):
        pass

        # Test whether concatenating topologies work
