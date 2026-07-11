# Copyright 2023 Multiscale Modeling of Fluid Materials, TU Munich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License and limitations under the License.

import jax.numpy as jnp
from jax_md import space
from chemtrain.data import preprocessing
from pathlib import Path
import pytest

class TestMappingAla2:
    @pytest.fixture
    def setup_problem(self, datafiles):
        # Directory containing test mapping data for alanine dipeptide
        data_dir = Path(datafiles)
        
        # Load all-atom and coarse-grained data
        Ala2_AT_F = jnp.load(str(data_dir / 'forces_ala2_AT.npy'))
        Ala2_AT_R = jnp.load(str(data_dir / 'positions_ala2_AT.npy'))
        Ala2_CG_F = jnp.load(str(data_dir / 'forces_ala2_CG.npy'))
        Ala2_CG_R = jnp.load(str(data_dir / 'positions_ala2_CG.npy'))
        weights = jnp.load(str(data_dir / 'CG_weights_ala2.npy'))
        
        # Define periodic box
        box = jnp.identity(3) * 6
        displacement_fn, shift_fn = space.periodic_general(
            box=box, fractional_coordinates=False
        )
        
        return {
            'AT_F': Ala2_AT_F,
            'AT_R': Ala2_AT_R,
            'CG_F': Ala2_CG_F,
            'CG_R': Ala2_CG_R,
            'weights': weights,
            'displacement_fn': displacement_fn,
            'shift_fn': shift_fn
        }

    def test_map_ala2(self, setup_problem):
        # Get data from setup
        data = setup_problem
        
        # Map dataset: positions and forces
        mapped_R, mapped_F = preprocessing.map_dataset(
            data['AT_R'],
            data['displacement_fn'],
            data['shift_fn'],
            data['weights'],
            data['weights'],
            data['AT_F'],
        )

        assert mapped_R.shape == data['CG_R'].shape, \
            f"Mapped positions shape {mapped_R.shape} does not match expected {data['CG_R'].shape}"
        assert mapped_F.shape == data['CG_F'].shape, \
            f"Mapped forces shape {mapped_F.shape} does not match expected {data['CG_F'].shape}"

        assert jnp.allclose(mapped_R, data['CG_R'], rtol=1e-3, atol=1e-3), \
            "Mapped positions not close to expected CG positions"
        assert jnp.allclose(mapped_F, data['CG_F'], rtol=1e-3, atol=1e-3), \
            "Mapped forces not close to expected CG forces"
            
    def test_map_ala2_positions(self, setup_problem):
        # Get data from setup
        data = setup_problem
        
        # Map dataset: positions only
        mapped_R = preprocessing.map_dataset(
            data['AT_R'],
            data['displacement_fn'],
            data['shift_fn'],
            data['weights'],
        )

        assert mapped_R.shape == data['CG_R'].shape, \
            f"Mapped positions shape {mapped_R.shape} does not match expected {data['CG_R'].shape}"

        assert jnp.allclose(mapped_R, data['CG_R'], rtol=1e-3, atol=1e-3), \
            "Mapped positions not close to expected CG positions"
            
