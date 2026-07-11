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

from jax import numpy as jnp
from jax_sgmc.data.numpy_loader import NumpyDataLoader

from chemtrain.data import data_loaders

class TestForceMatching:

    def test_rng_key(self):

        loader = NumpyDataLoader(x=jnp.zeros((100, 10)))
    
        init_fn, get_batch_fn, release_fn = data_loaders.init_batch_functions(
            loader, 10, 2,
        )
    
        init_state = init_fn(random=True, rng_seed=11)
    
        _, batch1 = get_batch_fn(init_state)
        new_state, batch2 = get_batch_fn(init_state)
        
        assert 'rng' in batch1.keys(), "No rng key in batch."
        assert jnp.all(batch1['rng'] == batch2['rng']), "RNG key is not deterministic."
        
        _, batch3 = get_batch_fn(new_state)
        
        assert not jnp.all(batch1['rng'] == batch3['rng']), "RNG key did not change."