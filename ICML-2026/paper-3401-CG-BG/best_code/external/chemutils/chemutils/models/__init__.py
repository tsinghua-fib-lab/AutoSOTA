
from jax_md_mod import uncache

import e3nn_jax._src.scatter
from . import e3nn_mod

# Serializable version without scan loop and numpy.unique
e3nn_jax._src.scatter._distinct_but_small = e3nn_mod._distinct_but_small
uncache('e3nn_jax._src.scatter')
