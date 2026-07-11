:modulename:`model`
==========================

Models
------

**Chemtrain** does not require a specific model to be used.
Nevertheless, it provides some pre-defined models, such as a simple classical
force field in the :mod:`jax_md_mod.model.prior` module.

.. toctree::
   :titlesonly:

   prior
   neural_networks


Building Blocks
-----------------

These submodules contain utilities necessary for building advanced potential
models. For example, the :mod:`jax_md_mod.model.sparse_graph` module contains
functions to construct a sparse graph from a JAX, M.D
:class:`jax_md.partition.NeighborList`.

.. toctree::
   :titlesonly:

   dimenet_basis_util
   layers
   sparse_graph
