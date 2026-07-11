:modulename:`model.sparse_graph`
==============================================

.. currentmodule:: jax_md_mod.model.sparse_graph

.. automodule:: jax_md_mod.model.sparse_graph

Graph dataclass
------------------------

.. autoclass:: SparseDirectionalGraph
   :members:

Graph building
-------------------------
Functions to extract :class:`SparseDirectionalGraph` from molecular positions
in a box.

.. autofunction:: sparse_graph_from_neighborlist
.. autofunction:: angle_triplets
.. autofunction:: safe_angle_mask
.. autofunction:: angle

Dataset preprocessing
-------------------------
For direct molecular property prediction tasks (i.e. predicting potential
energy, band-gap, partial charges, ...), one can pre-compute the
:class:`SparseDirectionalGraph` for the whole dataset.

.. autofunction:: convert_dataset_to_graphs
.. autofunction:: pad_per_atom_quantities
