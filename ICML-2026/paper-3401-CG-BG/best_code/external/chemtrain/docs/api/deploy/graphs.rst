:modulename:`deploy.graphs`
============================

.. automodule:: chemtrain.deploy.graphs

.. currentmodule:: chemtrain.deploy.graphs

Graphs / Neighbor lists
------------------------

.. autoclass:: SimpleSparseNeighborList
   :members:
.. autoclass:: SimpleDenseNeighborList
   :members:

Neighbor List Statistics
------------------------
Neighborlist statistic are necessary to correctly allocate any buffer necessary
in neighbor list communications.

.. autoclass:: ListStatistics
   :members:
.. autoclass:: NeighborListStatistics
   :members:

Utility Functions
-----------------

.. autofunction:: compute_cell_list
.. autofunction:: prune_neighbor_list
.. autofunction:: prune_neighbor_list_dense
