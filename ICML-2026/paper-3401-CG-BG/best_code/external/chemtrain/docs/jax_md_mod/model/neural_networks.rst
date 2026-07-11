:modulename:`model.neural_networks`
=================================================

.. currentmodule:: jax_md_mod.model.neural_networks

.. automodule:: jax_md_mod.model.neural_networks

DimeNet++
-----------------
The :class:`DimeNetPP` directly takes a
:class:`jax_md_mod.model.sparse_graph.SparseDirectionalGraph` as input and
predicts per-atom quantities. As :class:`DimeNetPP` is a haiku Module, it needs
to be wrapped inside a hk.transform() before it can be applied.

We provide 2 interfaces to DimeNet++:
The function :func:`dimenetpp_neighborlist` serves as a interface to Jax M.D.
The resulting *apply* function can be directly used as a jax_md energy_fn,
e.g. to run molecular dynamics simulations.

For direct prediction of global molecular properties,
:func:`dimenetpp_property_prediction` can be used.

.. autoclass:: DimeNetPP
   :members:

   .. automethod:: __init__
   .. automethod:: __call__

.. autofunction:: dimenetpp_neighborlist

.. autofunction:: dimenetpp_property_prediction

Pairwise NN
----------------
:class:`PairwiseNN` implements a neural network, that parametrizes 2-body
interactions. The function :func:`pair_interaction_nn` initializes a pairwise
jax_md neighborlist energy_fn, as an alternative to classical tabulated
potentials.

.. autoclass:: PairwiseNN
   :members:

   .. automethod:: __init__
   .. automethod:: __call__

.. autofunction:: pair_interaction_nn
