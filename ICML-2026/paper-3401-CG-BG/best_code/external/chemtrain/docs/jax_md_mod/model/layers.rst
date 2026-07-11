:modulename:`model.layers`
========================================

.. currentmodule:: jax_md_mod.model.layers

.. automodule:: jax_md_mod.model.layers


Initializers
________________

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   OrthogonalVarianceScalingInit
   RBFFrequencyInitializer

DimeNet++ Layers
__________________

Basis Layers
.....................

.. autosummary::
   :toctree: _autosummary
   :template: haiku.rst

   SmoothingEnvelope
   RadialBesselLayer
   SphericalBesselLayer

.. _dimenet_building_blocks:

DimeNet++ Building Blocks
............................

.. autosummary::
   :toctree: _autosummary
   :template: haiku.rst

   ResidualLayer
   EmbeddingBlock
   OutputBlock
   InteractionBlock

Utility Functions
__________________

.. autosummary::
   :toctree: _autosummary
   :template: function.rst

    high_precision_segment_sum


