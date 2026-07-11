:modulename:`custom_quantity`
========================================

.. automodule:: jax_md_mod.custom_quantity

.. currentmodule:: jax_md_mod.custom_quantity


Common Quantities
------------------

.. autosummary::
   :toctree: _autosummary
   :template: function.rst

   energy_wrapper
   kinetic_energy_wrapper
   total_energy_wrapper
   temperature
   volume
   density


Structural Quantities
----------------------

Intra-Molecular
_______________

.. autosummary::
   :toctree: _autosummary
   :template: function.rst

   init_bond_angle_distribution
   init_bond_dihedral_distribution
   init_rmsd
   init_local_structure_index
   init_bond_length
   estimate_bond_constants
   angular_displacement
   dihedral_displacement


Inter-Molecular
________________

.. autosummary::
   :toctree: _autosummary
   :template: function.rst

   init_rdf
   init_adf_nbrs
   init_tcf_nbrs
   init_tetrahedral_order_parameter
   init_velocity_autocorrelation
   self_diffusion_green_kubo
   kinetic_energy_tensor
   virial_potential_part
   init_virial_stress_tensor
   init_pressure
   energy_under_strain
   init_sigma_born
   init_stiffness_tensor_stress_fluctuation


States
------

.. autosummary::
   :toctree: _autosummary
   :template: function.rst

   RDFParams
   ADFParams
   TCFParams
   BondAngleParams
   BondDihedralParams
