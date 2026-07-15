Basic PPFL Training
===================

``APPFL`` provides users with the capabilities of simulating and training PPFL on either a single machine, a cluster, or multiple heterogeneous machines. We refer

- **simulation** as running PPFL experiments on a single machine or a cluster without actual data decentralization
- **training** as running PPFL experiments on multiple (heterogeneous) machines with actual decentralization of client datasets

We provide examples for both simulation and training in the following sections. Particularly, simulating PPFL is useful for those who develop, test, and validate new models and algorithms for PPFL, whereas training PPFL for those who consider actual PPFL settings in practice.

.. toctree::
   :maxdepth: 1
   :titlesonly:

   user_simulation_serial
   user_simulation_mpi
   user_train

In additional to the examples above, we also provide instructions on how to define models, datasets, and configurations for your own FL applications.

.. toctree::
   :maxdepth: 1
   :titlesonly:

   user_model
   user_loss
   user_metric
   user_data
