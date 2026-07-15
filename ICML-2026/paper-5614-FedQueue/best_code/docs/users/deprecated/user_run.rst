Basic PPFL Training
===================

``APPFL`` provides users with the capabilities of simulating and training PPFL on either a single machine, a cluster, or multiple heterogeneous machines. We refer

- **simulation** as running PPFL experiments on a single machine or a cluster without actual data decentralization
- **training** as running PPFL experiments on multiple (heterogeneous) machines with actual decentralization of client datasets

Hence, we describe two types of PPFL run:

.. toctree::
   :maxdepth: 1
   :titlesonly:

   user_simulation
   user_train


:ref:`Simulating PPFL` is useful for those who develop, test, and validate new models and algorithms for PPFL, whereas :ref:`Training PPFL` for those who consider actual PPFL settings in practice.

Sample template
---------------

.. note::

    Before reading this section, it is highly recommended to check out :ref:`Tutorials` for more detailed examples in notebooks.


For either simulation or training, a skeleton of the script for running PPFL can be written as follows:

.. code-block:: python
    :linenos:

    from appfl.config import Config
    from omegaconf import OmegaConf

    def main():

        # load default configuration
        cfg = OmegaConf.structured(Config)

        # change configuration if needed
        ...

        # define model, loss, and data
        model = ... # user-defined model
        loss_fn = ... # user-defined loss function
        data = ... # user-defined datasets

        # The choice of PPFL runs

    if __name__ == "__main__":
        main()


Some remarks are made as follows:

- Line 8 loads the default configuration for PPFL run. See :ref:`How to set configuration` for more details.
- User-defined model, loss, and data can be read as in lines 14 to 16; see :ref:`How to define model` and :ref:`How to define datasets` for more details.
- Depending on our choice of PPFL run, we then call API functions in line 12.

.. toctree::
   :maxdepth: 1
   :titlesonly:

   user_model
   user_data
   ../notebooks/how_to_config
