Logging
=======

The log of the FL results looks like the following:

.. code-block::

        Iter     Local[s]    Global[s]     Valid[s]      Iter[s]   Elapsed[s]  TestAvgLoss TestAccuracy     Prim_res     Dual_res    Penal_min    Penal_max
           1        12.45         0.01         1.84        14.31        14.31     0.174392        94.32   8.9469e+00   0.0000e+00         0.00         0.00
           2        11.76         0.01         1.81        13.59        27.90     0.089002        97.15   3.1964e+00   0.0000e+00         0.00         0.00
      Device=cpu
      #Processors=5
      Dataset=MNIST
      #Clients=4
      Algorithm=fedavg
      Comm_Rounds=2
      Local_Epochs=1
      DP_Eps=False
      Clipping=False
      Elapsed_time=27.9
      BestAccuracy=97.15

The contents can be modified by revising ``log_title``, ``log_iteration``, and ``log_summary`` in ``src/appfl/misc/utils.py``.


FL results are recorded in ``.txt`` in a predefined directory.
To set the directory and the filename, for example, one can revise the configurations as follows:

.. code-block::

    # Loading Configurations
    from OmegaConf import OmegaConf
    from appfl.config import Config
    cfg = OmegaConf.structured(Config)
    # FL Outputs
    cfg.output_dirname = "./outputs"
    cfg.output_filename = "result"
