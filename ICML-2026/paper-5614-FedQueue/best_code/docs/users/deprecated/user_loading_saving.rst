Checkpointing
=============

Loading
-------
Pretrained ML models can be loaded in APPFL via ``torch.load()``.

The model parameters obtained from the pretrained model can be used as an initial guess of the FL model parameters in APPFL.
To use this feature, one should revise the corresponding fields in the configuration class ``Config``.
For example, suppose that we have a pretrained model stored as ``model_pretrained.pt`` in a ``examples/models`` directory.
Then, one can revise the configurations as follows:

.. code-block:: python

    # Loading Configurations
    from OmegaConf import OmegaConf
    from appfl.config import Config
    cfg = OmegaConf.structured(Config)
    # Loading Models
    cfg.load_model = True
    cfg.load_model_dirname = "./models"
    cfg.load_model_filename = "model_pretrained"

Saving
------
After federated learning, the resulting models can be stored via ``torch.save()``.
To use this feature, one should revise the configuration accordingly as well. See the following for an example:

.. code-block:: python

    # Saving Models
    cfg.save_model = True
    cfg.save_model_dirname = "./save_models"
    cfg.save_model_filename = "model"
    cfg.checkpoints_interval = 2

By setting ``checkpoints_interval = 2``, trained model will be saved for every 2 iteration.

.. note::

    When using ``docker container``, one can download the trained models via ``docker cp`` (https://docs.docker.com/engine/reference/commandline/cp/).

    For example, if my container ID is ``aa90d20f96c0d143012d2e6ca7d7820ed9ed8a36b163cddf8bfd6dd0e6228dab`` then

	.. code-block:: bash

		docker cp aa90d20f96c0d143012d2e6ca7d7820ed9ed8a36b163cddf8bfd6dd0e6228dab:/APPFL/save_models/ .
