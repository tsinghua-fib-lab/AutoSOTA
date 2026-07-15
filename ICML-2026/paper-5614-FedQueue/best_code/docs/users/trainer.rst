APPFL Trainer
=============

APPFL local trainer is the main building blocking of the APPFL client agent for training the model locally. All trainers are inherited from the base class ``BaseTrainer``. If user wants to implement a custom trainer, they need to inherit from the ``BaseTrainer`` and implement the abstract methods ``get_parameters`` and ``train``.

.. code:: python

    class BaseTrainer:
        def __init__(
            self,
            model: Optional[nn.Module]=None,
            loss_fn: Optional[nn.Module]=None,
            metric: Optional[Any]=None,
            train_dataset: Optional[Dataset]=None,
            val_dataset: Optional[Dataset]=None,
            train_configs: DictConfig = DictConfig({}),
            logger: Optional[Any]=None,
            **kwargs
        ):
            """
            BaseTrainer:
                Abstract base trainer for FL clients.
            Args:
                model: torch neural network model to train
                loss_fn: loss function (can be as nn.Module) for the model training
                metric: metric function for the model evaluation
                train_dataset: training dataset
                val_dataset: validation dataset
                train_configs: training configurations
                logger: logger for the trainer
            """

        @abc.abstractmethod
        def get_parameters(self) -> Union[Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
            """
            Return local model parameters and optional metadata to be send for
            server and used by the server aggregator.
            """

        @abc.abstractmethod
        def train(self):
            """
            Train the model.
            """

For the input parameters of the trainer (e.g., model, loss_fn), they are loaded from the client-specific configuration file as well as the configurations sent by the server, and processed by the ``ClientAgent`` before initializing the trainer.

.. note::

    - If you find that some parameters are useless for your usecase, for example, your trainer works well for a hardcoded loss function and evaluation metric function, just ignore the loss function part in the configuration file and it will be loaded as ``None`` in the trainer.
    - If you need additional parameters for your trainer, just put them under ``client_configs.train_configs`` in the configuration file and they will be passed to the trainer as ``**kwargs``.

    .. code:: yaml

        client_configs:
            train_configs:
                # Local trainer
                trainer: "VanillaTrainer"
                mode: "step"
                num_local_steps: 100
                optim: "Adam"
                optim_args:
                lr: 0.001
                # Loss function
                loss_fn_path: "./loss/celoss.py"
                loss_fn_name: "CELoss"
                # Client validation
                do_validation: True
                do_pre_validation: True
                metric_path: "./metric/acc.py"
                metric_name: "accuracy"
                # Differential privacy
                use_dp: False
                epsilon: 1
                clip_grad: False
                clip_value: 1
                clip_norm: 1
                # Data loader
                train_batch_size: 64
                val_batch_size: 64
                train_data_shuffle: True
                val_data_shuffle: False
                # Any configuration parameters you need
                ...
