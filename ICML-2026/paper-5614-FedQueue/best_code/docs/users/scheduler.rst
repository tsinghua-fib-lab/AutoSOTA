APPFL Scheduler
===============

APPFL scheduler is the interface between the communicator and the aggregator. Whenever the communicator receives the local model from a single client, it directly hands the local model to the scheduler, and the scheduler will decide when to aggregate the local model. Currently, APPFL supports three scheduler:

- :ref:`Synchronous Scheduler`
- :ref:`Vanilla Asynchronous Scheduler`
- :ref:`Compass Asynchronous Scheduler`

All schedulers are inherited from the base class ``BaseScheduler``. If user wants to implement a new scheduler, the user needs to inherit the ``BaseScheduler`` and implement the ``schedule`` and ``get_num_global_epochs`` methods.

.. note::

    The ``scheduler_configs`` should be passed in the server configuration YAML file under the key: ``server_configs.scheduler_kwargs``, and the type of scheduler to use should be passed under the key: ``server_configs.scheduler``.

.. code:: python

    class BaseScheduler:
        def __init__(
            self,
            scheduler_configs: DictConfig,
            aggregator: BaseAggregator,
            logger: Any
        ):
            """
            Initialize the scheduler.
            :param scheduler_configs: the configurations for the scheduler
            :param aggregator: the aggregator for aggregating the local models
            :param logger: the logger for logging the information
            """

        @abc.abstractmethod
        def schedule(self, client_id: Union[int, str], local_model: Union[Dict, OrderedDict], **kwargs) -> Union[Future, Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
            """
            Schedule the global aggregation for the local model from a client.
            :param local_model: the local model from a client
            :param client_idx: the index of the client
            :param kwargs: additional keyword arguments for the scheduler
            :return: the aggregated model or a future object for the aggregated model
            """

        @abc.abstractmethod
        def get_num_global_epochs(self) -> int:
            """
            Return the total number of global epochs for federated learning.
            """

        def get_parameters(self, **kwargs) -> Union[Future, Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
            """
            Return the global model to the clients. For the initial global model, the method can
            block until all clients have requested the initial global model to make sure all clients
            can get the same initial global model (if setting `same_init_model=True` in scheduler configs
            and `kwargs['init_model']=True`).
            :params `kwargs['init_model']` (default is `True`): whether to get the initial global model or not
            :return the global model or a `Future` object for the global model
            """

Synchronous Scheduler
---------------------

Synchronous scheduler waits for all clients to submit their local models before aggregating the local models.

.. code:: python

    class SyncScheduler(BaseScheduler):
        def __init__(
            self,
            scheduler_configs: DictConfig,
            aggregator: Any,
            logger: Any
        ):
            """
            Initialize the scheduler.
            :param scheduler_configs: the configurations for the scheduler
            :param aggregator: the aggregator for aggregating the local models
            :param logger: the logger for logging the information
            """

        def schedule(self, client_id: Union[int, str], local_model: Union[Dict, OrderedDict], **kwargs) -> Future:
            """
            Schedule a synchronous global aggregation for the local model from a client.
            The method will return a future object for the aggregated model, which will
            be set after all clients have submitted their local models for the global aggregation.
            :param client_id: the id of the client
            :param local_model: the local model from a client
            :param kwargs: additional keyword arguments for the scheduler
            :return: the future object for the aggregated model
            """

        def get_num_global_epochs(self) -> int:
            """
            Get the number of global epochs.
            :return: the number of global epochs
            """

Vanilla Asynchronous Scheduler
------------------------------

Vanilla asynchronous scheduler aggregates the local models from the clients as soon as the local model is received.

.. code:: python

    class AsyncScheduler(BaseScheduler):
        def __init__(
            self,
            scheduler_configs: DictConfig,
            aggregator: Any,
            logger: Any
        ):
            """
            Initialize the scheduler.
            :param scheduler_configs: the configurations for the scheduler
            :param aggregator: the aggregator for aggregating the local models
            :param logger: the logger for logging the information
            """

        def schedule(self, client_id: Union[int, str], local_model: Union[Dict, OrderedDict], **kwargs) -> Union[Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
            """
            Schedule an asynchronous global aggregation for the local model from a client.
            The method will return the aggregated model immediately after the local model is submitted.
            :param local_model: the local model from a client
            :param client_id: the index of the client
            :param kwargs: additional keyword arguments for the scheduler
            :return: global_model: the aggregated model
            """

        def get_num_global_epochs(self) -> int:
            """
            Return the total number of global epochs for federated learning.
            """

Compass Asynchronous Scheduler
------------------------------

Compass is COMputing Power Aware Scheduler, which is an asynchronous scheduler, which

- estimates and updates the computing power of each client on-the-fly;
- synchronizes the arrival of a group of client models by assigning different number of tasks according to estimated computing power;
- interacts with the server aggregator to update global model using one or a group of synchronized client local models.

.. code:: python

    class CompassScheduler(BaseScheduler):
        """
        Scheduler for `FedCompass` asynchronous federated learning algorithm.
        Paper reference: https://arxiv.org/abs/2309.14675
        """
        def __init__(
            self,
            scheduler_configs: DictConfig,
            aggregator: Any,
            logger: Any
        ):
            """
            Initialize the scheduler.
            :param scheduler_configs: the configurations for the scheduler
            :param aggregator: the aggregator for aggregating the local models
            :param logger: the logger for logging the information
            """

        def get_parameters(self, **kwargs) -> Union[Future, Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
            """
            Get the global model parameters for the clients.
            The `Compass` scheduler requires all clients to get the initial model at the same
            time to record a consistent start time for the clients. So we add a wrapper to the
            `get_parameters` method of the `BaseScheduler` class to record the start time.
            """

        def schedule(
                self,
                client_id: Union[int, str],
                local_model: Union[Dict, OrderedDict],
                **kwargs
            ) -> Union[Future, Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
            """
            Schedule an asynchronous global aggregation for the local model from a client
            using the `Compass` algorithm. The method will either return the current global model
            directly, or a `Future` object for the global model.
            :param `client_id`: the id of the client
            :param `local_model`: the local model from the client
            :param `kwargs`: additional keyword arguments for the scheduler
            :return: `global_model`: the global model and the number of local steps for the client
                in next round or a `Future` object for the global model
            """

        def get_num_global_epochs(self) -> int:
            """
            Return the total number of global epochs for federated learning.
            """

        def clean_up(self) -> None:
            """
            Optional function to clean up the scheduler states.
            """
