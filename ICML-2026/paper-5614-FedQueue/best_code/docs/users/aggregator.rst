APPFL Aggregator
================

Functionalities
---------------

The APPFL Aggregator is used for aggregating one or more client local model(s) to update the global model. Depending on the synchroneity of the FL algorithm, the aggregator can take

- only one client model to update the global model (for asynchronous FL)
- a list of client models to update the global model (for synchronous FL)
- one or a list of client models to update the global model (for asynchronous FL)


The aggregator has the following three main functionalities.

.. code:: python

    class BaseAggregator:

        @abc.abstractmethod
        def aggregate(self, *args, **kwargs) -> Union[Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
            """
            Aggregate local model(s) from clients and return the global model
            """
            pass

        @abc.abstractmethod
        def get_parameters(self, **kwargs) -> Union[Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
            """
            Return global model parameters
            """
            pass

        def set_client_sample_size(self, client_id: Union[str, int], sample_size: int):
            """
            Set the sample size of a client
            """
            if not hasattr(self, "client_sample_size"):
                self.client_sample_size = {}
            self.client_sample_size[client_id] = sample_size

To define a new aggregate, you need to inherit the ``BaseAggregator`` class and implement the ``aggregate`` and ``get_parameters`` methods. You can also define additional methods as needed.

Available Aggregators
---------------------

All available aggregators are defined in the ``appfl.algorithm.aggregator`` module, including:

- ``FedAvgAggregator``: [Synchronous] Federated Averaging (FedAvg) aggregator
- ``FedAvgMAggregator``: [Synchronous] Federated Averaging with Momentum (FedAvgM) aggregator
- ``FedYogiAggregator``: [Synchronous] Federated Yogi (FedYogi) aggregator
- ``FedAdamAggregator``: [Synchronous] Federated Adam (FedAdam) aggregator
- ``FedAdagradAggregator``: [Synchronous] Federated Adagrad (FedAdagrad) aggregator
- ``FedAsyncAggregator``: [Asynchronous] Federated Asynchronous (FedAsync) aggregator
- ``FedBuffAggregator``: [Asynchronous] Federated Buffered (FedBuff) aggregator
- ``FedCompassAggregator``: [Asynchronous] FedCompass aggregator
- ``ICEADMMAggregator``: [Synchronous] ICEADMM aggregator
- ``IIADMMAggregator``: [Synchronous] IIADMM aggregator
