import abc
from typing import Dict, Union, OrderedDict, Tuple


class BaseAggregator:
    def set_client_sample_size(self, client_id: Union[str, int], sample_size: int):
        """Set the sample size of a client"""
        if not hasattr(self, "client_sample_size"):
            self.client_sample_size = {}
        self.client_sample_size[client_id] = sample_size

    @abc.abstractmethod
    def aggregate(
        self, *args, **kwargs
    ) -> Union[Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
        """
        Aggregate local model(s) from clients and return the global model
        """
        pass

    @abc.abstractmethod
    def get_parameters(
        self, **kwargs
    ) -> Union[Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
        """Return global model parameters"""
        pass
