from abc import ABC, abstractmethod


class RatingMetric(ABC):

    def __init__(self, config, writer_queue, device):
        self.config = config
        self.writer_queue = writer_queue
        self.device = device

    @abstractmethod
    def rate(self, generations, baselines=None) -> None:
        pass

    def __call__(self, generations, baselines=None) -> None:
        return self.rate(generations, baselines)
