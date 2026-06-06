from abc import ABC, abstractmethod


### Baseline setup for different causal inference models
class BaselineModel(ABC):
    def __init__(self, hpo: bool = True):
        self.hpo = hpo

    @abstractmethod
    def estimate_ate(self, X, t, y): ...

    @abstractmethod
    def estimate_cate(self, X_train, t_train, y_train, X_test): ...
