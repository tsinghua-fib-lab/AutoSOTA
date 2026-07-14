from networks.network_interface import Network
from networks.layers import *
from networks.activation_function import *


class BP_network(Network):
    def __init__(self, config, name="BP_network") -> None:
        super().__init__(BP_layer, Softplus, Linear, config, name)

    def backward(self, _):
        self.loss.backward()

    def complete_task(self, _):
        pass
