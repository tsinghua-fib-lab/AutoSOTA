import torch.nn as nn
import torch.nn.functional as F

from networks.layer_interface import Layer


class DFC_layer(Layer):
    def __init__(self, in_features, out_features, activation_fn, name="DFC_layer"):
        super(DFC_layer, self).__init__(in_features, out_features, activation_fn, name)

    def backward(self):
        teaching_signal = self.r - self.r_ff

        self._weights.grad = (
            -2
            / self.r_prev.shape[0]
            * teaching_signal.t().mm(self.r_prev.view(self.r_prev.shape[0], -1))
        )
        self._bias.grad = -2 * teaching_signal.mean(dim=0)

    def compute_layerwise_jacobian(self):
        """
        Compute the Jacobian of this layer's output w.r.t. its input.
        """
        deriv = self.activation_derivative(self.v_ff)
        J = deriv.unsqueeze(-1) * self.weights.unsqueeze(0)

        return J


class BP_layer(Layer):
    def __init__(
        self, in_features, out_features, activation_fn, name="BP_layer"
    ) -> None:
        super(BP_layer, self).__init__(in_features, out_features, activation_fn, name)
