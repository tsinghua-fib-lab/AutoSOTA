from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from ..quantizers import get_quantizer


class LinearQERA(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        q_config: dict = None,
    ) -> None:
        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        self.q_config = q_config
        if q_config is None:
            self.bypass = True
        else:
            self.bypass = q_config.get("bypass", False)
            self.is_ptq = q_config.get("is_ptq", False)

            self._w_is_quantized = False

            self.A = None
            self.B = None

            x_quantizer_config = deepcopy(q_config["x_quantizer"])
            w_quantizer_config = deepcopy(q_config["w_quantizer"])

            self.x_quantizer = partial(
                get_quantizer(x_quantizer_config.pop("name")), **x_quantizer_config
            )
            self.w_quantizer = partial(
                get_quantizer(w_quantizer_config.pop("name")), **w_quantizer_config
            )

            if self.bias is not None:
                b_quantizer_config = deepcopy(
                    q_config.get("b_quantizer", q_config["x_quantizer"])
                )
                self.b_quantizer = partial(
                    get_quantizer(b_quantizer_config.pop("name")), **b_quantizer_config
                )

            A_out_quantizer_config = deepcopy(
                q_config.get("A_out_quantizer", q_config["x_quantizer"])
            )
            B_out_quantizer_config = deepcopy(
                q_config.get("B_out_quantizer", q_config["x_quantizer"])
            )
            self.A_out_quantizer = partial(
                get_quantizer(A_out_quantizer_config.pop("name")),
                **A_out_quantizer_config
            )
            self.B_out_quantizer = partial(
                get_quantizer(B_out_quantizer_config.pop("name")),
                **B_out_quantizer_config
            )

    def forward(self, x):
        if self.bypass:
            return F.linear(x, self.weight, self.bias)
        else:
            if self.is_ptq:
                x = self.x_quantizer(x)
                if not self._w_is_quantized:
                    self.weight.copy_(self.w_quantizer(self.weight))
                    if self.bias is not None:
                        self.bias.copy_(self.b_quantizer(self.bias))
                    self._w_is_quantized = True
                out = F.linear(x, self.weight, self.bias)
                if self.A is not None and self.B is not None:
                    xA = self.A_out_quantizer(torch.matmul(x, self.A))
                    xAB = self.B_out_quantizer(torch.matmul(xA, self.B))
                    return out + xAB
                else:
                    return out
            else:
                x = self.x_quantizer(x)
                w = self.w_quantizer(self.weight)
                bias = self.b_quantizer(self.bias) if self.bias is not None else None
                out = F.linear(x, w, bias)

                if self.A is not None and self.B is not None:
                    xA = self.A_out_quantizer(torch.matmul(x, self.A))
                    xAB = self.B_out_quantizer(torch.matmul(xA, self.B))
                    out += xAB
                    return out
                else:
                    return out

    @staticmethod
    def _get_quantizer_name(quantizer):
        if quantizer is None:
            return "None"
        elif isinstance(quantizer, partial):
            return quantizer.func.__name__
        else:
            return quantizer.__name__

    def __repr__(self):
        if self.bypass:
            txt = "{}(in_features={}, out_features={}, bias={}, bypass={})".format(
                self.__class__.__name__,
                self.in_features,
                self.out_features,
                self.bias is not None,
                self.bypass,
            )
        else:
            if self.A is None or self.B is None:
                txt = "{}(in_features={}, out_features={}, bias={}, is_ptq={}, QERA_enabled={}, x_quantizer={}, w_quantizer={})".format(
                    self.__class__.__name__,
                    self.in_features,
                    self.out_features,
                    self.bias is not None,
                    self.is_ptq,
                    self.A is not None and self.B is not None,
                    self._get_quantizer_name(self.x_quantizer),
                    self._get_quantizer_name(self.w_quantizer),
                )
            else:
                txt = "{}(in_features={}, out_features={}, bias={}, is_ptq={}, QERA_enabled={}, x_quantizer={}, w_quantizer={}, rank={}, xA_quantizer={}, xAB_quantizer={}".format(
                    self.__class__.__name__,
                    self.in_features,
                    self.out_features,
                    self.bias is not None,
                    self.is_ptq,
                    self.A is not None and self.B is not None,
                    self._get_quantizer_name(self.x_quantizer),
                    self._get_quantizer_name(self.w_quantizer),
                    self.A.shape[1],
                    self._get_quantizer_name(self.A_out_quantizer),
                    self._get_quantizer_name(self.B_out_quantizer),
                )
        return txt
