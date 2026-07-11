# MIT License
#
# Copyright (c) 2025 tummfm
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Union

import haiku as hk

import e3nn_jax as e3nn
from jax import nn


class ScaleShiftLayer(hk.Module):
    """Scales and shifts the input by learnable parameters."""

    def __init__(self,
                 scale: Union[float, bool] = True,
                 shift: Union[float, bool] = True):
        super().__init__()

        self.scale = self.get_param("scale", scale, 1.0)
        self.shift = self.get_param("shift", shift, 0.0)

    @staticmethod
    def get_param(name, value, default):
        if isinstance(value, bool):
            if not value:
                return default
            else:
                value = default

        return hk.get_parameter(name, shape=(), init=hk.initializers.Constant(value))

    def __call__(self, x: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        return nn.softplus(self.scale) * x + self.shift

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(scale={self.scale:.6f}, shift={self.shift:.6f})"
        )
