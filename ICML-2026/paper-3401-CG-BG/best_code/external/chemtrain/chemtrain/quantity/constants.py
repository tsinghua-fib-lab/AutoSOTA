# Copyright 2023 Multiscale Modeling of Fluid Materials, TU Munich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Chemtrain unit conversion utilities.

This module provides classes to convert between different units in chemistry.
Through JAX, M.D., chemtrain mostly relies on a
`self-consistent unit system <https://hoomd-blue.readthedocs.io/en/stable/units.html>`_.

"""

import math

import jax
try:
    from jax.typing import ArrayLike
except:
    from typing import Any
    ArrayLike = Any

class Constant:

    def __init__(self, factor: float = 1.0, exponent: int = 0):
        self.factor = factor
        self.exponent = exponent

    def __mul__(self, other):
        if isinstance(other, Constant):
            factor = other.factor * self.factor
            exponent = other.exponent + self.exponent
            exponent += int(math.log10(factor))
            factor /= 10 ** int(math.log10(factor))
            return Constant(factor, exponent)
        else:
            return self.multiply(other)

    def __rmul__(self, other):
        return self.multiply(other)

    def __truediv__(self, other):
        if isinstance(other, Constant):
            factor = other.factor * self.factor
            exponent = other.exponent + self.exponent
            exponent += int(math.log10(factor))
            factor /= 10 ** int(math.log10(factor))
            return Constant(factor, exponent)
        else:
            return self.divide(other)

    def __rtruediv__(self, other):
        return self.divide(other)

    def __pow__(self, power, modulo=None):
        factor = self.factor ** power
        exponent = power * self.exponent
        exponent += int(math.log10(factor ** 0.1))
        factor /= 10 ** int(math.log10(factor))
        return Constant(factor, exponent)

    def __repr__(self):
        return f"{self.factor} * 10^{self.exponent}"

    def multiply(self, value):
        return value * self.factor * (10 ** self.exponent)

    def divide(self, value):
        return value / self.factor * (0.1 ** self.exponent)


class Pressure:

    def __init__(self):
        self._pressure = 0.0

    @property
    def bar(self):
        """Pressure in bar."""
        return self._pressure * 16.60539067

    @bar.setter
    def bar(self, pressure):
        pressure = pressure / 16.60539067
        self._pressure = pressure

    @property
    def pressure(self):
        """Pressure in the base unit. """
        return self._pressure

    @pressure.setter
    def pressure(self, pressure):
        self._pressure = pressure

    def __repr__(self):
        return f"{self._pressure} kJ/(mol * nm^3)"


class Temperature:

    def __init__(self):
        self._temperature = 0.0

    @property
    def temperature(self):
        """Temperature in the base unit."""
        return self._temperature

    @temperature.setter
    def temperature(self, temperature):
        self._temperature = temperature

    @property
    def kelvin(self):
        """Temperature in kelvin. """
        return self._temperature / kb

    @kelvin.setter
    def kelvin(self, temperature):
        self._temperature = temperature * kb


class Density:

    def __init__(self):
        self._density = 0.0

    @property
    def density(self):
        """Density in the base unit."""
        return self._density

    @density.setter
    def density(self, density):
        self._density = density

    @property
    def g_per_cm3(self):
        """Density in grams per cm cubed. """
        return self._density * u.factor

    @g_per_cm3.setter
    def g_per_cm3(self, density):
        self._density = density / u.factor


kb = 0.0083145107 # Boltzmann constant in kJ/(mol * K)
u = Constant(1.6605, -27) # In kg
mol = Constant(6.02214076, 23)
