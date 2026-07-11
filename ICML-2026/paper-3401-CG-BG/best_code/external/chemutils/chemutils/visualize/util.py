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

"""Utilities to simplify plot creation."""

import functools

import matplotlib.pyplot as plt

import numpy as onp

def create_subplots(rows, cols):
    """Creates axes if not provided."""
    def decorator(fun):
        @functools.wraps(fun)
        def new_args(*args, ax=None, axes=None, **kwargs):
            if ax is not None:
                fig = ax.get_figure()
                return fig, fun(ax, *args, **kwargs)
            if axes is not None:
                fig = onp.ravel(axes)[0].get_figure()
                return fig, fun(axes, *args, **kwargs)

            fig, axes = plt.subplots(rows, cols, layout="constrained")
            return fig, fun(axes, *args, **kwargs)

        return new_args
    return decorator
