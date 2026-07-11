..
   Software Name : learning-parities-with-product-networks
   SPDX-FileCopyrightText: Copyright (c) 2026 Orange S.A.
   SPDX-License-Identifier: MIT

   This software is distributed under the MIT License,
   see the "LICENSE.md" file for more details or https://opensource.org/licenses/MIT

   Author: Guillaume Larue, guillaume.larue@orange.com
   Software description: Source code of the paper "Learning High-Dimensional Parity Functions with Product Networks"

Study C — Impact of :math:`\alpha` (learning rate)
===================================================

Sweeps the learning rate :math:`\alpha \in [0.01, 100]` for a fixed
network size :math:`N=100` and optimal Bernoulli probability of ones
:math:`p_e = 1/N = 0.01`, to characterise how :math:`\alpha` affects
convergence.

Run with::

   python studies/run_study_C.py

Results are visualised in :doc:`/notebooks/plot_study_C`.

.. literalinclude:: run_study_C.py
   :language: python
   :caption: studies/run_study_C.py