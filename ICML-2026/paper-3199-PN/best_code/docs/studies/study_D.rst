..
   Software Name : learning-parities-with-product-networks
   SPDX-FileCopyrightText: Copyright (c) 2026 Orange S.A.
   SPDX-License-Identifier: MIT

   This software is distributed under the MIT License,
   see the "LICENSE.md" file for more details or https://opensource.org/licenses/MIT

   Author: Guillaume Larue, guillaume.larue@orange.com
   Software description: Source code of the paper "Learning High-Dimensional Parity Functions with Product Networks"

Study D — Optimal :math:`\alpha` for different :math:`N`
=========================================================

For each network size :math:`N \in [10, 1000]`, sweeps the learning rate
:math:`\alpha \in [0.1, 10]` with Bernoulli probability of ones
:math:`p_e = 1/N`, to locate the optimal :math:`\alpha` as a function
of :math:`N`.

Run with::

   python studies/run_study_D.py

Results are visualised in :doc:`/notebooks/plot_study_D`.

.. literalinclude:: run_study_D.py
   :language: python
   :caption: studies/run_study_D.py