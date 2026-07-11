..
   Software Name : learning-parities-with-product-networks
   SPDX-FileCopyrightText: Copyright (c) 2026 Orange S.A.
   SPDX-License-Identifier: MIT

   This software is distributed under the MIT License,
   see the "LICENSE.md" file for more details or https://opensource.org/licenses/MIT

   Author: Guillaume Larue, guillaume.larue@orange.com
   Software description: Source code of the paper "Learning High-Dimensional Parity Functions with Product Networks"

Study A — Impact of :math:`p_e` on convergence
===============================================

Sweeps the Bernoulli probability of ones :math:`p_e` over a log-scale range
:math:`[0.001, 0.1]` for a fixed network size :math:`N=100` and learning
rate :math:`\alpha=0.1`, measuring how :math:`p_e` affects convergence speed.

Run with::

   python studies/run_study_A.py

Results are visualised in :doc:`/notebooks/plot_study_A`.

.. literalinclude:: run_study_A.py
   :language: python
   :caption: studies/run_study_A.py
