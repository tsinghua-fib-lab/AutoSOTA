..
   Software Name : learning-parities-with-product-networks
   SPDX-FileCopyrightText: Copyright (c) 2026 Orange S.A.
   SPDX-License-Identifier: MIT

   This software is distributed under the MIT License,
   see the "LICENSE.md" file for more details or https://opensource.org/licenses/MIT

   Author: Guillaume Larue, guillaume.larue@orange.com
   Software description: Source code of the paper "Learning High-Dimensional Parity Functions with Product Networks"

Study B — Optimal :math:`p_e` for different :math:`N`
======================================================

For each network size :math:`N \in [10, 1000]`, sweeps the Bernoulli
probability of ones :math:`p_e` (with fixed :math:`\alpha=0.1`) to locate the optimal :math:`p_e` as a
function of :math:`N`.

Run with::

   python studies/run_study_B.py

Results are visualised in :doc:`/notebooks/plot_study_B`.

.. literalinclude:: run_study_B.py
   :language: python
   :caption: studies/run_study_B.py