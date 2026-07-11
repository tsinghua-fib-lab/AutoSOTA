..
   Software Name : learning-parities-with-product-networks
   SPDX-FileCopyrightText: Copyright (c) 2026 Orange S.A.
   SPDX-License-Identifier: MIT

   This software is distributed under the MIT License,
   see the "LICENSE.md" file for more details or https://opensource.org/licenses/MIT

   Author: Guillaume Larue, guillaume.larue@orange.com
   Software description: Source code of the paper "Learning High-Dimensional Parity Functions with Product Networks"

Study E — Impact of :math:`p_e` under larger batch size
===========================================================

Reproduces the sweep of Study A (:math:`p_e \in [0.001, 0.1]`,
:math:`N=100`, :math:`\alpha=0.1`) under a larger batch size, to assess
the robustness of the convergence behaviour with respect to the Bernoulli
probability of ones :math:`p_e`.

Run with::

   python studies/run_study_E.py

Results are visualised in :doc:`/notebooks/plot_study_E`.

.. literalinclude:: run_study_E.py
   :language: python
   :caption: studies/run_study_E.py