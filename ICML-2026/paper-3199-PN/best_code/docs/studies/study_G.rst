..
   Software Name : learning-parities-with-product-networks
   SPDX-FileCopyrightText: Copyright (c) 2026 Orange S.A.
   SPDX-License-Identifier: MIT

   This software is distributed under the MIT License,
   see the "LICENSE.md" file for more details or https://opensource.org/licenses/MIT

   Author: Guillaume Larue, guillaume.larue@orange.com
   Software description: Source code of the paper "Learning High-Dimensional Parity Functions with Product Networks"
   
Study G — Joint impact of :math:`\alpha`, :math:`p_e`, and batch size for different :math:`N`
==============================================================================================

For each network size :math:`N \in [10, 1000]`, combines a grid of
learning rates :math:`\alpha`, Bernoulli probability of ones :math:`p_e`
and batch sizes, to map out the joint influence of these
three hyperparameters on convergence.

Run with::

   python studies/run_study_G.py

Results are visualised in :doc:`/notebooks/plot_study_G`.

.. literalinclude:: run_study_G.py
   :language: python
   :caption: studies/run_study_G.py