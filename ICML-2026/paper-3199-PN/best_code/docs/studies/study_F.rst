..
   Software Name : learning-parities-with-product-networks
   SPDX-FileCopyrightText: Copyright (c) 2026 Orange S.A.
   SPDX-License-Identifier: MIT

   This software is distributed under the MIT License,
   see the "LICENSE.md" file for more details or https://opensource.org/licenses/MIT

   Author: Guillaume Larue, guillaume.larue@orange.com
   Software description: Source code of the paper "Learning High-Dimensional Parity Functions with Product Networks"

Study F — Distribution of weights during training
==================================================

Records weight statistics at regular intervals throughout training for several values
of the initial weight Bernoulli probability :math:`p_w`, to characterise
how the weight distribution evolves.

Run with::

   python studies/run_study_F.py

Results are visualised in :doc:`/notebooks/plot_study_F`.

.. literalinclude:: run_study_F.py
   :language: python
   :caption: studies/run_study_F.py