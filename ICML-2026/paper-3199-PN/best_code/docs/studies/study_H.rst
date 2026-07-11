..
   Software Name : learning-parities-with-product-networks
   SPDX-FileCopyrightText: Copyright (c) 2026 Orange S.A.
   SPDX-License-Identifier: MIT

   This software is distributed under the MIT License,
   see the "LICENSE.md" file for more details or https://opensource.org/licenses/MIT

   Author: Guillaume Larue, guillaume.larue@orange.com
   Software description: Source code of the paper "Learning High-Dimensional Parity Functions with Product Networks"

Study H — Product Node vs MLP: Sparse vs Full-Table Training
=============================================================

Compares a **product node** with a standard **MLP** under two training regimes:

- **Sparse:** Bernouilli sampled inputs with :math:`p_e = 1/N` (very few active bits per example).
- **Full:** inputs sampled uniformly from the complete truth table (:math:`2^N` rows).

Both models are evaluated on the **full truth table** (generalisation accuracy).

The product node should generalise perfectly from very few examples thanks to its
multiplicative inductive bias, while the MLP is expected to require substantially
more coverage to learn the parity function.

Run with::

   python studies/run_study_H.py

Results are visualised in :doc:`/notebooks/plot_study_H`.

.. literalinclude:: run_study_H.py
   :language: python
   :caption: studies/run_study_H.py
