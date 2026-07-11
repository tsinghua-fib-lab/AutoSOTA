..
   Software Name : learning-parities-with-product-networks
   SPDX-FileCopyrightText: Copyright (c) 2026 Orange S.A.
   SPDX-License-Identifier: MIT

   This software is distributed under the MIT License,
   see the "LICENSE.md" file for more details or https://opensource.org/licenses/MIT

   Author: Guillaume Larue, guillaume.larue@orange.com
   Software description: Source code of the paper "Learning High-Dimensional Parity Functions with Product Networks"

Run All Studies
===============

Convenience script that executes every individual study script in sequence.
Each study is run as a subprocess so that its global state (seeds, device
selection, etc.) is fully isolated. :math:`\quad`

Run all studies::

   python studies/run_all_studies.py

Run a subset (e.g. only A, C, and F)::

   python studies/run_all_studies.py A C F

.. literalinclude:: run_all_studies.py
   :language: python
   :caption: studies/run_all_studies.py
