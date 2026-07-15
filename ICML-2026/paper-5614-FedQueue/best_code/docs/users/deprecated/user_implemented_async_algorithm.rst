Asynchronous algorithms
=======================

Various asynchronous global model update methods @ FL server
------------------------------------------------------------
We have implemented various asynchronous global model update methods at the FL server. These methods are based on the following papers:

- |ServerFedAsynchronous|_: asynchronous federated learning which updates the global model once receives one local model update from a client, with a staleness factor applied.
- |ServerFedBuffer|_: buffered asynchronous federated learning which updates the global model once receives a batch of local model updates from clients, with a staleness factor applied.
- |ServerFedCompass|_: computing power-aware asynchronous federated learning which assigns various number of local steps to each client based on its computing power.

.. |ServerFedAsynchronous| replace:: ``ServerFedAsynchronous``
.. _ServerFedAsynchronous: https://arxiv.org/abs/1903.03934
.. |ServerFedBuffer| replace:: ``ServerFedBuffer``
.. _ServerFedBuffer: https://arxiv.org/abs/2106.06639
.. |ServerFedCompass| replace:: ``ServerFedCompass``
.. _ServerFedCompass: https://arxiv.org/abs/2309.14675

One can set which algorithm to use by setting ``servername`` in ``appfl/config/fed/fedasync.py`` (e.g., ``cfg.fed.servername = 'ServerFedAsynchronous'``).
One can also configure the hyperparameters for each algorithm, as shown in ``appfl/config/fed/fedasync.py``.

.. literalinclude:: /../src/appfl/config/fed/fedasync.py
    :language: python
    :lines: 41-54
    :caption: configurations of asynchronous global update methods

In asynchronous federated learning algorithms, as the server may update the global model before all clients finish their local updates, the local updates from clients may become stale. To mitigate this issue, asynchronous FL algorithms apply a staleness factor to penalize the stale local updates as follows:

``global_model_parameters`` = (1 - ``staleness_factor``) * ``global_model_parameters`` + ``staleness_factor`` * ``local_model_parameters``

where

``staleness_factor`` = ``alpha`` * ``staleness_function``(``t_global`` - ``t_local``)

- ``alpha`` is a hyperparameter to control the staleness factor
- ``t_global`` is the global model timestamp
- ``t_local`` is the local model timestamp
- ``staleness_function`` is a function to compute the staleness factor. We have implemented three staleness functions ``constant``, ``polynomial``, and ``hinge``, as shown below.

.. literalinclude:: /../src/appfl/algorithm/server_fed_asynchronous.py
    :language: python
    :lines: 42-53
    :caption: staleness functions

The application of staleness factor may cause the global model to drift away from training data of slower clients (known as client drift). To mitigate this issue,

- |ServerFedBuffer|_ employs a size-K buffer to store the local model updates from clients. The server updates the global model once it receives a batch of local model updates from clients. The size of the buffer K can be set by ``cfg.fed.args.K``.

- |ServerFedCompass|_ automatically and dynamically assigns various numbers of local steps to each client based on its computing power to make a group of clients send local models back almost simultaneously to reduce the usage global update frequency. The maximum ratio of the number of local steps for clients within a same group can be set by ``cfg.fed.args.q_ratio``, which will affect the grouping behavior of the FedCompass algorithm.

For more details of those asynchronous FL algorithms, please refer to the papers.
