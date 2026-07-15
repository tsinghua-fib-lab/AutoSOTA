Synchronous algorithms
======================

Various synchronous global model update methods @ FL server
-----------------------------------------------------------
We have implemented various synchronous global model update methods taken place at the FL server given local model parameters received from clients, which are

- ``ServerFedAvg``            : averaging local model parameters to update global model parameters
- ``ServerFedAvgMomentum``    : ServerFedAvg with a momentum
- ``ServerFedAdagrad``        : use of the adaptive gradient (Adagrad) algorithm for a global update at a server
- ``ServerFedAdam``           : use of the adaptive moment estimation (Adam) algorithm for a global update
- ``ServerFedYogi``           : use of the Yogi algorithm for a global update

One can set which algorithm to use by setting ``servername`` in ``cfg.fed`` (e.g., ``cfg.fed.servername='ServerFedAvgMomentum'``).
One can also configure the hyperparameters for each algorithm, as shown in ``appfl/config/federated.py``.

.. literalinclude:: /../src/appfl/config/fed/federated.py
    :language: python
    :lines: 31-40
    :caption: configurations of synchronous global update methods

Roughly speaking, the global update is done as follow:

``global_model_parameter`` += (``server_learning_rate`` * ``m``) / ( sqrt(``v``) + ``server_adapt_param``)

where

``server_learning_rate``: learning rate for the global update

``server_adapt_param``: adaptivity parameter

``m``: momentum

- ``m`` = ``server_momentum_param_1`` * ``m`` + (1- ``server_momentum_param_1``) * ``PseudoGrad``
- ``PseudoGrad`` : pseudo gradient obtained by averaging differences of global and local model parameters

``v``: variance

- For ``ServerFedAdagrad``:
    ``v`` = ``v`` + (``PseudoGrad``)^2
- For ``ServerFedAdam``:
    ``v`` = ``server_momentum_param_2`` * ``v`` + (1- ``server_momentum_param_2``)* (``PseudoGrad``)^2
- For ``ServerFedYogi``:
    ``v`` = ``v`` - (1 - ``server_momentum_param_2`` )* (``PseudoGrad``)^2 sign(``v`` - (``PseudoGrad``)^2)


See the following paper for more details on the above global update techniques:

Reddi, S., Charles, Z., Zaheer, M., Garrett, Z., Rush, K., Konečný, J., Kumar, S. and McMahan, H.B., 2020. Adaptive federated optimization. arXiv preprint arXiv:2003.00295
