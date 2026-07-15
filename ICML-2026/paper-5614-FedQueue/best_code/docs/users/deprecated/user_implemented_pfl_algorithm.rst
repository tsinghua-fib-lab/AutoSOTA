Personalization algorithms
==========================

Various methods for privacy-preserving personalization @ FL server and client
-----------------------------------------------------------------------------
APPFL supports implementation of personalization schemes, wherein certain layers are marked as personalized and remain local to the client. Personalized layers are updated locally at each client and do not participate in the FL process, thereby ensuring data localization and privacy.

In personalized FL, model layers are either 'shared' or 'personalized'. The server maintains a global copy of shared layers, and updates it just like classical FL. On the other hand, the clients receive shared layer weights from the server on each server epoch, using it to replace its local shared layer weights, and carries out local updates as usual. Following the same, the clients only communicate their shared layer weights back to the server. Optionally, the client can add differentially private noise to the communicated weights for privacy-preservation. Personalization has been shown to combat data heterogeneity, and enjoys **lower total noise** added to the model during training since noise is only added to shared layers.

Personalization can be enabled by populating the configuration parameter ``cfg.p_layers`` with **PyTorch layer names** of the layers meant to be personalized. Furthermore, one should use one of the following as the base class for client optimizer:

- |PersonalizedClientOptim|_: personalization layers (loss captures full dataset)
- |PersonalizedClientStepOptim|_: personalization layers (loss captures stochastic minibatch)

Both methods support `differential privacy <https://arxiv.org/abs/2312.00036>`_ when the ``cfg.fed.args.use_dp`` flag is set.

This is an example configuration:

.. code-block:: python

    cfg.personalization = True
    cfg.p_layers = ["personalized_layer_1", "personalized_layer_2"]
    cfg.fed.clientname = "PersonalizedClientOptim"
    cfg.fed.args.use_dp = True
    cfg.fed.args.clip_value = 100.
    cfg.fed.args.clip_norm = 1 # laplace mechanism requires 1-norm

A detailed example is available in our Github repository: `examples/personalization_fedloadforecast.py`

.. |PersonalizedClientOptim| replace:: ``PersonalizedClientOptim``
.. _PersonalizedClientOptim: https://arxiv.org/abs/2309.13194
.. |PersonalizedClientStepOptim| replace:: ``PersonalizedClientStepOptim``
.. _PersonalizedClientStepOptim: https://arxiv.org/abs/2309.13194

Implementation of `differential privacy <https://link.springer.com/chapter/10.1007/978-3-540-79228-4_1>`_ requires that client-to-server updates are clipped to a constant upper bound. This can be achieved by setting ``cfg.fed.args.clip_value`` and ``cfg.fed.args.clip_norm``, and specifying the norm-type by setting ``cfg.fed.args.clip_norm``.
Currently, APPFL supports `Laplace mechanism <https://link.springer.com/chapter/10.1007/978-3-540-79228-4_1>`_ for differential privacy, which requires usage of 1-norm for privacy guarantees to be meaningful.

The final privacy budget is given as ``cfg.fed.args.clip_value`` / ``cfg.fed.args.clip_norm``.

Please see ``How to set configuration`` for more details about setting configuration parameters. Theoretical details are included in the linked papers.
