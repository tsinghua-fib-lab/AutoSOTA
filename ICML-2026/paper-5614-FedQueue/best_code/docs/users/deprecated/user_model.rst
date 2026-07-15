How to define model
===================

User-defined models can be anything derived from ``torch.nn.Module``.
For example, we can define a convolutional neural network (CNN) as follows:

.. literalinclude:: /../examples/models/cnn.py
    :language: python
    :caption: User-defined convolutionsl neural network model: examples/models/cnn.py

In the code, one can create the CNN model and the loss function instances as follows:

.. code-block:: python

    model = CNN()
    loss_fn = torch.nn.CrossEntropyLoss()

Note that the ``loss_fn`` can be anything derived from ``torch.nn.Module`` as well.
