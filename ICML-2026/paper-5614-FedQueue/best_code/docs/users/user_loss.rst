How to define loss
==================

User-defined loss can be anything derived from ``torch.nn.Module`` with defined ``forward`` method, which takes the model output ``prediction`` and label ``target`` as input and returns the loss value. The loss value should be a scalar tensor. For example, we can define the mean absolute scaled error (MASE) loss as follows:

.. code-block:: python

    import torch
    import torch.nn as nn

    class MASELoss(nn.Module):
        '''Mean Absolute Scaled Error Loss'''
        def __init__(self, min_number=1e-8):
            super(MASELoss, self).__init__()
            self.min_number = min_number # floor for denominator to prevent inf losses

        def forward(self, prediction, target):
            numerator = torch.mean( torch.abs(prediction-target) )
            denominator = torch.mean( torch.abs(torch.diff(target),n=1) )
            denominator = torch.maximum(denominator,torch.mul(torch.ones_like(denominator),self.min_number))
            return torch.divide(numerator,denominator)

To use the loss function during the training, you need to provide the absolute/relative path to the loss definition file and the name of the loss class. For example, to use the MASE loss defined above, you can add the following lines to the server configuration file:

.. code-block:: yaml

    client_configs:
        train_configs:
            ...
            # Loss function
            loss_fn_path: "<path_to_mase_loss>.py"
            loss_fn_name: "MASELoss"
            loss_fn_kwargs:
                min_number: 1e-8
        ...
